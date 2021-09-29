import torch
from transformers import *  # https://github.com/huggingface/transformers


class BERT():
    def __init__(self, model='base', min_score=0, obj_classes=None, rel_classes=None, triplet2str=None, device='cpu'):
        model_ind = 0  # 0: Bert
        MODELS = [(BertForMaskedLM, BertTokenizer, 'bert-%s-uncased' % model)]
        print('initializing Bert %s model with threshold %f' % (MODELS[model_ind][2], min_score))
        self.tokenizer = MODELS[model_ind][1].from_pretrained(MODELS[model_ind][2])
        self.device = device
        self.model = MODELS[model_ind][0].from_pretrained(MODELS[model_ind][2]).to(self.device)
        self.model.eval()
        self.min_score = min_score
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.triplet2str = triplet2str


    def my_simple_tokenizer(self, s):
        tokenized_text = s.replace(',', ' ,').split(' ')
        tokenized_text = ['[CLS]'] + tokenized_text + ['.', '[SEP]']
        return tokenized_text


    def predict_token(self, text, masked_index, classes, topk=5, verbose=False):
        # tokenized_text = tokenizer.tokenize("[CLS] %s .[SEP]" % text)  # this tokenizer fails for some words like giraffe and surboard
        tokenized_text = self.my_simple_tokenizer(text)
        tokenized_text[masked_index] = '[MASK]'
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_tensors = [1] * len(tokenized_text)
        if verbose:
            print('BERT:', list(zip(tokenized_text, indexed_tokens, segments_tensors)))
        data = (torch.tensor([indexed_tokens]).to(self.device), torch.tensor([segments_tensors]).to(self.device))
        predictions = self.model(*data)[0].squeeze()[masked_index]  # segments_tensors
        pairs = []
        if len(predictions) > 0:

            predicted_entries = torch.topk(predictions, k=topk)
            for i, (score, ind) in enumerate(zip(*predicted_entries)):
                if score < self.min_score:
                    break
                text = self.tokenizer.decode([ind])

                if len(text) < 2 or text.startswith('#'):
                    continue

                if classes is not None and text not in classes:
                    if text.endswith('s'):
                        # check for plurals
                        text2 = text[:-1]
                        if text2 in classes:
                            pairs.append((text2, score))  # add a word without the 's' on the end, since it often returns plurals and VG classes are singular
                        else:
                            continue
                else:
                    pairs.append((text, score))

                if len(pairs) >= topk:
                    break
        return pairs


    def context(self, triplet, gt_classes, gt_rels):
        context = ''
        for o1, o2, R in gt_rels:
            tri_str = '{}_{}_{}'.format(self.obj_classes[gt_classes[o1]],
                                        self.rel_classes[R],
                                        self.obj_classes[gt_classes[o2]])
            if tri_str == triplet:
                continue
            context += ', ' + tri_str
        return context


    def bert_score(self, text, masked_index, target_text, verbose=False):
        with torch.no_grad():
            tokenized_text = self.my_simple_tokenizer(text)
            tokenized_text[masked_index] = '[MASK]'
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_tensors = [1] * len(tokenized_text)
            if verbose:
                print('BERT:', list(zip(tokenized_text, indexed_tokens, segments_tensors)))
            data = (torch.tensor([indexed_tokens]).to(self.device),
                    torch.tensor([segments_tensors]).to(self.device))
            predictions = self.model(*data)[0].squeeze()[masked_index]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids([target_text])
            score = predictions[indexed_tokens[0]]

            return score.item()


    def bert_score_triplet(self, triplet, gt_classes, gt_rels, is_subject, verbose=False):

        triplet = self.triplet2str(triplet)  # transform to words
        subj, R_name, obj = triplet.split('_')

        R_words = R_name.split(' ')
        context = self.context(triplet, gt_classes, gt_rels)
        if is_subject:
            masked_index = 2
            article1 = 'the'  # np.random.choice(['a', 'an'])  # a or an will determine if the word starts from a vowel and consonant
            article2 = 'the'
            verb = 'is' if R_name.find('ing') >= 0 else ''
        else:
            masked_index = 4 + len(R_words)
            article1 = 'the'
            article2 = 'the'  # np.random.choice(['a', 'an'])  # a or an will determine if the word starts from a vowel and consonant
            if R_name.find('ing') >= 0:
                verb = 'are' if subj in ['men', 'people'] else 'is'
                masked_index += 1
            else:
                verb = ''

        query = '{} {} {} {} {} {}{}'.format(article1, subj, verb, R_name, article2,
                                             obj, context).replace('_', ' ').replace('  ', ' ')

        score = self.bert_score(query, masked_index,target_text=subj if is_subject else obj,
                                verbose=verbose)
        if verbose:
            print('\nquery:', query, 'masked_index:', masked_index, '\nscore:', score, '\n')

        return score
