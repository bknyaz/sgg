# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import sys
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detector.engine import train_one_epoch
import detector.utils as utils
import detector.transforms as T
from dataloaders.visual_genome import VG
from PIL import Image
from lib.pytorch_misc import save_checkpoint, get_smallest_lr
from config import BOX_SCALE

VG.split = sys.argv[1]
data_dir = sys.argv[2]
save_dir = sys.argv[3]
checkpoint_name = '%s_maskrcnn_res50fpn.pth' % VG.split

class VGLoader(VG):
    def __init__(self, mode, data_dir, transforms):
        super(VGLoader, self).__init__(mode, data_dir, num_val_im=5000, filter_duplicate_rels=True,
                                            min_graph_size=-1,
                                            max_graph_size=-1,
                                            filter_non_overlap=False)
        self.transforms = transforms


    def __getitem__(self, idx):
        index = idx

        img = Image.open(self.filenames[index]).convert('RGB')
        w, h = img.size

        gt_boxes = self.gt_boxes[index].copy()

        if VG.split == 'stanford':
            # makes boxes scale the same as images
            gt_boxes = gt_boxes / (BOX_SCALE / max(w, h))

        if self.is_train:
            # crop boxes that are too large.
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(None, h)
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(None, w)

        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)

        target = {}
        target["boxes"] = gt_boxes
        target["labels"] = torch.from_numpy(self.gt_classes[index]).long()
        # target["masks"] = masks  # no mask annotations
        target["image_id"] = torch.tensor([idx])
        target["area"] = (gt_boxes[:, 3] - gt_boxes[:, 1]) * (gt_boxes[:, 2] - gt_boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(self.gt_classes[index]),), dtype=torch.int64)  # suppose all instances are not crowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def get_model_optimizer(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    model.roi_heads.mask_predictor = None  # no masks in these datasets

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,  # didn't tune these values but looks good
                                momentum=0.9, weight_decay=0.0005)

    start_epoch = -1
    if os.path.exists(checkpoint_name):
        print('loading the model and optimizer state from %s' % checkpoint_name)
        state_dict = torch.load(checkpoint_name)
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch']

    return model, optimizer, start_epoch


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 151 if VG.split == 'stanford' else 1704

    # use our dataset and defined transformations
    dataset = VGLoader('train', data_dir, get_transform(train=True))
    # dataset_test = GQALoader('val', get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=3 if VG.split == 'stanford' else 2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1, shuffle=False, num_workers=4,
    #     collate_fn=utils.collate_fn)

    # get the model using our helper function
    model, optimizer, start_epoch = get_model_optimizer(num_classes)
    print('start_epoch', start_epoch)

    # move model to the right device
    model.to(device)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(start_epoch + 1, num_epochs):
        print('\nepoch %d, smallest lr %f\n' % (epoch, get_smallest_lr(optimizer)))
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        try:
            print("\nCheckpointing to %s" % os.path.join(save_dir, checkpoint_name))
            save_checkpoint(model, optimizer, save_dir, checkpoint_name, {'epoch': epoch})
            print('done!\n')
        except Exception as e:
            print('error saving checkpoint', e)

        # update the learning rate
        lr_scheduler.step(epoch)
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)  # some issues with evaluation (check coco_eval code)

    print("That's it!")
    
if __name__ == "__main__":
    main()
