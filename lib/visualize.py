
import numpy as np
import networkx as nx
import cv2
import matplotlib.pyplot as plt
from dataloaders.visual_genome import filter_dups
from config import BOX_SCALE, IM_SCALE

rnd = np.random.RandomState(12345)
node_colors_fixed = rnd.randint(low=1, high=255, size=(1000, 3)).astype(np.uint8)

def get_color(obj, obj_name, obj_names_orig=None, fmt='array', alpha=255):

    assert fmt in ['array', 'string'], fmt

    if (obj_names_orig is None and obj_name == 'person') or (
            obj_names_orig is not None and obj_names_orig[obj] == 'person'):
        color = (30, 220, 0)  # BGR
    elif (obj_names_orig is None and obj_name == 'surfboard') or (
            obj_names_orig is not None and obj_names_orig[obj] == 'surfboard'):
        color = (0, 250, 200)
    elif (obj_names_orig is None and obj_name == 'wave') or (
            obj_names_orig is not None and obj_names_orig[obj] == 'wave'):
        color = (220, 30, 0)
    else:
        color = node_colors_fixed[obj]

    if fmt == 'string':
        return "#" + ''.join(["%0.2X" % c for c in color[::-1]]) + ("%0.2X" % alpha)
    else:
        return color


def draw_boxes(im, obj_class_names, bboxes, fontscale=0.5, lw=4, rels=None, torch_detector=False):

    if torch_detector:
        # resize both the image and boxes
        k = 512. / np.max(im.shape)
        im = cv2.resize(im, (int(im.shape[1] * k), int(im.shape[0] * k)))
        bboxes = bboxes.copy() * k
    else:
        bboxes = bboxes.copy() / BOX_SCALE * max(im.shape)

    im = ((im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)
    for obj, (cls, bbox) in enumerate(zip(obj_class_names, bboxes)):
        if rels is not None and (np.sum([rel[0] == obj for rel in rels]) +
                                 np.sum([rel[1] == obj for rel in rels])) == 0:
            continue
        bbox = np.round(bbox.copy()).astype(np.int)
        bbox[0] = np.clip(bbox[0], 1, im.shape[1] - 2)
        bbox[2] = np.clip(bbox[2], 1, im.shape[1] - 2)
        bbox[1] = np.clip(bbox[1], 1, im.shape[0] - 2)
        bbox[3] = np.clip(bbox[3], 1, im.shape[0] - 2)
        color = get_color(obj, cls)[::-1]  # RGB
        color = (int(color[0]), int(color[1]), int(color[2]))   # to get around numpy-cv2 issue
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, lw)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + len(cls) * int(fontscale * 20), bbox[1] + int(fontscale ** 0.5 * 30)), color, -1)
        cv2.putText(im, cls, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 255, 255), 2, cv2.LINE_AA)
    return im



def show_nx(objs, boxes, rels, train_set, test_set, perturbed_nodes=None,
            edge_label_pos=0.5, obj_names_orig=None, name=None, fontsize=22, torch_detector=False):
    G = nx.DiGraph()
    node_labels = {}
    node_size = []
    rels = filter_dups(rels, random_edge=False)
    node_colors = []  # BGR
    edgecolors = []
    linewidth = []

    if torch_detector:
        k = 512. / np.max(boxes)
        boxes = boxes * k

    for obj, cls in enumerate(objs):
        obj_name = train_set.ind_to_classes[cls]
        G.add_node(obj, label=obj_name)
        node_labels[obj] = obj_name

        if torch_detector:
            node_size.append(10 * ((boxes[obj][2] - boxes[obj][0]) * (boxes[obj][3] - boxes[obj][1]) ** 0.1))
        else:
            node_size.append(2000)

        # predefined colors for the paper
        node_colors.append(get_color(obj, obj_name, obj_names_orig=obj_names_orig))

        # highlight perturbed nodes
        if perturbed_nodes is not None and obj in perturbed_nodes:
            edgecolors.append([0, 0, 0, 255])
            linewidth.append(8)
        else:
            edgecolors.append([200, *node_colors[-1]])
            linewidth.append(1)

    edge_labels = {}
    edges = {}
    for rel_id, rel in enumerate(rels):
        triplet = '{}_{}_{}'.format(objs[rel[0]], rel[2], objs[rel[1]])
        is_zs = triplet in test_set.triplet_counts
        key = '{}_{}'.format(rel[1], rel[0])

        # heuristic to select a single edge between a pair of nodes for visualization
        if key in edges:
            if is_zs or edge_labels[(rel[1], rel[0])].split('-')[0] != 'near':  # for the figures in the paper
                G.remove_edge(rel[1], rel[0])
                del edge_labels[(rel[1], rel[0])]
            else:
                continue
        assert (rel[1], rel[0]) not in edge_labels, (rel, edge_labels, key, edges)
        edges['{}_{}'.format(rel[0], rel[1])] = rel_id
        G.add_edge(*rel[:2], color='red' if is_zs else 'blue', weight=3. if is_zs else 1.)
        G[rel[0]][rel[1]]['color'] = 'red' if triplet not in train_set.triplet_counts else 'blue'
        G[rel[0]][rel[1]]['weight'] = 8. if is_zs else (2. if triplet not in train_set.triplet_counts else 1.)
        edge_labels[tuple(rel[:2])] = '{}-{}'.format(train_set.ind_to_predicates[rel[2]],
                                                     train_set.triplet_counts[triplet] if triplet in train_set.triplet_counts else 0)

    pos = nx.circular_layout(G)
    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    weights = [G[u][v]['weight'] for u, v in edges]

    fig, ax = plt.subplots(figsize=(10, 5))
    nx.draw(G, pos=pos, with_labels=False, node_size=node_size,
            node_color=np.array(node_colors)[:, ::-1] / 255., alpha=0.6,
            edge_color=colors, width=weights, edgecolors=np.array(edgecolors)[:, ::-1] / 255., linewidths=np.array(linewidth),
            arrowstyle='-|>',
            arrowsize=35,
            # connectionstyle='angle3,angleA=90,angleB=0',  # does not properly place edge labels
            ax=ax)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_weight='bold',
                            font_size=max(fontsize, min(24, 50 / len(node_size))))
    nx.draw_networkx_edge_labels(G, pos=pos,
                                 label_pos=edge_label_pos,
                                 edge_labels=edge_labels, font_color='black',
                                 font_size=fontsize - 4)
    plt.xlim(-1.5, 2.5)
    plt.ylim(-1.2, 1.2)
    plt.tight_layout()
    if name is not None:
        plt.savefig('%s.png' % name, transparent=True, bbox_inches='tight')
    plt.show()