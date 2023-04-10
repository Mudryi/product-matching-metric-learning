from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import gc
import torch
import cv2
import pandas as pd
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from config import params

validation_dataset_path = "development_test_data"
batch_size = params['test_batch_size']

transform = {'mcs_eval': transforms.Compose([transforms.Resize(params['image_size']),
                                             transforms.transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             # transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                             #                      std=(0.5, 0.5, 0.5)),
                                             ])}


def report_gpu():
    torch.cuda.empty_cache()
    gc.collect()


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


gallery_labels = pd.read_csv(os.path.join(validation_dataset_path, 'gallery.csv'))['product_id'].values
query_labels = pd.read_csv(os.path.join(validation_dataset_path, 'queries.csv'))['product_id'].values


class SubmissionDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None, with_bbox=False,
                 **transform_params):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox
        self.transform_params = transform_params

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y + h, x:x + w, :]

        img = Image.fromarray(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imlist)


def compute_precision_at_k(ranked_targets: np.ndarray,
                           k: int) -> float:
    assert k >= 1
    assert ranked_targets.size >= k, ValueError('Relevance score length < k')
    return np.mean(ranked_targets[:k])


def compute_average_precision(ranked_targets: np.ndarray,
                              gtp: int) -> float:
    assert gtp >= 1
    out = [compute_precision_at_k(ranked_targets, k + 1) for k in range(ranked_targets.size) if ranked_targets[k]]
    if len(out) == 0:
        return 0.0
    else:
        return np.sum(out) / gtp


def calculate_map(ranked_retrieval_results: np.ndarray) -> float:
    class_average_precisions = []

    class_ids, class_counts = np.unique(gallery_labels, return_counts=True)
    class_id2quantity_dict = dict(zip(class_ids, class_counts))

    for gallery_indices, query_class_id in zip(ranked_retrieval_results, query_labels):
        current_retrieval = gallery_labels[gallery_indices] == query_class_id
        gpt = class_id2quantity_dict[query_class_id]
        class_average_precisions.append(compute_average_precision(current_retrieval, gpt))

    mean_average_precision = np.mean(class_average_precisions)
    return mean_average_precision


gallery_dataset = SubmissionDataset(root=validation_dataset_path,
                                    annotation_file=os.path.join(validation_dataset_path, 'gallery.csv'),
                                    transforms=transform['mcs_eval'],
                                    )

gallery_loader = torch.utils.data.DataLoader(gallery_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=4
                                             )

query_dataset = SubmissionDataset(root=validation_dataset_path,
                                  annotation_file=os.path.join(validation_dataset_path, 'queries.csv'),
                                  transforms=transform['mcs_eval'],
                                  with_bbox=True,
                                  )

query_loader = torch.utils.data.DataLoader(query_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=4
                                           )


def product_matching_validation(model, embedding_size=1024):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gallery_embeddings = np.zeros((len(gallery_dataset), embedding_size))
    query_embeddings = np.zeros((len(query_dataset), embedding_size))

    report_gpu()
    model.to(device)

    with torch.no_grad():
        for i, images in tqdm(enumerate(gallery_loader),
                              total=len(gallery_loader)):
            images = images.to(device)
            outputs = model.extract_feat(images)
            outputs = outputs.data.cpu().numpy()
            gallery_embeddings[i * batch_size:(i * batch_size + batch_size), :] = outputs

        report_gpu()
        for i, images in tqdm(enumerate(query_loader),
                              total=len(query_loader)):
            images = images.to(device)
            outputs = model.extract_feat(images)
            outputs = outputs.data.cpu().numpy()
            query_embeddings[i * batch_size:(i * batch_size + batch_size), :] = outputs

    gallery_embeddings = normalize(gallery_embeddings)
    query_embeddings = normalize(query_embeddings)
    distances = pairwise_distances(query_embeddings, gallery_embeddings)
    sorted_distances = np.argsort(distances, axis=1)[:, :1000]
    return calculate_map(sorted_distances)
