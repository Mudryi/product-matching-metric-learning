import cv2
import torch.utils.data as data
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import pandas as pd
import torch
import os
import numpy as np
from torchvision import transforms
from tqdm.auto import tqdm
from src.config import product_batch_size

validation_dataset_path = "datasets/product_matching_evaluation"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


class SubmissionDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms=None, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

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
    gallery_labels = pd.read_csv(os.path.join(validation_dataset_path, 'gallery.csv'))['product_id'].values
    query_labels = pd.read_csv(os.path.join(validation_dataset_path, 'queries.csv'))['product_id'].values

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
                                    transforms=transforms.Compose([transforms.Resize((224, 224)),
                                                                   transforms.ToTensor()])
                                    )

gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=product_batch_size,
                                             shuffle=False, pin_memory=True)

query_dataset = SubmissionDataset(root=validation_dataset_path,
                                  annotation_file=os.path.join(validation_dataset_path, 'queries.csv'),
                                  transforms=transforms.Compose([transforms.Resize((224, 224)),
                                                                 transforms.ToTensor()]),
                                  with_bbox=True)

query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=product_batch_size,
                                           shuffle=False, pin_memory=True)


def product_matching_validation(model, embedding_size=768):
    gallery_embeddings = np.zeros((len(gallery_dataset), embedding_size))
    query_embeddings = np.zeros((len(query_dataset), embedding_size))

    model.to(device)
    with torch.no_grad():
        for i, images in tqdm(enumerate(gallery_loader),
                              total=len(gallery_loader)):
            images = images.to(device)

            outputs, _ = model(images)
            outputs = outputs[:, 0, :]

            outputs = outputs.data.cpu().numpy()
            gallery_embeddings[i * product_batch_size:(i * product_batch_size + product_batch_size), :] = outputs

        for i, images in tqdm(enumerate(query_loader),
                              total=len(query_loader)):
            images = images.to(device)

            outputs, _ = model(images)
            outputs = outputs[:, 0, :]

            outputs = outputs.data.cpu().numpy()
            query_embeddings[i * product_batch_size:(i * product_batch_size + product_batch_size), :] = outputs

    gallery_embeddings = normalize(gallery_embeddings)
    query_embeddings = normalize(query_embeddings)
    distances = pairwise_distances(query_embeddings, gallery_embeddings)
    sorted_distances = np.argsort(distances, axis=1)[:, :1000]
    return calculate_map(sorted_distances)
