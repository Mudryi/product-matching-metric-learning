import cv2
import random
from torch.utils.data import Dataset
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from albumentations import ImageOnlyTransform
import albumentations.augmentations.geometric.functional as F
from albumentations import (
    Resize,
    Compose,
    Normalize,
    Affine,
    RandomBrightnessContrast,
    MotionBlur,
    CLAHE,
    RandomCrop,
    Rotate,
    Flip,
    ToGray,
    ColorJitter,
)
from config import params


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img

class LandmarkDataset(Dataset):
    # (width, height)
    size_template = {
        'SS': ((384, 256), (256, 256), (256, 384)),
        'SS2': ((480, 320), (480, 352), (384, 384), (352, 480), (320, 480)),
        'S': ((512, 384), (448, 448), (384, 512)),
        'S2': ((512, 352), (512, 384), (448, 448), (384, 512), (352, 512)),
        'SM2': ((736, 512), (704, 544), (608, 608), (544, 704), (512, 736)),
        'M': ((800, 608), (704, 704), (608, 800)),
        'M2': ((800, 544), (800, 608), (704, 704), (608, 800), (544, 800)),
        'L': ((800, 608), (800, 800), (608, 800)),
        'L2': ((800, 544), (800, 608), (800, 800), (608, 800), (544, 800)),
    }

    def __init__(self,
                 paths,
                 class_ids=None,
                 transform=None,
                 aspect_gids=None,
                 scale='S'
                 ):
        self.paths = paths
        self.class_ids = class_ids
        self.aspect_gids = aspect_gids
        self.transform = transform
        self.scale = scale

    def __getitem__(self, index):
        img_path = str(self.paths[index])
        img = read_image(img_path)
        img = cv2.resize(img, params['image_size'])
        # assert img is not None, f'path: {img_path} is invalid.'
        # img = img[..., ::-1]

        # if self.aspect_gids is not None:
        #     gid = self.aspect_gids[index]
        #     img = cv2.resize(img, self.size_template[self.scale][gid])

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        img_id = img_path.split('/')[-1].replace('.jpg', '')

        if self.class_ids is None:
            return img_id, img
        else:
            target = torch.tensor(self.class_ids[index]).long()
            return img_id, img, target

    def __len__(self):
        return len(self.paths)


def prepare_grouped_loader_from_df(df,
                                   transform,
                                   batch_size,
                                   scale='S',
                                   is_train=True,
                                   num_workers=4
                                   ):
    class_ids = df['class'].values if 'class' in df.columns else None
    dataset = LandmarkDataset(paths=df['path'].values,
                              class_ids=class_ids,
                              # aspect_gids=df['aspect_gid'].values,
                              aspect_gids=None,
                              transform=transform,
                              scale=scale)
    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    # gb_sampler = torch_custom.GroupedBatchSampler(sampler=sampler,
    #                                               group_ids=df['aspect_gid'].values,
    #                                               batch_size=batch_size,
    #                                               drop_uneven=is_train)

    loader = DataLoader(dataset=dataset,
                        # batch_sampler=sampler,
                        batch_size=batch_size,
                        pin_memory=True,
                        num_workers=num_workers
                        )
    return loader


def make_train_loaders(params,
                       data_root,
                       use_clean_version=False,
                       train_transform=None,
                       eval_transform=None,
                       scale='S',
                       limit_samples_per_class=-1,
                       test_size=0.1,
                       num_workers=4,
                       seed=77777
                       ):
    df = pd.read_csv("train.csv")
    # df = df.groupby('class').filter(lambda x: len(x) > 16) # 16 because we need at leat 16 samples of class in batch (4 classes, 64 batch size)
    # le = LabelEncoder()
    # df["class"] = le.fit_transform(df["class"])
    df['path'] = df['name'].apply(lambda x: f'{data_root}/{x}')

    train_split, val_split = train_test_split(df, test_size=test_size, random_state=seed)

    data_loaders = dict()
    data_loaders['train'] = prepare_grouped_loader_from_df(
        train_split, train_transform, params['batch_size'],
        scale=scale, is_train=True, num_workers=num_workers)
    data_loaders['val'] = prepare_grouped_loader_from_df(
        val_split, eval_transform, params['test_batch_size'],
        scale=scale, is_train=False, num_workers=num_workers)

    return data_loaders

def build_transforms(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225),
                     divide_by=255.0,
                     scale_limit=0.0,
                     shear_limit=0,
                     rotate_limit=0,
                     brightness_limit=0.0,
                     contrast_limit=0.0,
                     clahe_p=0.0,
                     blur_p=0.0,
                     ):
    norm = Normalize(mean=mean, std=std, max_pixel_value=divide_by)

    train_transform = Compose([
        # Affine(rotate=(-rotate_limit, rotate_limit), shear=(-shear_limit, shear_limit), mode='constant'),
        # RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit),
        # MotionBlur(p=blur_p),
        # CLAHE(p=clahe_p),
        Rotate(limit=90),
        Flip(),
        RandomCrop(height=int(params['image_size'][0]/2), width=int(params['image_size'][1]/2)),
        ColorJitter(),
        # ToGray(),
        Resize(params['image_size'][0], params['image_size'][1]),
        # RandomCropThenScaleToOriginalSize(limit=scale_limit, p=1.0),
        norm,
        ])
    eval_transform = Compose([Resize(params['image_size'][0], params['image_size'][1]), norm])

    return train_transform, eval_transform


class RandomCropThenScaleToOriginalSize(ImageOnlyTransform):
    """Crop a random part of the input and rescale it to some size.
    Args:
        limit (float): maximum factor range for cropping region size.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        pad_value (int): pixel value for padding.
        p (float): probability of applying the transform. Default: 1.
    """

    def __init__(self, limit=0.1, interpolation=cv2.INTER_LINEAR, pad_value=0, p=1.0):
        super(RandomCropThenScaleToOriginalSize, self).__init__(p)
        self.limit = limit
        self.interpolation = interpolation
        self.pad_value = pad_value

    def apply(self, img, height_scale=1.0, width_scale=1.0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR,
              pad_value=0, pad_loc_seed=None, **params):
        img_height, img_width = img.shape[:2]
        crop_height, crop_width = int(img_height * height_scale), int(img_width * width_scale)
        crop = self.random_crop(img, crop_height, crop_width, h_start, w_start, pad_value, pad_loc_seed)
        return F.resize(crop, img_height, img_width, interpolation)

    def get_params(self):
        height_scale = 1.0 + random.uniform(-self.limit, self.limit)
        width_scale = 1.0 + random.uniform(-self.limit, self.limit)
        return {'h_start': random.random(),
                'w_start': random.random(),
                'height_scale': height_scale,
                'width_scale': width_scale,
                'pad_loc_seed': random.random()}

    def update_params(self, params, **kwargs):
        if hasattr(self, 'interpolation'):
            params['interpolation'] = self.interpolation
        if hasattr(self, 'pad_value'):
            params['pad_value'] = self.pad_value
        params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
        return params

    @staticmethod
    def random_crop(img, crop_height, crop_width, h_start, w_start, pad_value=0, pad_loc_seed=None):
        height, width = img.shape[:2]

        if height < crop_height or width < crop_width:
            img = _pad_const(img, crop_height, crop_width, value=pad_value, center=False, pad_loc_seed=pad_loc_seed)

        y1 = max(int((height - crop_height) * h_start), 0)
        y2 = y1 + crop_height
        x1 = max(int((width - crop_width) * w_start), 0)
        x2 = x1 + crop_width
        img = img[y1:y2, x1:x2]
        return img


def _pad_const(x, target_height, target_width, value=255, center=True, pad_loc_seed=None):
    random.seed(pad_loc_seed)
    height, width = x.shape[:2]

    if height < target_height:
        if center:
            h_pad_top = int((target_height - height) / 2.0)
        else:
            h_pad_top = random.randint(a=0, b=target_height - height)
        h_pad_bottom = target_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < target_width:
        if center:
            w_pad_left = int((target_width - width) / 2.0)
        else:
            w_pad_left = random.randint(a=0, b=target_width - width)
        w_pad_right = target_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    x = cv2.copyMakeBorder(x, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right,
                           cv2.BORDER_CONSTANT, value=value)
    return x