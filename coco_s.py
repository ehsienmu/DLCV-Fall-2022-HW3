from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os

# # from transformers import BertTokenizer
from tokenizers import Tokenizer

from cat_utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 224


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    # RandomRotation(),
    # tv.transforms.Lambda(under_max),
    # tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
    #                           0.8, 1.5], saturation=[0.2, 1.5]),
    # tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    # tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(self._process(val['image_id'], ann['images']), val['caption'])
                      for val in ann['annotations']]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]
            
        self.tokenizer = Tokenizer.from_file('./hw3_data/caption_tokenizer.json')
        self.tokenizer.enable_padding(length=64)
        self.max_length = max_length + 1

    def _process(self, image_id, images):
        # val = str(image_id)
        for val in images:
            if val['id'] == image_id:
                return val['file_name']
        assert False

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))
        image = image.convert("RGB") #####
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode(caption)
        caption = np.array(caption_encoded.ids)
        cap_mask = (
            1 - np.array(caption_encoded.attention_mask)).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'images', 'train')
        train_file = os.path.join(
            config.dir, 'train.json')
        data = CocoCaption(train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'images', 'val')
        val_file = os.path.join(
            config.dir, 'val.json')
        data = CocoCaption(val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")