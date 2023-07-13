import random
from torch.utils.data import Dataset
import numpy as np
import math

class UnlabelDataset(Dataset):
    def __init__(self,data, labels, augmentations,slice_width=256,slide_width=256,sample_pos=None):
        self.data = data
        self.labels = labels
        self.sample_pos = sample_pos
        self.slice_width = slice_width
        self.slide_width = slide_width
        self.augmentations = augmentations
        self.images = list()
        self.masks = list()
        self.make_dataset()

    def add_data(self,image,mask):
        image = np.expand_dims(image,-1)
        auged_data = self.augmentations(
            image=image,
            mask=mask
        )
        auged_image = auged_data['image']
        auged_mask = auged_data['mask']
        auged_image = auged_image.transpose(2, 0, 1)
        self.images.append(auged_image)
        self.masks.append(auged_mask)
    def make_dataset(self):
        if self.sample_pos is not None:
            all_sample_pos = range(self.data.shape[2])
            unlabel_sample_pos = set(all_sample_pos) - set(self.sample_pos)
        else:
            unlabel_sample_pos = range(self.data.shape[2])

        for step in range(int(2 + (self.data.shape[1]-self.slice_width)/(self.slide_width))):
            crossline_start = step * self.slide_width
            if crossline_start + self.slice_width < self.data.shape[1]:
                crossline_end = crossline_start + self.slice_width
            else:
                crossline_end = self.data.shape[1]
                crossline_start = crossline_end - self.slice_width
            for inline in unlabel_sample_pos:
                mask = self.labels[:, crossline_start:crossline_end, inline].astype(np.int64)
                image = self.data[:, crossline_start:crossline_end, inline]
                self.add_data(image, mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

class FewLabelDataset(Dataset):
    def __init__(self,data, labels, augmentations,slice_width=256,slide_width=256,sample_pos=None):
        self.data = data
        self.labels = labels
        self.sample_pos = sample_pos
        self.slice_width = slice_width
        self.slide_width = slide_width
        self.augmentations = augmentations
        self.images = list()
        self.masks = list()
        self.make_dataset()

    def over_sample(self,unlabel_num):
        zip_data = list(zip(self.images,self.masks))
        repeats = math.ceil(unlabel_num / len(self.images))
        zip_data = zip_data*repeats
        over_smp_zip = random.sample(zip_data,unlabel_num)
        images,masks = zip(*over_smp_zip)
        self.images = images
        self.masks = masks

    def add_data(self, image, mask):
        image = np.expand_dims(image, -1)
        auged_data = self.augmentations(
            image=image,
            mask=mask
        )
        auged_image = auged_data['image']
        auged_mask = auged_data['mask']
        auged_image = auged_image.transpose(2, 0, 1)
        self.images.append(auged_image)
        self.masks.append(auged_mask)
    def make_dataset(self):
        label_sample_pos = self.sample_pos
        print('******sample_position*****',label_sample_pos)
        for step in range(int(2 + (self.data.shape[1] - self.slice_width) / (self.slide_width))):
            crossline_start = step * self.slide_width
            if crossline_start + self.slice_width < self.data.shape[1]:
                crossline_end = crossline_start + self.slice_width
            else:
                crossline_end = self.data.shape[1]
                crossline_start = crossline_end - self.slice_width
            for inline in label_sample_pos:
                mask = self.labels[:, crossline_start:crossline_end, inline].astype(np.int64)
                image = self.data[:, crossline_start:crossline_end, inline]
                self.add_data(image, mask)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class STDataset(Dataset):
    def __init__(self,data, labels, pseudo_labels,augmentations,slice_width=256,slide_width=256,sample_pos=None):
        self.data = data
        self.labels = labels
        self.pseudo_labels = pseudo_labels
        self.sample_pos = sample_pos
        self.slice_width = slice_width
        self.slide_width = slide_width
        self.augmentations = augmentations
        self.images = list()
        self.masks = list()
        self.make_dataset()
    def add_data(self, image, mask):
        image = np.expand_dims(image, -1)
        auged_data = self.augmentations(
            image=image,
            mask=mask
        )
        auged_image = auged_data['image']
        auged_mask = auged_data['mask']
        auged_image = auged_image.transpose(2, 0, 1)
        self.images.append(auged_image)
        self.masks.append(auged_mask)
    def make_dataset(self):
        label_sample_pos = self.sample_pos

        for step in range(int(2 + (self.data.shape[1]-self.slice_width)/(self.slide_width))):
            crossline_start = step * self.slide_width
            if crossline_start + self.slice_width < self.data.shape[1]:
                crossline_end = crossline_start + self.slice_width
            else:
                crossline_end = self.data.shape[1]
                crossline_start = crossline_end - self.slice_width
            for inline in label_sample_pos:
                mask = self.labels[:, crossline_start:crossline_end, inline].astype(np.int64)
                image = self.data[:, crossline_start:crossline_end, inline]
                self.add_data(image, mask)

        if self.sample_pos is not None:
            all_sample_pos = range(self.data.shape[2])
            unlabel_sample_pos = set(all_sample_pos) - set(self.sample_pos)
        else:
            unlabel_sample_pos = range(self.data.shape[2])

        for step in range(int(2 + (self.data.shape[1]-self.slice_width)/(self.slide_width))):
            crossline_start = step * self.slide_width
            if crossline_start + self.slice_width < self.data.shape[1]:
                crossline_end = crossline_start + self.slice_width
            else:
                crossline_end = self.data.shape[1]
                crossline_start = crossline_end - self.slice_width
            for inline in unlabel_sample_pos:
                mask = self.pseudo_labels[:, crossline_start:crossline_end, inline].astype(np.int64)
                image = self.data[:, crossline_start:crossline_end, inline]
                self.add_data(image, mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

