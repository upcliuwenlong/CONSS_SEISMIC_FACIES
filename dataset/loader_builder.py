from albumentations import Compose, PadIfNeeded
from torch.utils.data import DataLoader
from dataset.dataset import UnlabelDataset, FewLabelDataset, STDataset
from utils.utils import get_logger
import numpy as np

def get_seismic(cfg_dataset):
    all_data,all_labels = None,None
    if cfg_dataset["name"] in ["f3_st","f3_semi"]:
        train_data_path = cfg_dataset["train_data_path"]
        test1_data_path = cfg_dataset["test1_data_path"]
        test2_data_path = cfg_dataset["test2_data_path"]
        # inline*crossline*depth: 401*701*255
        train_data = np.load(train_data_path)
        # inline*crossline*depth: 200*701*255
        test1_data = np.load(test1_data_path)
        # inline*crossline*depth: 601*200*255
        test2_data = np.load(test2_data_path)

        train_labels_path = cfg_dataset["train_labels_path"]
        test1_labels_path = cfg_dataset["test1_labels_path"]
        test2_labels_path = cfg_dataset["test2_labels_path"]
        train_labels = np.load(train_labels_path)
        test1_labels = np.load(test1_labels_path)
        test2_labels = np.load(test2_labels_path)

        # inline*crossline*depth->depth*crossline*inline: 255*901*601
        all_data = np.concatenate([np.concatenate([test1_data, train_data], axis=0), test2_data],
                                  axis=1).transpose(2, 1, 0)
        all_labels = np.concatenate([np.concatenate([test1_labels, train_labels], axis=0), test2_labels],
                                    axis=1).transpose(2, 1, 0)
    elif cfg_dataset["name"] in ["seam_st","seam_semi"]:
        train_data_path = cfg_dataset["train_data_path"]
        train_labels_path = cfg_dataset["train_labels_path"]

        # depth*crossline*inline: 1006*782*590
        train_data = np.load(train_data_path)['data']
        # 1-6->0-5
        train_labels = np.load(train_labels_path)['labels'] - 1

        all_data = train_data
        all_labels = train_labels
    return all_data,all_labels


def build_loader(cfg):
    sup_loader, semi_loader, validate_loader=None,None,None
    logger = get_logger('seis_facies_ident')
    logger.propagate = False
    dataset = cfg["dataset"]
    if dataset["name"] == "seam_semi":
        train_data, train_labels = get_seismic(cfg_dataset=dataset)

        SLICE_WIDTH = dataset["slice_width"]
        BATCH_SIZE = dataset["batch_size"]
        logger.info(f"Make semi dataset...")
        semi_dataset = UnlabelDataset(data=train_data, labels=train_labels,
                                      augmentations=Compose([PadIfNeeded(1024, SLICE_WIDTH, p=1)]),
                                      slice_width=SLICE_WIDTH,
                                      sample_pos=dataset["sample_position"])

        logger.info(f"Semi dataset: {len(semi_dataset)}")

        logger.info(f"Make sup dataset...")
        sup_dataset = FewLabelDataset(data=train_data, labels=train_labels,
                                      augmentations=Compose([PadIfNeeded(1024, SLICE_WIDTH, p=1)]),
                                      slice_width=SLICE_WIDTH,
                                      sample_pos=dataset["sample_position"])
        logger.info(f"Sup dataset: {len(sup_dataset)}")
        sup_dataset.over_sample(len(semi_dataset))
        logger.info(f"Over sample sup dataset: {len(sup_dataset)}")

        sup_loader = DataLoader(sup_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        semi_loader = DataLoader(semi_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        validate_loader = DataLoader(semi_dataset, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0,
                                     drop_last=True)

    elif dataset["name"] == "f3_semi" :
        train_data,train_labels = get_seismic(cfg_dataset=dataset)
        SLICE_WIDTH = dataset["slice_width"]
        BATCH_SIZE = dataset["batch_size"]
        logger.info(f"Make semi dataset...")
        semi_dataset = UnlabelDataset(data=train_data, labels=train_labels,
                                      augmentations=Compose([PadIfNeeded(256, SLICE_WIDTH, p=1)]),
                                      slice_width=SLICE_WIDTH,
                                      sample_pos=dataset["sample_position"])

        logger.info(f"Semi dataset: {len(semi_dataset)}")

        logger.info(f"Make sup dataset...")
        sup_dataset = FewLabelDataset(data=train_data, labels=train_labels,
                                         augmentations=Compose([PadIfNeeded(256, SLICE_WIDTH, p=1)]),
                                         slice_width=SLICE_WIDTH,
                                         sample_pos=dataset["sample_position"])
        logger.info(f"Sup dataset: {len(sup_dataset)}")
        sup_dataset.over_sample(len(semi_dataset))
        logger.info(f"Over sample sup dataset: {len(sup_dataset)}")

        sup_loader = DataLoader(sup_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        semi_loader = DataLoader(semi_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        validate_loader = DataLoader(semi_dataset, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0,
                                     drop_last=True)

    return sup_loader, semi_loader, validate_loader

def build_st_loader(cfg):
    st_loader, val_loader=None,None
    logger = get_logger('seis_facies_ident')
    logger.propagate = False
    dataset = cfg["dataset"]
    if dataset["name"] == "seam_st":
        train_data, train_labels = get_seismic(cfg_dataset=dataset)
        SLICE_WIDTH = dataset["slice_width"]
        BATCH_SIZE = dataset["batch_size"]
        pseudo_labels = np.load(dataset["pseudo_label_path"])['prediction']
        logger.info(f"Make ST dataset...")
        st_dataset = STDataset(data=train_data, labels=train_labels, pseudo_labels=pseudo_labels,
                               sample_pos=dataset["sample_position"],
                               augmentations=Compose([PadIfNeeded(1024, SLICE_WIDTH, p=1)]), slice_width=SLICE_WIDTH)
        st_loader = DataLoader(st_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        logger.info(f"ST dataset: {len(st_dataset)}")

        val_dataset = UnlabelDataset(data=train_data, labels=train_labels,
                                     augmentations=Compose([PadIfNeeded(1024, SLICE_WIDTH, p=1)]),
                                     slice_width=SLICE_WIDTH,
                                     sample_pos=dataset["sample_position"])
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0,
                                drop_last=True)


    elif dataset["name"] == "f3_st":
        train_data, train_labels = get_seismic(cfg_dataset=dataset)
        pseudo_labels = np.load(dataset["pseudo_label_path"])['prediction']
        SLICE_WIDTH = dataset["slice_width"]
        BATCH_SIZE = dataset["batch_size"]
        logger.info(f"Make ST dataset...")
        st_dataset = STDataset(data=train_data, labels=train_labels, pseudo_labels=pseudo_labels,
                               sample_pos=dataset["sample_position"],
                               augmentations=Compose([PadIfNeeded(256, SLICE_WIDTH, p=1)]), slice_width=SLICE_WIDTH)
        st_loader = DataLoader(st_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        logger.info(f"ST dataset: {len(st_dataset)}")

        val_dataset = UnlabelDataset(data=train_data, labels=train_labels,
                                     augmentations=Compose([PadIfNeeded(256, SLICE_WIDTH, p=1)]),
                                     slice_width=SLICE_WIDTH,
                                     sample_pos=dataset["sample_position"])
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0,
                                drop_last=True)
    return st_loader, val_loader
