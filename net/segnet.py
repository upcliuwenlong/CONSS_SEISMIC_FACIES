import importlib
import os
import yaml
import segmentation_models_pytorch as smp
from tensorboardX import SummaryWriter
from datetime import datetime
import torch.nn as nn
import shutil
import torch
from utils.utils import get_logger

def prepare_settings(cfg_path):
    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)
    CURRENT_TIME = datetime.now().strftime("%b%d_%H%M%S")
    LOG_DIR = os.path.join(os.getcwd() + "/runs", CURRENT_TIME +
                           "_{}".format(cfg["train"]["dir_suffix"]))
    os.makedirs(LOG_DIR)
    logger = get_logger("seis_facies_ident", log_path=LOG_DIR)
    logger.propagate = False
    writer = SummaryWriter(log_dir=LOG_DIR)
    shutil.copy(cfg_path, LOG_DIR)
    sce_loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=cfg["train"]["smooth_factor"], ignore_index=-1)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"cfg":cfg,"logger":logger,"writer":writer,"sce_loss":sce_loss,
            "DEVICE":DEVICE,"ROUND_NUM":6,"LOG_DIR":LOG_DIR}

def init_model(settings):
    logger = settings["logger"]
    DEVICE = settings["DEVICE"]
    model = SegNet()
    model_total_param = sum([param.nelement() for param in model.parameters()])
    logger.info(f"Model total parameters {model_total_param}")
    # use DataParallel if more than 1 GPU available
    if torch.cuda.device_count() > 1 and not DEVICE.type == "cpu":
        model = torch.nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
    logger.info(f"Sending the model to {DEVICE}")
    model = model.to(DEVICE)
    return model

class SegNet(nn.Module):
    def __init__(self, model_name='DeepLabV3Plus', in_channels=1, classes=6, decoder_channels=256, upsampling=4, rep_channels=128,encoder_name="resnet101"):
        super(SegNet, self).__init__()
        model_module = getattr(importlib.import_module("segmentation_models_pytorch"),model_name)
        model = model_module(encoder_name=encoder_name,in_channels=in_channels,classes=classes)

        self.encoder = model.encoder
        self.decoder = model.decoder

        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
        )

        self.representation = nn.Sequential(
            nn.Conv2d(decoder_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, rep_channels, 1),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
        )
    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        prediction = self.classifier(decoder_output)
        representation = self.representation(decoder_output)
        return prediction, representation