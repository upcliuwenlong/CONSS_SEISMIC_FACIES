import os
import yaml
import segmentation_models_pytorch as smp
from tensorboardX import SummaryWriter
from datetime import datetime
import torch.nn as nn
import shutil
import torch
from net.segnet import SegNet
from utils.utils import get_logger

def prepare_settings(cfg_path):
    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)
    CURRENT_TIME = datetime.now().strftime("%b%d_%H%M%S")
    LOG_DIR = os.path.join(os.getcwd() + "/runs", CURRENT_TIME + "_{}".format(cfg["train"]["dir_suffix"]))
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
    cfg_train = settings["cfg"]["train"]
    CLASSES = cfg_train["classes"]
    model_l = SingleNetwork(classes=CLASSES)
    model_r = SingleNetwork(classes=CLASSES)
    model_total_param = sum([param.nelement() for param in model_l.parameters()])
    logger.info(f"Model total parameters {model_total_param}")
    # use DataParallel if more than 1 GPU available
    if torch.cuda.device_count() > 1 and not DEVICE.type == "cpu":
        model_l = torch.nn.DataParallel(model_l)
        model_r = torch.nn.DataParallel(model_r)
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
    logger.info(f"Sending the model to {DEVICE}")
    model_l = model_l.to(DEVICE)
    model_r = model_r.to(DEVICE)
    return model_l, model_r

class SingleNetwork(SegNet):
    def __init__(self,model_name="DeepLabV3Plus", in_channels=1, classes=6, decoder_channels=256, upsampling=4,encoder_name="resnet101"):
        super(SingleNetwork, self).__init__(model_name=model_name,in_channels=in_channels,classes=classes,decoder_channels=decoder_channels,
                                            upsampling=upsampling,encoder_name=encoder_name)

        self.business_layer = []
        self.business_layer.append(self.classifier)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        prediction = self.classifier(decoder_output)
        return prediction

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)









