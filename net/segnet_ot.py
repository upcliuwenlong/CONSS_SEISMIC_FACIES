from net.decoders import *
from utils.utils import get_logger
import os
import yaml
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.decoders.deeplabv3.decoder as decoders
from tensorboardX import SummaryWriter
from datetime import datetime
import shutil

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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"cfg":cfg,"logger":logger,"writer":writer,
            "DEVICE":DEVICE,"ROUND_NUM":6,"LOG_DIR":LOG_DIR}

def init_model(settings):
    logger = settings["logger"]
    DEVICE = settings["DEVICE"]
    cfg_train = settings["cfg"]["train"]
    model_s = SegNet()
    model_t = SegNet()
    model_total_param = sum([param.nelement() for param in model_s.parameters()])
    logger.info(f"Model total parameters {model_total_param}")
    logger.info(f"Sending the model to {DEVICE}")
    model_s = model_s.to(DEVICE)
    model_t = model_t.to(DEVICE)
    return model_s,model_t

class SegNet(nn.Module):
    def __init__(self, in_channels = 1, classes = 6, decoder_channels = 256, upscale = 4, encoder_name="resnet101"):
        super(SegNet, self).__init__()

        self.encoder = smp.encoders.get_encoder(encoder_name,in_channels=in_channels,depth=5,
                                                weights="imagenet",output_stride=16)

        self.decoder = decoders.DeepLabV3PlusDecoder(encoder_channels=self.encoder.out_channels,out_channels=decoder_channels,
                                                     atrous_rates=(12,24,36),output_stride=16)

        self.classifier_decoder = upsample(decoder_channels,classes,upscale=upscale)

        self.aux_decoders = nn.ModuleList([FeatureNoiseDecoder(upscale=upscale,conv_in_ch=decoder_channels,num_classes=classes),
                                           DropOutDecoder(upscale=upscale,conv_in_ch=decoder_channels,num_classes=classes),
                                           FeatureDropDecoder(upscale=upscale,conv_in_ch=decoder_channels,num_classes=classes),
                                           VATDecoder(upscale=upscale,conv_in_ch=decoder_channels,num_classes=classes)
                                           ])

        init_weight(self.aux_decoders, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(*features)
        main_decoder_out = self.classifier_decoder(features)
        if self.training and features.requires_grad:
            aux_decoder_outs = [aux_decoder(features) for aux_decoder in  self.aux_decoders]
            return main_decoder_out, aux_decoder_outs
        else:
            return main_decoder_out,None


