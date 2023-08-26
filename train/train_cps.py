import torch
from torch import nn
import os
from dataset.loader_builder import build_loader
from net.cps import prepare_settings, init_model, init_weight
from utils.utils import fix_seed, runningScore
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BEST_METRIC = 0

def main(cfg_path):
    fix_seed(2023)
    settings = prepare_settings(cfg_path)
    logger = settings["logger"]
    cfg = settings["cfg"]
    logger.info(cfg)
    LR = cfg["train"]["lr"]
    bn_eps = cfg["train"]["bn_eps"]
    bn_momentum = cfg["train"]["bn_momentum"]

    model_l, model_r = init_model(settings)

    init_weight(model_l.business_layer, nn.init.kaiming_normal_,
                nn.BatchNorm2d, bn_eps,bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model_r.business_layer, nn.init.kaiming_normal_,
                nn.BatchNorm2d, bn_eps,bn_momentum,
                mode='fan_in', nonlinearity='relu')


    optimizer_l = torch.optim.Adam(params=model_l.parameters(),lr=LR)
    optimizer_r = torch.optim.Adam(params=model_r.parameters(),lr=LR)

    lr_scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_l, step_size=4, gamma=0.1)
    lr_scheduler_r = torch.optim.lr_scheduler.StepLR(optimizer_r, step_size=4, gamma=0.1)

    sup_loader, semi_loader, valid_loader= build_loader(cfg)
    for epoch in range(cfg["train"]["max_epoch"]):
        train(cfg,model_l,model_r,optimizer_l,optimizer_r,lr_scheduler_l,lr_scheduler_r,sup_loader,semi_loader,epoch,settings)
        validate(cfg,model_l,valid_loader,epoch,settings)


def train(cfg,model_l,model_r,optimizer_l,optimizer_r,lr_scheduler_l,lr_scheduler_r,sup_loader,semi_loader,epoch,settings):
    model_l.train()
    model_r.train()

    logger = settings["logger"]
    DEVICE = settings["DEVICE"]
    ROUND_NUM = settings["ROUND_NUM"]
    sce_loss = settings["sce_loss"]
    writer = settings["writer"]

    cfg_train = cfg["train"]
    MAX_EPOCH = cfg_train["max_epoch"]
    CLASSES = cfg_train["classes"]

    sup_running_metrics_train = runningScore(CLASSES)
    sup_running_metrics_train.reset()
    semi_running_metrics_train = runningScore(CLASSES)
    semi_running_metrics_train.reset()

    sup_loader_iter = iter(sup_loader)
    semi_loader_iter = iter(semi_loader)

    for i_train in range(len(sup_loader)):
        sup_images, sup_labels = sup_loader_iter.next()
        sup_images, sup_labels = sup_images.to(DEVICE).to(torch.float32), sup_labels.to(DEVICE).to(torch.int64)

        semi_images,semi_labels = semi_loader_iter.next()
        semi_images, semi_labels = semi_images.to(DEVICE).to(torch.float32), semi_labels.to(DEVICE).to(torch.int64)

        optimizer_l.zero_grad()
        optimizer_r.zero_grad()

        pred_sup_l = model_l(sup_images)
        pred_unsup_l = model_l(semi_images)
        pred_sup_r = model_r(sup_images)
        pred_unsup_r = model_r(semi_images)

        pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
        pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
        _, max_l = torch.max(pred_l.detach(), dim=1)
        _, max_r = torch.max(pred_r.detach(), dim=1)
        cps_loss = sce_loss(pred_l,max_r) + sce_loss(pred_r,max_l)
        sup_loss_l = sce_loss(pred_sup_l, sup_labels)
        sup_loss_r = sce_loss(pred_sup_r, sup_labels)
        loss = cps_loss + sup_loss_l + sup_loss_r

        sup_running_metrics_train.update(sup_labels.detach().cpu().numpy(), pred_sup_l.detach().max(1)[1].cpu().numpy())
        semi_running_metrics_train.update(semi_labels.detach().cpu().numpy(), pred_unsup_l.detach().max(1)[1].cpu().numpy())

        loss.backward()
        optimizer_l.step()
        optimizer_r.step()

        logger.info(
            f'Train: epoch [{epoch}/{MAX_EPOCH - 1}] iteration [{i_train}] sup_loss {round(sup_loss_l.item(), ROUND_NUM)} '
            f'cps_loss {round(cps_loss.item(), ROUND_NUM)} lr {round(optimizer_l.param_groups[0]["lr"],ROUND_NUM)}')

    lr_scheduler_l.step()
    lr_scheduler_r.step()

    train_sup_score, train_sup_class_iou = sup_running_metrics_train.get_scores()
    for key in train_sup_score.keys():
        if key != 'Class Accuracy':
            writer.add_scalar('train_sup/' + key, train_sup_score[key], epoch)
            logger.info(f'Train_sup: {key} {round(train_sup_score[key], ROUND_NUM)}')

    train_semi_score, train_semi_class_iou = semi_running_metrics_train.get_scores()
    for key in train_semi_score.keys():
        if key != 'Class Accuracy':
            writer.add_scalar('train_semi/' + key, train_semi_score[key], epoch)
            logger.info(f'Train_semi: {key} {round(train_semi_score[key], ROUND_NUM)}')

def validate(cfg,model,valid_loader,epoch,settings):
    global BEST_METRIC

    logger = settings["logger"]
    DEVICE = settings["DEVICE"]
    ROUND_NUM = settings["ROUND_NUM"]
    sce_loss = settings["sce_loss"]
    writer = settings["writer"]
    LOG_DIR = settings["LOG_DIR"]

    cfg_train = cfg["train"]
    CLASSES = cfg_train["classes"]
    MAX_EPOCH = cfg_train["max_epoch"]

    running_metrics_val = runningScore(CLASSES)
    running_metrics_val.reset()
    METRIC = cfg["valid"]["metric"]

    model.eval()
    with torch.no_grad():
        for i_val, (images_val, labels_val) in enumerate(valid_loader):
            images_val, labels_val = images_val.to(DEVICE).to(torch.float32), labels_val.to(DEVICE).to(torch.int64)
            outputs_val = model(images_val)
            loss = sce_loss(outputs_val, labels_val)
            pred = outputs_val.detach().max(1)[1].cpu().numpy()
            gt = labels_val.detach().cpu().numpy()
            running_metrics_val.update(gt, pred)

            logger.info(
                f'Validation epoch [{epoch }/{MAX_EPOCH - 1}] iteration [{i_val}] loss: {round(loss.item(), ROUND_NUM)}')

        val_score, val_class_iou = running_metrics_val.get_scores()

        for key in val_score.keys():
            if key != 'Class Accuracy':
                writer.add_scalar('val/' + key, val_score[key], epoch)
                logger.info(f'Validation: {key} {round(val_score[key], ROUND_NUM)}')

        MODEL_PATH = os.path.join(LOG_DIR, f"epoch" + str(epoch) + "_model.pkl")
        torch.save(model, MODEL_PATH)
        # save checkpoint
        LAST_MODEL_PATH = os.path.join(LOG_DIR, f"last_model.pkl")
        torch.save(model, LAST_MODEL_PATH)
        logger.info(f'Save checkpoint {LAST_MODEL_PATH}')
        if val_score[METRIC] >= BEST_METRIC:
            BEST_METRIC = val_score[METRIC]
            BEST_MODEL_PATH = os.path.join(LOG_DIR, f"best_model.pkl")
            torch.save(model, BEST_MODEL_PATH)
            logger.info(f'Save checkpoint {BEST_MODEL_PATH}')

if __name__ == '__main__':
    cfg_dir = "/volume/CONSS/conf/"
    cfg_path = cfg_dir + "seam_cps.yaml"
    main(cfg_path)