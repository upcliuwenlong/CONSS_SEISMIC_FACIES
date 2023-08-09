import os
from dataset.loader_builder import build_loader
from net.segnet_ot import prepare_settings, init_model
from utils.loss import SCEloss, STloss, REloss, CONloss
import torch
from utils.utils import fix_seed, runningScore
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
BEST_METRIC = 0

def ema(student,teacher):
    _contrast_momentum = 0.995
    for mean_param, param in zip(teacher.parameters(), student.parameters()):
        mean_param.data.mul_(_contrast_momentum).add_(other=param.data, alpha=1 - _contrast_momentum)

def main(cfg_path):
    fix_seed(2023)
    settings = prepare_settings(cfg_path)
    logger = settings["logger"]
    cfg = settings["cfg"]
    logger.info(cfg)
    LR = cfg["train"]["lr"]
    model_s,model_t,model_e = init_model(settings)
    optimizer = torch.optim.Adam(params=[{"params":model_s.parameters()},
                                         {"params":model_e.parameters()}], lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    sup_loader, semi_loader, valid_loader = build_loader(cfg)
    for epoch in range(cfg["train"]["max_epoch"]):
        train(cfg, model_s, model_t ,model_e, optimizer, lr_scheduler, sup_loader, semi_loader, epoch, settings)
        validate(cfg, model_s, valid_loader, epoch, settings)

def train(cfg, model_s, model_t, model_e, optimizer, lr_scheduler, sup_loader, semi_loader, epoch, settings):
    model_s.train()
    model_t.train()
    model_e.train()

    if epoch == 1:
        model_t.load_state_dict(model_s.state_dict())

    logger = settings["logger"]
    DEVICE = settings["DEVICE"]
    ROUND_NUM = settings["ROUND_NUM"]
    writer = settings["writer"]
    LOG_DIR = settings["LOG_DIR"]

    cfg_train = cfg["train"]
    CLASSES = cfg_train["classes"]
    MAX_EPOCH = cfg_train["max_epoch"]
    CON = cfg_train["con"]
    RE = cfg_train["re"]

    sup_running_metrics_train = runningScore(CLASSES)
    sup_running_metrics_train.reset()
    semi_running_metrics_train = runningScore(CLASSES)
    semi_running_metrics_train.reset()

    sup_loader_iter = iter(sup_loader)
    semi_loader_iter = iter(semi_loader)

    for i_train in range(len(sup_loader)):
        sup_images, sup_labels = sup_loader_iter.next()
        sup_images, sup_labels = sup_images.to(DEVICE).to(torch.float32), sup_labels.to(DEVICE).to(torch.int64)

        semi_images, semi_labels = semi_loader_iter.next()
        semi_images, semi_labels = semi_images.to(DEVICE).to(torch.float32), semi_labels.to(DEVICE).to(torch.int64)

        optimizer.zero_grad()

        sup_logits, _ = model_s(sup_images)
        sup_loss = SCEloss(sup_logits, sup_labels)

        if RE:
            re_sup_map = model_e(sup_images,sup_logits.detach())
            re_loss = REloss(re_sup_map,sup_labels,sup_logits.detach())
        else:
            re_loss = torch.tensor(0.0)

        semi_logits_s, semi_aux_logits_s = model_s(semi_images)

        st_loss = torch.tensor(0.0)
        con_loss = torch.tensor(0.0)

        if epoch > 0:
            with torch.no_grad():
                semi_logits_t,_ = model_t(semi_images)
                if RE:
                    re_semi_map = model_e(semi_images,semi_logits_t)
                    re_semi_map = torch.round(torch.sigmoid(re_semi_map)).squeeze(1)
                else:
                    re_semi_map = torch.ones(semi_images.size()).to(semi_images.device).squeeze(1)
                pseudo_label = semi_logits_t.argmax(1).to(semi_logits_t.device)

            st_loss = STloss(semi_logits_s, re_semi_map, pseudo_label)

            if CON:
                con_loss = sum([CONloss(logits,re_semi_map,F.softmax(semi_logits_t.detach(), dim=1))
                                for logits in semi_aux_logits_s]) / len(semi_aux_logits_s)

        loss = sup_loss + re_loss + st_loss + 10 * con_loss
        writer.add_scalar("sup_loss", round(sup_loss.item(), ROUND_NUM), i_train + epoch * len(sup_loader))
        writer.add_scalar("re_loss", round(re_loss.item(), ROUND_NUM), i_train + epoch * len(sup_loader))
        writer.add_scalar("st_loss", round(st_loss.item(), ROUND_NUM), i_train + epoch * len(sup_loader))
        writer.add_scalar("con_loss", round(con_loss.item(), ROUND_NUM), i_train + epoch * len(sup_loader))

        loss.backward()
        optimizer.step()

        if epoch > 0:
            ema(model_s,model_t)
            torch.save(model_e, os.path.join(LOG_DIR, f"epoch" + str(epoch) + "_model_e.pkl"))


        logger.info(
            f"Train: epoch [{epoch}/{MAX_EPOCH - 1}] iteration [{i_train}] sup_loss {round(sup_loss.item(), ROUND_NUM)} "
            f"re_loss {round(re_loss.item(), ROUND_NUM)} "
            f"st_loss {round(st_loss.item(), ROUND_NUM)} "
            f"con_loss {round(con_loss.item(), ROUND_NUM)} "
            f"lr {round(optimizer.param_groups[0]['lr'], ROUND_NUM)}")

    lr_scheduler.step()

    sup_running_metrics_train.update(sup_labels.detach().cpu().numpy(), sup_logits.detach().max(1)[1].cpu().numpy())

    semi_running_metrics_train.update(semi_labels.detach().cpu().numpy(),
                                      semi_logits_s.detach().max(1)[1].cpu().numpy())


    train_sup_score, train_sup_class_iou = sup_running_metrics_train.get_scores()
    for key in train_sup_score.keys():
        if key != "Class Accuracy":
            writer.add_scalar("train_sup/" + key, train_sup_score[key], epoch)
            logger.info(f"Train_sup: {key} {round(train_sup_score[key], ROUND_NUM)}")

    train_semi_score, train_semi_class_iou = semi_running_metrics_train.get_scores()
    for key in train_semi_score.keys():
        if key != "Class Accuracy":
            writer.add_scalar("train_semi/" + key, train_semi_score[key], epoch)
            logger.info(f"Train_semi: {key} {round(train_semi_score[key], ROUND_NUM)}")


def validate(cfg, model, valid_loader, epoch, settings):
    global BEST_METRIC

    logger = settings["logger"]
    DEVICE = settings["DEVICE"]
    ROUND_NUM = settings["ROUND_NUM"]
    writer = settings["writer"]
    LOG_DIR = settings["LOG_DIR"]

    cfg_train = cfg["train"]
    CLASSES = cfg_train["classes"]
    MAX_EPOCH = cfg_train["max_epoch"]

    running_metrics_val = runningScore(CLASSES)
    running_metrics_val.reset()
    METRIC = cfg["valid"]["metric"]
    with torch.no_grad():
        model.eval()
        for i_val, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(DEVICE).to(torch.float32), labels.to(DEVICE).to(torch.int64)
            logtits,_ = model(images)
            loss = SCEloss(logtits, labels)
            pred = logtits.detach().max(1)[1].cpu().numpy()
            gt = labels.detach().cpu().numpy()
            running_metrics_val.update(gt, pred)

            logger.info(
                f"Validation epoch [{epoch}/{MAX_EPOCH - 1}] iteration [{i_val}] loss: {round(loss.item(), ROUND_NUM)}")

        val_score, val_class_iou = running_metrics_val.get_scores()

        for key in val_score.keys():
            if key != "Class Accuracy":
                writer.add_scalar("val/" + key, val_score[key], epoch)
                logger.info(f"Validation: {key} {round(val_score[key], ROUND_NUM)}")

        MODEL_PATH = os.path.join(LOG_DIR, f"epoch" + str(epoch) + "_model.pkl")
        torch.save(model, MODEL_PATH)
        # save checkpoint
        LAST_MODEL_PATH = os.path.join(LOG_DIR, f"last_model.pkl")
        torch.save(model, LAST_MODEL_PATH)
        logger.info(f"Save checkpoint {LAST_MODEL_PATH}")
        if val_score[METRIC] >= BEST_METRIC:
            BEST_METRIC = val_score[METRIC]
            BEST_MODEL_PATH = os.path.join(LOG_DIR, f"best_model.pkl")
            torch.save(model, BEST_MODEL_PATH)
            logger.info(f"Save checkpoint {BEST_MODEL_PATH}")

if __name__ == "__main__":
    cfg_dir = "/volume/OT_114/conf/"
    cfg_path = cfg_dir + "seam_mt.yaml"
    main(cfg_path)
