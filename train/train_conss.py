import os
from dataset.loader_builder import build_loader
from net.segnet import prepare_settings, init_model
from utils.loss import contra_loss
import torch
from utils.utils import fix_seed, runningScore, label_onehot, generate_aug_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BEST_METRIC = 0


def main(cfg_path):
    fix_seed(2023)
    settings = prepare_settings(cfg_path)
    logger = settings["logger"]
    cfg = settings["cfg"]
    logger.info(cfg)
    LR = cfg["train"]["lr"]
    model = init_model(settings)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    sup_loader, semi_loader, valid_loader = build_loader(cfg)
    for epoch in range(cfg["train"]["max_epoch"]):
        train(cfg, model, optimizer, lr_scheduler, sup_loader, semi_loader, epoch, settings)
        validate(cfg, model, valid_loader, epoch, settings)


def train(cfg, model, optimizer, lr_scheduler, sup_loader, semi_loader, epoch, settings):
    model.train()

    logger = settings["logger"]
    DEVICE = settings["DEVICE"]
    ROUND_NUM = settings["ROUND_NUM"]
    sce_loss = settings["sce_loss"]
    writer = settings["writer"]

    cfg_train = cfg["train"]
    CLASSES = cfg_train["classes"]
    MAX_EPOCH = cfg_train["max_epoch"]
    CONTRA_LOSS = cfg_train["contra_loss"]
    WEAK_TH = cfg_train["contrastive"]["weak_threshold"]
    STRONG_TH = cfg_train["contrastive"]["strong_threshold"]
    TEMP = cfg_train["contrastive"]["temperature"]
    AUG_MODE = cfg_train["contrastive"]["aug_mode"]
    NUM_NEG = cfg_train["contrastive"]["num_negatives"]
    NUM_QE = cfg_train["contrastive"]["num_queries"]
    sup_running_metrics_train = runningScore(CLASSES)
    sup_running_metrics_train.reset()
    semi_running_metrics_train = runningScore(CLASSES)
    semi_running_metrics_train.reset()

    sup_loader_iter = iter(sup_loader)
    semi_loader_iter = iter(semi_loader)

    for i_train in range(len(sup_loader)):
        sup_images, sup_labels = sup_loader_iter.next()
        sup_images, sup_labels = sup_images.to(DEVICE).to(torch.float32), sup_labels.to(DEVICE).to(torch.int64)

        # transductive semi-supervised learning: unlabeled data is also prediction target
        semi_images, semi_labels = semi_loader_iter.next()
        # semi_labels are only used for evaluation but not visible to the model
        semi_images, semi_labels = semi_images.to(DEVICE).to(torch.float32), semi_labels.to(DEVICE).to(torch.int64)

        optimizer.zero_grad()

        with torch.no_grad():
            pred_semi_temp, _ = model(semi_images)
            prob_max_semi_temp, labels_semi_temp = torch.max(torch.softmax(pred_semi_temp, dim=1), dim=1)
            semi_aug_images, semi_aug_labels, semi_aug_prob_max = \
                generate_aug_data(semi_images, labels_semi_temp, prob_max_semi_temp, mode=AUG_MODE)

        pred_sup, rep_sup = model(sup_images)
        pred_semi, rep_semi = model(semi_aug_images)

        sup_running_metrics_train.update(sup_labels.detach().cpu().numpy(), pred_sup.detach().max(1)[1].cpu().numpy())
        semi_running_metrics_train.update(semi_labels.detach().cpu().numpy(),
                                          pred_semi_temp.detach().max(1)[1].cpu().numpy())

        rep_all = torch.cat((rep_sup, rep_semi))

        # supervised loss
        sup_loss = sce_loss(pred_sup, sup_labels)

        # contrastive loss for labeled data and unlabeled data
        if CONTRA_LOSS:
            with torch.no_grad():
                semi_aug_mask = semi_aug_prob_max.ge(WEAK_TH).float()
                mask_all = torch.cat(((sup_labels.unsqueeze(1) >= 0).float(), semi_aug_mask.unsqueeze(1)))
                label_all = torch.cat((label_onehot(sup_labels, CLASSES), label_onehot(semi_aug_labels, CLASSES)))

                prob_sup = torch.softmax(pred_sup, dim=1)
                prob_semi = torch.softmax(pred_semi, dim=1)
                prob_all = torch.cat((prob_sup, prob_semi))
            # unsupervised loss
            con_loss = contra_loss(rep_all, label_all, mask_all, prob_all, STRONG_TH, TEMP, NUM_QE, NUM_NEG)
        else:
            con_loss = torch.tensor(0)

        loss = sup_loss + con_loss

        writer.add_scalar("sup_loss", round(sup_loss.item(), ROUND_NUM), i_train + epoch * len(sup_loader))
        writer.add_scalar("con_loss", round(con_loss.item(), ROUND_NUM), i_train + epoch * len(sup_loader))

        loss.backward()
        optimizer.step()

        logger.info(
            f"Train: epoch [{epoch}/{MAX_EPOCH - 1}] iteration [{i_train}] sup_loss {round(sup_loss.item(), ROUND_NUM)} "
            f"con_loss {round(con_loss.item(), ROUND_NUM)} lr {round(optimizer.param_groups[0]['lr'], ROUND_NUM)}")

    lr_scheduler.step()

    train_sup_score, train_sup_class_iou = sup_running_metrics_train.get_scores()
    for key in train_sup_score.keys():
        if key not in ["Class Accuracy", "IoU"]:
            writer.add_scalar("train_sup/" + key, train_sup_score[key], epoch)
            logger.info(f"Train_sup: {key} {round(train_sup_score[key], ROUND_NUM)}")

    train_semi_score, train_semi_class_iou = semi_running_metrics_train.get_scores()
    for key in train_semi_score.keys():
        if key not in ["Class Accuracy", "IoU"]:
            writer.add_scalar("train_semi/" + key, train_semi_score[key], epoch)
            logger.info(f"Train_semi: {key} {round(train_semi_score[key], ROUND_NUM)}")


def validate(cfg, model, valid_loader, epoch, settings):
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
    with torch.no_grad():
        model.eval()
        for i_val, (images_val, labels_val) in enumerate(valid_loader):
            images_val, labels_val = images_val.to(DEVICE).to(torch.float32), labels_val.to(DEVICE).to(torch.int64)
            outputs_val, _ = model(images_val)
            loss = sce_loss(outputs_val, labels_val)
            pred = outputs_val.detach().max(1)[1].cpu().numpy()
            gt = labels_val.detach().cpu().numpy()
            running_metrics_val.update(gt, pred)

            logger.info(
                f"Validation epoch [{epoch}/{MAX_EPOCH - 1}] iteration [{i_val}] loss: {round(loss.item(), ROUND_NUM)}")

        val_score, val_class_iou = running_metrics_val.get_scores()

        for key in val_score.keys():
            if key not in ["Class Accuracy", "IoU"]:
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
    cfg_dir = "/conf/"
    cfg_path = cfg_dir + "seam_conss.yaml"
    main(cfg_path)
