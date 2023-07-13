import os
from dataset.loader_builder import build_st_loader
from net.segnet import prepare_settings, init_model
import torch
from utils.utils import fix_seed, runningScore, generate_aug_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BEST_METRIC = 0

def main(cfg_path, model_cfg_path):
    fix_seed(2023)
    settings = prepare_settings(cfg_path, model_cfg_path)
    logger = settings["logger"]
    cfg = settings["cfg"]
    logger.info(cfg)
    LR = cfg["train"]["lr"]
    model = init_model(settings)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    train_loader, val_loader= build_st_loader(cfg)
    for epoch in range(cfg["train"]["max_epoch"]):
        train(cfg, model, optimizer, lr_scheduler, train_loader, epoch, settings)
        validate(cfg, model, val_loader, epoch, settings)

def train(cfg, model, optimizer, lr_scheduler, train_loader, epoch, settings):
    model.train()

    logger = settings["logger"]
    DEVICE = settings["DEVICE"]
    ROUND_NUM = settings["ROUND_NUM"]
    sce_loss = settings["sce_loss"]
    writer = settings["writer"]
    CLASSES = settings["model_cfg"]["classes"]

    cfg_train = cfg["train"]
    MAX_EPOCH = cfg_train["max_epoch"]

    running_metrics_train = runningScore(CLASSES)
    running_metrics_train.reset()

    train_loader_iter = iter(train_loader)

    for i_train in range(len(train_loader)):
        train_images, train_labels = train_loader_iter.next()
        train_images, train_labels = train_images.to(DEVICE).to(torch.float32), train_labels.to(DEVICE).to(torch.int64)

        train_images, train_labels, _ = \
            generate_aug_data(train_images, train_labels, train_labels, mode="classmix")

        optimizer.zero_grad()

        pred_st, _ = model(train_images)

        running_metrics_train.update(train_labels.detach().cpu().numpy(), pred_st.detach().max(1)[1].cpu().numpy())

        loss = sce_loss(pred_st, train_labels)
        writer.add_scalar('loss', round(loss.item(), ROUND_NUM), i_train + epoch * len(train_loader))

        loss.backward()
        optimizer.step()

        logger.info(
            f'Train: epoch [{epoch}/{MAX_EPOCH - 1}] iteration [{i_train}] loss {round(loss.item(), ROUND_NUM)} ')

    lr_scheduler.step()

    train_score, train_sup_class_iou = running_metrics_train.get_scores()
    for key in train_score.keys():
        if key != 'Class Accuracy':
            writer.add_scalar('train/' + key, train_score[key], epoch)
            logger.info(f'Train: {key} {round(train_score[key], ROUND_NUM)}')


def validate(cfg,model,valid_loader,epoch,settings):
    global BEST_METRIC
    cfg_train = cfg["train"]

    logger = settings["logger"]
    DEVICE = settings["DEVICE"]
    ROUND_NUM = settings["ROUND_NUM"]
    sce_loss = settings["sce_loss"]
    writer = settings["writer"]
    LOG_DIR = settings["LOG_DIR"]
    CLASSES = settings["model_cfg"]["classes"]

    cfg_train = cfg["train"]
    MAX_EPOCH = cfg_train["max_epoch"]

    running_metrics_val = runningScore(CLASSES)
    running_metrics_val.reset()
    METRIC = cfg["valid"]["metric"]

    model.eval()
    with torch.no_grad():
        for i_val, (images_val, labels_val) in enumerate(valid_loader):
            images_val, labels_val = images_val.to(DEVICE).to(torch.float32), labels_val.to(DEVICE).to(torch.int64)
            outputs_val,_ = model(images_val)
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


if __name__ == "__main__":
    cfg_dir = "/volume/CONSS/conf/"
    cfg_path = cfg_dir + "seam_st.yaml"
    model_cfg_path = cfg_dir + "model.yaml"
    main(cfg_path, model_cfg_path)