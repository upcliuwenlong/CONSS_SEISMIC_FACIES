import argparse
import cv2
from utils.utils import *
import warnings
import os
import torch.nn.functional as F
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def predict(args):
    logger = get_logger('Predict SEAM AI')
    logger.propagate = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ClASSES = args.classes
    SLIDE_WINDOW = args.slide_window
    SLICE_WIDTH = args.slice_width
    MODEL_PATH = args.model_path
    cut_index = MODEL_PATH.rfind('/')
    LOG_PATH = MODEL_PATH[:cut_index]
    MODEL_NAME = MODEL_PATH[cut_index+1:-4]
    logger.info(f'Model {MODEL_NAME}')
    logger.info(f'Loading data...')
    train_data = np.load(args.train_data_path)['data']
    # label：1-6 -> 0-5
    train_labels = np.load(args.labels_data_path)['labels'] - 1

    pred_labels = np.ones_like(train_data) * -1

    shape = tuple([ClASSES] + list(pred_labels.shape))
    pred_probs = np.zeros(shape)

    logger.info(f'Data loaded...')
    # load model:
    model = torch.load(MODEL_PATH)
    total = sum([param.nelement() for param in model.parameters()])
    logger.info(f"Model total parameters {total}")
    # Send to GPU if available
    logger.info(f"Sending the model to {DEVICE}")
    model = model.to(DEVICE)
    logger.info(f'Predict start...')
    SHOW_IMG_FLAG = True
    for step in range(1 + int(pred_labels.shape[1] / SLIDE_WINDOW)):
        input_crossline_start = SLIDE_WINDOW * step
        if input_crossline_start + SLICE_WIDTH < pred_labels.shape[1]:
            input_crossline_end = input_crossline_start + SLICE_WIDTH
        else:
            input_crossline_end = pred_labels.shape[1]
            input_crossline_start = pred_labels.shape[1] - SLICE_WIDTH
        for inline in range(train_labels.shape[2]):
            logger.info(
                f'crossline: {str(input_crossline_start) + "-" + str(input_crossline_end)}, inline: {str(inline)}')
            img = train_data[:, input_crossline_start:input_crossline_end, inline]
            # padding
            input = cv2.copyMakeBorder(img, 9, 9, 0, 0, cv2.BORDER_REPLICATE)
            # hw->chw
            input = np.expand_dims(input, 0)
            # chw->bchw
            input = np.expand_dims(input, 0)
            input = torch.from_numpy(input).to(DEVICE).float()
            output = model(input)
            output = output[0] if isinstance(output, tuple) else output
            output = F.softmax(output, dim=1)[0, :, 9:-9, :]
            output = output.detach().cpu().numpy()
            if SHOW_IMG_FLAG:
                show_img = np.argmax(output, axis=0)
                cv2.imwrite(LOG_PATH + '/' + MODEL_NAME + '_prediction_seam_ai.png', show_img * 32)
                SHOW_IMG_FLAG = False
            pred_probs[:, :, input_crossline_start:input_crossline_end, inline] = \
                output[:, :, : SLIDE_WINDOW]
        pred_labels[:, input_crossline_start:input_crossline_end, :] = \
            np.argmax(
                pred_probs[:, :, input_crossline_start:input_crossline_end, :],
                axis=0)
    np.savez_compressed(LOG_PATH + '/' + MODEL_NAME + '_prediction_seam_ai.npz', prediction=pred_labels.astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--slice_width', nargs='?', type=int, default=256,
                        help='Slice width')
    parser.add_argument('--slide_window', nargs='?', type=int, default=256,
                        help='Slide window')
    parser.add_argument('--classes', nargs='?', type=int, default=6,
                        help='Classes')
    parser.add_argument('--model_path', nargs='?', type=str,
                        default='/volume/CONSS_114/runs/seam_sup_DeepLabV3Plus/best_model.pkl')
    parser.add_argument('--train_data_path', nargs='?', type=str,
                        default='/volume/dataset/seam_ai/data_train.npz',
                        help='Train data path')
    parser.add_argument('--labels_data_path', nargs='?', type=str,
                        default='/volume/dataset/seam_ai/labels_train.npz',
                        help='Labels data path')
    args = parser.parse_args()
    predict(args)