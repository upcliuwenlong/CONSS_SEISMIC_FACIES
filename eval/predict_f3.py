import cv2
import torch
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import get_logger, runningScore
import os
import torch.nn.functional as F
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def predict(args):
    logger = get_logger('Predict F3')
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

    # inline*crossline*depth: 401*701*255
    train_labels = np.load(args.train_labels_path)
    # inline*crossline*depth: 200*701*255
    test1_labels = np.load(args.test1_labels_path)
    # inline*crossline*depth: 601*200*255
    test2_labels = np.load(args.test2_labels_path)
    # inline*crossline*depth->depth*crossline*inline: 255*901*601
    train_labels = np.concatenate([np.concatenate([test1_labels, train_labels], axis=0), test2_labels],axis=1).transpose(2, 1, 0)
    # data
    train_data = np.load(args.train_data_path)
    test1_data = np.load(args.test1_data_path)
    test2_data = np.load(args.test2_data_path)
    train_data = np.concatenate([np.concatenate([test1_data, train_data], axis=0), test2_data], axis=1).transpose(2, 1, 0)

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
    for step in range(1+int(pred_labels.shape[1] / SLIDE_WINDOW)):
        input_crossline_start = SLIDE_WINDOW * step
        if input_crossline_start+SLICE_WIDTH < pred_labels.shape[1]:
            input_crossline_end = input_crossline_start+SLICE_WIDTH
        else:
            input_crossline_end = pred_labels.shape[1]
            input_crossline_start = pred_labels.shape[1]-SLICE_WIDTH
        for inline in range(train_labels.shape[2]):
            logger.info(f'crossline: {str(input_crossline_start)+"-"+str(input_crossline_end)}, inline: {str(inline)}')
            img = train_data[:, input_crossline_start:input_crossline_end, inline]
            #padding
            input = cv2.copyMakeBorder(img, 0, 1, 0, 0, cv2.BORDER_REPLICATE)
            # hw->chw
            input = np.expand_dims(input, 0)
            # chw->bchw
            input = np.expand_dims(input, 0)
            input = torch.from_numpy(input).to(DEVICE).float()
            output = model(input)
            output = output[0] if isinstance(output,tuple) else output
            output = F.softmax(output,dim=1)[0, :, :-1, :]
            output = output.detach().cpu().numpy()
            if SHOW_IMG_FLAG:
                show_img = np.argmax(output,axis=0)
                cv2.imwrite(LOG_PATH+'/'+MODEL_NAME+'_prediction_f3.png',show_img*32)
                SHOW_IMG_FLAG = False
            pred_probs[:, :, input_crossline_start:input_crossline_end, inline] = \
                output[:, :, : SLIDE_WINDOW]
        pred_labels[:, input_crossline_start:input_crossline_end, :] = \
            np.argmax(
                pred_probs[:, :, input_crossline_start:input_crossline_end, :],
                axis=0)
    np.savez_compressed(LOG_PATH+'/'+MODEL_NAME+'_prediction_f3.npz', prediction=pred_labels.astype(np.uint8))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--slice_width', nargs='?', type=int, default=256,
                        help='Slice width')
    parser.add_argument('--slide_window', nargs='?', type=int, default=256,
                        help='Slide window')
    parser.add_argument('--classes', nargs='?', type=int, default=6,
                        help='Classes')
    parser.add_argument('--model_path', nargs='?', type=str,
                        default='/volume/CONSS_114/runs/f3_sup_DeepLabV3Plus/best_model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--train_data_path', nargs='?', type=str,
                        default='/volume/dataset/f3/train/train_seismic.npy',
                        help='Train data path')
    parser.add_argument('--train_labels_path', nargs='?', type=str,
                        default='/volume/dataset/f3/train/train_labels.npy',
                        help='Train labels path')
    parser.add_argument('--test1_data_path', nargs='?', type=str,
                        default='/volume/dataset/f3/test_once/test1_seismic.npy',
                        help='Test1 data path')
    parser.add_argument('--test1_labels_path', nargs='?', type=str,
                        default='/volume/dataset/f3/test_once/test1_labels.npy',
                        help='Test1 labels path')
    parser.add_argument('--test2_data_path', nargs='?', type=str,
                        default='/volume/dataset/f3/test_once/test2_seismic.npy',
                        help='Test2 data path')
    parser.add_argument('--test2_labels_path', nargs='?', type=str,
                        default='/volume/dataset/f3/test_once/test2_labels.npy',
                        help='Test2 labels path')
    args = parser.parse_args()
    predict(args)