import argparse
from utils.utils import *
import seaborn as sns
import matplotlib.pyplot as plt

def eval(args):
    logger = get_logger('Evaluate SEAM AI')
    logger.propagate = False
    LABEL_PATH = args.labels_path
    PREDICATION_PATH = args.predication_path
    cut_index = PREDICATION_PATH.rfind('/')
    LOG_PATH = PREDICATION_PATH[:cut_index]
    PREDICT_NAME = PREDICATION_PATH[cut_index + 1:-4]
    logger.info(f'{PREDICT_NAME}')
    CLASSES = args.classes
    labels_data = np.load(LABEL_PATH)['labels'] - 1
    predication = np.load(PREDICATION_PATH)['prediction']

    y_list = list(range(labels_data.shape[2]))
    ex_list = [0,100,200,300,400,500,589]
    for e in ex_list:
        y_list.remove(e)

    labels_data = labels_data[:, :, y_list]
    predication = predication[:, :, y_list]
    eval = runningScore(CLASSES)
    eval.update(labels_data,predication)
    eval_score, eval_class_iou = eval.get_scores()
    xticks = ['B/O', 'SMA', 'MTD', 'SMB', 'SV', 'SCS']
    yticks = ['B/O', 'SMA', 'MTD', 'SMB', 'SV', 'SCS']
    plt.figure(figsize=(5, 5), dpi=300)
    ax = sns.heatmap(eval.confusion_matrix / eval.confusion_matrix.sum(axis=1).reshape(CLASSES,1),
                     annot=True,
                     fmt=".3f",
                     linewidths=2,
                     square=True,
                     cbar=False,
                     xticklabels=xticks,
                     yticklabels=yticks, annot_kws={"fontsize": 8})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    ax.set_title('Class Accuracy', fontsize=8, family='DejaVu Sans')
    plt.tight_layout()
    plt.savefig(LOG_PATH + '/confusion_matrix.png', bbox_inches='tight', pad_inches=0,dpi=200)

    eval_txt = open(LOG_PATH + '/eval_score.txt', 'w')
    for key in eval_score.keys():
        eval_txt.write(key + ':' + str(eval_score[key]) + '\n')
        logger.info(f'{key} {eval_score[key]}')
    eval_txt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--labels_path', nargs='?', type=str,
                        default='/volume/dataset/seam_ai/labels_train.npz',
                        help='Labels path')
    parser.add_argument('--predication_path', nargs='?', type=str,
                        default='/volume/CONSS_114/runs/f3_conss_DeepLabV3Plus/best_model_prediction_f3.npz',
                        help='Predication path')
    parser.add_argument('--classes', nargs='?', type=int, default=6,
                        help='Classes')
    args = parser.parse_args()
    eval(args)