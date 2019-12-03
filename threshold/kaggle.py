import os
import cv2
import numpy as np
import pandas as pd
from tqdm import trange
from torch.utils.data import DataLoader
import torch



# TODO: do
def write_list_to_file(*args):
    pass


# TODO: do
def read_list_from_file(*args):
    pass


def main():
    pass


## https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97225#latest-570515
## https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/evaluation

#############################################################################################

def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start

    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i - 1], length[i]])

    rle = ' '.join([str(r) for r in rle])
    return rle


def kaggle_metric_one(predict, truth):
    if truth.sum() == 0:
        if predict.sum() == 0:
            return 1
        else:
            return 0

    # ----
    predict = predict.reshape(-1)
    truth = truth.reshape(-1)

    intersect = predict * truth
    union = predict + truth
    dice = 2.0 * intersect.sum() / union.sum()
    return dice


def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predict = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = 1
            num += 1
    return predict, num


def compute_metric(test_id, test_truth, test_probability):
    test_num = len(test_truth)
    truth = test_truth.reshape(test_num, -1)
    probability = test_probability.reshape(test_num, -1)

    loss = - truth * np.log(probability) - (1 - truth) * np.log(1 - probability)
    loss = loss.mean()

    t = (truth > 0.5).astype(np.float32)
    p = (probability > 0.5).astype(np.float32)
    t_sum = t.sum(-1)
    p_sum = p.sum(-1)
    neg_index = np.where(t_sum == 0)[0]
    pos_index = np.where(t_sum >= 1)[0]

    dice_neg = (p_sum == 0).astype(np.float32)
    dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1) + 1e-12)
    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]
    dice = np.concatenate([dice_pos, dice_neg])

    dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
    dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
    dice = dice.mean()

    return loss, dice, dice_neg, dice_pos


def compute_kaggle_lb(test_id, test_truth, test_probability, threshold, min_size):
    test_num = len(test_truth)

    kaggle_pos = []
    kaggle_neg = []
    for b in trange(test_num, leave=False):
        truth = test_truth[b, 0]
        probability = test_probability[b, 0]

        if truth.shape != (1024, 1024):
            truth = cv2.resize(truth, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            truth = (truth > 0.5).astype(np.float32)

        if probability.shape != (1024, 1024):
            probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

        # -----
        predict, num_component = post_process(probability, threshold, min_size)

        score = kaggle_metric_one(predict, truth)

        if truth.sum() == 0:
            kaggle_neg.append(score)
        else:
            kaggle_pos.append(score)

    kaggle_neg = np.array(kaggle_neg)
    kaggle_pos = np.array(kaggle_pos)
    kaggle_neg_score = kaggle_neg.mean()
    kaggle_pos_score = kaggle_pos.mean()
    kaggle_real_score = np.vstack([kaggle_pos, kaggle_neg]).mean()
    kaggle_score = 0.7886 * kaggle_neg_score + (1 - 0.7886) * kaggle_pos_score

    return kaggle_real_score, kaggle_score, kaggle_neg_score, kaggle_pos_score


###################################################################################


def do_evaluate(net, test_loader):
    test_id = []
    test_probability = []
    test_truth = []
    test_num = 0

    for b, (input, truth, infor) in enumerate(test_loader):
        net.eval()
        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit = net(input)
            probability = torch.sigmoid(logit)

            # batch_size, C, H, W = probability.shape
            # if H!=1024 or W!=1024:
            #     probability = F.interpolate(probability,size=(1024,1024), mode='bilinear', align_corners=False)

        # ---
        batch_size = len(infor)
        test_id.extend([i.image_id for i in infor])
        test_probability.append(probability.data.cpu().numpy())
        test_truth.append(truth.data.cpu().numpy())
        test_num += batch_size

    assert (test_num == len(test_loader.dataset))

    test_truth = np.concatenate(test_truth)
    test_probability = np.concatenate(test_probability)

    return test_probability, test_truth, test_id


def find_theshold(test_truth, test_probability, threshold_list, min_area_list, basename):
    os.makedirs("thresholds_dump", exist_ok=True)
    test_id = None

    loss, dice, dice_neg, dice_pos = \
        compute_metric(test_id, test_truth, test_probability)
    print("Loss, Dice, Dice neg, Dice Pos")
    print(loss, dice, dice_neg, dice_pos)
    fname = "thresholds_dump/{}_find_threshold_dump.npy".format(basename)
    print("Will Save into", fname)

    scores = []
    total_steps = len(thresh_list) * len(min_area_list)
    step = 0
    for min_size in min_area_list:
        for threshold in threshold_list:
            print("step:", step, "/", total_steps)
            step += 1
            kaggle_real_score, kaggle_score, kaggle_neg_score, kaggle_pos_score = \
                compute_kaggle_lb(test_id, test_truth, test_probability, threshold, min_size)
            print(threshold, min_size, kaggle_score, kaggle_neg_score, kaggle_pos_score)

            scores.append([threshold, min_size, kaggle_real_score, kaggle_score, kaggle_neg_score, kaggle_pos_score])

    print("Saved into", fname)
    np.save(fname, np.array(scores, dtype=np.float32))

    # ===================================================================
    if 0:

        image_id = test_id
        encoded_pixel = []
        for b in range(num_test):
            probability = test_probability[b, 0]
            if probability.shape != (1024, 1024):
                probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

            predict, num_predict = post_process(probability, threshold, min_size)

            if num_predict == 0:
                encoded_pixel.append('-1')
            else:
                r = run_length_encode(predict)
                encoded_pixel.append(r)

        df = pd.DataFrame(list(zip(image_id, encoded_pixel)), columns=['ImageId', 'EncodedPixels'])
        df.to_csv(csv_file, columns=['ImageId', 'EncodedPixels'], index=False)

        print('\n')
        print('threshold = %0.5f\n' % (threshold))
        print('min_size  = %d\n' % (min_size))
        print('\n')
        print('id_file      = %s\n' % (id_file))
        print('predict_file = %s\n' % (predict_file))
        print('csv_file     = %s\n' % (csv_file))
        print('\n')
        print('test_id = %d\n' % (len(test_id)))
        print('test_probability = %s\n' % (str(test_probability.shape)))
        print('\n')


####################################################################################
'''
threshold = 0.900000
min_size  = 3500

kaggle_neg_score  :  0.983333
kaggle_pos_score  :  0.286285
kaggle_score      :  0.835977

'''

# main #################################################################
if __name__ == '__main__':
    basename = "accum_bug"
    truth_file_path = "model_preds/{}_truth.npy".format(basename)
    probability_file_path = "model_preds/{}_probability.npy".format(basename)
    test_truth = np.load(truth_file_path)
    test_probability = np.load(probability_file_path)

    thresh_list = np.arange(0.2, 0.9999, 0.05)
    min_area_list = np.arange(0, 7000, 100)
    find_theshold(test_truth, test_probability, thresh_list, min_area_list, "second_" + basename)
