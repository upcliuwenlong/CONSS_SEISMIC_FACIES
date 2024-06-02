import torch
import torch.nn.functional as F
from utils.utils import negative_index_sampler

def contra_loss(rep, label, mask, prob, strong_threshold=1.0, temp=0.5, num_queries=128, num_negatives=128):
    # high confidence pixels
    high_cfd_pixel = label * mask
    # b x h x w x feature_dimension
    rep = rep.permute(0, 2, 3, 1)
    strong_feat_list = []
    weak_feat_list = []
    strong_feat_num_list = []
    feat_center_list = []
    for i in range(label.shape[1]):
        # high confidence pixel for current class
        high_cfd_pixel_cur = high_cfd_pixel[:, i]
        prob_seg = prob[:, i, :, :]
        # unlabeled: weak< weak_mask <strong , labeled: weak_mask <strong
        weak_mask = (prob_seg < strong_threshold) * high_cfd_pixel_cur.bool()
        # strong_mask > strong
        strong_mask = (prob_seg > strong_threshold) * high_cfd_pixel_cur.bool()
        if strong_mask.sum() == 0 or weak_mask.sum() == 0:
            continue
        feat_center_list.append(torch.mean(rep[strong_mask], dim=0, keepdim=True))
        strong_feat_list.append(rep[strong_mask])
        weak_feat_list.append(rep[weak_mask])
        strong_feat_num_list.append(int(strong_mask.sum().item()))
    if len(strong_feat_num_list) < 2 or len(weak_feat_list) < 2:
        # classes < 2
        return torch.tensor(0.0)
    else:
        con_loss = torch.tensor(0.0)
        feat_centers = torch.cat(feat_center_list)
        class_num = len(strong_feat_num_list)
        classes = torch.arange(class_num)

        for i in range(class_num):
            # sample query vectors for this class
            query_idx = torch.randint(len(weak_feat_list[i]), size=(num_queries,))
            query_feat = weak_feat_list[i][query_idx]
            with torch.no_grad():
                # generate index for the current query class
                class_index = torch.cat(([classes[i:], classes[:i]]))

                # similarity between classes
                classes_sim = torch.cosine_similarity(feat_centers[class_index[0]].unsqueeze(0), feat_centers[class_index[1:]], dim=1)
                sample_prob = torch.softmax(classes_sim / temp, dim=0)

                # get negative samples
                negative_dist = torch.distributions.categorical.Categorical(probs=sample_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(sample_prob))], dim=1)
                negative_num_list = strong_feat_num_list[i+1:] + strong_feat_num_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)

                negative_feat_all = torch.cat(strong_feat_list[i+1:] + strong_feat_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, rep.shape[3])

                # positive cat negative
                positive_feat = feat_centers[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            sim_logits = torch.cosine_similarity(query_feat.unsqueeze(1), all_feat, dim=2)
            # InfoNCE loss
            con_loss = con_loss + F.cross_entropy(sim_logits / temp, torch.zeros(num_queries).long().to(sim_logits.device))
        return con_loss / class_num
