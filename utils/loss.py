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
        weak_mask = (prob_seg < strong_threshold) * high_cfd_pixel_cur.bool()  # select hard queries
        # strong_mask > strong
        strong_mask = (prob_seg > strong_threshold) * high_cfd_pixel_cur.bool()
        if strong_mask.sum() == 0 or weak_mask.sum() == 0:
            continue
        feat_center_list.append(torch.mean(rep[strong_mask], dim=0, keepdim=True))
        strong_feat_list.append(rep[strong_mask])
        weak_feat_list.append(rep[weak_mask])
        strong_feat_num_list.append(int(strong_mask.sum().item()))
    # contrastive loss
    if len(strong_feat_num_list) < 2 or len(weak_feat_list) < 2:
        # classes < 2
        return torch.tensor(0.0)
    else:
        con_loss = torch.tensor(0.0)
        feat_centers = torch.cat(feat_center_list)
        class_num = len(strong_feat_num_list)
        classes = torch.arange(class_num)

        for i in range(class_num):
            # sample queries
            query_idx = torch.randint(len(weak_feat_list[i]), size=(num_queries,))
            query_feat = weak_feat_list[i][query_idx]
            with torch.no_grad():
                # generate index for the current query class
                class_index  = torch.cat(([classes[i:], classes[:i]]))

                # similarity between classes
                classes_sim = torch.cosine_similarity(feat_centers[class_index[0]].unsqueeze(0), feat_centers[class_index[1:]], dim=1)
                sample_prob = torch.softmax(classes_sim / temp, dim=0)

                negative_dist = torch.distributions.categorical.Categorical(probs=sample_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(sample_prob))], dim=1)
                negative_num_list = strong_feat_num_list[i+1:] + strong_feat_num_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)

                # sample negative
                negative_feat_all = torch.cat(strong_feat_list[i+1:] + strong_feat_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, rep.shape[3])

                # positive | negative
                positive_feat = feat_centers[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            sim_logits = torch.cosine_similarity(query_feat.unsqueeze(1), all_feat, dim=2)
            con_loss = con_loss + F.cross_entropy(sim_logits / temp, torch.zeros(num_queries).long().to(sim_logits.device))
        return con_loss / class_num

def single_contra_loss(rep, label, mask, temp=0.5, num_queries=128, num_negatives=128):
    # confidence pixels
    high_cfd_pixel = label * mask
    # b x h x w x feature_dimension
    rep = rep.permute(0, 2, 3, 1)
    strong_feat_list = []
    weak_feat_list = []
    strong_feat_num_list = []
    feat_center_list = []
    for i in range(label.shape[1]):
        # high confidence pixel for current class
        cfd_pixel_cur = high_cfd_pixel[:, i]
        weak_mask = cfd_pixel_cur.bool()  # select hard queries
        strong_mask = cfd_pixel_cur.bool()
        if strong_mask.sum() == 0 or weak_mask.sum() == 0:
            continue
        feat_center_list.append(torch.mean(rep[strong_mask], dim=0, keepdim=True))
        strong_feat_list.append(rep[strong_mask])
        weak_feat_list.append(rep[weak_mask])
        strong_feat_num_list.append(int(strong_mask.sum().item()))
    # contrastive loss
    if len(strong_feat_num_list) < 2 or len(weak_feat_list) < 2:
        # classes < 2
        return torch.tensor(0.0)
    else:
        con_loss = torch.tensor(0.0)
        feat_centers = torch.cat(feat_center_list)
        class_num = len(strong_feat_num_list)
        classes = torch.arange(class_num)

        for i in range(class_num):
            # sample queries
            query_idx = torch.randint(len(weak_feat_list[i]), size=(num_queries,))
            query_feat = weak_feat_list[i][query_idx]
            with torch.no_grad():
                # generate index for the current query class
                class_index  = torch.cat(([classes[i:], classes[:i]]))

                # similarity between classes
                classes_sim = torch.cosine_similarity(feat_centers[class_index[0]].unsqueeze(0), feat_centers[class_index[1:]], dim=1)
                sample_prob = torch.softmax(classes_sim / temp, dim=0)

                negative_dist = torch.distributions.categorical.Categorical(probs=sample_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(sample_prob))], dim=1)
                negative_num_list = strong_feat_num_list[i+1:] + strong_feat_num_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)

                # sample negative
                negative_feat_all = torch.cat(strong_feat_list[i+1:] + strong_feat_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, rep.shape[3])

                # positive | negative
                positive_feat = feat_centers[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            sim_logits = torch.cosine_similarity(query_feat.unsqueeze(1), all_feat, dim=2)
            con_loss = con_loss + F.cross_entropy(sim_logits / temp, torch.zeros(num_queries).long().to(sim_logits.device))
        return con_loss / class_num


def CONloss(input, ue_map, soft_pseudo_label):
    input = F.softmax(input, dim=1)
    loss_mat = F.mse_loss(input, soft_pseudo_label, reduction='none')
    mask = (ue_map == 1).unsqueeze(1).expand_as(loss_mat)
    loss_mat = loss_mat[mask]
    if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(input.device)
    return loss_mat.mean()

def SCEloss(input, label,smooth_factor=0,ignore_index=255):
    return smp.losses.SoftCrossEntropyLoss(smooth_factor=smooth_factor, ignore_index=ignore_index)(input, label)

def CEloss(inputs, labels, ignore_index=255):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(inputs, labels)

def STloss(logits, ue_map, pseudo_label):
    pseudo_label = torch.where(ue_map == 1, pseudo_label, 255)
    st_loss = CEloss(logits, pseudo_label, 255)  #
    return st_loss

def REloss(con_map, labels_l, logits):
    # 1 is correct 0 is error
    label_for_re= torch.where(logits.argmax(1)==labels_l, 1.0, 0.0).to(con_map.device)

    re_loss = nn.BCEWithLogitsLoss(reduction="none")(con_map, label_for_re.unsqueeze(1))

    zero_loss = re_loss.squeeze(1)[label_for_re == 0].numel()
    one_loss = re_loss.squeeze(1)[label_for_re == 1].numel()
    mul_fact = one_loss / zero_loss
    re_loss.squeeze(1)[label_for_re == 0] *= mul_fact

    return re_loss.mean()