
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def get_shot(cls_num_list):
    # FIXME here follow SADE (https://github.com/vanint/sade-agnosticlt)
    
    shot = {}
    cls_num_list = torch.tensor(cls_num_list)
    many_shot = cls_num_list > 100
    few_shot = cls_num_list < 20
    medium_shot = ~many_shot & ~few_shot
    
    shot['many_shot'] = many_shot
    shot['few_shot'] = few_shot
    shot['medium_shot'] = medium_shot
    return shot

def calibration(preds, labels, confidences, num_bins=15):
    assert(len(confidences) == len(preds))
    assert(len(confidences) == len(labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(labels[selected] == preds[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


@torch.no_grad()
def evaluate_all_metric(data_loader, model, device, cls_num):
    '''
        dataloader: [images(bz, 3, 224, 224), labels(bz, 1)]
        model: M(images) -> logits (bz, nClasses)
        device: cpu or gpu
        cls_num: list, each class images number
    '''
    
    model.eval()
    nClasses = len(cls_num)
    shot = get_shot(cls_num)
    many_shot = shot['many_shot']
    medium_shot = shot['medium_shot']
    few_shot = shot['few_shot']

    predList = np.array([])
    cfdsList = np.array([])
    grndList = np.array([])
    for images, labels in tqdm(data_loader):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            logits = model(images)
            cfds, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            cfds = cfds.detach().squeeze().cpu().numpy()
            preds = preds.detach().squeeze().cpu().numpy()
            cfdsList = np.concatenate((cfdsList, cfds))
            predList = np.concatenate((predList, preds))
            grndList = np.concatenate((grndList, labels))

    cali = calibration(predList, grndList, cfdsList, num_bins=15)
    ece = cali['expected_calibration_error']
    mce = cali['max_calibration_error']
    
    cfd_per_class = [0] * nClasses
    pdt_per_class = [0] * nClasses
    rgt_per_class = [0] * nClasses
    acc_per_class = [0] * nClasses
    gts_per_class = [0] * nClasses
    
    cfd_map = [[0] * nClasses for _ in range(nClasses)]
    cfd_cnt = [[0] * nClasses for _ in range(nClasses)]
    
    for c, g, p in zip(cfdsList, grndList, predList):
        cfd_map[int(p)][int(g)] += c
        cfd_cnt[int(p)][int(g)] += 1
        gts_per_class[int(g)] += 1
        pdt_per_class[int(p)] += 1
        if g == p:
            cfd_per_class[int(g)] += c
            rgt_per_class[int(g)] += 1
            
    for i in range(nClasses):
        cnt = rgt_per_class[i]
        if cnt != 0:
            acc_per_class[i] = np.round(cnt/gts_per_class[i] * 100, decimals=2)
            cfd_per_class[i] = np.round(cfd_per_class[i]/cnt * 100, decimals=2)
    
    for i in range(nClasses):
        for j in range(nClasses):
            if cfd_cnt[i][j] != 0:
                cfd_map[i][j] = cfd_map[i][j] / cfd_cnt[i][j]
    
    avg_acc = np.sum(rgt_per_class) / np.sum(gts_per_class)
    acc_per_class = np.array(acc_per_class)
    many_shot_acc = acc_per_class[many_shot].mean()
    medium_shot_acc = acc_per_class[medium_shot].mean()
    few_shot_acc = acc_per_class[few_shot].mean()

    cls_num = np.array(cls_num)
    pdt_per_class = np.array(pdt_per_class)
    gts_per_class = np.array(gts_per_class)
    q = pdt_per_class / np.sum(pdt_per_class)
    pt = gts_per_class / np.sum(gts_per_class)
    ps = cls_num / np.sum(cls_num)

    pdc_s = np.sum(pt * np.log(pt + 1e-6) - pt * np.log(ps + 1e-6))
    pdc_t = np.sum(pt * np.log(pt + 1e-6) - pt * np.log(q + 1e-6))

    # 'cfd_per_class': cfd_per_class,
    # 'rgt_per_class': rgt_per_class,
    # 'acc_per_class': acc_per_class,
    # 'gts_per_class': gts_per_class,
    # 'pdt_per_class': pdt_per_class,
    # 'cfd_map': cfd_map,
    # 'cfd_cnt': cfd_cnt 
    
    result = {
        'avg_acc': np.round(avg_acc*100, decimals=2).tolist(),
        'ece': np.round(ece*100, decimals=2).tolist(),
        'mce': np.round(mce*100, decimals=2).tolist(),
        'many' : np.round(many_shot_acc, decimals=2).tolist(),
        'medium' : np.round(medium_shot_acc, decimals=2).tolist(),
        'few' : np.round(few_shot_acc, decimals=2).tolist(),
        'pdc': np.round(float(pdc_t / pdc_s), decimals=2),
        'cfd_map': np.array(cfd_map),
        'cfd_cnt': np.array(cfd_cnt) 
    }
    return result

