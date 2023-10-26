import yaml
import numpy as np

P = 1.5


def score_single(gt, preds):
    gt = {tuple([float(i.strip()) for i in k.strip('()').split(',')]): v for k, v in gt.items()}
    if preds is None:
        preds = dict()
    else:
        preds = {tuple([float(i.strip()) for i in k.strip('()').split(',')]): v for k, v in preds.items()}
    M1M3_error_scores = {}
    M2_error = 0
    table = np.array([list(i) for i in gt.values()])
    table_size = table.max(axis=0)
    boxes = np.array([list(i) for i in gt.keys()])
    for gt_idx in table:  # default to M3 group
        M1M3_error_scores[tuple(gt_idx)] = len(table[table[:, 0] == gt_idx[0]]) ** P + \
                                           len(table[table[:, 1] == gt_idx[1]]) ** P
    matched_gt_points = np.zeros(len(table), dtype=bool)
    for point, idx in preds.items():
        point_box_idx = np.logical_and(
            np.logical_and(point[0] >= boxes[:, 0], point[0] <= boxes[:, 2]),
            np.logical_and(point[1] >= boxes[:, 1], point[1] <= boxes[:, 3]),
        )
        if np.sum(point_box_idx) == 1:  # M1
            gt_idx = table[point_box_idx][0]
            error = abs(gt_idx[0] - idx[0]) ** P + \
                    abs(gt_idx[1] - idx[1]) ** P
            if matched_gt_points[point_box_idx]:
                M1M3_error_scores[tuple(gt_idx)] = min(error, M1M3_error_scores[tuple(gt_idx)])
                M2_error += np.sum(table_size ** P)
            else:
                M1M3_error_scores[tuple(gt_idx)] = error
                matched_gt_points[point_box_idx] = True
        elif np.sum(point_box_idx) == 0:  # M2
            M2_error += np.sum(table_size ** P)
        else:
            centers_creator = lambda b: np.array([[(x[0] + x[2]) / 2, (x[1] + x[3]) / 2] for x in b])
            box_centers = centers_creator(boxes[point_box_idx])
            dists = np.linalg.norm(box_centers - point)
            best_box_idx = np.argmin(dists)
            gt_idx = table[point_box_idx][best_box_idx]
            error = abs(gt_idx[0] - idx[0]) ** P + \
                    abs(gt_idx[1] - idx[1]) ** P
            if matched_gt_points[point_box_idx][best_box_idx]:
                M1M3_error_scores[tuple(gt_idx)] = min(error, M1M3_error_scores[tuple(gt_idx)])
                M2_error += np.sum(table_size ** P)
            else:
                M1M3_error_scores[tuple(gt_idx)] = error
                matched_gt_points[point_box_idx][best_box_idx] = True
    unmatched_gt_points = len(matched_gt_points) - sum(matched_gt_points)
    M2_error += unmatched_gt_points * np.sum(table_size ** P)
    return M2_error + sum(M1M3_error_scores.values())


def main(gt, pred):
    with open(gt, 'r') as f:
        gt_ds = yaml.safe_load(f)
    with open(pred, 'r') as f:
        pred_ds = yaml.safe_load(f)
    imgs = list(gt_ds.keys())
    error_rates = {}
    for img in imgs:
        error_rates[img] = score_single(gt_ds[img], pred_ds.get(img, None))
    mean_score = np.mean(list(error_rates.values()))
    return mean_score


if __name__ == '__main__':
    score = main(
        './test/labels/labels.yaml',
        './output/preds.yaml'
    )
    print(score)
