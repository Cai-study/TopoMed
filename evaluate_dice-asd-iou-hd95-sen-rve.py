import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm
from utils import read_list, read_nifti
from utils import config
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='mmwhs_ct2mr')
    parser.add_argument('--exp', type=str, default="fully")
    parser.add_argument('--folds', type=int, default=1)
    parser.add_argument('--split', type=str, default='test_mr')
    parser.add_argument('--modality', type=str, default='MR')

    args = parser.parse_args()

    config = config.Config(args.task)

    ids_list = read_list(args.split, task=args.task)
    results_all_folds = []

    txt_path = "./logs/" + args.exp + "/evaluation_res-dice-asd-iou-hd95-sen-rve_AA-LAC-LVC-MYO.txt"
    print('--txt_path', txt_path)
    print("\nEvaluating...")
    fw = open(txt_path, 'w')

    for fold in range(1, args.folds + 1):

        test_cls = [i for i in range(1, config.num_cls)]
        values = np.zeros((len(ids_list), len(test_cls), 6))

        for idx, data_id in enumerate(tqdm(ids_list)):
            pred = read_nifti(
                os.path.join("./logs/", args.exp, "fold" + str(fold), "predictions", f'{data_id}.nii.gz'))
            lb_path = os.path.join(config.save_dir, 'npy', f'{data_id}_label.npy')
            label = np.load(lb_path)

            padding_flag = label.shape[0] < config.patch_size[0] or label.shape[1] < config.patch_size[1] or \
                           label.shape[2] < config.patch_size[2]

            if padding_flag:
                pw = max((config.patch_size[0] - label.shape[0]) // 2 + 1, 0)
                ph = max((config.patch_size[1] - label.shape[1]) // 2 + 1, 0)
                pd = max((config.patch_size[2] - label.shape[2]) // 2 + 1, 0)
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            for i in test_cls:
                pred_i = (pred == i)
                label_i = (label == i)

                if pred_i.sum() > 0 and label_i.sum() > 0:
                    dice = metric.binary.dc(pred_i, label_i) * 100
                    asd = metric.binary.asd(pred_i, label_i)
                    iou = metric.binary.jc(pred_i, label_i) * 100
                    hd95 = metric.binary.hd95(pred_i, label_i)
                    sen = metric.binary.sensitivity(pred_i, label_i) * 100
                    rve = np.abs(pred_i.sum() - label_i.sum()) / label_i.sum() * 100  # 手动计算 RVE (以百分比表示)
                    values[idx][i - 1] = np.array([dice, asd, iou, hd95, sen, rve])
                elif pred_i.sum() > 0 and label_i.sum() == 0:
                    values[idx][i - 1] = np.array([0, 128, 0, 128, 0, 100])
                elif pred_i.sum() == 0 and label_i.sum() > 0:
                    values[idx][i - 1] = np.array([0, 128, 0, 128, 0, 100])
                elif pred_i.sum() == 0 and label_i.sum() == 0:
                    values[idx][i - 1] = np.array([100, 0, 100, 0, 100, 0])


        values = values[:, [3, 1, 2, 0], :]

        values_mean_cases = np.mean(values, axis=0)
        results_all_folds.append(values)


        fw.write(f"Fold {fold}\n")
        fw.write("Dice:\n" + str(np.round(values_mean_cases[:, 0], 1)) + '\n')
        fw.write("ASD:\n" + str(np.round(values_mean_cases[:, 1], 1)) + '\n')
        fw.write("IoU:\n" + str(np.round(values_mean_cases[:, 2], 1)) + '\n')
        fw.write("HD95:\n" + str(np.round(values_mean_cases[:, 3], 1)) + '\n')
        fw.write("Sensitivity:\n" + str(np.round(values_mean_cases[:, 4], 1)) + '\n')
        fw.write("RVE:\n" + str(np.round(values_mean_cases[:, 5], 1)) + '\n')

        fw.write(f"Average Dice: {np.mean(values_mean_cases, axis=0)[0]:.2f}\n")
        fw.write(f"Average ASD: {np.mean(values_mean_cases, axis=0)[1]:.2f}\n")
        fw.write(f"Average IoU: {np.mean(values_mean_cases, axis=0)[2]:.2f}\n")
        fw.write(f"Average HD95: {np.mean(values_mean_cases, axis=0)[3]:.2f}\n")
        fw.write(f"Average Sen: {np.mean(values_mean_cases, axis=0)[4]:.2f}\n")
        fw.write(f"Average RVE: {np.mean(values_mean_cases, axis=0)[5]:.2f}\n\n")


        print(f"Fold {fold}")
        print("Dice:\n", np.round(values_mean_cases[:, 0], 1))
        print("ASD:\n", np.round(values_mean_cases[:, 1], 1))
        print("IoU:\n", np.round(values_mean_cases[:, 2], 1))
        print("HD95:\n", np.round(values_mean_cases[:, 3], 1))
        print("Sensitivity:\n", np.round(values_mean_cases[:, 4], 1))
        print("RVE:\n", np.round(values_mean_cases[:, 5], 1))
        print(
            f"Averages -> Dice: {np.mean(values_mean_cases, axis=0)[0]:.2f}, ASD: {np.mean(values_mean_cases, axis=0)[1]:.2f}")

    results_all_folds = np.array(results_all_folds)

    fw.write('\n\nAll folds\n')
    results_folds_mean = results_all_folds.mean(0)

    for i in range(results_folds_mean.shape[0]):
        fw.write(f"Case {ids_list[i]}\n")
        fw.write(f"\tDice: {np.round(results_folds_mean[i][:, 0], 2).tolist()}\n")
        fw.write(f"\tASD: {np.round(results_folds_mean[i][:, 1], 2).tolist()}\n")
        fw.write(f"\tIoU: {np.round(results_folds_mean[i][:, 2], 2).tolist()}\n")
        fw.write(f"\tHD95: {np.round(results_folds_mean[i][:, 3], 2).tolist()}\n")
        fw.write(f"\tSen: {np.round(results_folds_mean[i][:, 4], 2).tolist()}\n")
        fw.write(f"\tRVE: {np.round(results_folds_mean[i][:, 5], 2).tolist()}\n")
        fw.write(f"\tAverage Dice: {np.mean(results_folds_mean[i], axis=0)[0]:.2f}\n")
        fw.write(f"\tAverage ASD: {np.mean(results_folds_mean[i], axis=0)[1]:.2f}\n")
        fw.write(f"\tAverage IoU: {np.mean(results_folds_mean[i], axis=0)[2]:.2f}\n")
        fw.write(f"\tAverage HD95: {np.mean(results_folds_mean[i], axis=0)[3]:.2f}\n")
        fw.write(f"\tAverage Sen: {np.mean(results_folds_mean[i], axis=0)[4]:.2f}\n")
        fw.write(f"\tAverage RVE: {np.mean(results_folds_mean[i], axis=0)[5]:.2f}\n\n")

    mean_metrics = results_folds_mean.mean(0)
    std_metrics = np.std(results_all_folds.mean(1).mean(1), axis=0)


    dice_str = '\t'.join([str(round(x, 1)) for x in mean_metrics[:, 0].tolist()])
    asd_str = '\t'.join([str(round(x, 1)) for x in mean_metrics[:, 1].tolist()])
    iou_str = '\t'.join([str(round(x, 1)) for x in mean_metrics[:, 2].tolist()])
    hd95_str = '\t'.join([str(round(x, 1)) for x in mean_metrics[:, 3].tolist()])
    sen_str = '\t'.join([str(round(x, 1)) for x in mean_metrics[:, 4].tolist()])
    rve_str = '\t'.join([str(round(x, 1)) for x in mean_metrics[:, 5].tolist()])

    fw.write('Final Dice of each class:\n' + dice_str + '\n')
    fw.write('Final ASD of each class:\n' + asd_str + '\n')
    fw.write('Final IoU of each class:\n' + iou_str + '\n')
    fw.write('Final HD95 of each class:\n' + hd95_str + '\n')
    fw.write('Final Sen of each class:\n' + sen_str + '\n')
    fw.write('Final RVE of each class:\n' + rve_str + '\n\n')

    final_avg_dice = round(mean_metrics.mean(0)[0], 2)
    final_avg_asd = round(mean_metrics.mean(0)[1], 2)
    final_avg_iou = round(mean_metrics.mean(0)[2], 2)
    final_avg_hd95 = round(mean_metrics.mean(0)[3], 2)
    final_avg_sen = round(mean_metrics.mean(0)[4], 2)
    final_avg_rve = round(mean_metrics.mean(0)[5], 2)

    fw.write(f'Final Avg Dice: {final_avg_dice} ± {round(std_metrics[0], 2)}\n')
    fw.write(f'Final Avg ASD: {final_avg_asd} ± {round(std_metrics[1], 2)}\n')
    fw.write(f'Final Avg IoU: {final_avg_iou} ± {round(std_metrics[2], 2)}\n')
    fw.write(f'Final Avg HD95: {final_avg_hd95} ± {round(std_metrics[3], 2)}\n')
    fw.write(f'Final Avg Sen: {final_avg_sen} ± {round(std_metrics[4], 2)}\n')
    fw.write(f'Final Avg RVE: {final_avg_rve} ± {round(std_metrics[5], 2)}\n')

    print("\n=============================================")
    print("Final Result for Copy-Pasting (AA, LAC, LVC, MYO):")
    print("Dice:\t", dice_str)
    print("ASD:\t", asd_str)
    print("IoU:\t", iou_str)
    print("HD95:\t", hd95_str)
    print("Sen:\t", sen_str)
    print("RVE:\t", rve_str)
    print("---------------------------------------------")
    print(f'Final Avg Dice: {final_avg_dice} ± {round(std_metrics[0], 2)}')
    print(f'Final Avg ASD: {final_avg_asd} ± {round(std_metrics[1], 2)}')
    print(f'Final Avg IoU: {final_avg_iou} ± {round(std_metrics[2], 2)}')
    print(f'Final Avg HD95: {final_avg_hd95} ± {round(std_metrics[3], 2)}')
    print(f'Final Avg Sen: {final_avg_sen} ± {round(std_metrics[4], 2)}')
    print(f'Final Avg RVE: {final_avg_rve} ± {round(std_metrics[5], 2)}')
    print("=============================================\n")

    fw.close()


    csv_path = txt_path.replace('.txt', '.csv')


    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f_csv:
        writer = csv.writer(f_csv)

        writer.writerow(['Metric', 'AA', 'LAC', 'LVC', 'MYO', 'Average'])
        writer.writerow(['Dice'] + [round(x, 1) for x in mean_metrics[:, 0]] + [final_avg_dice])
        writer.writerow(['ASD'] + [round(x, 1) for x in mean_metrics[:, 1]] + [final_avg_asd])
        writer.writerow(['IoU'] + [round(x, 1) for x in mean_metrics[:, 2]] + [final_avg_iou])
        writer.writerow(['HD95'] + [round(x, 1) for x in mean_metrics[:, 3]] + [final_avg_hd95])
        writer.writerow(['Sen'] + [round(x, 1) for x in mean_metrics[:, 4]] + [final_avg_sen])
        writer.writerow(['RVE'] + [round(x, 1) for x in mean_metrics[:, 5]] + [final_avg_rve])

