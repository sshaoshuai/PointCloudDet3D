import os, re
import glob
import sys
import copy
import json
import math
import gc
# matplotlib.use('Agg')
import threading
import concurrent.futures
from multiprocessing import Process
import numpy as np
from matplotlib import pyplot as plt
import torch

# Default evaluation dict format
# All values are 1D_LISTs
proto_exec_time_dict = {
    'End-to-end': [],
    'PreProcess': [],
    'VFE': [],
    'MapToBEV': [],
    'AnchorMask': [],
    'RPN-stage-1': [],
    'RPN-stage-2': [],
    'RPN-stage-3': [],
    'RPN-finalize': [],
    'RPN-total': [],
    'Pre-stage-1': [],
    'Post-stage-1': [],
    'PostProcess': [],
}

proto_AP_types_dict = {
    "aos": [],
    "3d": [],
    "bev": [],
    "image": [],
}

# Rows will be image bev 3d, cols will be easy medium hard
proto_AP_dict = {
    'Car': copy.deepcopy(proto_AP_types_dict),
    'Pedestrian': copy.deepcopy(proto_AP_types_dict),
    'Cyclist': copy.deepcopy(proto_AP_types_dict),
}

proto_mAP_dict = {
    "aos":0.0,
    '3d': 0.0,
    'bev': 0.0,
    'image': 0.0,
}

proto_eval_dict = {
    'method': 1,  # VAL
    'slice_size_perc': 100,  # VAL
    'min_slice_overlap_perc': 2,  # VAL
    'num_slices': 1,  # VAL
    'rpn_stg_exec_seqs': [],  # 2D_LIST
    'deadline_sec': 0.1,  # VAL
    'deadlines_missed': 0,  # VAL
    'deadlines_diffs': [],  # 1D_LIST
    'exec_times': proto_exec_time_dict,  # DICT
    'exec_time_stats': proto_exec_time_dict,  # DICT
    "AP": proto_AP_dict,
    "mAP": proto_mAP_dict,
}

# method number to method name
method_num_to_str = ['Baseline-1', 'Baseline-2', 'Baseline-3',
                     'Imprecise-no-slice', 'Imprecise-slice']


def merge_eval_dicts(eval_dicts):
    merged_ed = copy.deepcopy(eval_dicts[0])
    # use np.concatenate on a 2D list if you want to merge multiple 2D arrays
    for k, v in merged_ed.items():
        if not isinstance(v, dict):
            merged_ed[k] = [e[k] for e in eval_dicts]
    for k1 in ['exec_times', 'exec_time_stats', 'mAP']:
        for k2 in merged_ed[k1].keys():
            merged_ed[k1][k2] = [e[k1][k2] for e in eval_dicts]

    for cls in merged_ed['AP'].keys():
        for eval_type in merged_ed['AP'][cls].keys():
            merged_ed['AP'][cls][eval_type] = \
                [e['AP'][cls][eval_type] for e in eval_dicts]

    return merged_ed


inp_dir = sys.argv[1]

# each experiment has multiple eval dicts
def load_eval_dict(path):
    print('Loading', path)
    with open(path, 'r') as handle:
        eval_d = json.load(handle)
    eval_d['deadline_msec'] = int(eval_d['deadline_sec'] * 1000)

    # Copy AP dict with removing threshold info, like @0.70
    AP_dict_json = eval_d["eval_results_dict"]
    AP_dict = copy.deepcopy(proto_AP_dict)
    for cls_metric, AP in AP_dict_json.items():
        cls_metric, difficulty = cls_metric.split('/')
        if cls_metric == 'recall':
            continue
        cls, metric = cls_metric.split('_')
        AP_dict[cls][metric].append(AP)
    for v in AP_dict.values():
        for v2 in v.values():
            v2.sort() # sort according to difficulty
    eval_d["AP"] = AP_dict

    # Calculate mAP values
    eval_d["mAP"] = copy.deepcopy(proto_mAP_dict)
    for metric in eval_d["mAP"].keys():
        mAP, cnt = 0.0, 0
        for v in eval_d["AP"].values():
            mAP += sum(v[metric])  # hard medium easy
            cnt += len(v[metric])  # 3
        if cnt > 0:
            eval_d["mAP"][metric] = mAP / cnt
    return eval_d


exps_dict = {}
# load eval dicts
with concurrent.futures.ThreadPoolExecutor() as executor:
    futs = []
    for path in glob.glob(inp_dir + "/eval_dict_*"):
        futs.append(executor.submit(load_eval_dict, path))
    for f in concurrent.futures.as_completed(futs):
        ed = f.result()
        k = method_num_to_str[ed['method']-1]
        if k not in exps_dict:
            exps_dict[k] = []
        exps_dict[k].append(ed)

for exp, evals in exps_dict.items():
    # Sort according to deadlines
    evals.sort(key=lambda e: e['deadline_sec'])
    evals.sort(key=lambda e: e['deadline_msec'])
    print(exp)
    for e in evals:
        mAP_image, mAP_bev, mAP_3d = e["mAP"]['image'], e["mAP"]['bev'], e["mAP"]['3d']
        print('\tdeadline:', e['deadline_sec'], "\tmissed:", e['deadlines_missed'],
              f"\tmAP (image, bev, 3d):\t{mAP_image:.2f},\t{mAP_bev:.2f},\t{mAP_3d:.2f}")

merged_exps_dict = {}
for k, v in exps_dict.items():
    merged_exps_dict[k] = merge_eval_dicts(v)

# for plotting
colors = ['green', 'red', 'blue']
procs = []


# compare deadlines misses
def plot_func_dm(exps_dict):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    for exp_name, evals in exps_dict.items():
        x = [e['deadline_msec'] for e in evals]
        y = [e['deadlines_missed'] for e in evals]
        l2d = ax.plot(x, y, label=exp_name)
        ax.scatter(x, y, color=l2d[0].get_c())
    ax.invert_xaxis()
    ax.legend(fontsize='medium')
    ax.set_ylabel('Missed deadlines', fontsize='large')
    ax.set_xlabel('Deadline (msec)', fontsize='large')
    ax.grid('True', ls='--')
    fig.suptitle("Number of missed deadlines over different deadlines", fontsize=16)
    plt.savefig("exp_plots/deadlines_missed.jpg")


procs.append(Process(target=plot_func_dm, args=(exps_dict,)))
procs[-1].start()


def plot_func_eted(exps_dict):
    # compare execution times end to end
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    for exp_name, evals in exps_dict.items():
        x = [e['deadline_msec'] for e in evals]
        y = [e['exec_time_stats']['End-to-end'][1] for e in evals]
        l2d = ax.plot(x, y, label=exp_name)
        ax.scatter(x, y, color=l2d[0].get_c())
    ax.invert_xaxis()
    ax.legend(fontsize='medium')
    ax.set_ylabel('End-to-end time (msec)', fontsize='large')
    ax.set_xlabel('Deadline (msec)', fontsize='large')
    ax.grid('True', ls='--')
    fig.suptitle("Average end-to-end time over different deadlines", fontsize=16)
    plt.savefig("exp_plots/end-to-end_deadlines.jpg")


procs.append(Process(target=plot_func_eted, args=(exps_dict,)))
procs[-1].start()

def plot_func_hist(plot_dict, filename_prefix):
    for exp_name, merged_evals in plot_dict.items():
        h, w = len(merged_evals) // 2, 2
        if h == 0:
            h, w = 1, 1
        fig, axs = plt.subplots(h, w, figsize=(16, h * 4))
        if w == 1:
            axs = [axs]
        axs = np.concatenate(axs)

        for (i, ax), (key, val) in zip(enumerate(axs), merged_evals.items()):
            # Use 99 percentile
            x = np.concatenate(val)
            if len(x) > 0:
                perc99 = np.percentile(x, 99, interpolation='lower')
                # print(f"{exp_name} {key} 99 perc:{perc99}, max: {max(x)}")
                x = [et for et in x if et < perc99]
                ax.hist(x, 33)
            ax.set_xlabel(key + ' execution time bins (ms)')
            # #plot first instance timeline
            # ax.bar(np.arange(len(mrg_exe_times[key][0])), mrg_exe_times[key][0])
            # ax.set_xlabel('deadline ' + str(merged_evals['deadline_msec'][0]) + key +
            #               ' execution timeline (sample ID)')

        fig.suptitle("Execution time histogram of " + exp_name, fontsize=16)
        plt.savefig("exp_plots/" + filename_prefix + exp_name + "_exec_time_hist.jpg")


# better approach: use plot dict
def fill_plot_dict(plot_dict, exp_dict, keys, pred_keys=None):
    if pred_keys is None:
        pred_keys = list()
    for exp in exp_dict.keys():
        if exp not in plot_dict:
            plot_dict[exp] = {}
        for k in keys:
            if len(pred_keys) == 0:
                plot_dict[exp][k] = exp_dict[exp][k]
            elif len(pred_keys) == 1:
                plot_dict[exp][k] = \
                    exp_dict[exp][pred_keys[0]][k]
            elif len(pred_keys) == 2:
                plot_dict[exp][k] = \
                    exp_dict[exp][pred_keys[0]][pred_keys[1]][k]

plot_dict = {}
exec_keys_to_plot = [
    'PreProcess', 'RPN-stage-1', 'RPN-stage-2', 'RPN-stage-3',
    'RPN-finalize', 'PostProcess', 'End-to-end']
#    'RPN-finalize', 'RPN-total', 'Predict', 'End-to-end']
fill_plot_dict(plot_dict, merged_exps_dict, exec_keys_to_plot, ['exec_times'])
procs.append(Process(target=plot_func_hist, args=(plot_dict,  "ALL_", )))
procs[-1].start()
plot_dict.clear()

#exec_keys_to_plot = ['PFE', 'PillarGen', 'PillarPrep',
#                     'PillarFeatureNet', 'PillarScatter']
#fill_plot_dict(plot_dict, merged_exps_dict, exec_keys_to_plot, ['exec_times'])
#fill_plot_dict(plot_dict, merged_exps_dict, ['num_voxels'],)
#procs.append(Process(target=plot_func_hist, args=(plot_dict,  "PFE_", )))
#procs[-1].start()
#
#exec_keys_to_plot = ['exec_keys_to_plot']

def plot_func_sorted(plot_dict, x_key, filename_prefix):
    for exp_name, merged_evals in plot_dict.items():
        h = (len(merged_evals) - 1) // 2
        fig, axs = plt.subplots(h, 2, figsize=(16, h * 4))
        axs = np.concatenate(axs)

        x = np.concatenate(merged_evals[x_key])
        del merged_evals[x_key]
        sorted_indexes = np.argsort(x)
        # downsample indexes until array length is small enough
        target_size = 250
        if len(sorted_indexes) > target_size:
            mask = [True] + ([False] * math.floor(len(sorted_indexes) // target_size))
            mask = mask * math.ceil(len(sorted_indexes) / len(mask))
            sorted_indexes = sorted_indexes[mask[:len(sorted_indexes)]]

        x = x[sorted_indexes]

        for (i, ax), (key, val) in zip(enumerate(axs), merged_evals.items()):
            # Use 99 percentile
            y = np.concatenate(val)
            if len(y) > 0:
                y = y[sorted_indexes]
                perc99_idx = int(len(y) * 0.99)
                print(f"{exp_name} {key} 99 perc:{y[perc99_idx]}, max: {max(x)}")
                ax.plot(x[:perc99_idx], y[:perc99_idx])
                ax.grid('True', ls='--')
                ax.set_ylabel(key + ' execution time (ms)')
            ax.set_xlabel(x_key)
            # #plot first instance timeline
            # ax.bar(np.arange(len(mrg_exe_times[key][0])), mrg_exe_times[key][0])
            # ax.set_xlabel('deadline ' + str(merged_evals['deadline_msec'][0]) + key +
            #               ' execution timeline (sample ID)')

        fig.suptitle(exp_name + " - Number of voxels vs. Execution Time of PFE Phases", fontsize=16)
        plt.savefig("exp_plots/" + filename_prefix + exp_name + "_voxel_pfe_bar.jpg")

# plot dict was already filled except end to end
#fill_plot_dict(plot_dict, merged_exps_dict, ['End-to-end'], ['exec_times'])
#procs.append(Process(target=plot_func_sorted, args=(plot_dict, 'num_voxels', "", )))
#procs[-1].start()

# compare mAP for all types
fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
for ax, eval_type in zip(axs, proto_mAP_dict.keys()):
    for exp_name, evals in merged_exps_dict.items():
        x = evals['deadline_msec']
        y = evals['mAP'][eval_type]
        l2d = ax.plot(x, y, label=exp_name)
        ax.scatter(x, y, color=l2d[0].get_c())
    ax.invert_xaxis()
    ax.legend(fontsize='medium')
    ax.set_ylabel(eval_type + ' mAP', fontsize='large')
    ax.set_xlabel('Deadline (msec)', fontsize='large')
    ax.grid('True', ls='--')
fig.suptitle("mean Average Precision over different deadlines", fontsize=16)
plt.savefig("exp_plots/mAP_deadlines.jpg")

for p in procs:
    p.join()

exit(0)

# compare averaged AP of car bus pedestrian classes over changing deadlines
def plot_avg_AP(diff_slc, exps_dict):
    selected_classes = ['bus', 'car', 'pedestrian']
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for i, ax in enumerate(axs):
        cls = selected_classes[i]
        for exp_name, exp_dict in exps_dict.items():
            if diff_slc:
                x = exp_dict['slice_size_perc']
            else:
                x = exp_dict['deadline_sec']
            y = exp_dict['cls_APs'][cls][-1]  # averaged APs
            ax.plot(x, y, label=exp_name)
        ax.invert_xaxis()
        ax.legend(fontsize='medium')
        ax.set_ylabel(cls + ' average AP', fontsize='large')
        if diff_slc:
            ax.set_xlabel('Slice size percentage')
            ax.set_xticks(list(range(10, 100, 5)))
        else:
            ax.set_xlabel('Deadline (sec)', fontsize='large')
        ax.grid('True', ls='--')
    fig.suptitle("Average Precision over different deadlines", fontsize=16)
    if diff_slc:
        plt.savefig("exp_plots/avg_AP_slice.jpg")
    else:
        plt.savefig("exp_plots/avg_AP_deadlines.jpg")


procs.append(Process(target=plot_avg_AP, \
                     args=(exps_dict,)))
procs[-1].start()

def plot_impr_rem_hist(exps_dict):
    for exp_name, exp_dict in exps_dict.items():
        if 'imprecise' not in exp_name:
            continue
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
        for i, ax in enumerate(axs):
            dist = exp_dict['exec_times'][f"impr{i + 1}"]
            perc99 = np.percentile(dist, 99, interpolation='lower')
            perc1 = np.percentile(dist, 1, interpolation='lower')
            dist_ = [et for et in dist if et < perc99 and et > perc1]
            ax.hist(dist_, 33)
            ax.set_xlabel(f"Remaining time to execute {i} stages (ms, 99pct)")
            ax.text((perc99 + perc1) / 2, 20, f"Max: {perc99:.5}")
            print(f"Remaining time to execute {i + 1} stages is 99 perc:{perc99}")
            print(f"Remaining time to execute {i + 1} stages is max:", max(dist))

        fig.suptitle("Remaining time historgram for three imprecise stage cases", fontsize=16)
        plt.savefig("exp_plots/impr_hist.jpg")


procs.append(Process(target=plot_impr_rem_hist, \
                     args=(exps_dict,)))
procs[-1].start()



for exp_name, exp_dict in exps_dict.items():
    if diff_slc:
        # RPN scaling
        fig, axs = plt.subplots(4, 1, figsize=(16, 12))
        x = exp_dict['slice_size_perc']
        # x = exp_dict['stage1_slice_size']
        for stg, ax in enumerate(axs[:-2]):
            stg_key = f"RPN-stage-{stg + 2}"
            mam_dict = exp_dict['exec_times']
            # last one does not do slicing
            y = [mam_dict[stg_key][1][i] / mam_dict[stg_key][1][-1] * 100 - x[i] \
                 for i in range(len(mam_dict[stg_key][1]) - 1)]
            ax.bar(x[:-1], y, width=0.4)
            ax.set_ylabel(f'RPN stage {stg + 2} time scale error percentage')
            ax.set_xticks(list(range(10, 65, 5)))
            ax.set_xlabel('Slice size percentage')

        axs[-2].scatter(x, exp_dict['stage1_slice_size'])
        axs[-2].set_ylabel('Stage 1 slice height')
        axs[-2].set_xlabel('Slice size percentage')
        axs[-2].set_xticks(list(range(10, 105, 5)))
        axs[-2].grid('True', ls='--')

        axs[-1].scatter(x, exp_dict['num_slices'])
        axs[-1].set_ylabel('Number of slices')
        axs[-1].set_xlabel('Slice size percentage')
        axs[-1].set_xticks(list(range(10, 105, 5)))
        axs[-1].grid('True', ls='--')

        fig.suptitle("RPN execution time scaling", fontsize=16)
        plt.savefig("exp_plots/rpn_scaling_perc.jpg")

        # num slices vs slice size