import glob
import numpy as np
import os
import json
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import random

all_clip_rates = []
fig = go.Figure()
all_colors = [list(np.random.choice(range(256), size=3)) for _ in range(50)]


def compute_variance(path, majority=0.53, metrics="eval_accuracy"):
    def plot_clip_rate_fig(seed_dirs, output_fn="clip_rate_over_time.pdf"):
        clip_rate = []
        for i in range(len(seed_dirs)):
            clip_rate.append([])
        for i, seed_dir in enumerate(seed_dirs):
            status_files = glob.glob(os.path.join(seed_dir, "gradClipMemoryJsons", "status*"))
            status_files.sort(key=lambda x: int(os.path.basename(x).strip(".json").strip("status_")))
            for status_file in status_files:
                with open(status_file, "r", encoding='utf-8') as f_in:
                    status_data = json.load(f_in)
                    keys = list(status_data.keys())
                    keys.remove("step")
                    denom = 0
                    numer = 0
                    for k in keys:
                        numer += status_data[k]["clipped_num"]
                        denom += status_data[k]["n_element"]
                    clip_rate[i].append(numer / denom)
        clip_rate = np.array(clip_rate)
        mean_clip_rate = np.mean(clip_rate, axis=0)
        # print(mean_clip_rate)
        all_clip_rates.append(mean_clip_rate)
        plt.plot([(x + 1) * 20 for x in range(len(mean_clip_rate))], mean_clip_rate)
        plt.xlabel("number of steps")
        plt.ylabel("clipping rates")
        plt.savefig(os.path.join(path, output_fn))
        print("clip rate over time plot has been saved at {}".format(os.path.join(path, output_fn)))

        plt.clf()

    def plot_norm_across_layer_over_time(seed_dirs, output_fn, plot_stat_name="current_param_norm"):
        fig.data = []
        norm_dict = {}
        # layer_param_names = [x for x in data_dict.keys() if "layer" in x]
        # param_types = [".".join(x.split(".")[4:]) for x in layer_param_names if "layer.0" in x]
        layer_param_names = None
        param_types = None
        NUM_LAYERS = 999
        num_steps = None
        for seed_dir_i, seed_dir in enumerate(seed_dirs):
            status_files = glob.glob(os.path.join(seed_dir, "gradClipMemoryJsons", "status*"))
            status_files.sort(key=lambda x: int(os.path.basename(x).strip(".json").strip("status_")))
            for status_file in status_files:
                with open(status_file, "r", encoding='utf-8') as f_in:
                    status_data = json.load(f_in)
                    parameter_names = list(status_data.keys())
                    parameter_names.remove("step")
                    # initialization
                    if layer_param_names is None:
                        layer_param_names = [x for x in parameter_names if "layer" in x]
                        param_types = [".".join(x.split(".")[4:]) for x in layer_param_names if "layer.0" in x]
                        NUM_LAYERS = len(set([int(x.split(".")[3]) for x in layer_param_names]))
                        for _seed_dir_i in range(len(seed_dirs)):
                            for layer_i in range(NUM_LAYERS):
                                if _seed_dir_i not in norm_dict:
                                    norm_dict[_seed_dir_i] = {}
                                norm_dict[_seed_dir_i][layer_i] = {param_type: [] for param_type in param_types}
                        num_steps = len(status_files)
                    for param_type in param_types:
                        subset_param_names = [x for x in layer_param_names if ".".join(x.split(".")[4:]) == param_type]
                        # assert len(subset_param_names) == NUM_LAYERS, f"{subset_param_names}"
                        if len(subset_param_names) != NUM_LAYERS:
                            print(
                                f"{subset_param_names} does not have mismatch number of params for {NUM_LAYERS} layers"
                                f" (has {len(subset_param_names)} layers)")
                            for layer_i in range(NUM_LAYERS):
                                norm_dict[seed_dir_i][layer_i].pop(param_type)
                            continue
                        for layer_i in range(NUM_LAYERS):
                            # key = f"bert.encoder.layer.{i}.{param_type}"
                            # key = f"roberta.encoder.layer.{layer_i}.{param_type}"
                            key = [x for x in subset_param_names if f"layer.{layer_i}." in x]
                            assert len(key) == 1, f"duplicate key {key} in layer {layer_i}, param_type: {param_type}"
                            key = key[0]
                            value = status_data[key][plot_stat_name]
                            norm_dict[seed_dir_i][layer_i][param_type].append(value)

                            # x = [0.1 * _num for _num in range(1, len(values) + 1)]
                            # discounted_values = []
                            # cur_value = 0
                            # for value_pair in values:
                            #     val, nelem = value_pair
                            #     cur_value = decay_value * cur_value + (1 - decay_value) * val / nelem
                            #     discounted_values.append(cur_value)
                            # discounted_values.append(val / nelem)
        # x = [(f_i + 1) / num_steps for f_i in range(num_steps)]
        steps = [(f_i + 1) * 20 for f_i in range(num_steps)]
        for param_type in norm_dict[0][0].keys():
            for layer_i in range(NUM_LAYERS):
                values = [norm_dict[x][layer_i][param_type] for x in norm_dict]
                values = np.array(values).mean(axis=0)
                fig.add_trace(
                    # go.Scatter(x=x, y=values,
                    go.Scatter(x=steps, y=values,
                               legendgroup=param_type,
                               legendgrouptitle_text=param_type,
                               name=f"layer.{layer_i}",
                               mode="lines",
                               line=dict(color=f"rgb{tuple(all_colors[layer_i])}"))
                )
        fig.update_layout(title=f"Task_{task}_{plot_stat_name}_across_layer_over_time",
                          # font_family="Courier New",
                          # font_color="black",
                          font=dict(
                              family="Arial",
                              size=18,
                              color="black"
                          ),
                          xaxis_title="#(Optimization Steps)",
                          yaxis_title="Grad Norm Value"

                          )
        os.makedirs(os.path.join(path, "norm_visualization"), exist_ok=True)
        fig.write_html(os.path.join(path, "norm_visualization", f"{output_fn}_{plot_stat_name}.html"))
        comp_output_path = os.path.join(path, "norm_visualization", f"component_figures_{output_fn.split('_')[0]}_{plot_stat_name}")
        os.makedirs(comp_output_path, exist_ok=True)
        fig.update_layout(title="")
        plt.rcParams["font.size"] = 15
        for param_type in norm_dict[0][0].keys():
            fig.data = []
            plt.clf()
            for layer_i in range(NUM_LAYERS):
                values = [norm_dict[x][layer_i][param_type] for x in norm_dict]
                values = np.array(values).mean(axis=0)
                # fig.add_trace(
                #     # go.Scatter(x=x, y=values,
                #     go.Scatter(x=steps, y=values,
                #                # legendgroup=param_type,
                #                # legendgrouptitle_text=param_type,
                #                name=f"layer.{layer_i}",
                #                mode="lines",
                #                line=dict(color=f"rgb{tuple(all_colors[layer_i])}"))
                # )
                plt.plot(steps, values, label=f"layer.{layer_i}", color=[x/255.0 for x in all_colors[layer_i]])
            # fig.write_html(os.path.join(comp_output_path, f"{param_type.replace('.', '_')}.html"))
            plt.xlabel("#(Optimization Steps)")
            plt.ylabel("Grad Norm Value")
            # plt.tight_layout()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                       ncol=4, fancybox=True, prop={"size": 10})
            plt.savefig(os.path.join(comp_output_path, f"{param_type.replace('.', '_')}.pdf"))
        layer_output_path = os.path.join(path, "norm_visualization", f"layer_figures_{output_fn.split('_')[0]}_{plot_stat_name}")
        os.makedirs(layer_output_path, exist_ok=True)
        fig.update_layout(title="")
        keys = list(norm_dict[0][0].keys())
        for layer_i in range(NUM_LAYERS):
            fig.data = []
            plt.clf()
            for param_type_i, param_type in enumerate(keys):
                values = [norm_dict[x][layer_i][param_type] for x in norm_dict]
                values = np.array(values).mean(axis=0)
                # fig.add_trace(
                #     # go.Scatter(x=x, y=values,
                #     go.Scatter(x=steps, y=values,
                #                # legendgroup=param_type,
                #                # legendgrouptitle_text=param_type,
                #                name=f"{param_type}",
                #                mode="lines",
                #                line=dict(color=f"rgb{tuple(all_colors[param_type_i])}"))
                # )
                plt.plot(steps, values, label=f"{param_type}", color=[x / 255.0 for x in all_colors[param_type_i]])
            ax = plt.gca()
            ax.set_ylim([0, 1.3])
            plt.xlabel("#(Optimization Steps)")
            plt.ylabel("Grad Norm Value")
            plt.tight_layout()
            # plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.5),
            #            ncol=2, fancybox=True, prop={"size": 10})
            # plt.subplots_adjust(top=0.7)
            # plt.show()
            # exit()
            plt.savefig(os.path.join(layer_output_path, f"{layer_i}.pdf"))
            # fig.write_html(os.path.join(layer_output_path, f"{layer_i}.html"))


        # clip_rate = np.array(clip_rate)
        # mean_clip_rate = np.mean(clip_rate, axis=0)
        # # print(mean_clip_rate)
        # all_clip_rates.append(mean_clip_rate)

    seed_dirs = glob.glob(os.path.join(path, "seed*"))

    accs = []
    failed_run_counter = 0
    failed_seed_dirs = []
    success_seed_dirs = []
    for seed_dir in seed_dirs:
        with open(os.path.join(seed_dir, "all_results.json"), "r", encoding='utf-8') as f_in:
            res = json.load(f_in)
            # if "eval_accuracy" in res:
            #     acc = res["eval_accuracy"]
            # else:
            #     acc = res["eval_matthews_correlation"]
            acc = res[metrics]
            accs.append(acc)
            if acc <= majority:
                failed_run_counter += 1
                failed_seed_dirs.append(seed_dir)
            else:
                success_seed_dirs.append(seed_dir)

    # print(f"std: {np.std(accs):.3f}, mean: {np.mean(accs):.3f}, max: {np.max(accs):.3f}, min: {np.min(accs):.3f} , failed_run_ratio: {failed_run_counter/len(accs): .2f}")
    print(failed_seed_dirs)
    print(
        f"min: {np.min(accs) * 100:.1f}, mean: {np.mean(accs) * 100:.1f}, max: {np.max(accs) * 100:.1f}, std: {np.std(accs) * 100:.1f} , failed_run_ratio: {failed_run_counter / len(accs) * 100: .1f}%")
    plot_clip_rate_fig(seed_dirs, "clip_rate_over_time.pdf")
    if len(failed_seed_dirs) > 0:
        plot_clip_rate_fig(failed_seed_dirs, "failed_clip_rate_over_time.pdf")
    else:
        print("no failed run, will not print figures for failed run")
    if len(success_seed_dirs) > 0:
        plot_clip_rate_fig(success_seed_dirs, "success_clip_rate_over_time.pdf")
    else:
        print("no success run, will not print figures for success run")

    if "baseline" not in path:
        if len(failed_seed_dirs) > 0:
            plot_norm_across_layer_over_time(failed_seed_dirs, "failed_norm_state_over_time", "current_param_norm")
            plot_norm_across_layer_over_time(failed_seed_dirs, "failed_norm_state_over_time", "previous_grad_norm")
        else:
            print("no failed run, will not print figures for failed run")
        if len(success_seed_dirs) > 0:
            plot_norm_across_layer_over_time(success_seed_dirs, "success_norm_state_over_time", "current_param_norm")
            plot_norm_across_layer_over_time(success_seed_dirs, "success_norm_state_over_time", "previous_grad_norm")
        else:
            print("no success run, will not print figures for failed run")


    return np.mean(accs), np.std(accs)


if __name__ == '__main__':
    # clip_values = ["999999", "0.0001", "1e-6", "0", "-1e-4"]
    # clip_values = ["1e5", "0.01", "0.05", "0.1", "0.5", "1", "5"]
    # clip_values = ["none"]
    clip_values = ["1e5", "0.05"]
    # clip_values = ["0.05"]
    # task = "rte"
    # task = "mrpc"
    # task = "cola"
    majority_vote_map = {
        "rte": 0.53,
        "cola": 0,
        "mrpc": 0.75,
        "qnli": 0.50
    }
    metrics_map = {
        "rte": "eval_accuracy",
        "mrpc": "eval_f1",
        "cola": "eval_matthews_correlation",
        "qnli": "eval_accuracy"
    }
    # for task in ["rte", "mrpc", "cola"]:
    # for task in ["mrpc"]:
    for task in ["rte", "cola"]:
        model = "bert-large-uncased"
        # from: https://arxiv.org/pdf/2006.04884.pdf
        accs = []
        stds = []
        visualization_path = os.path.join("visualization", task)
        os.makedirs(visualization_path, exist_ok=True)
        for clip_val in clip_values:
        # for clip_val in ["1e-5" ,"3e-5","5e-5"]:
            try:
                path = f"output/output/output/pre_correction_{model}_{task}_group_clip_by_norm_{clip_val}"
                print("doing evaluation for", path)
                acc, std = compute_variance(path, majority=majority_vote_map[task], metrics=metrics_map[task])
                accs.append(acc)
                stds.append(std)
            except Exception as e:
                continue
        number_of_colors = 8

        # colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        #          for i inn range(number_of_colors)]
        colors = ["green", "purple", "blue", "black", "red"]
        for clip_tags, clip_ratios, acc, std, color in zip(clip_values, all_clip_rates, accs, stds, colors):
            plt.plot([(x + 1) * 20 for x in range(len(clip_ratios))], clip_ratios,
                     label=f"clip-value-{clip_tags} (acc:{acc:.3f}, std: {std:.3f}",
                     color=color)
        plt.legend()
        plt.xlabel("number of steps")
        plt.ylabel("clipping rates")
        plt.clf()
    # plt.savefig(os.path.join(visualization_path, "clip_rate_over_time_all.pdf"))
