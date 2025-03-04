# 我需要编写一个测试用例看看哪里出了问题
import torchmetrics
from typing import Tuple, Dict
import argparse
import yaml
import os
import torch
from torchmetrics.classification import MulticlassPrecision

def build_metric_fn(config: dict) -> Tuple[torchmetrics.Metric, Dict[str, float]]:
    common_metric_fn_kwargs = {"task": config["task"],
                               "compute_on_cpu": config["compute_on_cpu"],
                               "sync_on_compute": config["sync_on_compute"]}
    if config["task"] == "multiclass":
        assert "num_classes" in config, "num_classes must be provided for multiclass task"
        common_metric_fn_kwargs["num_classes"] = config["num_classes"]
    elif config["task"] == "multilabel":
        assert "num_labels" in config, "num_labels must be provided for multilabel task"
        common_metric_fn_kwargs["num_labels"] = config["num_labels"]


    metric_list = []
    for metric_class_name in config["target_metrics"]:
        if isinstance(metric_class_name, dict):
            # e.g., {"AUROC": {"average": macro}}
            assert len(metric_class_name) == 1, f"Invalid metric name: {metric_class_name}"
            metric_class_name, metric_fn_kwargs = list(metric_class_name.items())[0]
            metric_fn_kwargs.update(common_metric_fn_kwargs)
        else:
            metric_fn_kwargs = common_metric_fn_kwargs
        assert isinstance(metric_class_name, str), f"metric name must be a string: {metric_class_name}"
        assert hasattr(torchmetrics, metric_class_name), f"Invalid metric name: {metric_class_name}"
        metric_class = getattr(torchmetrics, metric_class_name)
        metric_fn = metric_class(**metric_fn_kwargs)
        if  metric_class_name == "Specificity" and not hasattr(metric_fn, "higher_is_better"):
            # 手动定义 high_is_better
            metric_fn.higher_is_better = True
        metric_list.append(metric_fn)
    
    print(f"metric_list: {metric_list}")
    metric_fn = torchmetrics.MetricCollection(metric_list)

    best_metrics = {
        k: -float("inf") if v.higher_is_better else float("inf")
        for k, v in metric_fn.items()
    }

    return metric_fn, best_metrics


def parse():
    parser = argparse.ArgumentParser('ECG downstream training')

    # parser.add_argument('--model_name',
    #                     default="mvt_larger_larger",
    #                     type=str,
    #                     help='resume from checkpoint')
    
    parser.add_argument('--ckpt_dir',
                        default="../weights/multiblock_epoch100.pth",
                        type=str,
                        metavar='PATH',
                        help='pretrained encoder checkpoint')
    
    parser.add_argument('--output_dir',
                        default="./output/finetuning",
                        type=str,
                        metavar='PATH',
                        help='output directory')
    
    parser.add_argument('--dataset',
                        default="ptbxl",
                        type=str,
                        help='dataset name')
    
    parser.add_argument('--data_dir',
                        default="/mount/ecg/ptb-xl-1.0.3/",
                        type=str,
                        help='dataset directory')
    
    parser.add_argument('--task',
                        default="multiclass",
                        type=str,
                        help='downstream task')

    parser.add_argument('--data_percentage',
                        default=1.0,
                        type=float,
                        help='data percentage (from 0 to 1) to use in few-shot learning')
    
    parser.add_argument('--use_class_weight',
                        default=False,
                        action='store_true',
                        help='use class weight in loss function')



    # Use parse_known_args instead of parse_args
    args, unknown = parser.parse_known_args()


    with open(os.path.realpath(f'../configs/downstream/finetuning/fine_tuning_ejepa.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config

if __name__ == "__main__":
    # 准备配置信息
    config = parse()
    config['metric']['task'] = config['task']
    config['metric']['num_classes'] = 2

    metric_fn, best_metric = build_metric_fn(config['metric'])

    # 准备测试数据
    '''
    outputs: tensor([[0.6966, 0.3034],
        [0.3629, 0.6371],
        [0.7717, 0.2283],
        [0.7749, 0.2251],
        [0.7908, 0.2092],
        [0.5725, 0.4275],
        [0.7102, 0.2898],
        [0.7097, 0.2903]], device='cuda:0')
    targets: tensor([0, 0, 0, 0, 0, 0, 1, 1], device='cuda:0')
    '''

    outputs = torch.tensor([[0.6966, 0.3034],
                            [0.3629, 0.6371],
                            [0.7102, 0.2898],
                            [0.7097, 0.2903]])
    targets = torch.tensor([0, 0, 1, 1])
    
    # 计算metric
    metric_fn(outputs, targets)
    print(metric_fn.compute())

    # 使用MultiClassPrecision计算指标
    mcp = torchmetrics.Precision(task='multiclass', num_classes=2, average='macro')
    mcp(outputs, targets)
    print(mcp.compute())

    # 使用官方的MultiClassPrecision计算指标
    official_mcp = MulticlassPrecision(num_classes=2)
    official_mcp(outputs, targets)
    print(official_mcp.compute())

    print("Test passed!")