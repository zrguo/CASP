import torch
import argparse


parser = argparse.ArgumentParser(description="CASP")
parser.add_argument(
    "--pseudolabel", type=str, default="", help="pseudolabel path"
)
parser.add_argument(
    "--quantile", type=float, default=0.95, help="stable score quantile"
)
parser.add_argument(
    "--selected_indice",
    type=str,
    default="",
    help="output path of selected label indice",
)
parser.add_argument(
    "--selected_label",
    type=str,
    default="",
    help="output path of selected labels",
)

args = parser.parse_args()

checkpoints = torch.load(args.pseudolabel)


def cal_stability(labels):
    res = []
    for i in range(labels.shape[0] - 1):
        res.append(torch.abs(labels[i + 1] - labels[i]).unsqueeze(0))
    res = torch.cat(res)
    res = torch.mean(res, dim=0)

    mean_label = torch.mean(labels, dim=0)
    threshold = torch.quantile(res, args.quantile)
    indices = torch.nonzero(res > threshold).squeeze()
    indice_labels = mean_label[indices]
    torch.save(indices, args.selected_indice)
    torch.save(indice_labels, args.selected_label)


if __name__ == "__main__":
    cal_stability(checkpoints)
