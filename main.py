import random
import torch
import argparse
from utils.util import *
from utils import base_model_train as base_train
from utils import train
from utils import pseudo_train
from utils.dataloader import getdataloader


parser = argparse.ArgumentParser(description="CASP")

parser.add_argument(
    "--backbone", type=str, default="latefusion", help="latefusion/earlyfusion"
)
parser.add_argument(
    "--pretrained_model", type=str, default="", help="pretrained model path"
)
parser.add_argument(
    "--datapath",
    type=str,
    default="",
    help="dataset path",
)
parser.add_argument(
    "--stage", type=str, default="pretrain", help="pretrain/contrastive/pseudo"
)
parser.add_argument(
    "--dataset", type=str, default="mosi", help="dataset to use (mosi/mosei/sims)"
)
parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
parser.add_argument(
    "--embed_dropout", type=float, default=0.25, help="embedding dropout"
)
parser.add_argument(
    "--res_dropout", type=float, default=0.1, help="residual block dropout"
)
parser.add_argument(
    "--out_dropout", type=float, default=0.0, help="output layer dropout"
)
parser.add_argument(
    "--nlevels",
    type=int,
    default=5,
    help="number of layers in the network (default: 5)",
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=8,
    help="number of heads for the transformer network (default: 5)",
)
parser.add_argument("--proj_dim", type=int, default=40)
parser.add_argument(
    "--batch_size", type=int, default=24, metavar="N", help="batch size (default: 24)"
)
parser.add_argument(
    "--clip", type=float, default=0.8, help="gradient clip value (default: 0.8)"
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="initial learning rate (default: 1e-3)"
)
parser.add_argument(
    "--optim", type=str, default="Adam", help="optimizer to use (default: Adam)"
)
parser.add_argument(
    "--num_epochs", type=int, default=15, help="number of epochs (default: 40)"
)
parser.add_argument(
    "--when", type=int, default=10, help="when to decay learning rate (default: 20)"
)
parser.add_argument("--intere", type=int, default=3)
parser.add_argument(
    "--log_interval",
    type=int,
    default=20,
    help="frequency of result logging (default: 30)",
)
parser.add_argument("--seed", type=int, default=666, help="random seed")
parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
parser.add_argument("--name", type=str, default="")
parser.add_argument("--pseudolabel", type=str, default="")
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if torch.cuda.is_available():
    use_cuda = True

setup_seed(args.seed)

dataloder, orig_dim = getdataloader(args)
train_loader = dataloder["train"]
valid_loader = dataloder["valid"]
test_loader = dataloder["test"]
hyp_params = args
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.when = args.when
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = (
    len(train_loader),
    len(valid_loader),
    len(test_loader),
)
hyp_params.criterion = "L1Loss"
hyp_params.orig_dim = orig_dim
hyp_params.output_dim = 1


if __name__ == "__main__":
    if hyp_params.stage == "pretrain":
        test_loss = base_train.initiate(
            hyp_params, train_loader, valid_loader, test_loader
        )
    elif hyp_params.stage == "contrastive":
        test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
    elif hyp_params.stage == "pseudo":
        test_loss = pseudo_train.initiate(
            hyp_params, train_loader, valid_loader, test_loader
        )
