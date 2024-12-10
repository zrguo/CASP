import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import torch
from torch import nn
import torch.nn.functional as F


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = test_truth_emo > 0
    predicted_label = test_preds_emo > 0
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0
    acc2 = accuracy_score(binary_truth, binary_preds)

    print("MAE: ", mae)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", acc2)
    return acc2


def count_parameters(model):
    trainable_params = 0
    total_params = 0
    trainable_params_list = list()
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_list.append(name)
    print(f"Total params: {total_params}, Trainable params: {trainable_params}")


def fix_para(model):
    for name, para in model.named_parameters():
        if "layer_norms" not in name:
            para.requires_grad = False
    return model


def random_drop(text, audio, vision):
    seed = random.random()
    if seed < 0.33:
        text = torch.zeros_like(text)
    elif seed < 0.66:
        audio = torch.zeros_like(audio)
    else:
        vision = torch.zeros_like(vision)
    return text, audio, vision


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, verbose=False):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        N = z_j.shape[0]

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )
        if self.verbose:
            print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose:
                print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = (
                torch.ones((2 * N,))
                .to(emb_i.device)
                .scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            )
            if self.verbose:
                print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose:
                print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss
