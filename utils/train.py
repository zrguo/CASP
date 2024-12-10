import torch
from torch import nn
from models import model as mm
from utils.util import *
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = torch.load(hyp_params.pretrained_model)
    model = fix_para(model)
    count_parameters(model)
    if hyp_params.use_cuda:
        model = model.cuda()
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.1, verbose=True
    )
    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]
    tune_cri = ContrastiveLoss()

    def train(model, optimizer, criterion, tune_cri):
        model.train()
        for i_batch, batch in enumerate(train_loader):
            idx = batch["idx"]
            text, audio, vision, batch_Y = (
                batch["text"],
                batch["audio"],
                batch["vision"],
                batch["label"],
            )
            text_aug, audio_aug, vision_aug = random_drop(text, audio, vision)
            eval_attr = batch_Y.unsqueeze(-1)
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = (
                        text.cuda(),
                        audio.cuda(),
                        vision.cuda(),
                        eval_attr.cuda(),
                    )
                    text_aug, audio_aug, vision_aug = (
                        text_aug.cuda(),
                        audio_aug.cuda(),
                        vision_aug.cuda(),
                    )

            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model

            preds, rep = net([text, audio, vision])
            preds_aug, rep_aug = net([text_aug, audio_aug, vision_aug])
            loss = tune_cri(rep, rep_aug)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch in enumerate(loader):
                text, audio, vision, batch_Y = (
                    batch["text"],
                    batch["audio"],
                    batch["vision"],
                    batch["label"],
                )
                eval_attr = batch_Y.unsqueeze(-1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = (
                            text.cuda(),
                            audio.cuda(),
                            vision.cuda(),
                            eval_attr.cuda(),
                        )

                batch_size = text.size(0)

                net = nn.DataParallel(model) if batch_size > 10 else model
                preds, _ = net([text, audio, vision])
                total_loss += criterion(preds, eval_attr).item()

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    def save_checkpoints():
        results = []
        with torch.no_grad():
            for idx, batch in enumerate(train_loader):
                text, audio, vision = batch["text"], batch["audio"], batch["vision"]

                if hyp_params.use_cuda:
                    text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()

                preds, _ = model([text, audio, vision])
                results.append(preds)
        results = torch.cat(results).squeeze(-1)
        return results

    best_acc = 0
    re = save_checkpoints()
    checkpoint = [re.unsqueeze(0)]
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion, tune_cri)
        val_loss, r, t = evaluate(model, criterion, test=False)
        acc2 = eval_senti(r, t)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 50)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f}".format(
                epoch, duration, val_loss
            )
        )
        print("-" * 50)

        if best_acc < acc2:
            print(f"Saved model at {hyp_params.name}!")
            torch.save(model, hyp_params.name)
            best_acc = acc2

        if epoch % hyp_params.intere == 0:
            results = save_checkpoints()
            checkpoint.append(results.unsqueeze(0))
    checkpoint = torch.cat(checkpoint).cpu()
    print(checkpoint.shape)
    torch.save(checkpoint, hyp_params.pseudolabel)
