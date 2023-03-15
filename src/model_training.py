import gc
import os
import torch
from src.competition_evaluation import product_matching_validation
import neptune.new as neptune
from tqdm.auto import tqdm
from torch import nn, optim


def report_gpu():
    torch.cuda.empty_cache()
    gc.collect()


def fine_tune_ViT_triplet(encoder,
                          dataloaders,
                          n_epochs,
                          lr,
                          batch_size):
    run = neptune.init_run(
        project="vmudryi/Product-matching",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0Njk1ZDE5ZS02ODhhLTRhMmYtYjRhNC0wZTlhNjBkYWYzNTUifQ==",
    )
    run["epochs"] = n_epochs
    run['lr'] = lr
    run["dataset/train"] = len(dataloaders["train"])
    run["dataset/eval"] = len(dataloaders["eval"])
    run['batch_size'] = batch_size
    run_id = run["sys/id"].fetch()
    os.mkdir(run_id)

    encoder.cuda()

    optimizer_encoder = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.9))
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    best_mape = 0

    for e in range(n_epochs):
        loop = tqdm(dataloaders['train'], leave=True)
        for batch_i, (anchor, pos, neg) in enumerate(loop):
            encoder.train()
            anchor, pos, neg = anchor.cuda(), pos.cuda(), neg.cuda()

            optimizer_encoder.zero_grad()

            anchor_embeddings, _ = encoder(anchor)
            pos_embeddings, _ = encoder(pos)
            neg_embeddings, _ = encoder(neg)

            loss = criterion(anchor_embeddings, pos_embeddings, neg_embeddings)
            loss.backward()
            optimizer_encoder.step()
            run["train/loss"].append(loss.item())

            if batch_i % 300 == 0:
                encoder.eval()
                report_gpu()

                target_mape = product_matching_validation(encoder)
                run["eval/MAPE"].append(target_mape)
                print("Epoch: {}/{}, Batch = {}   MAPE:  {:.6f}. ".format(e, n_epochs, batch_i, target_mape))

                if best_mape < target_mape:
                    best_mape = target_mape
                    torch.save(encoder.state_dict(), run_id + '/encoder_{}.pt'.format(e + 1))

        encoder.eval()
        report_gpu()

        val_loss = 0.
        loop_eval = tqdm(dataloaders['eval'], leave=True)
        for _, (anchor, pos, neg) in enumerate(loop_eval):
            anchor, pos, neg = anchor.cuda(), pos.cuda(), neg.cuda()
            with torch.no_grad():
                anchor_embeddings, _ = encoder(anchor)
                pos_embeddings, _ = encoder(pos)
                neg_embeddings, _ = encoder(neg)
                loss = criterion(anchor_embeddings, pos_embeddings, neg_embeddings)
                val_loss += loss.item()

        run["eval/loss"].append(val_loss / len(dataloaders['eval']))
        print("Epoch: {}/{}   Val CE Loss: {:.5f}.".format(e, n_epochs, val_loss / len(dataloaders['eval'])))
    run.stop()
    return encoder
