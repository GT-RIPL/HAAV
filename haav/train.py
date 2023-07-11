import random
from data import (
    ImageDetectionsField, TextField, TxtCtxField, VisCtxField, RawField
)
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import (
    Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory,
    ViewGenerator
)
from utils import contrastive_loss, count_parameters
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW
import torch
from torch import nn
from torch.nn import NLLLoss
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import accuracy
import argparse, os, pickle
from tqdm import tqdm
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from pathlib import Path
import itertools
import json


random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_metrics(model, dataloader, text_field):
    model.eval()

    gen, gts = {}, {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), dynamic_ncols=True, smoothing=0.05) as pbar:
        for it, data in enumerate(dataloader):
            txt_ctx = {
                k1: {
                    k2: v2.to(device, non_blocking=True)
                    for k2, v2 in v1.items()
                }
                for k1, v1 in data["txt_ctx"].items()
            }
            vis_ctx = {
                k: v.to(device, non_blocking=True)
                for k, v in data["vis_ctx"].items()
            }
            obj = data["object"].to(device, non_blocking=True)

            (out, _), _ = model(
                obj=obj, vis_ctx=vis_ctx, txt_ctx=txt_ctx, max_len=args.seq_len, mode="rl",
                eos_idx=text_field.vocab.stoi['<eos>'], beam_size=5, out_size=1,
            )

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(data["text"], caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()

    running_loss_word = 0.0
    running_loss_con = 0.0
    running_acc = 0.0
    ema = 0.9
    with tqdm(desc=f'Epoch {e}', unit='it', total=len(dataloader), dynamic_ncols=True, smoothing=0.05) as pbar:
        for it, data in enumerate(dataloader):
            txt_ctx = {
                k1: {
                    k2: v2.to(device, non_blocking=True)
                    for k2, v2 in v1.items()
                }
                for k1, v1 in data["txt_ctx"].items()
            }
            vis_ctx = {
                k: v.to(device, non_blocking=True)
                for k, v in data["vis_ctx"].items()
            }
            obj = data["object"].to(device, non_blocking=True)
            captions = data["text"].to(device, non_blocking=True)

            # word_logp: N x S x vocab
            # view_cls, view_cls_ema: N x n_views x d
            word_logp, view_cls, view_cls_ema = model(
                obj=obj, vis_ctx=vis_ctx, txt_ctx=txt_ctx, seq=captions, mode="xe"
            )
            
            word_logp = word_logp[:, :-1].contiguous()
            captions = captions[:, 1:].contiguous()
            loss_word = loss_fn(
                word_logp.reshape(-1, len(text_field.vocab)),
                captions.reshape(-1, )
            )

            loss_con = contrastive_loss(
                view_cls, view_cls_ema, model.memory, args.tau
            )
            
            optim.zero_grad()
            loss = loss_word + args.w_con * loss_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            model.ema_update()
            model.enqueue(view_cls_ema)
            
            with torch.no_grad():
                acc = accuracy(
                    word_logp.reshape(-1, len(text_field.vocab)),
                    captions.reshape(-1, ), ignore_index=text_field.vocab.stoi['<pad>']
                )
            running_loss_word = ema * running_loss_word + (1 - ema) * loss_word.item()
            running_loss_con = ema * running_loss_con + (1 - ema) * loss_con.item()
            running_acc = ema * running_acc + (1 - ema) * acc.item()
            pbar.set_postfix({
                "loss_word": running_loss_word,
                "loss_con": running_loss_con,
                "acc": running_acc
            })
            pbar.update()

            step = e*len(dataloader) + it
            writer.add_scalar(f'xe/loss_word', loss_word.item(), step)
            writer.add_scalar(f'xe/loss_con', loss_con.item(), step)
            writer.add_scalar(f'xe/acc', acc.item(), step)
            
    ret = {
        "loss": running_loss_word,
        "acc": running_acc,
    }
    
    return ret


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    model.train()

    running_baseline = .0
    running_loss_word = 0.0
    running_loss_con = 0.0
    ema = 0.9

    tokenizer_pool = multiprocessing.Pool()
    beam_size = 5
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), dynamic_ncols=True, smoothing=0.05) as pbar:
        for it, data in enumerate(dataloader):
            txt_ctx = {
                k1: {
                    k2: v2.to(device, non_blocking=True)
                    for k2, v2 in v1.items()
                }
                for k1, v1 in data["txt_ctx"].items()
            }
            vis_ctx = {
                k: v.to(device, non_blocking=True)
                for k, v in data["vis_ctx"].items()
            }
            obj = data["object"].to(device, non_blocking=True)

            (out, word_logprob), (view_cls, view_cls_ema) = model(
                obj=obj, vis_ctx=vis_ctx, txt_ctx=txt_ctx, max_len=args.seq_len, mode="rl",
                eos_idx=text_field.vocab.stoi['<eos>'], beam_size=beam_size, out_size=beam_size,
            )

            caps_gen = text_field.decode(out.view(-1, args.seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in data["text"])))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(obj.shape[0], beam_size)
            baseline = torch.sum(reward, dim=-1, keepdim=True) - reward
            baseline = baseline / (beam_size - 1)
            loss_word = -torch.sum(word_logprob, dim=-1)/torch.sum(out!=0, dim=-1)
            loss_word = torch.mean((reward - baseline) * loss_word)

            loss_con = contrastive_loss(
                view_cls, view_cls_ema, model.memory, args.tau
            )

            optim.zero_grad()
            loss = loss_word + args.w_con * loss_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            model.ema_update()
            model.enqueue(view_cls_ema)

            running_loss_word = ema * running_loss_word + (1 - ema) * loss_word.item()
            running_loss_con = ema * running_loss_con + (1 - ema) * loss_con.item()
            running_baseline = ema * running_baseline + (1 - ema) * reward.mean().item()
            pbar.set_postfix({
                "loss_word": running_loss_word,
                "loss_con": running_loss_con,
                "baseline": running_baseline
            })
            pbar.update()

            step = e*len(dataloader) + it
            writer.add_scalar('scst/loss_word', loss_word.item(), step)
            writer.add_scalar('scst/loss_con', loss_con.item(), step)
            writer.add_scalar('scst/baseline', reward.mean().item())

    tokenizer_pool.close()

    ret = {
        "loss": running_loss_word,
        "baseline": running_baseline
    }

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HAAV Training')
    parser.add_argument('--exp_name', type=str, default='haav')
    parser.add_argument('--stage', required=True, choices=["xe", "scst"])
    parser.add_argument('--dropout_c', type=float, default=0.1)
    parser.add_argument('--dropout_s', type=float, default=0.1)
    parser.add_argument('--dropout_v', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--ema', type=float, default=0.999)
    parser.add_argument('--tau', type=float, default=0.06)
    parser.add_argument('--K', type=int, default=8000)
    parser.add_argument('--txt_ctx', default="txt_ctx.hdf5")
    parser.add_argument('--vis_ctx', default="vis_ctx.hdf5")
    parser.add_argument('--obj_file', type=str, default="oscar.hdf5")
    parser.add_argument(
        '--txt_keys', nargs='+', type=str,
        default=("whole", "five"), choices=("whole", "five", "nine")
    )
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default="./datasets")
    parser.add_argument('--eval_set', type=str, default="test", choices=["val", "test"])
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.stage == "xe":
        args.bs = 50
        args.warmup = round(10000 / args.bs * 50 * args.train_ratio)
        args.lr = 5e-5
        args.wd = 0.05
        args.w_con = 0.05
    elif args.stage == "scst":
        args.bs = 40
        args.warmup = 200
        args.lr = 1e-5
        args.wd = 0.0
        args.w_con = 0.2
    else:
        raise ValueError
    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("outputs")/f"{args.exp_name}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print('HAAV Training')
    print(args)

    device = torch.device(args.device)
    writer = SummaryWriter(log_dir=args.save_dir/"tensorboard")

    # Create the dataset
    object_field = ImageDetectionsField(
        obj_file=args.dataset_root/args.obj_file, max_detections=50
    )
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
        remove_punctuation=True, nopoints=False
    )
    txt_ctx_filed = TxtCtxField(
        ctx_file=args.dataset_root/args.txt_ctx, k=args.topk
    )
    vis_ctx_filed = VisCtxField(
        ctx_file=args.dataset_root/args.vis_ctx
    )

    fields = {
        "object": object_field, "text": text_field, "img_id": RawField(),
        "txt_ctx": txt_ctx_filed, "vis_ctx": vis_ctx_filed
    }
    dset = args.dataset_root/"annotations"
    dataset = COCO(fields, dset, dset, train_ratio=args.train_ratio)
    train_dataset, val_dataset, test_dataset = dataset.splits
    if args.eval_set == "val":
        eval_dataset = val_dataset
    elif args.eval_set == "test":
        eval_dataset = test_dataset

    fields = {
        "object": object_field, "text": RawField(), "img_id": RawField(),
        "txt_ctx": txt_ctx_filed, "vis_ctx": vis_ctx_filed
    }
    dict_dataset_eval = eval_dataset.image_dictionary(fields)
    dict_dataset_train = train_dataset.image_dictionary(fields)
    
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    
    # build vocabulary
    vocab_file = 'vocab/vocab_coco.pkl'
    if not os.path.isfile(vocab_file):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open(vocab_file, 'wb'))
    else:
        text_field.vocab = pickle.load(open(vocab_file, 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(
        N=3, padding_idx=0, d_model=args.dim, d_ff=args.dim*4, dropout_s=args.dropout_s,
        attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m}
    )
    decoder = MeshedDecoder(
        vocab_size=len(text_field.vocab), max_len=54, N_dec=3, n_layers=3,
        d_model=args.dim, d_ff=args.dim*4, dropout_v=args.dropout_v,
        padding_idx=text_field.vocab.stoi['<pad>']
    )
    view_gen = ViewGenerator(
        d_obj=2054, d_vis=vis_ctx_filed.fdim_grid, d_txt=txt_ctx_filed.fdim, d_out=args.dim,
        txt_keys=args.txt_keys, topk=args.topk, drop_rate=args.dropout_c
    )
    model = Transformer(
        bos_idx=text_field.vocab.stoi['<bos>'], m=args.ema, K=args.K,
        encoder=encoder, decoder=decoder, view_gen=view_gen
    ).to(device)
    print(f"Number of trainable parameters: {count_parameters(model):,}")

    if args.stage == "scst":
        fname = args.save_dir/"ckpt_best.pth"
        data = torch.load(fname, map_location="cpu")
        model.load_state_dict(data['model'])
        best_cider = data['best_cider']
        print(f"Resume from epoch {data['epoch']} with CIDEr = {best_cider}")
    else:
        print("Train from scratch")
        best_cider = .0
    
    # optimizer
    no_decay = [
        n for n, m in model.named_modules()
        if any(isinstance(m, nd) for nd in [nn.LayerNorm, nn.BatchNorm1d])
    ]
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
                    any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
        {'params': [p for n, p in model.named_parameters() if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optim = AdamW(grouped_parameters, lr=args.lr, eps=1e-8)
    
    if args.stage == "xe":
        num_training_steps = len(train_dataset)//args.bs
    elif args.stage == "scst":
        num_training_steps = len(dict_dataset_train)//args.bs
    else:
        raise ValueError
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup, num_training_steps=num_training_steps
    )

    # Initial conditions
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    for e in range(args.num_epochs):
        # training epoch
        if args.stage == "xe":
            dataloader_train = DataLoader(
                train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True
            )
            train_xe(model, dataloader_train, optim, text_field)
        elif args.stage == "scst":
            dataloader_train = DataLoader(
                dict_dataset_train, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True
            )
            train_scst(model, dataloader_train, optim, cider_train, text_field)

        # Validation scores
        with torch.no_grad():
            dataloader_eval = DataLoader(
                dict_dataset_eval, batch_size=10, shuffle=False, num_workers=args.workers, drop_last=False
            )
            eval_scores = evaluate_metrics(model, dataloader_eval, text_field)
            print("Eval scores", eval_scores)
            eval_cider = eval_scores['CIDEr']
            writer.add_scalar('metrics/cider', eval_scores['CIDEr'], e)
            writer.add_scalar('metrics/bleu1', eval_scores['BLEU'][0], e)
            writer.add_scalar('metrics/bleu4', eval_scores['BLEU'][3], e)
            writer.add_scalar('metrics/meteor', eval_scores['METEOR'], e)
            writer.add_scalar('metrics/rouge', eval_scores['ROUGE'], e)
            writer.add_scalar('metrics/spice', eval_scores['SPICE'], e)

        # Prepare for next epoch
        best = False
        if eval_cider >= best_cider:
            best_cider = eval_cider
            best = True
            with open(args.save_dir/"best_scores.json", "w") as f:
                json.dump(eval_scores, f)

        torch.save({
            'epoch': e,
            'eval_cider': eval_cider,
            "eval_scores": eval_scores,
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_cider': best_cider,
        }, args.save_dir/'ckpt_last.pth')

        if best:
            copyfile(args.save_dir/'ckpt_last.pth', args.save_dir/'ckpt_best.pth')
