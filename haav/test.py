import random
from data import (
    ImageDetectionsField, TextField, TxtCtxField, VisCtxField, RawField
)
from data import COCO, DataLoader
import evaluation
from models.transformer import (
    Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory,
    ViewGenerator
)
import torch
import argparse, os, pickle
from tqdm import tqdm
import numpy as np
import itertools
from pathlib import Path
import itertools


random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_metrics(model, dataloader, text_field):
    model.eval()

    gen, gts = {}, {}
    with tqdm(desc='COCO image captioning', unit='it', total=len(dataloader), dynamic_ncols=True, smoothing=0.05) as pbar:
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
                obj=obj, vis_ctx=vis_ctx, txt_ctx=txt_ctx, max_len=20, mode="rl",
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HAAV Training')
    parser.add_argument('--dataset_root', type=str, default="./datasets")
    parser.add_argument('--ckpt', type=str, default='ckpt/ckpt.pth')
    parser.add_argument('--txt_ctx', default="txt_ctx.hdf5")
    parser.add_argument('--vis_ctx', default="vis_ctx.hdf5")
    parser.add_argument('--obj_file', type=str, default="oscar.hdf5")
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    args.dataset_root = Path(args.dataset_root)

    print('HAAV Testing')
    print(args)

    device = torch.device(args.device)

    # Create the dataset
    object_field = ImageDetectionsField(
        obj_file=args.dataset_root/args.obj_file, max_detections=50
    )
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
        remove_punctuation=True, nopoints=False
    )
    txt_ctx_filed = TxtCtxField(
        ctx_file=args.dataset_root/args.txt_ctx, k=8
    )
    vis_ctx_filed = VisCtxField(
        ctx_file=args.dataset_root/args.vis_ctx
    )

    fields = {
        "object": object_field, "text": text_field, "img_id": RawField(),
        "txt_ctx": txt_ctx_filed, "vis_ctx": vis_ctx_filed
    }
    dset = args.dataset_root/"annotations"
    dataset = COCO(fields, dset, dset)
    train_dataset, val_dataset, test_dataset = dataset.splits

    fields = {
        "object": object_field, "text": RawField(), "img_id": RawField(),
        "txt_ctx": txt_ctx_filed, "vis_ctx": vis_ctx_filed
    }
    dict_dataset_eval = test_dataset.image_dictionary(fields)
    dict_dataset_train = train_dataset.image_dictionary(fields)
    
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
        N=3, padding_idx=0, d_model=256, d_ff=256*4, dropout_s=0.0,
        attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': 40}
    )
    decoder = MeshedDecoder(
        vocab_size=len(text_field.vocab), max_len=54, N_dec=3, n_layers=3,
        d_model=256, d_ff=256*4, dropout_v=0.0,
        padding_idx=text_field.vocab.stoi['<pad>']
    )
    view_gen = ViewGenerator(
        d_obj=2054, d_vis=vis_ctx_filed.fdim_grid, d_txt=txt_ctx_filed.fdim, d_out=256,
        txt_keys=("whole", "five"), topk=8, drop_rate=0.0
    )
    model = Transformer(
        bos_idx=text_field.vocab.stoi['<bos>'], K=8000,
        encoder=encoder, decoder=decoder, view_gen=view_gen
    ).to(device)

    data = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(data['model'])
    
    # Validation scores
    with torch.no_grad():
        dataloader_eval = DataLoader(
            dict_dataset_eval, batch_size=10, shuffle=False, num_workers=args.workers, drop_last=False
        )
        eval_scores = evaluate_metrics(model, dataloader_eval, text_field)
        print("Eval scores", eval_scores)
