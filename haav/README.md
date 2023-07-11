# Train HAAV image captioning model

## Setup

Make sure to first follow the instructions in the [ctx](../ctx/) folder to download or re-build the visual and textual context.

```bash
mkdir datasets && cd datasets

# Download COCO caption annotations
gdown --fuzzy https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing
unzip annotations.zip
rm annotations.zip

# Download object features
wget https://www.dropbox.com/s/0h67c6ezwnderbd/oscar.hdf5

# Link cross-modal context
ln -s ../../ctx/outputs/image_features/vis_ctx.hdf5
ln -s ../../ctx/outputs/retrieved_captions/txt_ctx.hdf5
```

## Training

HAAV is a lightweight model trained from scratch on MS-COCO only. The model is first trained with the standard cross-entropy loss and then fine-tuned with SCST loss. We train HAAV on a single A-40 GPU.

Cross-entropy training of HAAV on GPU 0 (or any available GPU on your machine).

```Bash
python train.py --device 0 --stage xe
```

SCST training of HAAV on GPU 1 (or any available GPU on your machine).

```Bash
python train.py --device 1 --stage scst
```

\<Note\> After paper acceptance, we find that using cosine lr scheduler with warm-up achieves better performance. We thus follow this new training receipe for the released code.

## Evaluation

First, create a folder named _ckpt_ and download the model checkpoint from [here](https://www.dropbox.com/s/pnh98cu4488rrz0/ckpt.pth) to the _ckpt_ folder. And then, use the following command to evaluate the trained model on the Karpathy test set:

```Bash
python test.py --devices 0
```

|          |  B-4 | METEOR | CIDEr | SPICE |
| -------- | :---:| :----: | :---: | :---: |
| paper    | 41.0 |   30.2 | 141.5 |  23.9 |
| released | 41.2 |   30.3 | 142.3 |  23.9 |



## Citations

Please cite our work if you find this repo useful.

```BibTeX
@inproceedings{kuo2023hierarchical,
    title={HAAV: Hierarchical Aggregation of Augmented Views for Image Captioning},
    author={Chia-Wen Kuo and Zsolt Kira},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
}
```

HAAV is a follow-on work of our [Xmodal-Ctx](https://github.com/GT-RIPL/Xmodal-Ctx) and the codebase is built upon it. Consider also citing Xmodal-Ctx if you find this repo useful.

```BibTeX
@inproceedings{kuo2022pretrained,
    title={Beyond a Pre-Trained Object Detector: Cross-Modal Textual and Visual Context for Image Captioning},
    author={Chia-Wen Kuo and Zsolt Kira},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2022}
}
```
