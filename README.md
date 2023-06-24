# Contactless Human Interaction Detection in Multi-Person Scenes
[[paper]](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-023-2418-0)

If you find our work or the codebase inspiring and useful to your research, please cite
```bibtex
@article{li2023contactless,
  title={Contactless Interaction Recognition and Interactor Detection in Multi-Person Scenes.},
  author={Jiacheng Li, Ruize Han, Wei Feng, Haomin Yan, Song Wang},
  journal={Frontiers of Computer Science},
  year={2023}
}
```

### Project Structure
#### Dataset
**Download**
[BaiduYun](https://pan.baidu.com/s/1g7wP7AbgFVnx-swIxYOTPA?pwd=xge3) (extract code: xge3)
The extracted directory structure is shown below:
```
.
├── annotations
│   ├── seq10
│   ├── seq101
│   ├── seq102
...
│   ├── seq97
│   ├── seq98
│   └── seq99
├── annotations.zip
├── features
│   ├── seq10
│   ├── seq101
│   ├── seq102
...
│   ├── seq96
│   ├── seq97
│   ├── seq98
│   └── seq99
├── features.7z
├── features_skeleton
│   ├── seq10
│   ├── seq101
│   ├── seq102
...
│   ├── seq97
│   ├── seq98
│   └── seq99
├── features_skeleton.zip
├── frames
│   ├── seq10
│   ├── seq101
│   ├── seq102
...
│   ├── seq96
│   ├── seq97
│   ├── seq98
│   └── seq99
├── S120-1.zip
├── S120-2.zip
├── S120-3.zip
├── S120-4.zip

```
For the zip files S120-1, S120-2, and S120-3, you need to create a new folder named **frames**, and put all the extracted files from these three zip files into this folder.

We provide the following dataset resources:
* The original video frames (in the frames folder)
* Manual annotation for video-level and MOT-method-generated frame-level tracking annotations. (in the annotations folder)
* Bounding box features (in the features folder)
* Skeleton features (in the features-skeleton folder)

#### Code Folder
```
.
├── backbone.py
├── base_model.py
├── collective.py
├── config.py
├── data
│   ├── DistantDatasetV2 -> /{YOUR_SAVE_PATH}/DistantDatasetV2
├── dataset.py
├── dataset_script.py
├── display.py
├── distant.py
├── distant_utils.py
├── evaluate.py
├── gcn_bank.py
├── gcn_model.py
├── LICENSE
├── models
│   ├── GPNN_HICO.py
│   └── __init__.py
├── myConvLSTM.py
├── README.md
├── reference
│   ├── convlstm.py
│   ├── Convolutional_LSTM_PyTorch
│   ├── CTR_GCN
│   ├── MS_G3D
│   └── SlowFastNetworks
├── result -> /{YOUR_SAVE_PATH}/HIT
├── re.txt
├── S3D_G.py
├── test.json
├── train_distant_stage1.py
├── train_distant_stage2.py
├── train.json
├── train_net.py
├── units
│   ├── ConvLSTM.py
│   ├── __init__.py
│   ├── LinkFunction.py
│   ├── MessageFunction.py
│   ├── ReadoutFunction.py
│   └── UpdateFunction.py
├── utils.py
└── volleyball.py
```

### Experiment Results&Models
The reference model checkpoint and results referred to as `Ours` in this article are uploaded to [BaiduYun](https://pan.baidu.com/s/1g7wP7AbgFVnx-swIxYOTPA?pwd=xge3) (extract code: xge3) as `final.tar.xz`.

### Dependence
We provide environment.yml file exported by conda environment management. You can clone our Python environment by `conda env create -f environment.yml`.

### Training&Testing
**Training:**
Stage1:
Run the following command to get the stage1 model checkpoint:
`python train_distant_stage1.py`
The model checkpoint is saved to `/{CODE_PATH}/HIT/result/[Distant_stage1_OneGroupV5Net_stage1]<xxx>/stage1_epochx_x.pth`.
Stage2:
Before the stage2 training, please edit the `cfg.stage1_model_path` in 'train_distant_stage2.py' as the checkpoint path you obtained in the last step. Then you can run the following command to start the second step of training:
`python train_distant_stage2.py`
The results of the proposed method are in folder '/{CODE_PATH}/HIT/result/[Distant_stage2_OneGroupV5Net_stage2]<xxx>'.

**Testing:**
Uncomment the last line of `train_distant_stage2.py` and comment on the line of `train_net(cfg)`.

Then edit the address of saved model checkpoints in cfg.stage2_model_path='' as `/{CODE_PATH}/HIT/result/[Distant_stage2_OneGroupV5Net_stage2]<xxx>/stage2_epochx_x.pth` in `train_distant_stage2.py`.

Then just run `python train_distant_stage2.py`, and you will get all the evaluation results as shown in the article.
