# PoseFormer with WAO and OHJM loss
Our code is built on top of [PoseFormer](https://github.com/zczcwh/PoseFormer).

### Environment

The code is developed and tested under the following environment

* Python 3.8.2
* PyTorch 1.7.1
* CUDA 11.0

You can install dependecies by running this script:
```bash
pip install -r requirements.txt
```

### Dataset

Before you run the code, please read the [document](data/README.md) to download the preprocessed Human3.6M dataset

### Checkpoints

If you want to run evaluation, please read the [document](checkpoint/README.md) to download essential checkpoints.

### Evaluating pre-trained models

To evaluate the checkpoint, please run the following script
```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 81 -c checkpoint --evaluate poseformer_mpjpe.pth
```

Change the `--evaluate` parameter with different checkpoint to obtain the errors of those models.

### Training new models

* To train a model from scratch, run:

```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 81 -lr 0.00004 -lrd 0.99 --exp ohjm_wao
```

* If you want to train with MPJPE only, run:
```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 81 -lr 0.00004 -lrd 0.99 --exp mpjpe --lambda-wao 0 --lambda-ohjm 1.0 --m-ohjm 1.0 
```

* If you want to train with MPJPE + WAO, run:
```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 81 -lr 0.00004 -lrd 0.99 --exp mpjpe_wao --m-ohjm 1.0 
```

* If you want to train with OHJM, run:
```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 81 -lr 0.00004 -lrd 0.99 --exp mpjpe_wao --lambda-wao 0 --lambda-ohjm 1.0 
```