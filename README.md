This repository holds the codes used in [Connecting the Dots: Detecting Adversarial Perturbations Using Context Inconsistency, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680392.pdf).

## Key Dependencies

1. Python2

2. TensorFlow 1.5.0 （CUDA 9.0), other version might be okay (e.g, TensorFlow 1.3.0)

3. PyTorch 1.3.0

## Installation 

1. Clone the SIN repository
  
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/sli057/SCEME.git
  ```

2. Build the Cython modules

(You may need to change "arch=" according to you GPU type.)
  ```Shell
  cd $SCEME_ROOT/lib
  make
  ```

## Step1: Build SCEME and train context-aware Faster R-CNN

We provided the pre-trained model on VOC0712 dataset for both Faster R-CNN and the context-aware Faster R-CNN, you could download them from [Dropbox](https://www.dropbox.com/sh/zeu90jxstipabnv/AABd5exXwn65LcrPY8UZQe9fa?dl=0)

Faster R-CNN: output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wo_context/VGGnet_wo_context.ckpt

Context-ware Faster R-CNN: output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wt_context/VGGnet_wt_context.ckpt

1. Test with the pre-trained models 
    ```Shell
    cd context_model
    python test_FasterRCNN.py --net_final '../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wo_context/VGGnet_wo_context.ckpt'
    python test_context_model.py --net_final '../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wt_context/VGGnet_wt_context.ckpt'
    
    ```
    
2. If you want to train your own models
    ```Shell
    cd context_model
    python train_FasterRCNN.py --train_set YOUR_DATASET
    python train_context_model.py  --train_set YOUR_DATASET
    
    ```
    
## Step2: Adversarial attacks on Faster RCNN
We provide both digital (FGSM +IFGSM ) and physical attack codes.

### Generate perturbations
```
cd attack_detector

```

1. digital miscategorization attack
```
python digital_attack.py --attack_type 'miscls'
```
2. digital hiding attack
```
python digital_attack.py --attack_type 'hiding'
```
3. digital appearing attack
```
python digital_attack.py --attack_type 'appear'
```
4. physical miscategorization attack
```
python physical_attack.py --attack_type 'miscls'
```
5. physical hiding attack
```
python physical_attack.py --attack_type 'hiding'
```
6. physical appearing attack
```
python physical_attack.py --attack_type 'appear'
```

### Collect the generated perturbations
```
cd script_extract_files
python extract_attack.py
```

## Step 3: Collect context profiles

```
cd context_profile
python get_context_profiles.py
```
Note that is is not necessary to collect all the context profiles, just stop the running if you have got enough training/testing samples.

```
python get_dataset.py
```


## Step 4: Adversarial detection via AutoEncoders
The AutoEncoder is trained and tested with PyTorch
### Train the AutoEncoders with the collected benign context profiles.
```
cd detect_attacks
python run_training_testing.py --mode 'train'

```
### Test the reconstruction error on both benign and perturbed context profiles.
```
python run_training_testing.py --mode 'test'
```
### Calculate the ROC-AUC.
python test_ROC-AUC.py


### References

[Faster R-CNN tf version](https://github.com/smallcorgi/Faster-RCNN_TF)

[Context-aware Faster R-CNN](https://github.com/choasup/SIN)

[Physical perturbation generation](https://github.com/evtimovi/robust_physical_perturbations)


