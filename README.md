# :framed_picture: pruning-image-classification

## Introduction

In this tutorial, we first deal with a simple OCR model built with the Functional API. Apart from combining CNN and RNN, it also illustrates how you can instantiate a new layer and use it as an "Endpoint layer" for implementing **Connectionist Temporal Classification loss**. Thereafter, we implement pruned versions of the baseline model in order to evalute them.

## Installation

`pip install -r requirements.txt`

## Datas

The Dataset used in this tutorial is made of 1040  captcha images and it can be downloaded [here](https://github.com/IsmaelMekene/pruning-image-classification/blob/main/data/captchas.zip)

<p align="center">
  <img title= "Data Visualisation" src="https://github.com/IsmaelMekene/pruning-image-classification/blob/main/data/viz_captcha.png" alt="Visualize fee captchas">
</p>


## Models

### 1. Baseline Model

- **Building**

The baseline model is a combination of CNN and RNN while it builds in a CTC (Connectionist Temporal Classification) loss.
*View [here](https://github.com/IsmaelMekene/pruning-image-classification/blob/main/models/baseline_model.py)*


      Model: "ocr_model_v1"
      __________________________________________________________________________________________________
      Layer (type)                    Output Shape         Param #     Connected to                     
      ==================================================================================================
      image (InputLayer)              [(None, 200, 50, 1)] 0                                            
      __________________________________________________________________________________________________
      Conv1 (Conv2D)                  (None, 200, 50, 32)  320         image[0][0]                      
      __________________________________________________________________________________________________
      pool1 (MaxPooling2D)            (None, 100, 25, 32)  0           Conv1[0][0]                      
      __________________________________________________________________________________________________
      Conv2 (Conv2D)                  (None, 100, 25, 64)  18496       pool1[0][0]                      
      __________________________________________________________________________________________________
      pool2 (MaxPooling2D)            (None, 50, 12, 64)   0           Conv2[0][0]                      
      __________________________________________________________________________________________________
      reshape (Reshape)               (None, 50, 768)      0           pool2[0][0]                      
      __________________________________________________________________________________________________
      dense1 (Dense)                  (None, 50, 64)       49216       reshape[0][0]                    
      __________________________________________________________________________________________________
      dropout (Dropout)               (None, 50, 64)       0           dense1[0][0]                     
      __________________________________________________________________________________________________
      bidirectional (Bidirectional)   (None, 50, 256)      197632      dropout[0][0]                    
      __________________________________________________________________________________________________
      bidirectional_1 (Bidirectional) (None, 50, 128)      164352      bidirectional[0][0]              
      __________________________________________________________________________________________________
      label (InputLayer)              [(None, None)]       0                                            
      __________________________________________________________________________________________________
      dense2 (Dense)                  (None, 50, 20)       2580        bidirectional_1[0][0]            
      __________________________________________________________________________________________________
      ctc_loss (CTCLayer)             (None, 50, 20)       0           label[0][0]                      
                                                                       dense2[0][0]                     
      ==================================================================================================
      Total params: 432,596
      Trainable params: 432,596
      Non-trainable params: 0
      __________________________________________________________________________________________________
      
      
      
- **Training**

The training of the baseline model was supposed to be done over 100 epochs however due to the earlystoping callbacks it has bee shortcutted to 13.



      Epoch 1/100
      59/59 [==============================] - 23s 397ms/step - loss: 0.2436 - val_loss: 0.1349
      Epoch 2/100
      59/59 [==============================] - 23s 394ms/step - loss: 0.1178 - val_loss: 0.1337
      Epoch 3/100
      59/59 [==============================] - 23s 393ms/step - loss: 0.1014 - val_loss: 0.1062
      Epoch 4/100
      59/59 [==============================] - 23s 392ms/step - loss: 0.1024 - val_loss: 0.1193
      Epoch 5/100
      59/59 [==============================] - 23s 393ms/step - loss: 0.0818 - val_loss: 0.1211
      Epoch 6/100
      59/59 [==============================] - 23s 394ms/step - loss: 0.0708 - val_loss: 0.1420
      Epoch 7/100
      59/59 [==============================] - 23s 393ms/step - loss: 0.0738 - val_loss: 0.1229
      Epoch 8/100
      59/59 [==============================] - 23s 397ms/step - loss: 0.0623 - val_loss: 0.1237
      Epoch 9/100
      59/59 [==============================] - 23s 394ms/step - loss: 0.0931 - val_loss: 0.1063
      Epoch 10/100
      59/59 [==============================] - 23s 392ms/step - loss: 0.0942 - val_loss: 0.1236
      Epoch 11/100
      59/59 [==============================] - 23s 393ms/step - loss: 0.0673 - val_loss: 0.1114
      Epoch 12/100
      59/59 [==============================] - 23s 393ms/step - loss: 0.0450 - val_loss: 0.1333
      Epoch 13/100
      59/59 [==============================] - 23s 392ms/step - loss: 0.0628 - val_loss: 0.1284
      
      
      The average test loss is:  0.09358327692517868
      
      
- **Tensorboard**

<p align="center">
  <img title= "Baseline model" src="https://github.com/IsmaelMekene/pruning-image-classification/blob/main/data/loss_captcha.svg">
</p>


  ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `train`

  ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `validation` 


- **Inference**

The prediction over the testing set with the baseline model can finally be done.

<p align="center">
  <img title= "Prediction Baseline model" src="https://github.com/IsmaelMekene/pruning-image-classification/blob/main/data/pred_captcha.png">
</p>


### 2. Pruning

#### In progress ...
