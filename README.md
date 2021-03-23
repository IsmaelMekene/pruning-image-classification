# pruning-image-classification

## Introduction

In rhis tutorial, we first deal with a simple OCR model built with the Functional API. Apart from combining CNN and RNN, it also illustrates how you can instantiate a new layer and use it as an "Endpoint layer" for implementing **Connectionist Temporal Classification loss**. Thereafter, we implement pruned versions of the baseline model in order to evalute them.

## Installation

`pip install -r requirements.txt`

## Datas

The Dataset used in this tutorial is made of 1040  captcha images and it can be downloaded [here](https://github.com/IsmaelMekene/pruning-image-classification/blob/main/data/captchas.zip)

<p align="center">
  <img title= "Data Visualisation" src="https://github.com/IsmaelMekene/pruning-image-classification/blob/main/data/viz_captcha.png" alt="Visualize fee captchas">
</p>


## Model

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
      
      
      
