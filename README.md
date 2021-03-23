# pruning-image-classification

## Introduction

In rhis tutorial, we first deal with a simple OCR model built with the Functional API. Apart from combining CNN and RNN, it also illustrates how you can instantiate a new layer and use it as an "Endpoint layer" for implementing CTC loss. Thereafter, we implement pruned versions of the baseline model in order to evalute them.

## Installation

`pip install -r requirements.txt`

## Datas

The Dataset used in this tutorial is made of 1040  captcha images and it can be downloaded [here](https://github.com/IsmaelMekene/pruning-image-classification/blob/main/data/captchas.zip)

<p align="center">
  <img title= "Data Visualisation" src="https://github.com/IsmaelMekene/pruning-image-classification/blob/main/data/viz_captcha.png" alt="Visualize fee captchas">
</p>
