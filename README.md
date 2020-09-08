# MLB Pitch Analytics
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

In January 2020, the MLB confirmed that the Houston Astros had used technology to steal signs in the 2017 and 2018 seasons. Statistics show that their plate discipline and runs produced were significantly better when comparing the 2016 and 2017 seasons and their home vs. away games. An important takeaway from this scandal is that being able to predict a pitch is hugely valuable to a team's success.

This repository details pitch classification, visualization, and prediction. To setup, first clone the respository with the git code below.

`$ git clone https://github.com/J-Douglas/MLB-Pitch-Analytics` 

## Classification

To train a custom classification model for a pitcher, call the code below.

`python train_classification_model.py`

To classify pitches, call the code below. You will be asked to specify the pitcher. The program will throw an error if a model has not be trained for the pitcher specified.

`python classification.py`

## Visualization

To visualize results, call the code below.

`python visualize.py`

![Marcus Stroman Pitch Visualization](https://github.com/J-Douglas/MLB-Pitch-Analytics/blob/master/Marcus%20Stroman/Marcus%20Stroman%20Pitch%20Visualization.png)![Nathan Eovaldi Pitch Visualization](https://github.com/J-Douglas/MLB-Pitch-Analytics/blob/master/Nathan%20Eovaldi/Nathan%20Eovalid%20Pitch%20Visaulization.png)

## Prediction

`python train_prediction_model.py`

`python prediction.py`
