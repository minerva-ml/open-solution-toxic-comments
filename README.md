# Starter code: Kaggle [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge 'Kaggle competition')

Here, at [Neptune](https://neptune.ml/ 'machine learning lab') we enjoy participating in the Kaggle competitions. [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge 'Kaggle competition') is especially interesting because it touches important issue of online harassment.

## Ensemble our predictions in the cloud!
You need to be registered to neptune.ml to be able to use our predictions for your ensemble models.

* click `start notebook` 
* choose `browse` button
* select the `neptune_ensembling.ipynb` file from this repository. 
* choose worker type: `gcp-large` is the reccomended one. 
* run first few cells to load our predictions on the held out validation set along with the labels
* grid search over many possible parameter options. the more runs you choose the longer it will run.
* train your second level, ensemble model (it should take less than an hour once you have the parameters)
* load our predictions on the test set
* feed our test set predictions to your ensemble model and get final predictions
* save your submission file 
* click on browse files and find your submission file to download it.

Running the notebook as is got **0.986+** on the LB.

## Disclaimer
In this open source solution you will find references to the neptune.ml. It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :wink:.

## The idea
We are contributing starter code that is easy to use and extend. We did it before with [Cdiscount’s Image Classification Challenge](https://github.com/deepsense-ai/cdiscount-starter) and we believe that it is correct way to open data science to the wider community and encourage more people to participate in Challenges. This starter is ready-to-use end-to-end solution. Since all computations are organized in separate steps, it is also easy to extend. Check [devbook.ipynb](https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/devbook.ipynb) for more information about different pipelines.

Now we want to go one step further and invite you to participate in the development of this analysis pipeline. At the later stage of the competition (early February) we will invite top contributors to join our team on Kaggle.

## Contributing
You are welcome to extend this pipeline and contribute your own models or procedures. Please refer to the [CONTRIBUTING](https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/CONTRIBUTING.md) for more details.

# Installation
### option 1: Neptune cloud
on the [neptune](https://neptune.ml/ 'machine learning lab') site
* log in: `neptune accound login`
* create new project named `toxic`: Follow the link `Projects` (top bar, left side), then click `New project` button. This action will generate project-key `TOX`, which is already listed in the `neptune.yaml`.

run setup commands
```bash
$ git clone https://github.com/neptune-ml/kaggle-toxic-starter.git
$ pip3 install neptune-cli
$ neptune login
```

start experiment 
```bash
$ neptune send --environment keras-2.0-gpu-py3 --worker gcp-gpu-medium --config best_configs/fasttext_gru.yaml -- train_evaluate_predict_cv_pipeline --pipeline_name fasttext_gru --model_level first
```
This should get you to **0.9852**
**Happy Training :)**

Refer to [Neptune documentation](https://docs.neptune.ml/cli/neptune_send/) and [Getting started: Neptune Cloud](https://github.com/neptune-ml/kaggle-toxic-starter/wiki/Getting-started:-Neptune-Cloud) for more.

### option 2: local install
Please refer to the [Getting started: local instance](https://github.com/neptune-ml/kaggle-toxic-starter/wiki/Getting-started:-local-instance) for installation procedure.

# Solution visualization
Below end-to-end pipeline is visualized. You can run exactly this one!
![pipeline_001](https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/imgs/log_reg_ensemble.png 'complex-ensemble')

We have also prepared something simpler to just get you started:

![pipeline_002](https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/imgs/glove_lstm_pipeline.png 'simple GLOVE LSTM')


# User support
There are several ways to seek help:
1. Read project's [Wiki](https://github.com/neptune-ml/kaggle-toxic-starter/wiki), where we publish descriptions about the code, pipelines and neptune.
2. Kaggle [discussion](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion) is our primary way of communication.
3. You can submit an [issue](https://github.com/neptune-ml/kaggle-toxic-starter/issues) directly in this repo.
