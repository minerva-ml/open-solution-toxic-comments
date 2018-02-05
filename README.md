# Starter code: Kaggle [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge 'Kaggle competition')

Here, at [Neptune](https://neptune.ml/ 'machine learning lab') we enjoy participating in the Kaggle competitions. [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge 'Kaggle competition') is especially interesting because it touches important issue of online harassment.

## Ensemble our predictions in the cloud!
If you are registered to neptune.ml you can use our predictions for your ensemble models.
Simply click `start notebook` and choose `browse` button, then select the `neptune_ensembling.ipynb` file from this repository. By running the first few cells you can load our predictions on the held out set along with the labels and train your second level, ensemble model.
Then predict with this model on competition test set. 
After saving your submission click on browse files select path to your submission and download it.

## The idea
We are contributing starter code that is easy to use and extend. We did it before with [Cdiscount’s Image Classification Challenge](https://github.com/deepsense-ai/cdiscount-starter) and we believe that it is correct way to open data science to the wider community and encourage more people to participate in Challenges. This starter is ready-to-use end-to-end solution. Since all computations are organized in separate steps, it is also easy to extend. Check [devbook.ipynb](https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/devbook.ipynb) for more information about different pipelines.

Now we want to go one step further and invite you to participate in the development of this analysis pipeline. At the later stage of the competition (early February) we will invite top contributors to join our team on Kaggle.

## Contributing
You are welcome to extend this pipeline and contribute your own models or procedures. Please refer to the [CONTRIBUTING](https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/CONTRIBUTING.md) for more details.

# Installation
### option 1: Neptune cloud
on the [neptune](https://neptune.ml/ 'machine learning lab') site
* register to receive $100 in GPU time
* log in
* create new project named `toxic`: Follow the link `Projects` (top bar, left side), then click `New project` button. This action will generate project-key `TOX`, which is already listed in the `neptune_config.yaml`.

run setup commands
```bash
$ git clone https://github.com/neptune-ml/kaggle-toxic-starter.git
$ pip3 install neptune-cli
$ neptune login
```

start experiment
```bash
$ neptune send experiment_manager.py --environment keras-2.0-gpu-py3 --worker gcp-gpu-medium --config neptune_config.yaml -- train_evaluate_predict_pipeline --pipeline_name glove_lstm
```
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
