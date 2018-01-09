# Starter code: Kaggle [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge 'Kaggle competition')

Here, at [Neptune](https://neptune.ml/ 'machine learning lab') we enjoy participating in the Kaggle competitions. [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge 'Kaggle competition') is especially interesting because it touches important issue of online harassment.


## The idea
We are contributing starter code that is easy to use and extend. We did it before with [Cdiscount’s Image Classification Challenge](https://github.com/deepsense-ai/cdiscount-starter) and we believe that it is correct way to open data science to the wider community and encourage more people to participate in Challenges.

Now we want to go one step further and invite you to participate in the development of this analysis pipeline. At the later stage of the competition (early February) we will invite top contributors to join our team on Kaggle.


## How to run?
This starter is ready-to-use end-to-end solution. Since all computations are organized in separate steps, it is also easy to extend this solution.

### Installation
This project assumes python 3.5.
1. Clone this repo

```bash
$ git clone https://github.com/neptune-ml/kaggle-toxic-starter.git
```
2. Install tensorflow and keras for your system or if you are planning on using cloud option just proceed to step 3 
```bash
$ pip3 install tensorflow-gpu
$ pip3 install Keras
```

3. Install other requirements
```bash
$ pip3 install -r requirements.txt
```
Note that [neptune](https://neptune.ml/ 'machine learning lab') (experiment monitoring and management system) is included in the requirements file.

### Run experiment
To run an experiment in the Neptune cloud use this command
```bash
$ neptune send experiment_manager.py --environment keras-2.0-gpu-py3 --worker gcp-gpu-medium --config neptune_config.yaml -- train_evaluate_predict_pipeline --pipeline_name glove_lstm
```

Check [Neptune documentation](https://docs.neptune.ml/cli/neptune_send/) for more options.


# Solution visualization
Below end-to-end pipeline is visualized. You can run exactly this one!
![pipeline_001](https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/imgs/log_reg_ensemble.png 'our initial pipeline')


# Contributing
You are welcome to extend this pipeline and contribute your own models or procedures. Please refer to the [CONTRIBUTING](https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/CONTRIBUTING.md) for more details.

# User support
There are two ways to reach us:
1. Kaggle [discussion](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion) is our primary way of communication.
2. You can submit an [issue](https://github.com/neptune-ml/kaggle-toxic-starter/issues) directly in this repo.
