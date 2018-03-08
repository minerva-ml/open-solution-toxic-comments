#!/usr/bin/env bash

#Train single models
neptune run \
--config best_configs/bad_word_logreg.yaml \
-- train_evaluate_predict_cv_pipeline -p bad_word_logreg -m first
neptune run \
--config best_configs/count_logreg.yaml \
-- train_evaluate_predict_cv_pipeline -p count_logreg -m first
neptune run \
--config best_configs/tfidf_logreg.yaml \
-- train_evaluate_predict_cv_pipeline -p tfidf_logreg -m first

neptune run \
--config best_configs/char_vdcnn.yaml \
-- train_evaluate_predict_cv_pipeline -p char_vdcnn -m first
neptune run \
--config best_configs/fasttext_gru.yaml \
-- train_evaluate_predict_cv_pipeline -p fasttext_gru -m first
neptune run \
--config best_configs/fasttext_lstm.yaml \
-- train_evaluate_predict_cv_pipeline -p fasttext_lstm -m first
neptune run \
--config best_configs/glove_gru.yaml \
-- train_evaluate_predict_cv_pipeline -p glove_gru -m first
neptune run \
--config best_configs/glove_lstm.yaml \
-- train_evaluate_predict_cv_pipeline -p glove_lstm -m first
neptune run \
--config best_configs/fasttext_scnn.yaml \
-- train_evaluate_predict_cv_pipeline -p fasttext_scnn -m first
neptune run \
--config best_configs/glove_scnn.yaml \
-- train_evaluate_predict_cv_pipeline -p glove_scnn -m first
neptune run \
--config best_configs/fasttext_dpcnn.yaml \
-- train_evaluate_predict_cv_pipeline -p fasttext_dpcnn -m first

neptune run \
--config best_configs/glove_dpcnn.yaml \
-- train_evaluate_predict_cv_pipeline -p glove_dpcnn -m first
neptune run \
--config best_configs/word2vec_gru.yaml \
-- train_evaluate_predict_cv_pipeline -p word2vec_gru -m first
neptune run \
--config best_configs/word2vec_lstm.yaml \
-- train_evaluate_predict_cv_pipeline -p word2vec_lstm -m first
neptune run \
--config best_configs/word2vec_dpcnn.yaml \
-- train_evaluate_predict_cv_pipeline -p word2vec_dpcnn -m first
neptune run \
--config best_configs/word2vec_scnn.yaml \
-- train_evaluate_predict_cv_pipeline -p word2vec_scnn -m first

#Copy single model predictions for stacking
neptune run \
--config best_configs/setup.yaml \
-- prepare_single_model_predictions_dir \
count_logreg \
bad_word_logreg \
tfidf_logreg \
char_vdcnn \
glove_gru \
glove_lstm \
glove_dpcnn \
glove_scnn \
word2vec_gru \
word2vec_lstm \
word2vec_dpcnn \
word2vec_scnn \
fasttext_gru \
fasttext_lstm \
fasttext_dpcnn \
fasttext_scnn

neptune run \
--config best_configs/xgboost_ensemble.yaml \
-- train_evaluate_predict_cv_pipeline -p xgboost_ensemble -m second