#!/usr/bin/env bash


#Train single models
neptune run \
--config best_configs/tfidf_logreg.yaml \
-- train_evaluate_predict_pipeline -p tfidf_logreg

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