#!/usr/bin/env bash

#Train single models
neptune run \
--config best_configs/count_logreg.yaml \
-- train_evaluate_predict_pipeline -p count_logreg
neptune run \
--config best_configs/bad_word_logreg.yaml \
-- train_evaluate_predict_pipeline -p bad_word_logreg
neptune run \
--config best_configs/bad_word_count_logreg.yaml \
-- train_evaluate_predict_pipeline -p bad_word_count_logreg
neptune run \
--config best_configs/tfidf_logreg.yaml \
-- train_evaluate_predict_pipeline -p tfidf_logreg
neptune run \
--config best_configs/hand_crafted_all_logreg.yaml \
-- train_evaluate_predict_pipeline -p hand_crafted_all_logreg

neptune run \
--config best_configs/char_vdcnn.yaml \
-- train_evaluate_predict_pipeline -p char_vdcnn

neptune run \
--config best_configs/glove_gru.yaml \
-- train_evaluate_predict_pipeline -p glove_gru
neptune run \
--config best_configs/glove_lstm.yaml \
-- train_evaluate_predict_pipeline -p glove_lstm
neptune run \
--config best_configs/glove_scnn.yaml \
-- train_evaluate_predict_pipeline -p glove_scnn
neptune run \
--config best_configs/glove_dpcnn.yaml \
-- train_evaluate_predict_pipeline -p glove_dpcnn

neptune run \
--config best_configs/fasttext_gru.yaml \
-- train_evaluate_predict_pipeline -p fasttext_gru
neptune run \
--config best_configs/fasttext_lstm.yaml \
-- train_evaluate_predict_pipeline -p fasttext_lstm
neptune run \
--config best_configs/fasttext_scnn.yaml \
-- train_evaluate_predict_pipeline -p fasttext_scnn
neptune run \
--config best_configs/fasttext_dpcnn.yaml \
-- train_evaluate_predict_pipeline -p fasttext_dpcnn

neptune run \
--config best_configs/word2vec_gru.yaml \
-- train_evaluate_predict_pipeline -p word2vec_gru
neptune run \
--config best_configs/word2vec_lstm.yaml \
-- train_evaluate_predict_pipeline -p word2vec_lstm
neptune run \
--config best_configs/word2vec_scnn.yaml \
-- train_evaluate_predict_pipeline -p word2vec_scnn
neptune run \
--config best_configs/word2vec_dpcnn.yaml \
-- train_evaluate_predict_pipeline -p word2vec_dpcnn

#Blend/copy single models
neptune run \
--config best_configs/setup.yaml \
-- blend_pipelines \
count_logreg \
bad_word_logreg \
bad_word_count_logreg \
tfidf_logreg \
hand_crafted_all_logreg \
char_vdcnn \
glove_gru \
glove_lstm \
glove_scnn \
glove_dpcnn \
word2vec_gru \
word2vec_lstm \
word2vec_scnn \
word2vec_dpcnn \
fasttext_gru \
fasttext_lstm \
fasttext_scnn \
fasttext_dpcnn \
--blended_name catboost_ensemble

#Train ensemble model
neptune run \
--config best_configs/catboost_ensemble.yaml \
-- train_evaluate_predict_pipeline -p catboost_ensemble