#!/usr/bin/env bash

: <<'comment'
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
--config best_configs/char_vdcnn.yaml \
-- train_evaluate_predict_pipeline -p char_vdcnn

neptune run \
--config best_configs/fasttext_gru.yaml \
-- train_evaluate_predict_pipeline -p fasttext_gru
neptune run \
--config best_configs/glove_gru.yaml \
-- train_evaluate_predict_pipeline -p glove_gru
neptune run \
--config best_configs/word2vec_gru.yaml \
-- train_evaluate_predict_pipeline -p word2vec_gru
comment

#Blend/copy single models
neptune run \
--config best_configs/setup.yaml \
-- blend_pipelines \
count_logreg \
bad_word_logreg \
tfidf_logreg \
char_vdcnn \
glove_gru \
word2vec_gru \
fasttext_gru \
--blended_name catboost_ensemble

#Train ensemble model
neptune run \
--config best_configs/catboost_ensemble.yaml \
-- train_evaluate_predict_pipeline -p catboost_ensemble