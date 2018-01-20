#!/usr/bin/env bash

#Train single models
neptune run experiment_manager.py \
--config best_configs/config_count_logreg.yaml \
-- train_evaluate_predict_pipeline -p count_logreg

neptune run experiment_manager.py \
--config best_configs/config_bad_word_logreg.yaml \
-- train_evaluate_predict_pipeline -p bad_word_tfidf_logreg

neptune run experiment_manager.py \
--config best_configs/config_bad_word_count_logreg.yaml \
-- train_evaluate_predict_pipeline -p bad_word_count_logreg

neptune run experiment_manager.py \
--config best_configs/config_tfidf_logreg.yaml \
-- train_evaluate_predict_pipeline -p tfidf_logreg

neptune run experiment_manager.py \
--config best_configs/config_char_vdcnn.yaml \
-- train_evaluate_predict_pipeline -p char_vdcnn

neptune run experiment_manager.py \
--config best_configs/config_word_lstm.yaml \
-- train_evaluate_predict_pipeline -p word_lstm

neptune run experiment_manager.py \
--config best_configs/config_glove_lstm.yaml \
-- train_evaluate_predict_pipeline -p glove_lstm

neptune run experiment_manager.py \
--config best_configs/config_glove_scnn.yaml \
-- train_evaluate_predict_pipeline -p glove_scnn

neptune run experiment_manager.py \
--config best_configs/config_glove_dpcnn.yaml \
-- train_evaluate_predict_pipeline -p glove_dpcnn

#Blend/copy single models
neptune run experiment_manager.py \
--config best_configs/config_blend.yaml \
-- blend_pipelines \
count_logreg_best \
bad_word_logreg_best \
bad_word_count_logreg_best \
tfidf_logreg_best \
char_vdcnn_best \
word_lstm_best \
glove_lstm_best \
glove_scnn_best \
glove_dpcnn_best \
--blended_name logreg_ensemble_best

#Train ensemble model
neptune run experiment_manager.py \
--config best_configs/config_logreg_ensemble.yaml \
-- train_evaluate_predict_pipeline -p logreg_ensemble