#!/usr/bin/env bash

#Train single models
<<COMMENT
neptune run experiment_manager.py \
--config best_configs/config_char_vdcnn.yaml \
-- train_evaluate_predict_pipeline -p char_vdcnn

neptune run experiment_manager.py \
--config best_configs/config_glove_dpcnn.yaml \
-- train_evaluate_predict_pipeline -p glove_dpcnn

neptune run experiment_manager.py \
--config best_configs/config_glove_scnn.yaml \
-- train_evaluate_predict_pipeline -p glove_scnn

neptune run experiment_manager.py \
--config best_configs/config_tfidf_logreg.yaml \
-- train_evaluate_predict_pipeline -p tfidf_logreg

neptune run experiment_manager.py \
--config best_configs/config_count_logreg.yaml \
-- train_evaluate_predict_pipeline -p count_logreg

neptune run experiment_manager.py \
--config best_configs/config_bad_word_count_logreg.yaml \
-- train_evaluate_predict_pipeline -p bad_word_count_logreg

neptune run experiment_manager.py \
--config best_configs/config_glove_lstm.yaml \
-- train_evaluate_predict_pipeline -p glove_lstm

neptune run experiment_manager.py \
--config best_configs/config_word_lstm.yaml \
-- train_evaluate_predict_pipeline -p word_lstm
COMMENT

neptune run experiment_manager.py \
--config best_configs/config_blend.yaml \
-- blend_pipelines bad_word_logreg_best bad_word_count_logreg_best char_vdcnn_best \
count_logreg_best glove_dpcnn_best glove_lstm_best glove_scnn_best_ml100 tfidf_logreg_best word_lstm_best_ml100 \
--blended_name logreg_ensemble_best

<<COMMENT
neptune run experiment_manager.py \
--config best_configs/config_logreg_ensemble.yaml \
-- train_evaluate_predict_pipeline -p logreg_ensemble
COMMENT