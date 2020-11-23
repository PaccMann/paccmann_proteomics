export OUTPUT_DIR=../trained_models/<where_to_store_models>
export MODEL_NAME=../trained_models/<model_name>
export TOKENIZER=../trained_models/<model_name>
export MODEL_TYPE=roberta
export TRAIN_FILE=../data/pretraining/<dataset_train>.txt
export EVAL_FILE=../data/pretraining/<dataset_eval>.txt
export BATCH_SIZE=4
export NUM_EPOCHS=1
export SAVE_STEPS=750
export SEED=1

python ../paccmann_proteomics/run_language_modeling.py \
--output_dir $OUTPUT_DIR \
--model_name_or_path $MODEL_NAME \
--model_type $MODEL_TYPE \
--tokenizer_name $TOKENIZER \
--train_data_file $TRAIN_FILE \
--eval_data_file $EVAL_FILE \
--logging_steps 400 \
--save_steps 400 \
--line_by_line \
--chunk_length 10000 \
--logging_dir $OUTPUT_DIR/logs \
--mlm \
--num_train_epochs $NUM_EPOCHS \
--learning_rate 1e-3 \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--block_size 512 \
--do_train \
--do_eval \
--overwrite_output_dir \
--chunk_length 1000000 \
--overwrite_cache \
