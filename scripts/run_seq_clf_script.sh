export MODEL_NAME=../trained_models/<model_name>
export EXP_DIR=../data/fine_tuning/
export TASK_NAME=pairwise-string
export OUTPUT_DIR=../trained_models/finetuning/
export BATCH_SIZE=8
export MAX_LENGTH=512
export NUM_EPOCHS=1
export SAVE_STEPS=750
export SEED=1

python ../paccmann_proteomics/run_sequence_classification.py \
--model_name_or_path $MODEL_NAME/$TASK_NAME \
--task_name $TASK_NAME \
--continue_from_checkpoint \
--data_dir $EXP_DIR/$TASK_NAME \
--cache_dir $EXP_DIR/$TASK_NAME \
--output_dir $MODEL_NAME/$TASK_NAME \
--max_seq_length $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--learning_rate 1e-5 \
--logging_dir $OUTPUT_DIR/$TASK_NAME \
--per_device_train_batch_size $BATCH_SIZE \
--seed $SEED \
--overwrite_output_dir \
--do_train \
--do_eval