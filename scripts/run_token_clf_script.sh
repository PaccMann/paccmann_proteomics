export MAX_LENGTH=512
export MODEL_NAME=../trained_models/<model_name>
export OUTPUT_DIR=../trained_models/finetuning/secondary_structure_pred
export BATCH_SIZE=32
export NUM_EPOCHS=1
export SAVE_STEPS=20
export SEED=1

python ../paccmann_proteomics/run_token_classification.py \
--data_dir ../data/fine_tuning/secondary_structure \
--labels ../data/fine_tuning/secondary_structurelabels.txt \
--model_name_or_path $MODEL_NAME \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--learning_rate=1e-05 \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval