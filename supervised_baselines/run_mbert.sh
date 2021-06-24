set -euxof pipefail

LOG_NAME=log_mbert.txt
WORKDIR_NAME=mbert
rm -f $LOG_NAME
rm -rf $WORKDIR_NAME

mkdir $WORKDIR_NAME

cd splits_txt
for lang in en fr jp ru zh pt; do
  cat lang_"$lang"_fold0_train.txt lang_"$lang"_fold0_dev.txt lang_"$lang"_test.txt >../$WORKDIR_NAME/data_lang_"$lang".txt
done
cd ..

for lang in en fr jp ru zh pt; do
  for seed in $(seq 0 4); do
    train=lang_"$lang"_fold"$seed"_train.txt
    valid=lang_"$lang"_fold"$seed"_dev.txt
    test=lang_"$lang"_test.txt

    echo "" >>$LOG_NAME
    echo "SPLIT $lang $seed" >>$LOG_NAME
    echo ""
    echo "SPLIT $lang $seed"

    cp splits_txt/"$train" $WORKDIR_NAME/train.txt
    cp splits_txt/"$valid" $WORKDIR_NAME/valid.txt
    cp splits_txt/"$test" $WORKDIR_NAME/test.txt

    TOKENIZERS_PARALLELISM=false python main.py --task_name train.txt --do_eval --do_train --eval_batch_size 32 \
      --data_dir $WORKDIR_NAME/ \
      --model bert-base-multilingual-uncased --max_seq_length 128 --train_batch_size 32 --alpha_param 20 \
      --gradient_accumulation_steps 1 \
      --beta_param 0.2 --learning_rate 5.0e-6 --num_train_epochs 10 --output_dir $WORKDIR_NAME/"$train" >>$LOG_NAME

    rm -rf "${WORKDIR_NAME:?}"/"$train"
  done
done
