. path.sh

gpu=$1
SAVE_DIR=exp/finetune_ja_char
W2V_PATH=exp/pretrain_6lang/checkpoint_best.pt
DATA_DIR=data/ja/char
label_type=char

CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/train.py $DATA_DIR --save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--wer-args '("data/ja/callhome.word.bin","data/ja/char/lexicon.txt",2,-1)' \
--post-process letter --train-subset train --valid-subset "valid" --no-epoch-checkpoints --best-checkpoint-metric uer \
--num-workers 4 --max-update 80000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path $W2V_PATH \
--labels $label_type --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000  --validate-interval 50 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage \
--warmup-steps 8000 --hold-steps 42000 --decay-steps 50000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1280000 --seed 2337 --ddp-backend no_c10d --update-freq 3 \
--log-interval 10 --log-format simple --save-interval 50
