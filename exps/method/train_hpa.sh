#!/usr/bin/env bash
# overall.
######################################################
CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}

SEED=28465820
CATEGORY=Table # choices=['Chair', 'Table', 'Lamp']
DATA_TRAIN=Table.train.npy
DATA_VAL=Table.test.npy
LEVEL=3
# Training.
LR=7.5e-5
WEIGHT_DECAY=1e-4
BATCH_SIZE=64   
EPOCHS=1000
LR_DROP=100
NUM_GUPS=8
NUM_WORKERS=32
SAVE_FREQ=1
PRINT_FREQ=20
TYPE_SCHED=cosine
# Network.
BASE_CAT=0
POSE_CAT=1
POSE_CAT_IN_ENCODER=1
SHARED_PRED=1
PRED_DETACH=1
TRAIN_MON=1
EVAL_MON=10
NOISE_CAT=0
NOISE_CAT_IN_ENCODER=1
NOISE_DIM=80
INS_CAT=0
INS_CAT_IN_ENCODER=1
INS_CAT_INTER_ONLY=0
INS_CAT_INTRA_ONLY=0
# Coefficient.
COEF_TRANS_L2=1.0
COEF_ROT_L2=0.0
COEF_ROT_CD=10.0
COEF_TRANS_CD=0.0
COEF_SHAPE_CD=1.0
COEF_PART_CD=0.0
#### Fixed above.
# Model.
MODEL=baseline2_trans
MODEL_VERSION=original      
BIPN=0
PRE_NORM=0
OA=0

ENC_LAYERS=8
DEC_LAYERS=6
INS_VERSION=v2
FILTER_ON=1
FILTER_THRESH=0.2
NUM_FILTER=1
# Decoder.
DECODE_ON=0
NUM_POS=1
NUM_QUERIES=1
COEF_DECODE=0.1
MEMORY_DETACH=0
FEAT_IN_DETACH=0
POSE_CAT_IN_DECODER_PRED=1
POSE_CAT_IN_DECODER_TRANS=1
NOISE_CAT_IN_DECODER_PRED=1
NOISE_CAT_IN_DECODER_TRANS=1

OUTPUT_DIR=${CURDIR}/checkpoints/train/${CATEGORY}


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_port=$((RANDOM + 10000)) --use_env --nproc_per_node=${NUM_GUPS} ${CURDIR}/execute.py --category ${CATEGORY} \
--train_data_fn ${DATA_TRAIN} --val_data_fn ${DATA_VAL} --level ${LEVEL} \
--learning-rate ${LR} --weight-decay ${WEIGHT_DECAY} \
--batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --lr_drop ${LR_DROP} --workers ${NUM_WORKERS} \
--save-freq ${SAVE_FREQ} --print-freq ${PRINT_FREQ}  --type-sched ${TYPE_SCHED} \
--base-cat ${BASE_CAT} --pose-cat ${POSE_CAT} --pose-cat-in-encoder ${POSE_CAT_IN_ENCODER} --shared-pred ${SHARED_PRED} \
--pred-detach ${PRED_DETACH} --train-mon ${TRAIN_MON} --eval-mon ${EVAL_MON} \
--noise-cat ${NOISE_CAT} --noise-cat-in-encoder ${NOISE_CAT_IN_ENCODER} --noise-dim ${NOISE_DIM}  \
--ins-cat ${INS_CAT} --ins-cat-in-encoder ${INS_CAT_IN_ENCODER} \
--ins-cat-inter-only ${INS_CAT_INTER_ONLY} --ins-cat-intra-only ${INS_CAT_INTRA_ONLY} \
--loss_weight_trans_l2 ${COEF_TRANS_L2} --loss_weight_rot_l2 ${COEF_ROT_L2} --loss_weight_trans_cd ${COEF_TRANS_CD} --loss_weight_rot_cd ${COEF_ROT_CD} \
--loss_weight_shape_cd ${COEF_SHAPE_CD} \
--loss_weight_part_cd ${COEF_PART_CD} \
--multiprocessing-distributed \
--seed ${SEED} --output-dir ${OUTPUT_DIR} \
--model ${MODEL} --model_version ${MODEL_VERSION} \
--bi_pn ${BIPN} \
--pre_norm ${PRE_NORM} --offset_attention ${OA} \
--enc_layers ${ENC_LAYERS} --dec_layers ${DEC_LAYERS} \
--ins-version ${INS_VERSION} --filter-on ${FILTER_ON} --filter-thresh ${FILTER_THRESH} --num-filter ${NUM_FILTER} \
--decode-on ${DECODE_ON} --num-pos ${NUM_POS} --num_queries ${NUM_QUERIES} \
--loss_weight_decode ${COEF_DECODE}  --memory-detach ${MEMORY_DETACH} --feat-in-detach ${FEAT_IN_DETACH} \
--pose-cat-in-decoder-pred ${POSE_CAT_IN_DECODER_PRED} --pose-cat-in-decoder-trans ${POSE_CAT_IN_DECODER_TRANS} \
--noise-cat-in-decoder-pred ${NOISE_CAT_IN_DECODER_PRED} --noise-cat-in-decoder-trans ${NOISE_CAT_IN_DECODER_TRANS}


# upload.
#python3 ${CURDIR}/file_uploading.py --local-dir ${OUTPUT_DIR} --dst-dir ${HDFS_DIR}












