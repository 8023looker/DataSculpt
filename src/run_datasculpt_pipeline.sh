#!/bin/bash

CONTEXT_LENGTH=$1 # context window length
DELTA=$2 # distance threshold
EPSILON=$3 # alteration threshold
ITER_T=$4 # number of iterations

echo $CONTEXT_LENGTH
echo $DELTA
echo $EPSILON
echo $ITER_T

# data preprocessing
python ./preprocessing/text_embedding.py

# semantic clustering + greedy MOCO policy
python ray_serverless.py \
        ${CONTEXT_LENGTH} \
        ${DELTA} \
        ${EPSILON} \
        ${ITER_T}

# python run.py \
#         ${CONTEXT_LENGTH} \
#         ${DELTA} \
#         ${EPSILON} \
#         ${ITER_T}