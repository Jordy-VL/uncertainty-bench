#!/bin/bash

#export CUDA_VISIBLE_DEVICES=-1

ood_single () {
    echo "eval $1"
    python3 ../evaluate.py $model -d CLINC_ood
    new=$model"_ood"
    mkdir $new
    cp $model/params.json $new
    cp $model/CLINC_oodeval.pickle $new/eval.pickle
    python3 ../ood.py $new
}

for model in /mnt/lerna/models/CLINC150_baseline_concrete_2DSNGP-15 /mnt/lerna/models/CLINC150_baseline_concrete_2DCNN
#/mnt/lerna/models/CLINC150_baseline_2DSNGP-1 /mnt/lerna/models/CLINC150_nodropout_2DSNGP-1 /mnt/lerna/models/CLINC150_baseline_2DSNGP-3 /mnt/lerna/models/CLINC150_nodropout_2DSNGP-3 /mnt/lerna/models/CLINC150_baseline_2DSNGP-5 /mnt/lerna/models/CLINC150_nodropout_2DSNGP-5 /mnt/lerna/models/CLINC150_baseline_2DSNGP-10 /mnt/lerna/models/CLINC150_nodropout_2DSNGP-10
#/mnt/lerna/models/CLINC150_baseline_2DSNGP-15 /mnt/lerna/models/CLINC150_nodropout_2DSNGP-15 #/home/jordy/uncertainty-bench/models/210525-191854_CLINC150
#CLINC150_baseline_cSNGP_nowd-6 /mnt/lerna/models/CLINC150_baseline_cSNGP-6 /mnt/lerna/models/CLINC150_baseline_cSNGP_nowd-4 /mnt/lerna/models/CLINC150_baseline_cSNGP-4 /mnt/lerna/models/CLINC150_baseline_cSNGP_nowd-2 /mnt/lerna/models/CLINC150_baseline_cSNGP-2 /mnt/lerna/models/CLINC150_baseline_cSNGP_nowd-1 /mnt/lerna/models/CLINC150_baseline_cSNGP-1 /mnt/lerna/models/CLINC150_baseline_cSNGP_nowd-0.5 /mnt/lerna/models/CLINC150_baseline_cSNGP-0.5
#/mnt/lerna/models/CLINC150_aleatorics_BERT /mnt/lerna/models/CLINC150_aleatorics_concrete_BERT /mnt/lerna/models/CLINC150_baseline_BERT /mnt/lerna/models/CLINC150_baseline_concrete_BERT /mnt/lerna/models/CLINC150_nodropout_BERT
#/mnt/lerna/models/CLINC150_aleatorics_concrete /mnt/lerna/models/CLINC150_baseline_concrete /mnt/lerna/models/CLINC150_aleatorics /mnt/lerna/models/CLINC150_baseline #/mnt/lerna/models/CLINC150_nodropout # 
do
    break
    ood_single $model
done


for model in /mnt/lerna/models/CLINC150_baseline_concrete_2DCNN_M5
#/mnt/lerna/models/CLINC150_baseline_2DCNN_M5 /mnt/lerna/models/CLINC150_nodropout_2DCNN_M5 
#/mnt/lerna/models/CLINC150_aleatorics_BERT_M5 /mnt/lerna/models/CLINC150_aleatorics_concrete_BERT_M5 /mnt/lerna/models/CLINC150_baseline_BERT_M5 /mnt/lerna/models/CLINC150_baseline_concrete_BERT_M5 /mnt/lerna/models/CLINC150_nodropout_BERT_M5
do
    for i in M0_ M1_ M2_ M3_ M4_
    do  
        #python3 get_single.py $model  
        python3 ../evaluate.py $model -i $i -d CLINC_ood
        new=$model"_ood"
        mkdir $new
        cp $model/params.json $new
        cp $model/$i"CLINC_oodeval".pickle $new/$i"eval.pickle"
    done
    # MAKE ENSEMBLE first?
    python3 ../compare.py $new
    python3 ../ood.py $new
    # python3 ../evaluate.py $model -d CLINC_ood
    # new=$model"_ood"
    # mkdir $new
    # cp $model/params.json $new
    # cp $model/CLINC_oodeval.pickle $new/eval.pickle
    # python3 ../ood.py $new
done