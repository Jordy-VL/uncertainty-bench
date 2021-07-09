for data in Reuters_multilabel AAPD imdb #20news CLINC150 
do 
    for model in /mnt/lerna/models/"$data"_aleatorics_concrete /mnt/lerna/models/"$data"_baseline_concrete /mnt/lerna/models/"$data"_aleatorics /mnt/lerna/models/"$data"_baseline /mnt/lerna/models/"$data"_nodropout
    do
        python3 ../evaluate.py $model -r
        #python3  ../compare.py $model -i raw
    done
done

#/mnt/lerna/models/"$data"_aleatorics_M5_concrete /mnt/lerna/models/"$data"_aleatorics_M5 /mnt/lerna/models/"$data"_baseline_M5_concrete /mnt/lerna/models/"$data"_baseline_M5 /mnt/lerna/models/"$data"_nodropout_M5
#ood="/mnt/lerna/models/"$data"_aleatorics_concrete_ood /mnt/lerna/models/"$data"_baseline_concrete_ood /mnt/lerna/models/"$data"_aleatorics_ood /mnt/lerna/models/"$data"_baseline_ood /mnt/lerna/models/"$data"_nodropout_ood "
#ensembles= /mnt/lerna/models/"$data"_aleatorics_M5_concrete_ood /mnt/lerna/models/"$data"_aleatorics_M5_ood /mnt/lerna/models/"$data"_baseline_M5_concrete_ood /mnt/lerna/models/"$data"_baseline_M5_ood /mnt/lerna/models/"$data"_nodropout_M5_ood 

# could also do it for BERT and compare :) 