#ensembles= /mnt/lerna/models/"$data"_aleatorics_M5_concrete_ood /mnt/lerna/models/"$data"_aleatorics_M5_ood /mnt/lerna/models/"$data"_baseline_M5_concrete_ood /mnt/lerna/models/"$data"_baseline_M5_ood /mnt/lerna/models/"$data"_nodropout_M5_ood  #/mnt/lerna/models/"$data"_aleatorics_M5_concrete /mnt/lerna/models/"$data"_aleatorics_M5 /mnt/lerna/models/"$data"_baseline_M5_concrete /mnt/lerna/models/"$data"_baseline_M5 /mnt/lerna/models/"$data"_nodropout_M5
#simple=/mnt/lerna/models/"$data"_aleatorics_concrete /mnt/lerna/models/"$data"_baseline_concrete /mnt/lerna/models/"$data"_aleatorics /mnt/lerna/models/"$data"_baseline /mnt/lerna/models/"$data"_nodropout
#ood="/mnt/lerna/models/"$data"_aleatorics_concrete_ood /mnt/lerna/models/"$data"_baseline_concrete_ood /mnt/lerna/models/"$data"_aleatorics_ood /mnt/lerna/models/"$data"_baseline_ood /mnt/lerna/models/"$data"_nodropout_ood "

for data in 20news CLINC150 imdb Reuters_multilabel AAPD
do 
    for model in /mnt/lerna/models/"$data"_aleatorics_M5_concrete_ood /mnt/lerna/models/"$data"_aleatorics_M5_ood /mnt/lerna/models/"$data"_baseline_M5_concrete_ood /mnt/lerna/models/"$data"_baseline_M5_ood /mnt/lerna/models/"$data"_nodropout_M5_ood 
    do
        echo $model
        
        # for m in 2 3 4
        # do
        #     # python3 ../compare.py $model -m $m
        #     python3 ../ood.py $model -m
        # done
        # python3 ../evaluate.py $model -r
        #python3  ../compare.py $model -i raw
    done
done