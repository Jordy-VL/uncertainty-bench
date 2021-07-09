for model in /mnt/lerna/models/20news_aleatorics_concrete /mnt/lerna/models/20news_aleatorics_M5_concrete /mnt/lerna/models/20news_baseline_concrete /mnt/lerna/models/20news_baseline_M5_concrete /mnt/lerna/models/AAPD_aleatorics_concrete /mnt/lerna/models/AAPD_aleatorics_M5_concrete /mnt/lerna/models/AAPD_baseline_concrete /mnt/lerna/models/AAPD_baseline_M5_concrete /mnt/lerna/models/CLINC150_aleatorics_concrete /mnt/lerna/models/CLINC150_aleatorics_M5_concrete /mnt/lerna/models/CLINC150_aleatorics_multivar-nofunc_concrete /mnt/lerna/models/CLINC150_baseline_concrete /mnt/lerna/models/CLINC150_baseline_M5_concrete /mnt/lerna/models/imdb_aleatorics_concrete /mnt/lerna/models/imdb_aleatorics_M5_concrete /mnt/lerna/models/imdb_baseline_concrete /mnt/lerna/models/imdb_baseline_M5_concrete /mnt/lerna/models/Reuters_multilabel_aleatorics_concrete /mnt/lerna/models/Reuters_multilabel_aleatorics_M5_concrete /mnt/lerna/models/Reuters_multilabel_baseline_concrete /mnt/lerna/models/Reuters_multilabel_baseline_M5_concrete
do     
    echo $model
    python3 ../get_logits.py $model -p
done

