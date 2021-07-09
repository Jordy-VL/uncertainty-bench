# Benchmark runs



## Base commands to reproduce

```console
python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatoric_M5_concrete_ood
```

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatoric_M5_ood

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_baseline_M5_concrete_ood

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_baseline_M5_ood

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False ood=[\"comp.graphics\",\"comp.sys.ibm.pc.hardware\",\"comp.windows.x\",\"rec.autos\",\"rec.sport.baseball\",\"sci.crypt\",\"sci.med\",\"soc.religion.christian\",\"talk.politics.mideast\",\"talk.religion.misc\"] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_nodropout_M5_ood

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatorics_concrete_ood

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_baseline_concrete_ood

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatorics_ood

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_baseline_ood

> python3 experiment.py with clf_default dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_nodropout_ood

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatoric_M5_concrete

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatoric_M5

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_baseline_M5_concrete

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_baseline_M5

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_nodropout_M5

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatorics_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_baseline_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatorics

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_baseline

> python3 experiment.py with clf_default dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_nodropout



> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatoric_M5_concrete

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatoric_M5

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_M5_concrete

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_M5

> python3 experiment.py with clf_default dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 ensemble=5 -n CLINC150_nodropout_M5

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatorics_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatorics

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline

> python3 experiment.py with clf_default dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_nodropout

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatoric_M5_concrete

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatoric_M5

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_M5_concrete

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_M5

> python3 experiment.py with clf_default dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 ensemble=5 -n CLINC150_nodropout_M5

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatorics_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatorics

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline

> python3 experiment.py with clf_default dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_nodropout



> python3 experiment.py with clf_default identifier=imdb dropout_nonlinear=0.5 weight_decay=0.0001 steps_per_epoch=2000 use_aleatorics=True ensemble=5 dropout_concrete=True max_vocabulary=20000 ood=5 -n imdb_aleatoric_5_True_ood

> python3 experiment.py with clf_default identifier=imdb dropout_nonlinear=0.5 weight_decay=0.0001 steps_per_epoch=2000 use_aleatorics=True ensemble=5 dropout_concrete=0 max_vocabulary=20000 ood=5 -n imdb_aleatoric_5_0_ood

> python3 experiment.py with clf_default ood=5 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_baseline_M5_concrete_ood

> python3 experiment.py with clf_default ood=5 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_baseline_M5_ood

> python3 experiment.py with clf_default ood=5 ensemble=5 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_nodropout_M5_ood

> python3 experiment.py with clf_default identifier=imdb dropout_nonlinear=0.5 weight_decay=0.0001 steps_per_epoch=2000 use_aleatorics=True ensemble=5 dropout_concrete=True max_vocabulary=20000 ood=5 -n imdb_aleatoric_5_True_ood

> python3 experiment.py with clf_default ood=5 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_baseline_M5_concrete_ood

> python3 experiment.py with clf_default identifier=imdb dropout_nonlinear=0.5 weight_decay=0.0001 steps_per_epoch=2000 use_aleatorics=True ensemble=5 dropout_concrete=0 max_vocabulary=20000 ood=5 -n imdb_aleatoric_5_0_ood

> python3 experiment.py with clf_default ood=5 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_baseline_M5_ood

> python3 experiment.py with clf_default ood=5 ensemble=5 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_nodropout_M5_ood

> python3 experiment.py with clf_default ood=None ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_aleatoric_M5_concrete

> python3 experiment.py with clf_default ood=None ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_aleatoric_M5

> python3 experiment.py with clf_default identifier=imdb dropout_nonlinear=0.5 weight_decay=0.0001 steps_per_epoch=2000 use_aleatorics=False embedding_dropout=0 dropout_concrete=True ensemble=5 -n imdb_baseline_M5_concrete

> python3home/azureuser/code/gordon/arkham/arkham/Bayes/Quantify/experiment.py with clf_default identifier=imdb dropout_nonlinear=0.5 weight_decay=0.0001 steps_per_epoch=2000 use_aleatorics=False ensemble=5 dropout_concrete=0 -n imdb_baseline_M5_0

> python3 experiment.py with clf_default identifier=imdb dropout=0 dropout_nonlinear=0 weight_decay=0 steps_per_epoch=2000 use_aleatorics=False ensemble=5 dropout_concrete=0 max_vocabulary=20000 -n imdb_nodropout_M5

> python3 experiment.py with clf_default ood=None ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_aleatoric_M5_concrete

> python3home/azureuser/code/gordon/arkham/arkham/Bayes/Quantify/experiment.py with clf_default identifier=imdb dropout_nonlinear=0.5 weight_decay=0.0001 steps_per_epoch=2000 use_aleatorics=False embedding_dropout=0 dropout_concrete=True -n imdb_baseline_0_True

> python3 experiment.py with clf_default identifier=imdb dropout_nonlinear=0.5 weight_decay=0.0001 steps_per_epoch=2000 seed=42 use_aleatorics=True ensemble=1 dropout_concrete=0 max_vocabulary=20000 -n imdb_aleatoric_True_0

> python3 experiment.py with clf_default identifier=imdb dropout_nonlinear=0.5 weight_decay=0.0001 steps_per_epoch=2000 seed=42 use_aleatorics=False ensemble=1 dropout_concrete=0 max_vocabulary=20000 -n imdb_aleatoric_False_0

> python3 experiment.py with clf_default identifier=imdb dropout=0 dropout_nonlinear=0 weight_decay=0 steps_per_epoch=2000 use_aleatorics=False ensemble=5 dropout_concrete=0 max_vocabulary=20000 -n imdb_nodropout_M5



> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatoric_M5_concrete_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatoric_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_baseline_M5_concrete_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_baseline_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_nodropout_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatoric_M5_concrete_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_baseline_M5_concrete_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatoric_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_baseline_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_nodropout_M5_ood

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatoric_M5_concrete

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatoric_M5

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_baseline_M5_concrete

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_baseline_M5

> python3 experiment.py with clf_default dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 ensemble=5 -n Reuters_multilabel_nodropout_M5

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatorics_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_baseline_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatorics

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_baseline

> python3 experiment.py with clf_default dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 ensemble=5 -n Reuters_multilabel_nodropout_M5

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_aleatoric_M5_concrete_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_aleatoric_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_baseline_M5_concrete_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_baseline_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_nodropout_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_aleatoric_M5_concrete_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_baseline_M5_concrete_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_aleatoric_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_baseline_M5_ood

> python3 experiment.py with clf_default ood=0 ensemble=5 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_nodropout_M5_ood

> python3 experiment.py with clf_default ood=None ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_aleatoric_M5_concrete

> python3 experiment.py with clf_default ood=None ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_aleatoric_M5

> python3 experiment.py with clf_default ood=None ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_baseline_M5_concrete

> python3 experiment.py with clf_default ensemble=5 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=20000 -n AAPD_baseline_M5

> python3 experiment.py with clf_default ood=None ensemble=5 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 -n AAPD_nodropout_M5

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=True identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=20000 -n AAPD_aleatorics_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=20000 -n AAPD_baseline_concrete

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=True identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=20000 -n AAPD_aleatorics

> python3 experiment.py with clf_default dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=20000 -n AAPD_baseline

> python3 experiment.py with clf_default dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=20000 -n AAPD_nodropout



> python3 experiment.py with clf_default ensemble=5 ood=None model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=20 max_document_len=50 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatorics_BERT_M5

> python3 experiment.py with clf_default model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=20 max_document_len=50 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatorics_BERT0.3_nowd

> python3 experiment.py with clf_default ensemble=5 ood=None model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=20 max_document_len=50 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatorics_concrete_BERT_M5

> python3 experiment.py with clf_default model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=20 max_document_len=50 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=True identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_aleatorics_concrete_BERT0.3_nowd

> python3 experiment.py with clf_default ensemble=5 ood=None model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=20 max_document_len=50 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_BERT_M5

> python3 experiment.py with clf_default model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=20 max_document_len=50 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_BERT0.3_nowd

> python3 experiment.py with clf_default ensemble=5 ood=None model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=20 max_document_len=50 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_concrete_BERT_M5

> python3 experiment.py with clf_default ensemble=5 ood=None model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=20 max_document_len=50 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_baseline_concrete_BERT_M5

> python3 experiment.py with clf_default ensemble=5 model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=5 max_document_len=50 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_nodropout_M5_BERT

> python3 experiment.py with clf_default model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=32 epochs=5 max_document_len=50 dropout_nonlinear=0 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 -n CLINC150_dropoutinside_BERT0.3



> python3 experiment.py with clf_default ensemble=5 ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=16 epochs=20 max_document_len=250 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=True identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatorics_concrete_BERT_M5_ood

> python3 experiment.py with clf_default ensemble=5 ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=16 epochs=20 max_document_len=250 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=True identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_aleatorics_concrete_BERT_M5_ood

> python3 experiment.py with clf_default ensemble=5 ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=16 epochs=20 max_document_len=250 dropout_nonlinear=0 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_dropout_BERT_M5_ood

> python3 experiment.py with clf_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=16 epochs=20 max_document_len=250 dropout_nonlinear=0 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 -n 20news_dropout_BERT0.3_ood



> python3 experiment.py with clf_default ensemble=5 ood=5 model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=8 epochs=20 max_document_len=350 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=True identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_aleatorics_concrete_BERT_M5_ood

> python3 experiment.py with clf_default ensemble=5 ood=5 model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=8 epochs=20 max_document_len=350 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=True identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_aleatorics_concrete_BERT_M5_ood

> python3 experiment.py with clf_default ensemble=5 ood=5 model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=8 epochs=20 max_document_len=350 dropout_nonlinear=0 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_dropout_BERT_M5_ood

> python3 experiment.py with clf_default ensemble=1 ood=5 model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=8 epochs=20 max_document_len=350 dropout_nonlinear=0 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=20000 -n imdb_dropout_BERT_ood



> python3 experiment.py with clf_default ensemble=5 ood=0 model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=16 epochs=20 max_document_len=200 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatorics_concrete_BERT_M5_ood

> python3 experiment.py with clf_default ood=0 model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=16 epochs=20 max_document_len=200 dropout_nonlinear=0.3 dropout=0.3 dropout_concrete=True weight_decay=0 use_aleatorics=True identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_aleatorics_concrete_BERT_ood

> python3 experiment.py with clf_default ensemble=5 ood=0 model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=16 epochs=20 max_document_len=200 dropout_nonlinear=0 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_dropout_BERT_M5_ood

> python3 experiment.py with clf_default ood=0 model=BERT_dense model_class=bert-base-uncased finetune=False learning_rate=0.00002 batch_size=16 epochs=20 max_document_len=200 dropout_nonlinear=0 dropout=0.3 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 -n Reuters_multilabel_dropout_BERT_ood



--------------------------------------------------------



#SNGP runs



> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=binary_categorical_crossentropy -n Reuters_multilabel_nodropout_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=binary_categorical_crossentropy -n Reuters_multilabel_baseline_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=binary_categorical_crossentropy -n Reuters_multilabel_baseline_concrete_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,15] -n Reuters_multilabel_nodropout_2DSNGP-15_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,15] -n Reuters_multilabel_baseline_2DSNGP-15_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=True loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,15] -n Reuters_multilabel_baseline_concrete_2DSNGP_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,1] -n Reuters_multilabel_nodropout_2DSNGP-1_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,3] -n Reuters_multilabel_nodropout_2DSNGP-3_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,5] -n Reuters_multilabel_nodropout_2DSNGP-5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,10] -n Reuters_multilabel_nodropout_2DSNGP-10_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,1] -n Reuters_multilabel_baseline_2DSNGP-1_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,3] -n Reuters_multilabel_baseline_2DSNGP-3_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,5] -n Reuters_multilabel_baseline_2DSNGP-5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,10] -n Reuters_multilabel_baseline_2DSNGP-10_ood

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=binary_categorical_crossentropy -n Reuters_multilabel_nodropout_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=binary_categorical_crossentropy -n Reuters_multilabel_baseline_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=binary_categorical_crossentropy -n Reuters_multilabel_baseline_concrete_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,15] -n Reuters_multilabel_nodropout_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,15] -n Reuters_multilabel_baseline_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=True loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,15] -n Reuters_multilabel_baseline_concrete_2DSNGP_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,1] -n Reuters_multilabel_nodropout_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,3] -n Reuters_multilabel_nodropout_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,5] -n Reuters_multilabel_nodropout_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,10] -n Reuters_multilabel_nodropout_2DSNGP-10

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,1] -n Reuters_multilabel_baseline_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,3] -n Reuters_multilabel_baseline_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,5] -n Reuters_multilabel_baseline_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=Reuters_multilabel steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers=[1,10] -n Reuters_multilabel_baseline_2DSNGP-10



> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=categorical_crossentropy -n 20news_nodropout_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=categorical_crossentropy -n 20news_baseline_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=categorical_crossentropy -n 20news_baseline_concrete_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,15] -n 20news_nodropout_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,15] -n 20news_baseline_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=True loss_fn=categorical_crossentropy_from_logits spec_norm_multipliers=[1,15] -n 20news_baseline_concrete_2DSNGP_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,1] -n 20news_nodropout_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,3] -n 20news_nodropout_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,5] -n 20news_nodropout_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,10] -n 20news_nodropout_2DSNGP-10

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,1] -n 20news_baseline_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,3] -n 20news_baseline_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,5] -n 20news_baseline_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,10] -n 20news_baseline_2DSNGP-10

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=5 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=categorical_crossentropy -n 20news_nodropout_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=5 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=categorical_crossentropy -n 20news_baseline_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=5 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=categorical_crossentropy -n 20news_baseline_concrete_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=5 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,15] -n 20news_nodropout_2DSNGP-15_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=5 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,15] -n 20news_baseline_2DSNGP-15_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=5 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=True loss_fn=categorical_crossentropy_from_logits spec_norm_multipliers=[1,15] -n 20news_baseline_concrete_2DSNGP_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=1 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,1] -n 20news_nodropout_2DSNGP-1_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=1 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,3] -n 20news_nodropout_2DSNGP-3_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=1 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,5] -n 20news_nodropout_2DSNGP-5_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=1 max_document_len=250 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,10] -n 20news_nodropout_2DSNGP-10_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=1 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,1] -n 20news_baseline_2DSNGP-1_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=1 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,3] -n 20news_baseline_2DSNGP-3_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=1 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,5] -n 20news_baseline_2DSNGP-5_ood

> python3 experiment.py with clf_default SNGP_default ood=['comp.graphics','comp.sys.ibm.pc.hardware','comp.windows.x','rec.autos','rec.sport.baseball','sci.crypt','sci.med','soc.religion.christian','talk.politics.mideast','talk.religion.misc'] ensemble=1 max_document_len=250 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=20news steps_per_epoch=None seed=42 max_vocabulary=30000 spec_norm_multipliers=[1,10] -n 20news_baseline_2DSNGP-10_ood



> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=100 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=categorical_crossentropy -n CLINC150_nodropout_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=100 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=categorical_crossentropy -n CLINC150_baseline_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=100 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 use_gp_layer=False spec_norm_multipliers=[0,0] loss_fn=categorical_crossentropy -n CLINC150_baseline_concrete_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=100 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,15] -n CLINC150_nodropout_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=100 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,15] -n CLINC150_baseline_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=100 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 use_gp_layer=True loss_fn=categorical_crossentropy_from_logits spec_norm_multipliers=[1,15] -n CLINC150_baseline_concrete_2DSNGP_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=100 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,1] -n CLINC150_nodropout_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=100 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,3] -n CLINC150_nodropout_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=100 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,5] -n CLINC150_nodropout_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=100 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,10] -n CLINC150_nodropout_2DSNGP-10

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=100 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,1] -n CLINC150_baseline_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=100 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,3] -n CLINC150_baseline_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=100 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,5] -n CLINC150_baseline_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=100 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=CLINC150 steps_per_epoch=None seed=42 max_vocabulary=20000 spec_norm_multipliers=[1,10] -n CLINC150_baseline_2DSNGP-10



> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=binary_categorical_crossentropy -n AAPD_nodropout_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=binary_categorical_crossentropy -n AAPD_baseline_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=binary_categorical_crossentropy -n AAPD_baseline_concrete_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,15]" -n AAPD_nodropout_2DSNGP-15_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,15]" -n AAPD_baseline_2DSNGP-15_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=True loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,15]" -n AAPD_baseline_concrete_2DSNGP_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,1]" -n AAPD_nodropout_2DSNGP-1_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,3]" -n AAPD_nodropout_2DSNGP-3_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,5]" -n AAPD_nodropout_2DSNGP-5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,10]" -n AAPD_nodropout_2DSNGP-10_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,1]" -n AAPD_baseline_2DSNGP-1_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,3]" -n AAPD_baseline_2DSNGP-3_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,5]" -n AAPD_baseline_2DSNGP-5_ood

> python3 experiment.py with clf_default SNGP_default ood=0 ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,10]" -n AAPD_baseline_2DSNGP-10_ood

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=binary_categorical_crossentropy -n AAPD_nodropout_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=binary_categorical_crossentropy -n AAPD_baseline_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=binary_categorical_crossentropy -n AAPD_baseline_concrete_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,15]" -n AAPD_nodropout_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,15]" -n AAPD_baseline_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 use_gp_layer=True loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,15]" -n AAPD_baseline_concrete_2DSNGP_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,1]" -n AAPD_nodropout_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,3]" -n AAPD_nodropout_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,5]" -n AAPD_nodropout_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,10]" -n AAPD_nodropout_2DSNGP-10

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,1]" -n AAPD_baseline_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,3]" -n AAPD_baseline_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,5]" -n AAPD_baseline_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=200 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=AAPD steps_per_epoch=None seed=42 max_vocabulary=30000 loss_fn=binary_crossentropy_from_logits spec_norm_multipliers="[1,10]" -n AAPD_baseline_2DSNGP-10



> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=categorical_crossentropy -n imdb_nodropout_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=categorical_crossentropy -n imdb_baseline_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=categorical_crossentropy -n imdb_baseline_concrete_2DCNN_M5

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=5 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=categorical_crossentropy -n imdb_nodropout_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=5 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=categorical_crossentropy -n imdb_baseline_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=5 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 use_gp_layer=False spec_norm_multipliers="[0,0]" loss_fn=categorical_crossentropy -n imdb_baseline_concrete_2DCNN_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=5 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,15]" -n imdb_nodropout_2DSNGP-15_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=5 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,15]" -n imdb_baseline_2DSNGP-15_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=5 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 use_gp_layer=True loss_fn=categorical_crossentropy_from_logits spec_norm_multipliers="[1,15]" -n imdb_baseline_concrete_2DSNGP_M5_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=1 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,1]" -n imdb_nodropout_2DSNGP-1_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=1 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,3]" -n imdb_nodropout_2DSNGP-3_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=1 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,5]" -n imdb_nodropout_2DSNGP-5_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=1 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,10]" -n imdb_nodropout_2DSNGP-10_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=1 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,1]" -n imdb_baseline_2DSNGP-1_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=1 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,3]" -n imdb_baseline_2DSNGP-3_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=1 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,5]" -n imdb_baseline_2DSNGP-5_ood

> python3 experiment.py with clf_default SNGP_default ood=5 ensemble=1 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,10]" -n imdb_baseline_2DSNGP-10_ood

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,15]" -n imdb_nodropout_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,15]" -n imdb_baseline_2DSNGP-15_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=5 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 use_gp_layer=True loss_fn=categorical_crossentropy_from_logits spec_norm_multipliers="[1,15]" -n imdb_baseline_concrete_2DSNGP_M5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,1]" -n imdb_nodropout_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,3]" -n imdb_nodropout_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,5]" -n imdb_nodropout_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=350 dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,10]" -n imdb_nodropout_2DSNGP-10

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,1]" -n imdb_baseline_2DSNGP-1

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,3]" -n imdb_baseline_2DSNGP-3

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,5]" -n imdb_baseline_2DSNGP-5

> python3 experiment.py with clf_default SNGP_default ood=None ensemble=1 max_document_len=350 dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=0.0001 use_aleatorics=False identifier=imdb steps_per_epoch=2000 seed=42 max_vocabulary=30000 spec_norm_multipliers="[1,10]" -n imdb_baseline_2DSNGP-10