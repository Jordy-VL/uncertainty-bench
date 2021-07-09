RUN=~/code/gordon/arkham/arkham/Bayes/Quantify
model="$1"
for i in M0_ M1_ M2_ M3_ M4_; do    python3 $RUN/evaluate.py $model -i $i ; done
for i in M0_ M1_ M2_ M3_ M4_; do    python3 $RUN/compare.py $model -i $i ; done
python3 $RUN/compare.py $model
