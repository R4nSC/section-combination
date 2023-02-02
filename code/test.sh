#!/bin/bash

if [ $# -ne 3 ]; then
  echo "指定された引数は$#個です" 1>&2
  echo "実行するには3個の引数が必要です" 1>&2
  exit 1
fi

# ensemble model experiment using cross validation
echo "Ensemble Model Experiment Using Cross-Validation"
echo "epoch: $2, batchsize: $3, model: pretrained"

python experiment_of_integrated_ensemble_model.py -n vgg16 -e $2 -b $3 -m 2 --cross-validation > ../logs/vgg16_integrated_cv1.txt

python experiment_of_integrated_ensemble_model.py -n vgg16 -e $2 -b $3 -m 2 --cross-validation --ensemble-add-allsection > ../logs/vgg16_integrated_cv1_add-allsection.txt

python experiment_of_integrated_ensemble_model.py -n resnet50 -e $2 -b $3 -m 2 --cross-validation > ../logs/resnet50_integrated_cv1.txt

python experiment_of_integrated_ensemble_model.py -n resnet50 -e $2 -b $3 -m 2 --cross-validation --ensemble-add-allsection > ../logs/resnet50_integrated_cv1_add-allsection.txt

for i in `seq 1 10`
do
    python experiment_of_ensemble_model.py -n resnet50 -e $2 -b $3 -m 4 > ../logs/resnet50_svm$i.txt
done
