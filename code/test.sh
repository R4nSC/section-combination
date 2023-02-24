#!/bin/bash

if [ $# -ne 3 ]; then
  echo "指定された引数は$#個です" 1>&2
  echo "実行するには3個の引数が必要です" 1>&2
  exit 1
fi

# ensemble model experiment using cross validation
echo "Ensemble Model Experiment Using Cross-Validation"
echo "epoch: $2, batchsize: $3, model: pretrained"

# for i in `seq 1 10`
# do
#     python experiment_of_ensemble_model.py -n vgg16 -e $2 -b $3 -m 1 --cross-validation > ../logs/vgg16_original_cv7$i.txt
# done

# python experiment_of_ensemble_model.py -n resnet50 -e $2 -b $3 -m 1 --cross-validation > ../logs/resnet50_original_cv5.txt

for i in `seq 1 10`
do
  # python experiment_of_ensemble_model.py -n vgg16 -d BIG2015 -e $2 -b $3 -m 6 --cross-validation > ../logs/BIG2015/vgg16_voting_cv$i.txt
  # python experiment_of_ensemble_model.py -n vgg16 -d BIG2015 -e $2 -b $3 -m 6 --cross-validation --ensemble-add-allsection > ../logs/BIG2015/vgg16_voting_add-allsection_cv$i.txt

  python experiment_of_ensemble_model.py -n vgg16 -d Malimg -e $2 -b $3 -m 6 --cross-validation > ../logs/Malimg/vgg16_voting_cv$i.txt
  # python experiment_of_ensemble_model.py -n vgg16 -d Malimg -e $2 -b $3 -m 6 --cross-validation --ensemble-add-allsection > ../logs/Malimg/vgg16_voting_add-allsection_cv$i.txt

  # python experiment_of_ensemble_model.py -n resnet50 -d BIG2015 -e $2 -b $3 -m 6 --cross-validation > ../logs/BIG2015/resnet50_voting_cv$i.txt
  # python experiment_of_ensemble_model.py -n resnet50 -d BIG2015 -e $2 -b $3 -m 6 --cross-validation --ensemble-add-allsection > ../logs/BIG2015/resnet50_voting_add-allsection_cv$i.txt

  python experiment_of_ensemble_model.py -n resnet50 -d Malimg -e $2 -b $3 -m 6 --cross-validation > ../logs/Malimg/resnet50_voting_cv$i.txt
  # python experiment_of_ensemble_model.py -n resnet50 -d Malimg -e $2 -b $3 -m 6 --cross-validation --ensemble-add-allsection > ../logs/Malimg/resnet50_voting_add-allsection_cv$i.txt
done

# for i in `seq 1 10`
# do
#     python experiment_of_ensemble_model.py -n resnet50 -e $2 -b $3 -m 4 > ../logs/resnet50_svm$i.txt
# done
