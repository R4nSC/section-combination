import torch
import time
import copy
import sys
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

sys.path.append('../')
from models import IntegratedEnsembleModel, IntegratedEnsembleModelAddAllsection
from evaluation_phase import calc_results_from_predicts_and_labels


def train_integrated_ensemble_model(params, model, data_loaders, datasets_size, criterion,
                optimizer, scheduler, num_epochs):
    # 学習開始時間の保持
    since = time.time()

    # 初期の事前学習時点でのパラメータを保存
    best_model_wts = copy.deepcopy(model.network.state_dict())
    best_acc = 0.0

    loss_dict = {"train": [], "val": []}
    acc_dict = {"train": [], "val": []}

    print("ensemble models - Train phase")
    print('-' * 10)
    print()

    # 定めた回数分だけ学習を繰り返す
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # 各エポックでTrainingとValidationを繰り返す
        for phase in ['train', 'val']:
            # 学習時と推論時で振る舞いの違うモジュールの振る舞いを変更する
            if phase == 'train':
                model.train()  # 学習モードに変更
            else:
                model.eval()  # 推論モードに変更

            running_loss = 0.0
            running_corrects = 0

            # 用意したデータセットで学習を繰り返す
            for data in data_loaders[phase]:
                # 勾配を０で初期化する
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, labels, input_size = return_output_of_model(params, model, data)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 学習フェーズの時のみ，誤差逆伝播とパラメータ更新を行う
                    if phase == 'train':
                        loss.backward()  # 誤差逆伝播にて勾配を求める
                        optimizer.step()  # optimizerでパラメータを更新

                # statistics
                running_loss += loss.item() * input_size
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / datasets_size[phase]
            epoch_acc = running_corrects / datasets_size[phase]

            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the models
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.network.state_dict())

        print()

    time_elapsed = time.time() - since
    print('classification models - Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print()

    # load best models weights
    model.network.load_state_dict(best_model_wts)

    # モデルのパラメータを保存
    # if args.model_save:
    #     model_path = '{}/{}'.format(MODEL_DIR, MODEL_FILE)
    #     torch.save(model.network.state_dict(), model_path)

    return model, loss_dict, acc_dict


# 分類タスクのテスト
def evaluate_integrated_ensemble_model(params, model, data_loader):
    all_pred = torch.zeros(1)
    all_label = torch.zeros(1)

    all_pred = all_pred.to(params.device)
    all_label = all_label.to(params.device)

    model.eval()
    with torch.no_grad():
        output_flg = True
        for data in data_loader:
            outputs, labels, _ = return_output_of_model(params, model, data)

            _, predicted = torch.max(outputs.data, 1)

            if output_flg:
                output_flg = False
                all_pred = predicted
                all_label = labels
            else:
                all_pred = torch.cat((all_pred, predicted), 0)
                all_label = torch.cat((all_label, labels), 0)

    # 分類精度の計算
    accuracy = calc_results_from_predicts_and_labels(params, all_pred, all_label, None)
    return accuracy


def return_output_of_model(params, model, data):
    # allsectionをアンサンブルに組み込む場合
    if params.args.ensemble_add_allsection:
        inputs_allsection, inputs_text, inputs_rdata, inputs_data, labels = data
        inputs_allsection = inputs_allsection.to(params.device)
        inputs_text = inputs_text.to(params.device)
        inputs_rdata = inputs_rdata.to(params.device)
        inputs_data = inputs_data.to(params.device)
        labels = labels.to(params.device)

        outputs = model(inputs_allsection, inputs_text, inputs_rdata, inputs_data)
    # allsectionをアンサンブルに組み込まない場合
    else:
        inputs_text, inputs_rdata, inputs_data, labels = data
        inputs_text = inputs_text.to(params.device)
        inputs_rdata = inputs_rdata.to(params.device)
        inputs_data = inputs_data.to(params.device)
        labels = labels.to(params.device)

        outputs = model(inputs_text, inputs_rdata, inputs_data)
    return outputs, labels, inputs_text.size(0)


# アンサンブル型モデルの学習を行う
def train_of_integrated_ensemble_model(params, data_loaders, datasets_size, evaluation_flag=False):
    if params.args.ensemble_add_allsection:
        model = IntegratedEnsembleModelAddAllsection(params)
    else:
        model = IntegratedEnsembleModel(params)

    if params.args.debug:
        print(model)

    model = model.to(params.device)

    criterion_ft = nn.CrossEntropyLoss().to(params.device)

    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    if params.args.model_use:
        model_ft = model
    else:
        model_ft, loss, acc = train_integrated_ensemble_model(params, model, data_loaders, datasets_size, criterion_ft,
                                          optimizer_ft, exp_lr_scheduler, num_epochs=params.args.epochs)

    classification_result = -1
    if evaluation_flag:
        classification_result = evaluate_integrated_ensemble_model(params, model_ft, data_loaders['test'])

    return model_ft, classification_result
