import torch
import sys
import time
import copy
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler

sys.path.append('../')
from evaluation_phase import multi_classification_test
from models import CNN


def train_model(params, model, data_loaders, datasets_size, criterion,
                optimizer, scheduler, num_epochs):
    # 学習開始時間の保持
    since = time.time()

    # 初期の事前学習時点でのパラメータを保存
    best_model_wts = copy.deepcopy(model.network.state_dict())
    best_acc = 0.0

    loss_dict = {"train": [], "val": []}
    acc_dict = {"train": [], "val": []}

    print("embedding models - Train phase")
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
                # optimizer.step()
                # scheduler.step()
                model.train()  # 学習モードに変更
            else:
                model.eval()  # 推論モードに変更

            running_loss = 0.0
            running_corrects = 0

            # 用意したデータセットで学習を繰り返す
            for data in data_loaders[phase]:

                # 入力データ(バッチサイズ分)
                inputs, labels, _ = data
                inputs = inputs.to(params.device)
                labels = labels.to(params.device)

                # 勾配を０で初期化する
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 学習フェーズの時のみ，誤差逆伝播とパラメータ更新を行う
                    if phase == 'train':
                        loss.backward()  # 誤差逆伝播にて勾配を求める
                        optimizer.step()  # optimizerでパラメータを更新

                # statistics
                running_loss += loss.item() * inputs.size(0)
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

    return model, loss_dict, acc_dict


def classification_model(params, model, data_loaders, datasets_size):
    criterion_ft = nn.CrossEntropyLoss().to(params.device)

    # Observe that all parameters are being optimized
    # weight_decay=10^-4
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    if params.args.model_use:
        model_ft = model
    else:
        model_ft, loss, acc = train_model(params, model, data_loaders, datasets_size, criterion_ft, optimizer_ft, exp_lr_scheduler, num_epochs=params.args.epochs)

        # if params.args.gui:
        #     train_visualization(loss, acc, params.args.epochs)

    return model_ft


def train_of_ensemble_model(params, data_loaders, datasets_size, section=None):
    cnn = CNN(params, section)

    cnn.model = classification_model(params, cnn.model, data_loaders, datasets_size)

    # モデルのパラメータを保存
    if params.args.model_save:
        torch.save(cnn.model.network.state_dict(), cnn.write_parameter_path)

    classification_result = multi_classification_test(params, cnn.model, data_loaders['test'])

    return cnn, classification_result
