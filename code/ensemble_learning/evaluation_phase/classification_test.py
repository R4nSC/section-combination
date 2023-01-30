import torch
from sklearn.metrics import classification_report

# モデルの予測と正解ラベルから分類精度を算出する
def calc_results_from_predicts_and_labels(params, predicts, labels, paths):
    # 全予測数（テストデータ数と同値）と正解数を求める
    pred_total = labels.size(0)
    pred_correct = (predicts == labels).sum().item()

    # 混同行列の可視化
    # if params.args.gui:
    #     multi_classification_result_visualization(labels, predicts)

    # 単純な分類精度（正解数/全予測数）の出力
    print('Accuracy of the network on the {:d} test images: {:.4f}\n'.format(pred_total, pred_correct / pred_total))

    # 分類に失敗したサンプルのパスを保存する
    # if params.args.result_analysis:
    #     memo_paths = params.yaml['log']['failed_sample']
    #     failed_classification_analysis(labels, predicts, paths, memo_paths)

    # classification_reportを利用した分類精度の出力（PredictionやRecallなども同時に算出）
    label_cpu = labels.cpu()
    pred_cpu = predicts.cpu()
    print(classification_report(label_cpu, pred_cpu, digits=4))

    # 簡易的な精度を返す
    return pred_correct / pred_total


# 分類タスクのテスト
def multi_classification_test(params, model, data_loader):
    all_pred = torch.zeros(1)
    all_label = torch.zeros(1)
    all_path = tuple()

    all_pred = all_pred.to(params.device)
    all_label = all_label.to(params.device)

    model.eval()
    with torch.no_grad():
        output_flg = True
        for data in data_loader:
            images, labels, paths = data

            images = images.to(params.device)
            labels = labels.to(params.device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            if output_flg:
                output_flg = False
                all_pred = predicted
                all_label = labels
                all_path = paths

                if params.args.debug:
                    print(outputs)
                    print(labels)
                    print(paths)
            else:
                all_pred = torch.cat((all_pred, predicted), 0)
                all_label = torch.cat((all_label, labels), 0)
                all_path = all_path + paths

    # 分類精度の計算
    accuracy = calc_results_from_predicts_and_labels(params, all_pred, all_label, all_path)
    return accuracy
