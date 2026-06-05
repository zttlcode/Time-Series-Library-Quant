from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
import SQTool.Tools as SQTools

warnings.filterwarnings('ignore')

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.59, 0.08, 0.08]))
        return criterion

    def _compute_metrics(self, trues, probs, predictions):
        """
        兼容二分类 / 多分类的指标计算
        """
        trues = np.asarray(trues).reshape(-1)
        predictions = np.asarray(predictions).reshape(-1)
        probs = np.asarray(probs)

        num_class = self.args.num_class
        out = {}

        # 基础指标
        out["accuracy"] = float(np.mean(predictions == trues))
        out["balanced_accuracy"] = float(balanced_accuracy_score(trues, predictions))

        if num_class == 2:
            # 二分类：默认把正类记为 1
            if probs.ndim == 2 and probs.shape[1] >= 2:
                pos_score = probs[:, 1]
            else:
                pos_score = probs.reshape(-1)

            out["precision"] = float(precision_score(trues, predictions, zero_division=0))
            out["recall"] = float(recall_score(trues, predictions, zero_division=0))
            out["f1"] = float(f1_score(trues, predictions, zero_division=0))

            try:
                out["auc"] = float(roc_auc_score(trues, pos_score))
            except Exception:
                out["auc"] = np.nan

            try:
                out["pr_auc"] = float(average_precision_score(trues, pos_score))
            except Exception:
                out["pr_auc"] = np.nan

        else:
            # 多分类：macro 平均
            out["precision_macro"] = float(precision_score(trues, predictions, average="macro", zero_division=0))
            out["recall_macro"] = float(recall_score(trues, predictions, average="macro", zero_division=0))
            out["f1_macro"] = float(f1_score(trues, predictions, average="macro", zero_division=0))

            try:
                y_true_bin = label_binarize(trues, classes=np.arange(num_class))
                out["auc_macro_ovr"] = float(roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovr"))
            except Exception:
                out["auc_macro_ovr"] = np.nan

            try:
                y_true_bin = label_binarize(trues, classes=np.arange(num_class))
                out["pr_auc_macro"] = float(average_precision_score(y_true_bin, probs, average="macro"))
            except Exception:
                out["pr_auc_macro"] = np.nan

        return out

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                # 报错用下面的ValueError: Expected input batch_size (1) to match target batch_size (0).
                # loss = criterion(pred, label.long().squeeze(-1).cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        # trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        trues = trues.flatten().cpu().numpy()

        metrics = self._compute_metrics(trues, probs, predictions)
        metrics["loss"] = float(total_loss)
        metrics["n"] = int(len(trues))

        self.model.train()
        return total_loss, metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                if hasattr(self.model, 'collect_batch'):
                    self.model.collect_batch(batch_x, label)
                # # 20250118 必须在这里把他放GPU上，但后面要放回CPU，没办法通过传参做，只能重新new个
                # criterion_tmp = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.59, 0.08, 0.08]).to(self.device))
                # loss = criterion_tmp(outputs, label.long().squeeze(-1))
                # criterion_tmp = None  # 为了防止内存泄漏，用完就清空
                loss = criterion(outputs, label.long().squeeze(-1))

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            # test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)
            #
            # print(
            #     "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
            #     .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            # early_stopping(-val_accuracy, self.model, path)
            vali_loss, val_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | "
                "Train Loss: {2:.3f} | Val Loss: {3:.3f} | Val Acc: {4:.3f} | Val F1: {5:.3f} | Val AUC: {6:.3f} | Val PR-AUC: {7:.3f} | "
                "Test Loss: {8:.3f} | Test Acc: {9:.3f} | Test F1: {10:.3f} | Test AUC: {11:.3f} | Test PR-AUC: {12:.3f}"
                .format(
                    epoch + 1, train_steps,
                    train_loss,
                    vali_loss,
                    val_metrics.get("accuracy", np.nan),
                    val_metrics.get("f1", val_metrics.get("f1_macro", np.nan)),
                    val_metrics.get("auc", val_metrics.get("auc_macro_ovr", np.nan)),
                    val_metrics.get("pr_auc", val_metrics.get("pr_auc_macro", np.nan)),
                    test_loss,
                    test_metrics.get("accuracy", np.nan),
                    test_metrics.get("f1", test_metrics.get("f1_macro", np.nan)),
                    test_metrics.get("auc", test_metrics.get("auc_macro_ovr", np.nan)),
                    test_metrics.get("pr_auc", test_metrics.get("pr_auc_macro", np.nan)),
                )
            )
            early_stopping(-val_metrics["accuracy"], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
        if hasattr(self.model, 'fit_xgb'):
            self.model.fit_xgb()
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, asset_code=None, pred_market=False, feature_plan_name=None, pred_live=False):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            # 20250118 这库没用预测代码，autoformer有，所以这里读不出文件，要改成这样，另外训练出的chekpoint和测试读的setting不一样，预测时要改
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        # trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        trues = trues.flatten().cpu().numpy()

        # # 打印每个样本的所有类别概率
        # if pred_live:
        #     probs = torch.softmax(preds, dim=1).cpu().numpy()
        #     print("class 1 prob:", probs[0][0])
        #     print("class 2 prob:", probs[0][1])
        #     print("class 3 prob:", probs[0][2])
        #     print("class 4 prob:", probs[0][3])

        metrics = self._compute_metrics(trues, probs, predictions)
        accuracy = metrics["accuracy"]

        if test:
            # 预测一次
            import pandas as pd
            # 用字典映射  UEAloader把ts文件中的分类转为了类别编码，这里把类别编码转回成类别名称
            if pred_market:
                test_loader.dataset.class_names = ["1", "2", "3"]
            else:
                test_loader.dataset.class_names = ["1", "2", "3", "4"]
            label_map = {i: name for i, name in enumerate(test_loader.dataset.class_names)}
            # 通过 map() 转换
            decoded_trues = list(map(label_map.get, trues))
            decoded_predictions = list(map(label_map.get, predictions))

            # 手动设置开关  is trues,predictions,predictions_market?
            if pred_market:
                # 预测行情分类，把trues,predictions,predictions_market合并在一起
                df = pd.read_csv('./results/'+asset_code+'_prd_result.csv')
                df['predictions_market'] = decoded_predictions
                df.to_csv('./results/'+asset_code+'_prd_result_tpp.csv', index=False)
            else:
                df = pd.DataFrame({'trues': decoded_trues, 'predictions': decoded_predictions})
                # 保存为CSV文件
                prob_df_path = SQTools.read_config("SQData", "inference_live")
                os.makedirs(prob_df_path, exist_ok=True)
                df.to_csv(prob_df_path + asset_code + '_prd_result.csv', index=False)

                # # 过滤出Column1为0或2的行
                # filtered_df = df[df['trues'].isin([0, 2])]
                # # 保存到新的CSV文件
                # filtered_df.to_csv('./results/filtered_output.csv', index=False)
                if pred_live:
                    # 预测概率
                    probs = torch.softmax(preds, dim=1).cpu().numpy()
                    # 四个类别概率，按你的类别顺序对应下标 0,1,2,3
                    if self.args.num_class == 2:
                        prob_df = pd.DataFrame({
                            'class1': [probs[0][0]],
                            'class2': [probs[0][1]],
                        })
                    else:
                        prob_df = pd.DataFrame({
                            'class1': [probs[0][0]],
                            'class2': [probs[0][1]],
                            'class3': [probs[0][2]],
                            'class4': [probs[0][3]],
                        })
                    prob_df_path = SQTools.read_config("SQData", "inference_live")
                    os.makedirs(prob_df_path, exist_ok=True)
                    # 保存概率文件
                    prob_df.to_csv(f'{prob_df_path}{asset_code}_prd_prob.csv', index=False)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        # f.write('accuracy:{}'.format(accuracy))
        if feature_plan_name is not None:
            f.write(f'feature_plan_name: {feature_plan_name}\n')
        f.write(f'accuracy: {metrics["accuracy"]}\n')
        f.write(f'balanced_accuracy: {metrics.get("balanced_accuracy", np.nan)}\n')

        if self.args.num_class == 2:
            f.write(f'precision: {metrics.get("precision", np.nan)}\n')
            f.write(f'recall: {metrics.get("recall", np.nan)}\n')
            f.write(f'f1: {metrics.get("f1", np.nan)}\n')
            f.write(f'auc: {metrics.get("auc", np.nan)}\n')
            f.write(f'pr_auc: {metrics.get("pr_auc", np.nan)}\n')
        else:
            f.write(f'precision_macro: {metrics.get("precision_macro", np.nan)}\n')
            f.write(f'recall_macro: {metrics.get("recall_macro", np.nan)}\n')
            f.write(f'f1_macro: {metrics.get("f1_macro", np.nan)}\n')
            f.write(f'auc_macro_ovr: {metrics.get("auc_macro_ovr", np.nan)}\n')
            f.write(f'pr_auc_macro: {metrics.get("pr_auc_macro", np.nan)}\n')
        f.write('\n')
        f.write('\n')
        f.close()
        return
