import numpy as np
import torch
import torch.nn as nn

from xgboost import XGBClassifier


class SequenceFeatureExtractor:
    """
    (B,T,F) -> (B,D)
    """

    def transform(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        last = x[:, -1, :]
        mean = np.mean(x, axis=1)
        std = np.std(x, axis=1)
        maxv = np.max(x, axis=1)
        minv = np.min(x, axis=1)
        delta = x[:, -1, :] - x[:, 0, :]

        feat = np.concatenate([
            last,
            mean,
            std,
            maxv,
            minv,
            delta
        ], axis=1)

        feat = np.nan_to_num(feat)

        return feat.astype(np.float32)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.num_class = configs.num_class

        self.extractor = SequenceFeatureExtractor()

        self.xgb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob' if self.num_class > 2 else 'binary:logistic',
            num_class=self.num_class if self.num_class > 2 else None,
            eval_metric='mlogloss' if self.num_class > 2 else 'logloss',
            tree_method='hist',
            random_state=42
        )

        # ---------------------------------------------------
        # 关键：
        # 给 optimizer 一个假的 parameter
        # 否则 self.model.parameters() 为空会报错
        # ---------------------------------------------------
        self.dummy_param = nn.Parameter(torch.zeros(1))

        self.is_fitted = False

        # 缓存训练数据
        self.train_x_cache = []
        self.train_y_cache = []

    def fit_xgb(self):

        X = np.concatenate(self.train_x_cache, axis=0)
        y = np.concatenate(self.train_y_cache, axis=0)

        if len(np.unique(y)) < 2:
            return

        self.xgb.fit(X, y)

        self.is_fitted = True

        # 训练完清空缓存
        self.train_x_cache = []
        self.train_y_cache = []

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        device = x_enc.device

        # ---------------------------------------------------
        # 特征提取
        # ---------------------------------------------------
        X = self.extractor.transform(x_enc)

        # ---------------------------------------------------
        # 训练阶段
        # ---------------------------------------------------
        if self.training:

            # label 无法从 forward 拿到
            # 所以只能做“在线伪训练”
            #
            # 方案：
            # 第一次 forward 时返回随机 logits
            # 后面在 train() 外部 test() 时再真正预测
            #
            # 因为你框架写死了 backward
            # 所以这里只能兼容
            #

            batch_size = X.shape[0]

            logits = torch.randn(
                batch_size,
                self.num_class,
                device=device,
                requires_grad=True
            )

            # 给 dummy_param 建图
            logits = logits + self.dummy_param * 0

            return logits

        # ---------------------------------------------------
        # 推理阶段
        # ---------------------------------------------------
        else:

            batch_size = X.shape[0]

            if not self.is_fitted:

                logits = torch.zeros(
                    batch_size,
                    self.num_class,
                    device=device
                )

                return logits

            probs = self.xgb.predict_proba(X)

            probs = np.clip(probs, 1e-7, 1.0)

            logits_np = np.log(probs)

            logits = torch.tensor(
                logits_np,
                dtype=torch.float32,
                device=device
            )

            return logits

    # -------------------------------------------------------
    # 真正训练XGB
    # -------------------------------------------------------
    def collect_batch(self, batch_x, label):

        X = self.extractor.transform(batch_x)

        y = label.detach().cpu().numpy()
        y = y.reshape(-1)

        self.train_x_cache.append(X)
        self.train_y_cache.append(y)

    # -------------------------------------------------------
    # 保存
    # -------------------------------------------------------
    def state_dict(self, *args, **kwargs):

        return {
            "dummy_param": self.dummy_param.data,
            "is_fitted": self.is_fitted,
            "xgb_model": self.xgb
        }

    # -------------------------------------------------------
    # 加载
    # -------------------------------------------------------
    def load_state_dict(self, state_dict, strict=True):

        self.dummy_param.data = state_dict["dummy_param"]

        self.is_fitted = state_dict["is_fitted"]

        self.xgb = state_dict["xgb_model"]