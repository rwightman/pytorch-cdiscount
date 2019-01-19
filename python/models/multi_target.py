import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTargetLoss(nn.Module):
    def __init__(self, learnable=0, fixed_scales=None, primary_scale=0.0,
                 loss_fn=nn.CrossEntropyLoss()):
        super(MultiTargetLoss, self).__init__()
        self.loss_fn = loss_fn
        self.fixed_scales = fixed_scales
        self.learnable = learnable
        self.primary_scale = primary_scale
        if learnable:
            params = [nn.Parameter(torch.FloatTensor([0.])) for _ in range(learnable)]
            self.log_var = nn.ParameterList(params)

    def forward(self, inputs, targets):
        if self.learnable:
            losses = []
            start_idx = 0
            if self.primary_scale:
                start_idx = 1
                losses += [self.primary_scale * self.loss_fn(inputs[0, targets[0]])]
            losses += [torch.exp(-lv) * self.loss_fn(i, t) + lv
                       for i, t, lv in zip(inputs[start_idx:], targets[start_idx:], self.log_var)]
            return torch.mean(torch.cat(losses))
        else:
            if not self.fixed_scales:
                n = len(inputs)
                scales = [1.0 / n] * n
            else:
                scales = self.fixed_scales
            losses = [s * self.loss_fn(i, t) for i, t, s in zip(inputs, targets, scales)]
            return sum(losses)


class MultiTargetLayer(nn.Module):
    def __init__(self, num_features, num_class_list):
        super(MultiTargetLayer, self).__init__()
        self.num_features = num_features
        assert isinstance(num_class_list, list)
        self.classifiers = nn.ModuleList([
            nn.Linear(self.num_features, nc) for nc in num_class_list])

    def forward(self, x):
        return [c(x) for c in self.classifiers]


class MultiTargetCascadeLayer(nn.Module):
    def __init__(self, num_features, num_class_list, act_fn=F.elu):
        super(MultiTargetCascadeLayer, self).__init__()
        self.num_features = num_features
        assert isinstance(num_class_list, list)
        num_prev = self.num_features
        layers = []
        for nc in num_class_list:
            layers.append(nn.Linear(num_prev, nc))
            num_prev = nc
        self.classifiers = nn.ModuleList(layers)
        self.act_fn = act_fn

    def forward(self, x):
        outputs = []
        for i, c in enumerate(self.classifiers):
            x = c(x)
            outputs.append(x)
            if i != len(self.classifiers) - 1:
                x = self.act_fn(x)
        return outputs


class MultiTargetModel(nn.Module):
    def __init__(self, model, num_class_list, sandwich=True, cascade=False):
        super(MultiTargetModel, self).__init__()
        self.model = model
        self.model.reset_classifier(num_classes=0)
        if sandwich:
            self.sandwich = nn.Linear(model.num_features, model.num_features)
        else:
            self.sandwich = None
        if cascade:
            self.classifiers = MultiTargetCascadeLayer(model.num_features, num_class_list)
        else:
            self.classifiers = MultiTargetLayer(model.num_features, num_class_list)

    def forward(self, x):
        x = self.model.forward_features(x)
        if self.sandwich is not None:
            x = self.sandwich(x)
        return self.classifiers(x)




