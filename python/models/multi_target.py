import torch.nn as nn


class MultiTargetLoss(nn.Module):
    def __init__(self, scales=None, loss_fn=nn.CrossEntropyLoss()):
        super(MultiTargetLoss, self).__init__()
        self.loss_fn = loss_fn
        self.scales = scales

    def forward(self, inputs, targets):
        if not self.scales:
            scales = [1.0] * len(inputs)
        else:
            scales = self.scales
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


class MultiTargetModel(nn.Module):
    def __init__(self, model, num_class_list):
        super(MultiTargetModel, self).__init__()
        self.model = model
        self.model.reset_classifier(num_classes=0)
        self.classifiers = MultiTargetLayer(model.num_features, num_class_list)

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.classifiers(x)



