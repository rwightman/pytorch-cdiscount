import torchvision.models
from .resnext101_32x4d import resnext101_32x4d
from .inception_v4 import inception_v4
from .inception_resnet_v2 import inception_resnet_v2
from .wrn50_2 import wrn50_2
from .my_densenet import densenet161, densenet121, densenet169, densenet201
from .my_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .fbresnet200 import fbresnet200
from .dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
#from .transformed_model import TransformedModel
from .load_checkpoint import load_checkpoint


def normalizer_from_model(model_name):
    if 'inception' in model_name:
        normalizer = 'le'
    elif 'dpn' in model_name:
        normalizer = 'dpn'
    else:
        normalizer = 'torchvision'
    return normalizer


model_config_dict = {
    'resnet18': {
        'model_name': 'resnet18', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'torchvision',
        'checkpoint_file': 'resnet18-5c106cde.pth', 'drop_first_class': False},
    'resnet34': {
        'model_name': 'resnet34', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'torchvision',
        'checkpoint_file': 'resnet34-333f7ec4.pth', 'drop_first_class': False},
    'resnet50': {
        'model_name': 'resnet50', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'torchvision',
        'checkpoint_file': 'resnet50-19c8e357.pth', 'drop_first_class': False},
    'resnet101': {
        'model_name': 'resnet101', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'torchvision',
        'checkpoint_file': 'resnet101-5d3b4d8f.pth', 'drop_first_class': False},
    'resnet152': {
        'model_name': 'resnet152', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'torchvision',
        'checkpoint_file': 'resnet152-b121ed2d.pth', 'drop_first_class': False},
    'densenet121': {
        'model_name': 'densenet121', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'torchvision',
        'checkpoint_file': 'densenet121-241335ed.pth', 'drop_first_class': False},
    'densenet169': {
        'model_name': 'densenet169', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'torchvision',
        'checkpoint_file': 'densenet169-6f0f7f60.pth', 'drop_first_class': False},
    'densenet201': {
        'model_name': 'densenet201', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'torchvision',
        'checkpoint_file': 'densenet201-4c113574.pth', 'drop_first_class': False},
    'densenet161': {
        'model_name': 'densenet161', 'num_classes': 1000, 'input_size': 224, 'normalizer': 'torchvision',
        'checkpoint_file': 'densenet161-17b70270.pth', 'drop_first_class': False},
    'dpn107': {
        'model_name': 'dpn107', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dualpathnet',
        'checkpoint_file': 'dpn107_extra-fc014e8ec.pth', 'drop_first_class': False},
    'dpn92_extra': {
        'model_name': 'dpn92', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dualpathnet',
        'checkpoint_file': 'dpn92_extra-1f58102b.pth', 'drop_first_class': False},
    'dpn92': {
        'model_name': 'dpn92', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dualpathnet',
        'checkpoint_file': 'dpn92-7d0f7156.pth', 'drop_first_class': False},
    'dpn68': {
        'model_name': 'dpn68', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dualpathnet',
        'checkpoint_file': 'dpn68-abcc47ae.pth', 'drop_first_class': False},
    'dpn68b': {
        'model_name': 'dpn68b', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dualpathnet',
        'checkpoint_file': 'dpn68_extra.pth', 'drop_first_class': False},
    'dpn68b_extra': {
        'model_name': 'dpn68b', 'num_classes': 1000, 'input_size': 299, 'normalizer': 'dualpathnet',
        'checkpoint_file': 'dpn68_extra.pth', 'drop_first_class': False},
    'inception_resnet_v2': {
        'model_name': 'inception_resnet_v2', 'num_classes': 1001, 'input_size': 299, 'normalizer': 'le',
        'checkpoint_file': 'inceptionresnetv2-d579a627.pth', 'drop_first_class': True},
}


def config_from_string(string, output_fn='log_softmax'):
    config = model_config_dict[string]
    config['output_fn'] = output_fn
    return config


def create_model(
        model_name='resnet50',
        pretrained=False,
        num_classes=1000,
        checkpoint_path='',
        **kwargs):

    if 'test_time_pool' in kwargs:
        test_time_pool = kwargs.pop('test_time_pool')
    else:
        test_time_pool = False

    if model_name == 'dpn68':
        model = dpn68(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn68b':
        model = dpn68b(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn92':
        model = dpn92(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn98':
        model = dpn98(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn131':
        model = dpn131(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn107':
        model = dpn107(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet152':
        model = resnet152(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet121':
        model = densenet121(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet161':
        model = densenet161(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet169':
        model = densenet169(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet201':
        model = densenet201(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'inception_resnet_v2':
        model = inception_resnet_v2(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'inception_v4':
        model = inception_v4(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'resnext101_32x4d':
        model = resnext101_32x4d(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'wrn50':
        model = wrn50_2(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'fbresnet200':
        model = fbresnet200(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    else:
        assert False and "Invalid model"

    if checkpoint_path and not pretrained:
        load_checkpoint(model, checkpoint_path)

    return model


def create_model_from_cfg(mc, checkpoint_path=''):
    if 'kwargs' not in mc:
        mc['kwargs'] = {}

    model = create_model(
        model_name=mc['model_name'],
        num_classes=mc['num_classes'],
        checkpoint_path=checkpoint_path if checkpoint_path else mc['checkpoint_file'],
        **mc['kwargs']
    )

    return model
