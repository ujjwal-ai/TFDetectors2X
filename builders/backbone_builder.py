import os
from loguru import logger
from backbones import vgg
from protos import backbones_pb2

BACKBONES = {
    'vgg11': vgg.vgg11,
    'vgg13': vgg.vgg13,
    'vgg16': vgg.vgg16,
    'vgg19': vgg.vgg19,
    'bn_vgg11': vgg.bn_vgg11,
    'bn_vgg13': vgg.bn_vgg13,
    'bn_vgg16': vgg.bn_vgg16,
    'bn_vgg19': vgg.bn_vgg19,
    'bn_vgg11b': vgg.bn_vgg11b,
    'bn_vgg13b': vgg.bn_vgg13b,
    'bn_vgg16b': vgg.bn_vgg16b,
    'bn_vgg19b': vgg.bn_vgg19b
}


def build_backbone(
        backbone_protoconfig
):
    """
    Constructes a backbone network with or without pretrained weights.
    :param backbone_protoconfig: A proto message of type BackBone defined in
    protos/backbones.proto.
    :return: A tf.keras.Model instance representing the backbone network.
    """
    if not isinstance(backbone_protoconfig,
                      backbones_pb2.BackBone):
        logger.error('The provided protoconfig to get_backbone(.) is not of '
                     'the message type BackBone defined in '
                     'protos/backbones.proto.')
        raise ValueError('The provided protoconfig to get_backbone(.) is not of '
                     'the message type BackBone defined in '
                     'protos/backbones.proto.')
    backbone_name = backbone_protoconfig.name
    use_pretrained = backbone_protoconfig.use_pretrained
    pretrained_path = backbone_protoconfig.pretrained_path
    if backbone_name not in BACKBONES.keys():
        logger.error('The backbone {} is not supported.'.format(backbone_name))
        raise ValueError(
            'The backbone {} is not supported.'.format(backbone_name))

    if use_pretrained:
        if not pretrained_path:
            logger.info('use_pretrained is set to True but pretrained_path '
                        'is provided as None. Therefore setting '
                        'pretrained_path to "~/pretrained_backbones".')
            pretrained_path = "~/pretrained_backbones"

    try:
        os.makedirs(pretrained_path, exist_ok=True)
    except OSError:
        logger.error('The pretrained path {} cannot be constructed.'.format(
            pretrained_path))
        raise OSError('The pretrained path {} cannot be constructed.'.format(
            pretrained_path))

    backbone_fn = BACKBONES[backbone_name]
    try:
        backbone = backbone_fn(pretrained=use_pretrained,
                               root=pretrained_path
                               )
        logger.info('Backbone {} constructed with pretrained '
                    'weights.'.format(backbone_name))
    except ValueError:
        logger.info('No pretrained model is available for the backbone {}. '
                    'Building without using a pretrained model.'.format(
            backbone_name))
        backbone = backbone_fn(pretrained=False)
        logger.info('Backbone {} constructed without pretrained '
                    'weights.'.format(backbone_name))

    return backbone
