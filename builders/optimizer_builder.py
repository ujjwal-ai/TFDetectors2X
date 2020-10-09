import tensorflow as tf
from builders.learning_rate_schedule_builder import build_learning_rate_schedule
from protos import optimizers_pb2


def build_optimizer(proto_config):
    """
    Builds an optimizer from the given proto configuration mesasge
    :param proto_config: A proto mesasge of type optimizers_pb2.Optimizer()
    :return: An instance that is a subclass of tf.keras.optimizers.Optimizer
    """
    if not isinstance(proto_config, optimizers_pb2.Optimizer):
        raise ValueError(
            'The proto_config argument to build_optimizer() must be of type as the message Optimizer defined in protos/optimizers.proto ')

    optimizer_type = proto_config.WhichOneof('Opt')
    if optimizer_type == 'adadelta':
        return build_adadelta(proto_config)
    elif optimizer_type == 'adagrad':
        return build_adagrad(proto_config)
    elif optimizer_type == 'adam':
        return build_adam(proto_config)
    elif optimizer_type == 'adamax':
        return build_adamax(proto_config)
    elif optimizer_type == 'nadam':
        return build_nadam(proto_config)
    elif optimizer_type == 'rmsprop':
        return build_rmsprop(proto_config)
    else:
        return build_sgd(proto_config)


def build_adadelta(proto_config):
    """
    Returns an instance of the Adadelta Optimizer
    :param proto_config: A proto mesasge of type optimizers_pb2.Optimizer()
    :return: An instance of the class tf.keras.optimizers.Adadelta
    """
    learning_rate_config = proto_config.adadelta.learning_rate_schedule
    learning_rate_schedule = build_learning_rate_schedule(learning_rate_config)
    rho = proto_config.adadelta.rho
    epsilon = proto_config.adadelta.epsilon
    optimizer = tf.keras.optimizers.Adadelta(
        learning_rate=learning_rate_schedule,
        rho=rho,
        epsilon=epsilon
    )
    return optimizer


def build_adagrad(proto_config):
    """
    Returns an instance of Adagrad Optimizer
    :param proto_config: A proto mesasge of type optimizers_pb2.Optimizer()
    :return: An instance of the class tf.keras.optimizers.Adagrad
    """
    learning_rate_config = proto_config.adagrad.learning_rate_schedule
    learning_rate_schedule = build_learning_rate_schedule(learning_rate_config)
    initial_accumulator_value = proto_config.adagrad.initial_accumulator_value
    epsilon = proto_config.adagrad.epsilon
    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=learning_rate_schedule,
        initial_accumulator_value=initial_accumulator_value,
        epsilon=epsilon
    )

    return optimizer


def build_adam(proto_config):
    """
    Returns an instance of the Adam Optimizer
    :param proto_config: A proto mesasge of type optimizers_pb2.Optimizer()
    :return: An instance of the class tf.keras.optimizers.Adam
    """
    learning_rate_config = proto_config.adam.learning_rate_schedule
    learning_rate_schedule = build_learning_rate_schedule(learning_rate_config)
    beta_1 = proto_config.adam.beta_1
    beta_2 = proto_config.adam.beta_2
    epsilon = proto_config.adam.epsilon
    amsgrad = proto_config.adam.amsgrad
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        amsgrad=amsgrad
    )

    return optimizer


def build_adamax(proto_config):
    """
    Returns an instance of the AdaMax Optimizer
    :param proto_config: A proto mesasge of type optimizers_pb2.Optimizer()
    :return: An instance of the class tf.keras.optimizers.Adamax
    """
    learning_rate_config = proto_config.adamax.learning_rate_schedule
    learning_rate_schedule = build_learning_rate_schedule(learning_rate_config)
    beta_1 = proto_config.adamax.beta_1
    beta_2 = proto_config.adamax.beta_2
    epsilon = proto_config.adamax.epsilon
    optimizer = tf.keras.optimizers.Adamax(
        learning_rate=learning_rate_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon
    )

    return optimizer


def build_ftrl(proto_config):
    """
    Returns an instance of the Ftrl Optimizer.
    :param proto_config: A proto mesasge of type optimizers_pb2.Optimizer()
    :return: An instance of the class tf.keras.optimizers.Ftrl
    """
    learning_rate_config = proto_config.ftrl.learning_rate_schedule
    learning_rate_schedule = build_learning_rate_schedule(learning_rate_config)
    initial_accumulator_value = proto_config.ftrl.initial_accumulator_value
    l1_regularization_strength = proto_config.ftrl.l1_regularization_strength
    l2_regularization_strength = proto_config.ftrl.l2_regularization_strength
    l2_shrinkage_regularization_strength = \
        proto_config.ftrl.l2_shrinkage_regularization_strength

    optimizer = tf.keras.optimizers.Ftrl(
        learning_rate=learning_rate_schedule,
        initial_accumulator_value=initial_accumulator_value,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength
    )

    return optimizer


def build_nadam(proto_config):
    """
    Returns an instance of the Nadam Optimizer.
    :param proto_config: A proto mesasge of type optimizers_pb2.Optimizer()
    :return: An instance of the class tf.keras.optimizers.Nadam
    """
    learning_rate_config = proto_config.nadam.learning_rate_schedule
    learning_rate_schedule = build_learning_rate_schedule(learning_rate_config)
    beta_1 = proto_config.nadam.beta_1
    beta_2 = proto_config.nadam.beta_2
    epsilon = proto_config.nadam.epsilon
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=learning_rate_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon
    )

    return optimizer


def build_rmsprop(proto_config):
    """
    Returns an instance of the RMSProp Optimizer.
    :param proto_config: A proto mesasge of type optimizers_pb2.Optimizer()
    :return: an instance of the class tf.keras.optimizers.RMSprop
    """
    learning_rate_config = proto_config.rmsprop.learning_rate_schedule
    learning_rate_schedule = build_learning_rate_schedule(learning_rate_config)
    rho = proto_config.rmsprop.rho
    momentum = proto_config.rmsprop.momentum
    epsilon = proto_config.rmsprop.epsilon
    centered = proto_config.rmsprop.centered
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate_schedule,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon,
        centered=centered
    )

    return optimizer


def build_sgd(proto_config):
    """
    Returns an instance of the SGD Optimizer.
    :param proto_config: A proto mesasge of type optimizers_pb2.Optimizer()
    :return: An instance of the class tf.keras.optimizers.SGD
    """
    learning_rate_config = proto_config.sgd.learning_rate_schedule
    learning_rate_schedule = build_learning_rate_schedule(learning_rate_config)
    momentum = proto_config.sgd.momentum
    nesterov = proto_config.sgd.nesterov

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate_schedule,
        momentum=momentum,
        nesterov=nesterov
    )

    return optimizer
