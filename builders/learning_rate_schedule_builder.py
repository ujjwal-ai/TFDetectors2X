import tensorflow as tf
from protos import learning_rates_pb2


def build_learning_rate_schedule(proto_config):
    """
    Builds a learning rate schedule from a proto configuration message.
    :param proto_config: A message of type
    learning_rates_pb2.LearningRateDecay().
    :return: A learning rate schedule. For constant rate, it is a scalar of
    type float. For other schedules it is an instance which is a subclass of
    tf.keras.optimizers.schedules.LearningRateSchedule.
    """
    if not isinstance(proto_config, learning_rates_pb2.LearningRateDecay):
        raise ValueError('The proto_config argument to '
                         'build_learning_rate_schedule() must be of the same '
                         'type as the message LearningRateDecay defined in '
                         'protos/learning_rates.proto.')

    decay_type = proto_config.WhichOneof('LearningRateSchedule')
    if decay_type == 'exponential_decay':
        return build_exponential_decay(proto_config)
    elif decay_type == 'inversetime_decay':
        return build_inversetime_decay(proto_config)
    elif decay_type == 'piecewiseconstant_decay':
        return build_piecewise_decay(proto_config)
    elif decay_type == 'polynomial_decay':
        return build_polynomial_recay(proto_config)
    else:
        return build_constant_learning_rate(proto_config)


def build_exponential_decay(proto_config):
    """
    Builds an exponential learning rate decay schedule
    :param proto_config:A message of type
    learning_rates_pb2.LearningRateDecay().
    :return: An instance of tf.keras.optimizers.schedules.ExponentialDecay
    """
    initial_learning_rate = proto_config.exponential_decay.initial_learning_rate
    if initial_learning_rate < 0:
        raise ValueError('The initial learning rate must be positive.')
    decay_steps = proto_config.exponential_decay.decay_steps
    if decay_steps < 0:
        raise ValueError('decay_steps must be positive.')
    decay_rate = proto_config.exponential_decay.decay_rate
    staircase = proto_config.exponential_decay.staircase

    learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )

    return learning_schedule


def build_inversetime_decay(proto_config):
    """
    Builds an inverse time learning rate decay schedule.
    :param proto_config: A message of type
    learning_rates_pb2.LearningRateDecay().
    :return:
    """
    initial_learning_rate = proto_config.inversetime_decay.initial_learning_rate
    if initial_learning_rate < 0:
        raise ValueError('The initial learning rate must be positive.')
    decay_steps = proto_config.inversetime_decay.decay_steps
    if decay_steps < 0:
        raise ValueError('decay_steps must be positive.')
    decay_rate = proto_config.inversetime_decay.decay_rate
    staircase = proto_config.inversetime_decay.staircase

    learning_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )

    return learning_schedule


def build_piecewise_decay(proto_config):
    """
    Builds  a piecewise constant learning rate decay schedule.
    :param proto_config: A message of type
    learning_rates_pb2.LearningRateDecay().
    :return: An instance of tf.keras.optimizers.schedules.PiecewiseConstantDecay
    """
    boundaries = proto_config.piecewiseconstant_decay.boundaries
    values = proto_config.piecewiseconstant_decay.values
    if len(values) - len(boundaries) != 1:
        raise ValueError('For piecewise constant decay there should be one '
                         'more number of values than boundaries.')

    learning_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values
    )

    return learning_schedule


def build_polynomial_recay(proto_config):
    """
    Builds a polynomial learning rate decay schedule.
    :param proto_config: A message of type
    learning_rates_pb2.LearningRateDecay().
    :return: An instance of f.keras.optimizers.schedules.PolynomialDecay
    """
    initial_learning_rate = proto_config.polynomial_decay.initial_learning_rate
    if initial_learning_rate < 0:
        raise ValueError('The initial learning rate must be positive.')
    decay_steps = proto_config.polynomial_decay.decay_steps
    if decay_steps < 0:
        raise ValueError('decay_steps must be positive.')
    end_learning_rate = proto_config.polynomial_decay.end_learning_rate
    if end_learning_rate < 0:
        raise ValueError('The end learning rate must be positive.')
    power = proto_config.polynomial_decay.power
    cycle = proto_config.polynomial_decay.cycle

    learning_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        end_learning_rate=end_learning_rate,
        decay_steps=decay_steps,
        power=power,
        cycle=cycle
    )

    return learning_schedule


def build_constant_learning_rate(proto_config):
    """
    Builds a constant learning rate value.
    :param proto_config: A message of type
    learning_rates_pb2.LearningRateDecay().
    :return: A scalar of type float.
    """
    learning_rate = proto_config.constant_rate.learning_rate
    return learning_rate
