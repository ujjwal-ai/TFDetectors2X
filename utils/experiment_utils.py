from loguru import logger
import os
from datetime import datetime

class HouseKeeping(object):
    def __init__(self,
                 experiment_dir,
                 experiment_name,
                 include_datetime=True
                 ):
        self._experiment_dir = experiment_dir
        if os.path.exists(experiment_dir):
            logger.debug('The experiment folder {} already exists.'.format(
                experiment_dir))
        else:
            try:
                os.makedirs(experiment_dir)
            except OSError:
                logger.error('The experiment folder {} cannot be '
                             'created.'.format(experiment_dir))
                raise OSError('The experiment folder {} cannot be '
                              'created.'.format(experiment_dir))

            logger.info('The experiment folder {} was successfully '
                        'created.'.format(experiment_dir))
        self._datatime = datetime
        if include_datetime:
            now = datetime.now().format('%m-%d-%Y-%H:%M:%S')
            experiment_name = '{}_{}'.format(experiment_name, now)
            self._datetime = now

        self._experiment_name = experiment_name

        experiment_folder = os.path.join(
            experiment_dir,
            experiment_name
        )
        self._experiment_folder = experiment_folder
        if os.path.exists(experiment_folder):
            logger.error('The experiment folder {} was already found. '
                         'Experiment folders cannot be overwritte.'.format(
                experiment_folder))
            raise OSError('The experiment folder {} was already found. '
                         'Experiment folders cannot be overwritte.'.format(
                experiment_folder))

    @property
    def experiment_dir(self):
        return self._experiment_dir

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def experiment_moment(self):
        return self._datetime

    @property
    def experiment_folder(self):
        return self._experiment_folder

