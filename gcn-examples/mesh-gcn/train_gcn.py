
import yaml
import os
import platform
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from core.utils import Logger, make_gif

from gcn_network import Model
from core.mesh_data import DataModule

'''
Build, train, and test a model.
'''

def main(experiment, trainer_args, model_args, data_args, misc_args):

    #callbacks
    callbacks=[]
    if trainer_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint(monitor="val_err",
                                            save_last=True,
                                            save_top_k=1,
                                            mode='min',
                                            filename='{epoch}'))
    if misc_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor="val_err",
                                        patience=5,
                                        strict=False))

    #logger
    if trainer_args['logger']:
        #save the configuration details
        exp_dir, exp_name = os.path.split(experiment)
        exp_name = os.path.splitext(exp_name)[0]

        logger = Logger(save_dir=os.path.join(trainer_args['default_root_dir'], exp_dir),
                        name=exp_name, default_hp_metric=False)

        config = {'train':trainer_args, 'model':model_args, 'data':data_args, 'misc':misc_args}
        logger.log_config(config)

        #add logger to trainer args
        trainer_args['logger'] = logger

    #setup datamodule
    datamodule = DataModule(**data_args)

    #build model
    model = Model(**model_args, data_info = datamodule.get_data_info())

    #build trainer
    trainer = Trainer(**trainer_args, callbacks=callbacks)

    #train model
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)

    #compute testing statistics
    if misc_args['compute_stats']:
        trainer.test(model=None if trainer_args['enable_checkpointing'] else model,
                        ckpt_path='best' if trainer_args['enable_checkpointing'] else None,
                        datamodule=datamodule)

    #make GIF
    if misc_args['make_gif']:
        make_gif(trainer, datamodule, None if trainer_args['enable_checkpointing'] else model)

    return



if __name__ == "__main__":

    #OS specific setup
    if platform.system() == 'Windows':
        os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

    if platform.system() == 'Darwin':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    #load YAML config
    experiment = Path('./gcn-examples/mesh-gcn/gcn_pool.yaml')


    #open YAML file
    with experiment.open() as file:
        config = yaml.safe_load(file)

    #extract args
    trainer_args = config['train']
    model_args = config['model']
    data_args = config['data']
    misc_args = config['misc']

    #run main script
    main(experiment, trainer_args, model_args, data_args, misc_args)
