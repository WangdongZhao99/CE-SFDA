def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.00005, 'ent_loss_wt': 0.8467, 'im': 0.2983,
                     'target_cls_wt': 1, 'beta': 10, 'alpha': 1},
            'CESFDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00005, 'ent_loss_wt': 0.8467, 'im': 0.2983,
                     'target_cls_wt': 1, 'aad_wt':0.01, 'beta': 10, 'alpha': 1},
            'COWA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00005, 'ent_loss_wt': 0.8467, 'im': 0.2983,
                     'target_cls_wt': 1, 'beta': 10, 'alpha': 1},
            'GKD': {'pre_learning_rate': 0.001, 'learning_rate': 0.00005, 'ent_loss_wt': 0.8467, 'im': 0.2983,
                     'target_cls_wt': 1, 'beta': 10, 'alpha': 1},
            'SCLM': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'TPDS': {'pre_learning_rate': 0.001, 'learning_rate': 0.00005, 'ent_loss_wt': 0.8467, 'im': 0.2983,
                     'target_cls_wt': 1, 'beta': 10, 'alpha': 1},
            'AaD': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'beta': 5, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
        }


class WISDM():
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }

        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514,
                     'target_cls_wt': 0.15, 'beta': 9, 'alpha': 1},
            'CESFDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514,
                     'target_cls_wt': 0.15, 'aad_wt':0.001, 'beta': 9, 'alpha': 1},
            'GKD': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514,
                     'target_cls_wt': 0.15, 'beta': 9, 'alpha': 1},
            'SCLM': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'TPDS': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'beta': 9, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},

        }

class HHAR():
    def __init__(self):
        super(HHAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }

        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.01, 'beta': 9, 'alpha': 1},#ours
            'CESFDA': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.01, 'aad_wt':0.02,'beta': 9, 'alpha': 1},#ours

            'GKD': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.01, 'beta': 9, 'alpha': 1},
            # 'SHOT': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'ent_loss_wt': 0.8467, 'im': 0.2983,
            #          'target_cls_wt': 0.01},#0.01
            'SCLM': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'TPDS': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'beta': 9, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.5514, 'TOV_wt': 0.2983},

        }   

class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'TPDS': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05, 'aad_wt':0.1, 'beta': 10, 'alpha': 1},
            'GKD': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'CESFDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.01,  'beta': 10, 'alpha': 1, 'aad_wt':0.025},

            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'beta': 10, 'alpha': 1},
            'SCLM': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }

        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514,
                     'target_cls_wt': 0.01},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'beta': 9, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},

        }
