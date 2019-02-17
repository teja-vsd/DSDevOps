import os


network_parameters_1_0 = {'dataset_path'            :           os.path.join('..', '..', 'data', 'processed_data'),
                          'random_seed'             :           123456,
                          'epochs'                  :           50,
                          'batch_size'              :           5,
                          'hidden1'                 :           50,
                          'hidden2'                 :           20,
                          'drp_rate'                :           0.5,
                          'weight_decay'            :           0.001,
                          'lr'                      :           0.001
                          }

config_parameters_1_0 = {'save_model'               :           True,
                         'model_save_path'          :           os.path.join('..', '..', 'models')
                         }
