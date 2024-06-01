import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import utils
import tensorflow as tf
from GSGM_uniform import GSGM
from deepsets_cond import DeepSetsAttClass
import time
import gc
import sys
from sklearn.metrics import roc_curve, auc

def evaluate(model,particles,jets,mask,idx=0,nsplit=200):
    part_split = np.array_split(particles,nsplit)
    jet_split = np.array_split(jets,nsplit)
    mask_split = np.array_split(mask,nsplit)

    start = time.time()
    print("Split size: {}".format(jet_split[idx].shape[0]))
    likelihoods_part,likelihoods_jet = model.get_likelihood(
        part_split[idx],jet_split[idx],mask_split[idx])
    Ns = np.sum(mask_split[idx],(1,2))
    
    end = time.time()
    print("Time for sampling {} events is {} seconds".format(particles.shape[0],end - start))
    
    return {'ll_part':likelihoods_part,'ll_jet': likelihoods_jet,'N': Ns}

def evaluate_classifier(num_feat,checkpoint_folder,data_path):
    #load the model
    from tensorflow import keras
    inputs, outputs = DeepSetsAttClass(
        num_feat,
        num_heads=2,
        num_transformer = 6,
        projection_dim = 128,
    )
    model = keras.Model(inputs=inputs,outputs=outputs)
    model.load_weights('{}'.format(checkpoint_folder)).expect_partial()
    
    #load the data
    data_bkg, _, _ = utils.DataLoader(data_path,
                                      ['gluon_tagging.h5'],
                                      'gluon_tagging',
                                      use_train=False,
                                      make_tf_data = False,
    )

    data_sig,_,_ = utils.DataLoader(data_path,
                                    ['top_tagging.h5'],
                                    'gluon_tagging',
                                    use_train=False,
                                    make_tf_data = False,
    )

    labels = np.concatenate([np.zeros(data_bkg.shape[0]),np.ones(data_sig.shape[0])],0)
    pred = model.predict(np.concatenate([data_bkg,data_sig],0))
    return labels,pred

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    utils.SetStyle()


    parser = argparse.ArgumentParser()

    #parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/TOP', help='Folder containing data and MC files')
    parser.add_argument('--data_folder', default='/pscratch/sd/n/nishank/data/TOPTAGGING/', help='Folder containing data and MC files')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--config', default='config_AD.json', help='Training parameters')

    parser.add_argument('--sample', action='store_true', default=False,help='Sample from the generative model')
    parser.add_argument('--nidx', default=0, type=int,help='Parallel sampling of the data')

    parser.add_argument('--sup', action='store_true', default=False,help='Plot only the ROC for classifier and density ratio')
    parser.add_argument('--ll', action='store_true', default=False,help='Load Max LL training model')
    parser.add_argument('--npart', default=100,type=int, help='Which particle is the anomaly')


    flags = parser.parse_args()
    npart = flags.npart
    config = utils.LoadJson(flags.config)
    model_name = config['MODEL_NAME']
    if flags.ll:
        model_name+='_ll'

    #processes = ['gluon_tagging','top_tagging']
    processes = ['gluon_tagging','top_tagging','HV']
    
    if flags.sample:    
        nll_qcd = {}
        model_gluon = GSGM(config=config,npart=npart,particle='gluon_tagging',ll_training=flags.ll)
        checkpoint_folder_gluon = '../checkpoints_{}/checkpoint'.format(model_name+ '_gluon_tagging')
        model_gluon.load_weights('{}'.format(checkpoint_folder_gluon)).expect_partial()




        model_top = GSGM(config=config,npart=npart,particle='top_tagging',ll_training=flags.ll)
        checkpoint_folder_top = '../checkpoints_{}/checkpoint'.format(model_name+ '_top_tagging')
        model_top.load_weights('{}'.format(checkpoint_folder_top)).expect_partial()
        nll_top = {}


    labels_top,likelihoods_top = evaluate_classifier(
                    config['NUM_FEAT'],
                    '../checkpoints_{}/checkpoint'.format(model_name+ '_class_gluon_tagging'),
                    flags.data_folder,
                )
    fpr, tpr, _ = roc_curve(labels_top,likelihoods_top, pos_label=1)
    print("Classifier AUC: {}".format(auc(fpr, tpr)))
    plt.plot(tpr,fpr,label="Supervised Classifier",
                         color='gray',
                         linestyle=utils.line_style['top_tagging_ll'])
    plt.yscale('log')
    plt.legend(frameon=False,fontsize=14)
    plt.savefig('roc_curve')

