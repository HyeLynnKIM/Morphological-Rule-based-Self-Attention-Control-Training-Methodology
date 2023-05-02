from KLUE_PN import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
ko = KoNET(firstTraining=True, use_attention_supervision=False)
ko.save_path = './pn_data/model1.ckpt'
# ko.Training(is_Continue=True, training_epoch=100)
ko.eval_nli_with_file()