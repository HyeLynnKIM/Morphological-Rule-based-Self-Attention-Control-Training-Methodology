from KLUE_STS import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
ko = KoNET(firstTraining=True, use_attention_supervision=True)
ko.save_path = './sts_data/model/sts_model.ckpt'
# ko.Training(is_Continue=True, training_epoch=500)
ko.eval_nli()