from KLUE_NLI import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
ko = KoNET(firstTraining=True, use_attention_supervision=True)
ko.save_path = './nli_data/model/nli_model_7ep.ckpt'
# ko.Training(is_Continue=True, training_epoch=500)
ko.eval_nli()