from loadlibs import *
# logging.set_verbosity_error()
import functions
import modules
import argparse


CFG = {
    # train setting
    'EPOCHS':20,
    'WARM_UP_EPOCHS':1,
    'BATCH_SIZE': 128,
    'VISIBLE_TQDM' : False,
    'PARALLEL' : True,
    'MIXED_PRECISION' : True, 
    'SEED': None,
    'LEARNING_RATE':0.003,
    'CONTINUE' : False,
    'CONTINUE2' : False,
    'CONTINUE3' : False,
    
    
    # augmentation
    'CUTMIX' : True, 
    'ONES' : False,
    'ROTATION': False,
    'TRANSLATION': False, 
    'NUM_AUGS_TYPE': 0, 
    'NUM_AUGS' : 3,
    
    # image
    'IMG_HEIGHT_SIZE':64,
    'IMG_WIDTH_SIZE':192,
    'GRAY_SCALE': True,
    'INPUT_CHANNEL': 1,

    # data loading 
    'NUM_WORKERS':2,
    'TEST_SIZE':0.2,
    
    # model
    'NUM_FIDUCIAL': 20,
    
        # cnn
    'PRETRAINED' : True,
    'CNN_TYPE': None, 
    'CNN_OUTPUT': None,            

        # rnn (transformer)
    # 'SEQ_HIDDEN_SIZE': 768,
    'SEQ_HIDDEN_SIZE': 1024,
    'SEQ_TYPE' : None,
    'SEQ_ACTIVATION' : torch.nn.GELU(approximate='tanh'),
    'SEQ_NUM_LAYERS': 2,
    'SEQ_BIDIRECTIONAL': True,
    'NUM_HEADS': 4,
        
            # rnn -> conformer encoder

        # pred    
    'NUM_CLASS': 2350,

    # etc
    'DROPOUT' : 0.2,
    'DROPPATH' : 0.2,
    'DROPBLOCK' : 0.2, 
    'DEVICE' :  'cuda' if torch.cuda.is_available() else 'cpu',
    'NUM_DEVICE' : None,
}

parser = argparse.ArgumentParser()
parser.add_argument("--contrast", dest="contrast", action='store', default=None)
parser.add_argument("--con", dest = "con", action='store_true')
parser.add_argument("--con2", dest = "con2", action='store_true')
parser.add_argument("--con3", dest = "con3", action='store_true')
parser.add_argument("--ep", dest="ep", action='store', default=CFG['EPOCHS'])
parser.add_argument("--bs", dest="bs", action="store", default=CFG['BATCH_SIZE'])
parser.add_argument("--cnn", dest="cnn", action="store", default='vgg')
parser.add_argument("--seq", dest="seq", action="store", default='dec')
parser.add_argument("--rot", dest="rot", action="store_true")
parser.add_argument("--trs", dest="trans", action="store_true")
parser.add_argument("--one", dest="one", action="store_true")
parser.add_argument("--tqdm_off", dest='tqdm_off', action='store_false')
parser.add_argument("-s", "--seed", dest="seed", action="store", default=42)
args = parser.parse_args()

# continue
# epochs
CFG['EPOCHS'] = int(args.ep)

# batch size
CFG['BATCH_SIZE'] = int(args.bs)

# augmentation
CFG['ROTATION'] = args.rot
CFG['TRANSLATION'] = args.trans
CFG['ONES'] = args.one
CFG['NUM_AUGS_TYPE'] = CFG['ROTATION'] + CFG['TRANSLATION']

# sequential
if args.seq == 'lstm':
    CFG['SEQ_TYPE'] = torch.nn.LSTM
elif args.seq == 'gru':
    CFG['SEQ_TYPE'] = torch.nn.GRU
elif args.seq == 'enc':
    CFG['SEQ_TYPE'] = torch.nn.TransformerEncoder
elif args.seq == 'dec':
    CFG['SEQ_TYPE'] = torch.nn.TransformerDecoder
    
# cnn
if args.cnn == 'vgg':
    CFG['CNN_TYPE'] = modules.VggNet
    # CFG['CNN_OUTPUT'] = 2048
    CFG['CNN_OUTPUT'] = 5760
elif args.cnn == 'reg':
    CFG['CNN_TYPE'] = modules.RegNet
    CFG['CNN_OUTPUT'] = 5760
elif args.cnn == 'res':
    CFG['CNN_TYPE'] = modules.ResNet
    CFG['CNN_OUTPUT'] = 1024
elif args.cnn == 'eff':
    CFG['CNN_TYPE'] = modules.EffnetV2
    CFG['CNN_OUTPUT'] = 5120
else:
    CFG['CNN_TYPE'] = modules.VggNet
    CFG['CNN_OUTPUT'] = 4096

CFG['SEED'] = int(args.seed)
CFG['VISIBLE_TQDM'] = args.tqdm_off


DIRECTORY = {
    'TRAIN_DIR' : "/home/gyuseonglee/workspace/2301_OCR/data/train.csv",
    'TEST_DIR'  : "/home/gyuseonglee/workspace/2301_OCR/data/test.csv",
    'SUBMIT_DIR' : "/home/gyuseonglee/workspace/2301_OCR/data/sample_submission.csv",
    
    'TRAIN_IMAGE_DIR' : "/home/gyuseonglee/workspace/2301_OCR/data/train",
    'TEST_IMAGE_DIR'  : "/home/gyuseonglee/workspace/2301_OCR/data/test",
    
    'CUTMIX' : "/home/gyuseonglee/workspace/2301_OCR/data/cutmix.csv",
    'ONES' : "/home/gyuseonglee/workspace/2301_OCR/data/ones.csv",
    'TWOS' : "/home/gyuseonglee/workspace/2301_OCR/data/twos.csv",
    'THREES' : "/home/gyuseonglee/workspace/2301_OCR/data/threes.csv",
    
}


TOOLS = {
    # model
    'MODEL' : modules.RecoModel,
    
    # optimizer
    "OPTIMIZER" : torch.optim.Adam,

    # scheduler - ReduceLROnPlateau
    "SCHEDULER" : torch.optim.lr_scheduler.ReduceLROnPlateau,
    "SCHEDULER_ARGS" : { 
        'mode':'max', 
        'factor':0.75, 
        'patience':1, 
        'threshold_mode':'abs',
        'min_lr':1e-5, 
        'verbose':True        
    },    
    # scheduler - CosineAnnealingWarmRestarts
    # 주의사항 : optimzer의 learning rate를 0 혹은 0에 아주 가까운 값으로 설정
    # "SCHEDULER" : modules.CosineAnnealingWarmUpRestarts,
    # "SCHEDULER_ARGS" : { 
    #     'T_0':CFG['EPOCHS']//3,
    #     'T_up':CFG['WARM_UP_EPOCHS'],
    #     'T_mult':1,
    #     'eta_max':CFG['LEARNING_RATE'],     
    #     'gamma': 0.33,
    # },   
    
    # warm_up scheduler
    "WARM_UP" : torch.optim.lr_scheduler.LinearLR,
    # "WARM_UP" : None,
    "WARM_UP_ARGS" :{
        'start_factor':CFG["LEARNING_RATE"]/CFG['WARM_UP_EPOCHS'], 
        'end_factor':CFG["LEARNING_RATE"], 
        'total_iters':CFG["WARM_UP_EPOCHS"],
    },

    # criterion
    "CRITERION" : torch.nn.CTCLoss,
    "CRITERION_ARGS" : {
        'blank':0,
    },
    
    "CONTRAST_WEIGHTS": {
        # "vgg" : "/home/gyuseonglee/workspace/contrastive/pretrained_VggNet_128_42/state_dicts.pt", # for repvgg_a2
        "vgg" : None,
        "reg" : "/home/gyuseonglee/workspace/contrastive/pretrained_RegNet_128_42/state_dicts.pt",
        "res" : None,
        "eff": None,
    },

    "CONTINUE_WEIGHTS": None,
}

# continue learning
if args.con == True:
    CFG['CONTINUE'] = True
    CFG['LEARNING_RATE'] /= 2
    if args.cnn == 'reg':
        TOOLS['CONTINUE_WEIGHTS'] = f"/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet-TransformerDecoder_{CFG['SEED']}/model.pt"
    elif args.cnn == 'vgg':
        TOOLS['CONTINUE_WEIGHTS'] = f"/home/gyuseonglee/workspace/2301_OCR/Aug-VggNet-TransformerDecoder_{CFG['SEED']}/model.pt"
if args.con2 == True:
    CFG['CONTINUE2'] = True
    CFG['LEARNING_RATE'] /= 2
    if args.cnn == 'reg':
        TOOLS['CONTINUE_WEIGHTS'] = f"/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet-TransformerDecoder_con20_{CFG['SEED']}/model.pt"
    elif args.cnn == 'vgg':
        TOOLS['CONTINUE_WEIGHTS'] = f"/home/gyuseonglee/workspace/2301_OCR/Aug-VggNet-TransformerDecoder_con20_{CFG['SEED']}/model.pt"
if args.con3 == True:
    CFG['CONTINUE3'] = True
    CFG['LEARNING_RATE'] /= 2
    if args.cnn == 'reg':
        TOOLS['CONTINUE_WEIGHTS'] = f"/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet-TransformerDecoder_con30_{CFG['SEED']}/model.pt"
    elif args.cnn == 'vgg':
        TOOLS['CONTINUE_WEIGHTS'] = f"/home/gyuseonglee/workspace/2301_OCR/Aug-VggNet-TransformerDecoder_con30_{CFG['SEED']}/model.pt"

# INFERENCE_MODEL = modules.ExpModel2
INFERENCE_MODELS = [
    # "/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet_TransformerDecoder_950317/model.pt",
    "/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet-TransformerDecoder_con55_1203/model.pt",
    "/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet-TransformerDecoder_con55_42/model.pt",
    "/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet-TransformerDecoder_con30_1203/model.pt",
    "/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet-TransformerDecoder_con30_42/model.pt",
    "/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet-TransformerDecoder_con20_33/model.pt",
    "/home/gyuseonglee/workspace/2301_OCR/Aug-RegNet-TransformerDecoder_con20_35/model.pt",
]


model_type = f"{CFG['CNN_TYPE'].__name__}-{CFG['SEQ_TYPE'].__name__}"
if CFG['NUM_AUGS_TYPE'] > 1:
    model_type = "Aug-" + model_type

if CFG['CONTINUE'] == True:
    model_type = model_type + "_con20" 
if CFG['CONTINUE2'] == True:
    model_type = model_type + "_con30" 
if CFG['CONTINUE3'] == True:
    model_type = model_type + "_con55"

OUTPUT = {
    'PRED_NAME' : f"{model_type}_{CFG['SEED']}_submission.csv",
    'OUTPUT_DIR' : "/home/gyuseonglee/workspace/2301_OCR",
    'OUTPUT_MODEL_NAME' : f"{model_type}_{CFG['SEED']}",
}
