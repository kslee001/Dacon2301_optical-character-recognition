from loadlibs import *
# logging.set_verbosity_error()
import functions
import modules
from configs import CFG, DIRECTORY, TOOLS, OUTPUT, args
from torchinfo import summary


if __name__ == '__main__':
    # fix seed
    functions.seed_everything(CFG['SEED']) 
    # apply actual batch size    
    if CFG['PARALLEL'] == True:
        CFG['NUM_DEVICE'] = torch.cuda.device_count()
        CFG['BATCH_SIZE']*= CFG['NUM_DEVICE']
        if CFG['NUM_DEVICE'] <= 1:
            CFG['PARALLEL'] = False
    # prepare data and dataloaders
    train, valid, test, submit = functions.prepare_data(configs=CFG, directory=DIRECTORY)
    idx2char, char2idx = functions.prepare_vocab(train)   
    train_loader, valid_loader, test_loader = functions.prepare_loader(
        train=train, 
        valid=valid, 
        test=test,
        configs=CFG
    )    
    # print configs
    functions.print_configs(CFG, train, valid, test)    

    # train setting
    # model = TOOLS['MODEL'](CFG) 
    model = torch.load('/home/gyuseonglee/workspace/2301_OCR/Aug-VggNet-TransformerEncoder_42/model.pt').module
    model.eval()
    model = model.cuda()
    model_summary = summary(
        model=model, 
        input_size=(CFG["BATCH_SIZE"], CFG["INPUT_CHANNEL"], CFG["IMG_HEIGHT_SIZE"],CFG["IMG_WIDTH_SIZE"]),
        verbose=1 # 0 : no output / 1 : print model summary / 2 : full detail(weight, bias layers)
    ).__repr__()
    if CFG['PARALLEL'] == True:
        model = torch.nn.parallel.DataParallel(model)
    if TOOLS['SCHEDULER'].__name__ == 'CosineAnnealingWarmUpRestarts':
        print("-- current lr scheduler is [CosineAnnealingWarmUpRestarts]")
        optimizer = TOOLS['OPTIMIZER'](params=model.parameters(), lr=0)
    else:
        optimizer = TOOLS['OPTIMIZER'](params=model.parameters(), lr=CFG["LEARNING_RATE"])
    scaler    = torch.cuda.amp.GradScaler() if CFG['DEVICE']=='cuda' else torch.GradScaler()
    scheduler = TOOLS['SCHEDULER'](optimizer=optimizer, **TOOLS['SCHEDULER_ARGS'])
    if (TOOLS['SCHEDULER'].__name__ != 'CosineAnnealingWarmUpRestarts') & (TOOLS['WARM_UP'] is not None):
        warm_up = TOOLS['WARM_UP'](optimizer=optimizer, **TOOLS['WARM_UP_ARGS'])
    else:
        warm_up = None
    criterion = nn.CTCLoss(blank=0) # idx 0 : '-'

    # prediction (single)
    predictions = functions.inference_single(
        model=model, 
        test_loader=test_loader, 
        idx2char=idx2char, 
        configs=CFG
    )
    
    pd.DataFrame(predictions).to_csv("prediction_test_raw.csv")
    submit['label'] = predictions
    submit['label'] = submit['label'].apply(functions.correct_prediction)
    submit.to_csv(
        f"{OUTPUT['OUTPUT_DIR']}/{OUTPUT['OUTPUT_MODEL_NAME']}/{OUTPUT['PRED_NAME']}", 
        index=False
    )
    