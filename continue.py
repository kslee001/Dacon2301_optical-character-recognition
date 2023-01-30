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
    model = TOOLS['MODEL'](CFG) 
    if CFG['CONTINUE'] == True:
        print("-- continue learning.. current weight is ")
        print(f"{TOOLS['CONTINUE_WEIGHTS']}")
        print(f"-- will be saved as : {OUTPUT['OUTPUT_MODEL_NAME']}\n")
        model = torch.load(TOOLS['CONTINUE_WEIGHTS'], map_location=CFG['DEVICE']).module
    elif TOOLS['CONTRAST_WEIGHTS'][args.cnn] is not None:
        model.cnn.load_state_dict(torch.load(TOOLS['CONTRAST_WEIGHTS'][args.cnn]))
        print("-- contrastive learning weights loaded")
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
    
    # train
    infer_model = functions.train_fn(
        model=model, 
        optimizer=optimizer, 
        scaler=scaler,
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        scheduler=scheduler, 
        warm_up=warm_up,
        criterion=criterion, 
        char2idx=char2idx, 
        idx2char=idx2char,
        configs=CFG
    )
    
    # save model & config information
    functions.save_model(
        best_model=infer_model, 
        output_folder=OUTPUT["OUTPUT_DIR"], 
        model_name=OUTPUT["OUTPUT_MODEL_NAME"]
    )
    functions.save_configs(
        configs=CFG, 
        directory_configs=DIRECTORY, 
        tool_configs=TOOLS, 
        output_configs=OUTPUT
    )
    functions.save_model_info(
        model=infer_model.__repr__(), 
        summary=model_summary, 
        output_configs=OUTPUT
    )

    # prediction (single)
    predictions = functions.inference_single(
        model=infer_model, 
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
    