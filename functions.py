from loadlibs import *
import modules

# Utils ---------------------------------------------
def seed_everything(seed):
    """
    ----------------------------------------------------------------------
    desc :
        seed를 parameter로 받은 뒤, 
        random seed가 필요한 모든 곳에 seed를 뿌려주는 function.
        main.py의 가장 윗부분에서 실행됨
    ----------------------------------------------------------------------
    return : 
        None
    ----------------------------------------------------------------------
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def make_dir(directory:str):
    """
    ----------------------------------------------------------------------
    desc :
        directory가 존재하지 않을 경우 새로 만들어 주는 함수
    ----------------------------------------------------------------------
    return : 
        None
    ----------------------------------------------------------------------
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def save_model(best_model, output_folder,model_name):
    """
    ----------------------------------------------------------------------
    desc :
        best_model(pytorch module),
        output_folder(string),
        model_name(string) 를 받은 뒤, 
        [output_folder] 경로에 [model_name.pt]으로 
        1. 모델 전체
        2. 모델 파라미터(state_dict)를 저장함
    ----------------------------------------------------------------------
    return : 
        None
    ----------------------------------------------------------------------
    """
    model_path =  output_folder + "/" + model_name + "/"
    make_dir(output_folder)
    make_dir(model_path)
    
    model_dir = model_path + "model.pt"
    model_state_dir = model_path + "state_dicts.pt"

    torch.save(
        best_model, 
        model_dir
    )
    torch.save(
        best_model.state_dict(),
        model_state_dir
    )
    

def save_configs(configs, directory_configs, tool_configs, output_configs):
    # save configs
    with open(f"{output_configs['OUTPUT_DIR']}/{output_configs['OUTPUT_MODEL_NAME']}/configs.txt", "w") as f:        
        # CFG
        f.write("CFG = {\n")
        for name, val in configs.items():
            if((type(val)==int) | (type(val)==float)):
                cur = "    '"+str(name)+"'"+ " : " + str(val) + ",\n"
            else:
                cur = "    '"+str(name)+"'"+ " : " + "'" + str(val) + "'" + ",\n"                
            f.write(cur)
        f.write("}\n\n")

        # directory
        f.write("DIRECTORY = {\n")
        for name, val in directory_configs.items():
            if((type(val)==int) | (type(val)==float)):
                cur = "    '"+str(name)+"'"+ " : " + str(val) + ",\n"
            else:
                cur = "    '"+str(name)+"'"+ " : " + "'" + str(val) + "'" + ",\n"                
            f.write(cur)
        f.write("}\n\n")        
     
        # TOOL
        f.write("TOOLS = {\n")
        for name, val in tool_configs.items():
            if((type(val)==int) | (type(val)==float)):
                cur = "    '"+str(name)+"'"+ " : " + str(val) + ",\n"
            else:
                cur = "    '"+str(name)+"'"+ " : " + "'" + str(val) + "'" + ",\n"                
            f.write(cur)
        f.write("}\n\n")

        # output
        f.write("OUTPUT = {\n")
        for name, val in tool_configs.items():
            if((type(val)==int) | (type(val)==float)):
                cur = "    '"+str(name)+"'"+ " : " + str(val) + ",\n"
            else:
                cur = "    '"+str(name)+"'"+ " : " + "'" + str(val) + "'" + ",\n"                
            f.write(cur)
        f.write("}\n\n")


def save_model_info(model, summary, output_configs):
    with open(f"{output_configs['OUTPUT_DIR']}/{output_configs['OUTPUT_MODEL_NAME']}/model_info.txt", "w") as f:        
        f.write("# ------------------------------------------- summary --------------------------------------------- \n")
        f.write(summary)
        f.write("\n")
        f.write("# ------------------------------------------------------------------------------------------------- \n")
        
        f.write("\n\n")
        
        f.write("# ------------------------------------------- full info ------------------------------------------- \n")
        f.write(model)
        f.write("\n")
        f.write("# ------------------------------------------------------------------------------------------------- \n")
        

# ------------------------------------------------------
def prepare_data(configs, directory):
    """
    ----------------------------------------------------------------------
    desc :
        데이터 (csv 파일) 불러오는 function.
        단순히 configs.py 에 작성된 DIRECTORY 내용을 통해 pd.DataFrame을 불러옴.
    ----------------------------------------------------------------------
    return : 
        train, valid, test, submit (pd.DataFrame)  
    ----------------------------------------------------------------------
    """
    data = pd.read_csv(directory["TRAIN_DIR"])
    data['img_path'] = data['img_path'].str.replace("./train", directory['TRAIN_IMAGE_DIR'])
    if configs['ONES'] ==True:
        ones = pd.read_csv(directory["ONES"])
        twos = pd.read_csv(directory["TWOS"])
        threes = pd.read_csv(directory["THREES"])
        data = pd.concat([data, ones, twos, threes], 0)

    # 1 character
    data['len'] = data['label'].str.len()
    train1 = data[data['len']==1]

    # GE 2 characters
    data = data[data['len']>1]
    if configs['CUTMIX'] == True:
        cutmix = pd.read_csv(directory['CUTMIX'])
        data = pd.concat([data, cutmix], 0)
    train2, valid, _, _ = train_test_split(data, data['len'], test_size=configs['TEST_SIZE'], random_state=configs['SEED'])
    train = pd.concat([train1, train2])
    test = pd.read_csv(directory["TEST_DIR"])
    test['img_path'] = test['img_path'].str.replace("./test", directory['TEST_IMAGE_DIR'])
    submit = pd.read_csv(directory["SUBMIT_DIR"])
    return train, valid, test, submit

def prepare_vocab(train):
    """
    ----------------------------------------------------------------------
    desc :
        train(pd.DataFrame) 을 받아, 
        1. idx2char(dictionary, idx를 넣으면 char 반환)
        2. char2idx(dictionary, char를 넣으면 idx 반환)를 return하는 함수
    ----------------------------------------------------------------------
    return : 
        idx2char(dictionary), char2idx(dictionary)
    ----------------------------------------------------------------------
    """
    # vocab
    train_vocab = [token for token in train['label']]
    train_vocab = "".join(train_vocab)
    letters = sorted(list(set(list(train_vocab))))
    vocabulary = ["-"] + letters
    idx2char = {k:v for k,v in enumerate(vocabulary, start=0)}
    char2idx = {v:k for k,v in idx2char.items()}
    
    return idx2char, char2idx

def prepare_loader(train, valid, test, configs):
    """
    ----------------------------------------------------------------------
    desc :
        1. train dataloader
        2. valid dataloader
        3. test dataloader 를 반환하는 함수
    ----------------------------------------------------------------------
    return : 
        train_loader, valid_loader, test_loader
    ----------------------------------------------------------------------
    """
    train_dataset=modules.BaseDataset(
        configs=configs,
        img_path_list=train['img_path'].values, 
        label_list=train['label'].values, 
        len_list=train['len'].values,
        mode='train',
    )
    valid_dataset = modules.BaseDataset(
            configs=configs,
            img_path_list=valid['img_path'].values, 
            label_list=valid['label'].values, 
            len_list=valid['len'].values,
            mode='valid',
    )
    train_aug_datasets = []
    valid_aug_datasets = []
    
    if configs['ROTATION'] == True:
        options = random.sample([0,1,2,3], k=configs['NUM_AUGS'])        
        for idx in range(configs['NUM_AUGS']):
            train_aug_datasets.append(
                modules.AugmentDataset(
                    configs=configs,
                    img_path_list=train['img_path'].values, 
                    label_list=train['label'].values, 
                    len_list=train['len'].values,
                    mode='rot',
                    aug_number=options[idx],
                )
            )
            valid_aug_datasets.append(
                modules.AugmentDataset(
                    configs=configs,
                    img_path_list=valid['img_path'].values, 
                    label_list=valid['label'].values, 
                    len_list=valid['len'].values,
                    mode='rot',
                    aug_number=options[idx],
                )
            )
        
    if configs['TRANSLATION'] == True:
        options = random.sample([0,1,2,3], k=configs['NUM_AUGS'])
        for idx in range(configs['NUM_AUGS']):
            train_aug_datasets.append(
                modules.AugmentDataset(
                    configs=configs,
                    img_path_list=train['img_path'].values, 
                    label_list=train['label'].values, 
                    len_list=train['len'].values,
                    mode='trs',
                    aug_number=options[idx],
                )
            )
            valid_aug_datasets.append(
                modules.AugmentDataset(
                    configs=configs,
                    img_path_list=valid['img_path'].values, 
                    label_list=valid['label'].values, 
                    len_list=valid['len'].values,
                    mode='trs',
                    aug_number=options[idx],
                )
            )
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, *train_aug_datasets])
    valid_dataset = torch.utils.data.ConcatDataset([valid_dataset, *valid_aug_datasets])
        
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size = configs['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=configs['NUM_WORKERS'],
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size = configs['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=configs['NUM_WORKERS'],
    )
    test_loader = DataLoader(
        dataset=modules.BaseDataset(
            configs=configs,
            img_path_list=test['img_path'].values, 
            label_list=None, 
            len_list=None,
            mode='test',
            ), 
        batch_size = configs['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=configs['NUM_WORKERS'],
    )
    return train_loader, valid_loader, test_loader

    
    
# train --------------------------------------------
def train_fn(model, optimizer, scaler, 
             train_loader, valid_loader, 
             scheduler, warm_up, criterion, char2idx, idx2char, configs):
    
    aux_criterion = torch.nn.MSELoss().to(configs['DEVICE'])
    def encode_text_batch(text_batch):
        """
        ----------------------------------------------------------------------
        desc :
            text_batch(=label)을 받아서 이를 idx로 반환하는 함수
            해당 label의 길이(length)에 관한 정보도 필요하므로, 이를 함께 반환함
        ----------------------------------------------------------------------
        return : 
            text_batch_targets, text_batch_targets_lens
        ----------------------------------------------------------------------
        """
        text_batch_targets_lens = [len(text) for text in text_batch]
        text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)
        
        text_batch_concat = "".join(text_batch)
        text_batch_targets = [char2idx[c] for c in text_batch_concat]
        text_batch_targets = torch.IntTensor(text_batch_targets)
        
        return text_batch_targets, text_batch_targets_lens
    
    def decode_predictions(yhat):
        text_batch_tokens = F.softmax(yhat, 2).argmax(2) # [T, batch_size]
        text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]
        
        text_batch_tokens_new = []
        for text_tokens in text_batch_tokens:
            text = [idx2char[idx] for idx in text_tokens]
            text = "".join(text)
            text_batch_tokens_new.append(text)
        return text_batch_tokens_new

    def compute_loss(text_batch, text_batch_logits):
        """
        text_batch: list of strings of length equal to batch size
        text_batch_logits: Tensor of size([T, batch_size, num_classes])
        """
        text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]  
        text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),), 
                                        fill_value=text_batch_logps.size(0), 
                                        dtype=torch.int32).to(configs['DEVICE']) # [batch_size] 

        text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch)
        loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

        return loss
    
    def validation(model, valid_loader):
        model.eval()
        preds = []
        labels = []
        val_loss = []
        acc = 0
        with torch.no_grad():
            valid_iterator = tq(valid_loader) if configs['VISIBLE_TQDM'] else valid_loader
            for img, label, length in valid_iterator:
                img = img.to(configs['DEVICE'])
                length = length.to(configs['DEVICE']).float()
                yhat, len_pred = model(img)
                
                # loss
                if configs['PARALLEL'] == True:
                    yhat = yhat.permute(1,0,2)
                loss = compute_loss(label, yhat)
                aux_loss = aux_criterion(length.float(), len_pred)

                loss = loss + aux_loss*0.15
                # acc
                pred = pd.Series(decode_predictions(yhat.cpu())).apply(correct_prediction).tolist()
                preds.extend(pred)
                labels.extend(label)
                val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)
        acc = [labels[idx] == preds[idx] for idx in range(len(labels))]
        acc = sum(acc)/len(acc)
        
        return _val_loss, acc
    
    def train_step(cur_img, label, length, train_loss):
        cur_img = cur_img.to(configs['DEVICE'])
        length = length.to(configs['DEVICE']).float()
        optimizer.zero_grad()
        
        # prediction
        yhat, len_pred = model(cur_img)
        if configs['PARALLEL'] == True:
            yhat = yhat.permute(1,0,2)
        
        # compute loss
        if configs['MIXED_PRECISION'] == True:
            with torch.autocast(device_type=configs['DEVICE'], dtype=torch.float16):
                loss = compute_loss(label, yhat)
        else:
            loss = compute_loss(label, yhat)
            
        aux_loss = aux_criterion(length, len_pred)
        loss = loss + aux_loss*0.15
        # scaled loss backward (optimizer step)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        # optimizer.step()
        train_loss.append(loss.item())
        return 
    
    model.to(configs['DEVICE'])

    best_acc = 0
    best_loss = 999999
    best_model = None
    for epoch in range(1, configs['EPOCHS']+1):
        model.train()
        train_loss = []

        """train"""
        train_iterator = tq(train_loader) if configs['VISIBLE_TQDM'] else train_loader
        for img, label, length in train_iterator:
            train_step(img, label, length, train_loss)
        _train_loss = np.mean(train_loss)
        
        
        """validation"""
        _val_loss, acc = validation(model, valid_loader)
        print(f'Epoch : [{epoch}] Train CTC Loss : [{_train_loss:.5f}] | Val CTC Loss : [{_val_loss:.5f}] | Val Accuracy : [{acc:.5f}]')

        if (warm_up is not None) & (epoch < configs['WARM_UP_EPOCHS']):
            warm_up.step()
        elif scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(acc)
        else:
            scheduler.step()
            
        # if best_loss > _val_loss:
        #     best_loss = _val_loss
        #     best_model = model
        if acc > best_acc:
            best_acc = acc
            best_model = model    
            
    return best_model



def inference(models, test_loader, idx2char, configs):
    print(f"-- number of models : {len(models)}")
    def decode_predictions(text_batch_logits):
        text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
        text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]
        text_batch_tokens_new = []
        for text_tokens in text_batch_tokens:
            text = [idx2char[idx] for idx in text_tokens]
            text = "".join(text)
            text_batch_tokens_new.append(text)
        return text_batch_tokens_new
    
    for idx in range(len(models)):
        model = torch.load(models[idx], map_location=configs['DEVICE']).module
        model = model.to(configs['DEVICE'])
        if configs['PARALLEL'] == True:
            model = torch.nn.parallel.DataParallel(model)
        model.eval()
        cur_probs = []
        probs_sum = torch.zeros(12, 74121, 2350)
        with torch.no_grad():
            if configs['VISIBLE_TQDM']:
                test_iterator = tq(test_loader)
            else:
                test_iterator = test_loader
            
            for batch in test_iterator:
                batch = batch.to(configs['DEVICE'])        
                yhat, _ = model(batch)
                if configs['PARALLEL'] == True:
                    yhat = yhat.permute(1,0,2)
                yhat = yhat.cpu().detach()
                cur_probs.append(yhat)
        cur_probs = torch.vstack(cur_probs).permute(1,0,2)
        probs_sum.add_(cur_probs)
        del cur_probs
    probs_sum/=len(models)    
    pred = decode_predictions(probs_sum.cpu()) 
       
    return pred


def inference_single(model, test_loader, idx2char, configs):
    def decode_predictions(yhat):
        text_batch_tokens = F.softmax(yhat, 2).argmax(2) # [T, batch_size]
        # print(f"test_batch_tokens.shape (should be [T, batch_size]) : {text_batch_tokens.shape}")
        text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]
        # print(f"test_batch_tokens.shape (should be [batch_size, T]) : {text_batch_tokens.shape}")
        
        text_batch_tokens_new = []
        for text_tokens in text_batch_tokens:
            text = [idx2char[idx] for idx in text_tokens]
            text = "".join(text)
            text_batch_tokens_new.append(text)
        return text_batch_tokens_new
    model.eval()
    preds = []
    with torch.no_grad():
        test_iterator = tq(test_loader) if configs['VISIBLE_TQDM'] else test_loader
        for image_batch in test_iterator:
            image_batch = image_batch.to(configs['DEVICE'])
            yhat, _ = model(image_batch)
            if configs['PARALLEL'] == True:
                yhat = yhat.permute(1,0,2)
            text_batch_pred = decode_predictions(yhat.cpu())
            
            preds.extend(text_batch_pred)
    return preds


def correct_prediction(word):
    def remove_duplicates(text):
        if len(text) > 1:
            letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
        elif len(text) == 1:
            letters = [text[0]]
        else:
            return ""
        return "".join(letters)
        
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word


def print_configs(configs, train, valid, test):
    print(f"SEED        : {configs['SEED']}")
    print(f"LR          : {configs['LEARNING_RATE']}")
    print(f"NUM_DEVICE  : {configs['NUM_DEVICE']}")
    print(f"BATCH_SIZE  : {configs['BATCH_SIZE']}")
    print(f"NUM_AUGS    : {configs['NUM_AUGS']}")
    print(f"ROTATION    : {configs['ROTATION']}")
    print(f"TRANSLATION : {configs['TRANSLATION']}")
    print(f"ONES        : {configs['ONES']}")
    print(f"CNN         : {configs['CNN_TYPE'].__name__}")
    print(f"SEQUEINTIAL : {configs['SEQ_TYPE'].__name__}")
    print(f"EPOCH       : {configs['EPOCHS']}")
    
    if configs['NUM_AUGS_TYPE'] >0:        
        print(f"len train   : {len(train) + len(train)*(configs['NUM_AUGS'])*(configs['NUM_AUGS_TYPE'])}")
        print(f"len valid   : {len(valid) + len(valid)*(configs['NUM_AUGS'])*(configs['NUM_AUGS_TYPE'])}")
    else:
        print(f"len train   : {len(train)}")
        print(f"len valid   : {len(valid)}")
    print(f"len test    : {len(test)}")
    