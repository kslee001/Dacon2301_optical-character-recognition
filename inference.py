from loadlibs import *
import functions
import modules
from configs import OUTPUT, CFG, TOOLS, DIRECTORY, INFERENCE_MODELS


train, valid, test, submit = functions.prepare_data(CFG, DIRECTORY)
idx2char, char2idx = functions.prepare_vocab(train)
_, _, test_loader = functions.prepare_loader(train, valid, test, CFG)
today = datetime.datetime.strftime(datetime.datetime.today(), '%y%m%d')
functions.print_configs(CFG, train, valid, test)    

CFG['PARALLEL'] = False
pred = functions.inference(
    models=INFERENCE_MODELS,
    test_loader=test_loader, 
    idx2char=idx2char, 
    configs=CFG,
)
print(len(pred))

submit['label'] = pred
submit['label'] = submit['label'].apply(functions.correct_prediction)
submit.to_csv(f'voting_submission_{today}.csv', index=False)


