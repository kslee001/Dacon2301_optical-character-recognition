import torch
import torch.nn as nn
import torch.nn.functional as F
from loadlibs import *

        
# -------------------------------------------------------------------------------
class RecoModel(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_type = configs['SEQ_TYPE'].__name__
        self.activate = torch.nn.LeakyReLU(inplace=True)
        seed_everything(configs['SEED'])

        self.tps = TPS_SpatialTransformerNetwork(configs=self.configs)
        self.cnn = configs['CNN_TYPE'](self.configs)
        self.cnn_linear = torch.nn.Linear(
            in_features=configs['CNN_OUTPUT'], 
            out_features=configs['SEQ_HIDDEN_SIZE'])
        self.pos_encoder = generate_PE
        
        if self.seq_type in ['LSTM', 'GRU']:
            self.rnn = configs['SEQ_TYPE'](
                num_layers=configs['SEQ_NUM_LAYERS'],
                input_size=configs['SEQ_HIDDEN_SIZE'],
                hidden_size=configs['SEQ_HIDDEN_SIZE'],
                bidirectional=configs['SEQ_BIDIRECTIONAL'],
                batch_first=True,
            )
            if (self.configs['SEQ_BIDIRECTIONAL'] == True):
                self.rnn_linear = torch.nn.Linear(
                    in_features=configs['SEQ_HIDDEN_SIZE']*2, 
                    out_features=configs['NUM_CLASS']) # num char
            else:
                self.rnn_linear = torch.nn.Linear(
                    in_features=configs['SEQ_HIDDEN_SIZE'], 
                    out_features=configs['NUM_CLASS']) # num char
                
        elif self.seq_type == 'TransformerEncoder':
            self.encoder = configs['SEQ_TYPE'](
                encoder_layer=torch.nn.TransformerEncoderLayer(
                    d_model=configs['SEQ_HIDDEN_SIZE'],
                    nhead=configs['NUM_HEADS'],
                    dim_feedforward=configs['SEQ_HIDDEN_SIZE']//2,
                    dropout=configs['DROPOUT'],
                    activation=configs['SEQ_ACTIVATION'],
                    batch_first=True,
                ),
                num_layers=configs['SEQ_NUM_LAYERS']
            )
            self.trf_linear = torch.nn.Linear(
                in_features=configs['SEQ_HIDDEN_SIZE'], 
                out_features=configs['NUM_CLASS']) # num char
            
        elif self.seq_type == 'TransformerDecoder':
            self.encoder = torch.nn.TransformerEncoder(
                encoder_layer=torch.nn.TransformerEncoderLayer(
                    d_model=configs['SEQ_HIDDEN_SIZE'],
                    nhead=configs['NUM_HEADS'],
                    dim_feedforward=configs['SEQ_HIDDEN_SIZE']//2,
                    dropout=configs['DROPOUT'],
                    activation=configs['SEQ_ACTIVATION'],
                    batch_first=True,
                ),
                num_layers=configs['SEQ_NUM_LAYERS']
            )
            self.decoder = configs['SEQ_TYPE'](
                decoder_layer=torch.nn.TransformerDecoderLayer(
                    d_model=configs['SEQ_HIDDEN_SIZE'],
                    nhead=configs['NUM_HEADS'],
                    dim_feedforward=configs['SEQ_HIDDEN_SIZE']//2,
                    dropout=configs['DROPOUT'],
                    activation=configs['SEQ_ACTIVATION'],
                    batch_first=True,
                ),
                num_layers=configs['SEQ_NUM_LAYERS']
            )
            self.trf_linear = torch.nn.Linear(
                in_features=configs['SEQ_HIDDEN_SIZE'], 
                out_features=configs['NUM_CLASS']) # num char

        # auxiliary loss
        self.len_cfr = torch.nn.Linear(self.configs['SEQ_HIDDEN_SIZE']*12, 1)

        
    def forward(self, x):
        # TPS - Spatial Transformation
        x = self.tps(x)

        # CNN 
        x = self.cnn(x) # visual feature
        x = x.permute(0,3,1,2)
        B, T = x.shape[:2]
        x = x.view(B, T, -1)
        x = self.cnn_linear(x)
            
        length = x.view(B, -1).clone()
        length = self.len_cfr(length).float()
        
        # RNN / Encoder / Decoder
        if   self.seq_type in ['LSTM', 'GRU']:
            x = self.rnn(x)[0] # contextual feature
            x = self.rnn_linear(x)
        elif self.seq_type == 'TransformerEncoder':
            pe = self.pos_encoder(length=T, d_model=self.configs['SEQ_HIDDEN_SIZE']).to(self.configs['DEVICE'])
            x = x.add_(pe)
            x = self.encoder(x) 
            x = self.trf_linear(x)
        elif self.seq_type == 'TransformerDecoder':
            # tgt_mask = self.generate_square_subsequent_mask(x.shape[1])
            pe = self.pos_encoder(length=T, d_model=self.configs['SEQ_HIDDEN_SIZE']).to(self.configs['DEVICE'])
            x = x.add_(pe)
            enc =self.encoder(x).add_(pe)
            x = self.decoder(tgt=x, memory=enc)
            x = self.activate(self.trf_linear(x))
            
    
        # output
        if self.configs["PARALLEL"] == True:
            return x, length
        else:
            return x.permute(1,0,2), length


    def generate_square_subsequent_mask(self, sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# -----------------------------------------------------------------
class VggNet(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.activate1 = torch.nn.SiLU(inplace=True)
        self.activate2 = torch.nn.GELU()
        
        """https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py"""
        """https://github.com/rwightman/pytorch-image-models/blob/3698e79ac5ea869e2e1ac9d22e8c84af8a99b57a/timm/models/byobnet.py#L1575"""
        net = timm.create_model(
            # "repvgg_a2",
            "repvgg_b2g4",
            pretrained=configs['PRETRAINED'],
            in_chans=configs['INPUT_CHANNEL'],
            num_classes=0, 
            output_stride=16, # 32
            drop_path_rate=configs['DROPPATH'],
            global_pool=None
        )
        net = [
            net.stem,
            net.stages,
            torch.nn.Conv2d(2560, 1440, kernel_size=3, stride=1, bias=False, padding='same'),
            torch.nn.LeakyReLU(inplace=True)
        ]

        self.cnn = nn.Sequential(*net)
    
    def forward(self, x):
        x = self.cnn(x)
        return x
       
       
class RegNet(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        net = timm.create_model(
            "regnety_120",
            pretrained=configs['PRETRAINED'],
            in_chans=1,
            num_classes=0, 
            output_stride=16, # 32
            drop_path_rate=configs['DROPPATH'],
            global_pool=None
        )
        net = torch.nn.ModuleList([
            net.stem,
            net.s1,
            net.s2,
            net.s3,
            net.s4,
            torch.nn.Conv2d(2240, 1440, 3, padding='same', bias=False),
            torch.nn.LeakyReLU(inplace=True),
        ])
        self.cnn = torch.nn.Sequential(*net)
    
    def forward(self, x):
        x = self.cnn(x)
        return x
    

class RegNetLarge(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        net = timm.create_model(
            "regnety_120",
            pretrained=configs['PRETRAINED'],
            in_chans=1,
            num_classes=0, 
            output_stride=16, # 32
            drop_path_rate=configs['DROPPATH'],
            global_pool=None
        )
        net = torch.nn.ModuleList([
            net.stem,
            net.s1,
            net.s2,
            net.s3,
            net.s4,
            torch.nn.Conv2d(2240, 1440, 3, padding='same', bias=False),
        ])
        self.cnn = torch.nn.Sequential(*net)
    
    def forward(self, x):
        x = self.cnn(x)
        return x
    

class ResNet(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        net = timm.create_model(
            "resnet18d",
            pretrained=configs['PRETRAINED'],
            in_chans=configs['INPUT_CHANNEL'],
            num_classes=0, 
            output_stride=16, # 32
            drop_path_rate=configs['DROPPATH'],
            global_pool=None
        )
        net = list(net.children())[:-3]
        self.cnn = nn.Sequential(
            *net,
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


class EffnetV2(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        net = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=configs['PRETRAINED'],
            in_chans=configs['INPUT_CHANNEL'],
            num_classes=0, 
            output_stride=16, # 32
            drop_path_rate=configs['DROPPATH'],
            global_pool=None
        )
        net = [
            net.conv_stem,
            net.bn1,
            net.blocks,
            net.conv_head,
            net.bn2
        ]
        self.net = torch.nn.Sequential(
            *net
        )
    def forward(self, x):
        x = self.net(x)
        return x
        

# --------------------------------------------------------------------------------------
def generate_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`
    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE
    
# --------------------------------------------------------------------------------------
class TPS_SpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, configs):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.configs = configs
        self.F = configs['NUM_FIDUCIAL']
        self.I_size = (configs['IMG_HEIGHT_SIZE'],configs['IMG_WIDTH_SIZE'])
        self.I_r_size = (configs['IMG_HEIGHT_SIZE'],configs['IMG_WIDTH_SIZE'])
        self.I_channel_num = configs['INPUT_CHANNEL']
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r


class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """
    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
        return batch_C_prime


class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # F+3 x F+3
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
            batch_size, 3, 2).float().cuda()), dim=1)  # batch_size x F+3 x 2
            # batch_size, 3, 2).float()), dim=1)  # batch_size x F+3 x 2
        batch_T = batch_inv_delta_C @ batch_C_prime_with_zeros  # batch_size x F+3 x 2
        batch_P_prime = batch_P_hat @ batch_T  # batch_size x n x 2
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2

        return batch_P_prime  # batch_size x n x 2



# --------------------------------------------------------------------
class BaseDataset(Dataset):
    def __init__(self, configs, img_path_list, label_list, len_list, mode=True):
        self.configs = configs
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.len_list = len_list
        self.mode = mode
        self.height = configs['IMG_HEIGHT_SIZE']
        self.width = configs['IMG_WIDTH_SIZE']
        
        self.blank = np.ones([configs['IMG_HEIGHT_SIZE'],configs['IMG_HEIGHT_SIZE']])*255
        self.blank[configs['IMG_HEIGHT_SIZE']//2, configs['IMG_HEIGHT_SIZE']//2] = 0
        self.blank = self.blank.astype(np.uint8)
        
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        image = self.img_path_list[idx]
        if self.configs['GRAY_SCALE'] == True:
            image = cv2.imread(image, 0)
            H, W = image.shape
            image = image.reshape(H, W, 1)
            # thresholding
            threshold, _ = cv2.threshold(image, image.mean()*0.85, 255, cv2.THRESH_BINARY)
            image[image > threshold] = 255
        else:
            image = cv2.imread(image)
            
        image = self.base_transform(image)
        if (self.mode == 'train') | (self.mode == 'valid'):
            label = self.label_list[idx]
            length = torch.tensor(self.len_list[idx])
            return image, label, length
        
        elif self.mode == 'test':
            return image
    
    # Image Augmentation
    def base_transform(self, image):
        shape = image.shape
        if shape[1] < self.configs['IMG_HEIGHT_SIZE']+8:
            image = A.Resize(height=self.configs['IMG_HEIGHT_SIZE'], width=self.configs['IMG_HEIGHT_SIZE'])(image=image)['image']
            image = cv2.hconcat([image, self.blank, self.blank])
        elif shape[1] < self.configs['IMG_HEIGHT_SIZE']*2+8:
            image = A.Resize(height=self.configs['IMG_HEIGHT_SIZE'], width=self.configs['IMG_HEIGHT_SIZE']*2)(image=image)['image']
            image = cv2.hconcat([image, self.blank])
        else:
            image = A.Resize(height=self.configs['IMG_HEIGHT_SIZE'], width=self.configs['IMG_HEIGHT_SIZE']*3)(image=image)['image']
        
        ops = A.Compose([
            ToTensorV2(),
        ])
        return ops(image=image)['image'].float()


class AugmentDataset(Dataset):
    def __init__(self, configs, img_path_list, label_list, len_list, mode=True, aug_number=0):
        self.configs = configs
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.len_list = len_list
        self.mode = mode
        self.height = configs['IMG_HEIGHT_SIZE']
        self.width = configs['IMG_WIDTH_SIZE']
        self.aug_number = aug_number
        
        self.blank = np.ones([configs['IMG_HEIGHT_SIZE'],configs['IMG_HEIGHT_SIZE']])*255
        self.blank[configs['IMG_HEIGHT_SIZE']//2, configs['IMG_HEIGHT_SIZE']//2] = 0
        self.blank = self.blank.astype(np.uint8)
        
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        image = self.img_path_list[idx]
        label = self.label_list[idx]
        length = torch.tensor(self.len_list[idx])
        if self.configs['GRAY_SCALE'] == True:
            image = cv2.imread(image, 0)
            H, W = image.shape
            image = image.reshape(H, W, 1)

        else:
            image = cv2.imread(image)
        
        if   self.mode == 'rot':
            random.seed(self.aug_number*self.configs['SEED'])
            image = self.rotation(image)
            random.seed(self.configs['SEED'])

        elif self.mode == 'trs':
            random.seed(self.aug_number*self.configs['SEED'])
            image = self.translation(image)
            random.seed(self.configs['SEED'])
        return image, label, length
            
    # Image Augmentation
    def rotation(self, image):
        shape = image.shape
        if shape[1] < self.configs['IMG_HEIGHT_SIZE']+8:
            image = A.Resize(height=self.configs['IMG_HEIGHT_SIZE'], width=self.configs['IMG_HEIGHT_SIZE'])(image=image)['image']
            image = cv2.hconcat([image, self.blank, self.blank])
        elif shape[1] < self.configs['IMG_HEIGHT_SIZE']*2+8:
            image = A.Resize(height=self.configs['IMG_HEIGHT_SIZE'], width=self.configs['IMG_HEIGHT_SIZE']*2)(image=image)['image']
            image = cv2.hconcat([image, self.blank])
        else:
            image = A.Resize(height=self.configs['IMG_HEIGHT_SIZE'], width=self.configs['IMG_HEIGHT_SIZE']*3)(image=image)['image']
        
        ops = A.Compose([
            AF(rotate = random.choice([-5, 5, -10, 10]), cval=255, p=1),
            AF(translate_percent = random.choice([[0.05, 0],[0,0.05],[-0.05, 0],[0, -0.05]]), cval=255, p=0.25, keep_ratio=True),
            ToTensorV2(),
        ])
        return ops(image=image)['image'].float()
    
    def translation(self, image):
        shape = image.shape
        if shape[1] < self.configs['IMG_HEIGHT_SIZE']+8:
            image = A.Resize(height=self.configs['IMG_HEIGHT_SIZE'], width=self.configs['IMG_HEIGHT_SIZE'])(image=image)['image']
            image = cv2.hconcat([image, self.blank, self.blank])
        elif shape[1] < self.configs['IMG_HEIGHT_SIZE']*2+8:
            image = A.Resize(height=self.configs['IMG_HEIGHT_SIZE'], width=self.configs['IMG_HEIGHT_SIZE']*2)(image=image)['image']
            image = cv2.hconcat([image, self.blank])
        else:
            image = A.Resize(height=self.configs['IMG_HEIGHT_SIZE'], width=self.configs['IMG_HEIGHT_SIZE']*3)(image=image)['image']
        
        ops = A.Compose([
            AF(translate_percent = random.choice([[0.05, 0],[0,0.05],[-0.05, 0],[0, -0.05]]), cval=255, p=1, keep_ratio=True),
            AF(rotate = random.choice([-5, 5, -10, 10]), cval=255, p=0.25),
            ToTensorV2(),
        ])
        return ops(image=image)['image'].float()

    
# ------------------------------------------------------------------------    
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
    
    
    
# ------------------------------------------------------------------------
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    
    
    
# -----------------------------------
class BaseLine(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_chars = configs["NUM_CLASS"]
        self.SEQ_hidden_size = configs["SEQ_HIDDEN_SIZE"]
        
        # CNN Backbone = 사전학습된 resnet18 활용
        # https://arxiv.org/abs/1512.03385
        resnet = resnet18(pretrained=True)
        # CNN Feature Extract
        resnet_modules = list(resnet.children())[:-3]
        self.feature_extract = nn.Sequential(
            *resnet_modules,
            nn.Conv2d(256, 256, kernel_size=(3,6), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(1024, self.SEQ_hidden_size)
        
        # RNN
        self.rnn = nn.RNN(input_size=self.SEQ_hidden_size, 
                            hidden_size=self.SEQ_hidden_size,
                            bidirectional=True, 
                            batch_first=True)
        self.linear2 = nn.Linear(self.SEQ_hidden_size*2, self.num_chars)
        
        
    def forward(self, x):
        # CNN
        x = self.feature_extract(x) # [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2) # [batch_size, width, channels, height]
         
        batch_size = x.size(0)
        T = x.size(1)
        x = x.view(batch_size, T, -1) # [batch_size, T==width, num_features==channels*height]
        x = self.linear1(x)
        
        # RNN
        x, hidden = self.rnn(x)
        
        output = self.linear2(x)
        output = output.permute(1, 0, 2) # [T==10, batch_size, num_classes==num_features]
        
        return output