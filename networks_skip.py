import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from . import residual_transformers
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG,vit_name,img_size,pre_trained_path, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[],pre_trained_trans=True,pre_trained_resnet=0):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'res_cnn':
        netG = residual_transformers.Res_CNN(residual_transformers.CONFIGS[vit_name], input_dim= input_nc, img_size=img_size, output_dim=1, vis=False)
    elif which_model_netG == 'resvit':
        print(vit_name)
        netG = residual_transformers.ResViT(residual_transformers.CONFIGS[vit_name],input_dim = input_nc,img_size=img_size, output_dim=1, vis=False)
        config_vit = residual_transformers.CONFIGS[vit_name]
        if pre_trained_resnet:
            pre_trained_model = residual_transformers.Res_CNN(residual_transformers.CONFIGS[vit_name], input_dim= input_nc, img_size=img_size, output_dim=1, vis=False)
            save_path = pre_trained_path
            print("pre_trained_path: ",save_path)
            pre_trained_model.load_state_dict(torch.load(save_path))

            pretrained_dict = pre_trained_model.state_dict()
            model_dict = netG.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            netG.load_state_dict(model_dict)

            print("Residual CNN loaded")

        if pre_trained_trans:
            print(config_vit.pretrained_path)
            netG.load_from(weights=np.load(config_vit.pretrained_path))
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


def define_D(input_nc, ndf, which_model_netD,vit_name,img_size,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
        
        
        
class GDLLoss(nn.Module):
    def __init__(self, alpha):
        super(GDLLoss, self).__init__()
        self.alpha = alpha

    def forward(self, gen_MRI, gt_MRI):
        """
        Calculates the sum of GDL losses between the predicted and ground truth images.
        @param gen_MRI: The predicted MRIs.
        @param gt_MRI: The ground truth images
        @return: The GDL loss.
        """
        
        # Create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        filter_x = torch.tensor([[[[1, -1]]]], dtype=torch.float32).to(gen_MRI.device)
        filter_y = torch.tensor([[1], [-1]], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).to(gen_MRI.device)
        print('gen_MRI shape:', gen_MRI.shape)
        print('filter_y shape:', filter_y.shape)
        print('gt_MRI shape:', gt_MRI.shape)
        print('filter_x shape:', filter_x.shape)
        
        gen_dy = torch.nn.functional.conv2d(gen_MRI, filter_y, bias=None, stride=(1,), padding='same', dilation=(1,))
        gt_dy = torch.nn.functional.conv2d(gt_MRI, filter_y, bias=None,stride=(1,), padding='same', dilation=(1,))
        gen_dx = torch.nn.functional.conv2d(gen_MRI, filter_x, bias=None, stride=(1,), padding='same', dilation=(1,))
        gt_dx = torch.nn.functional.conv2d(gt_MRI, filter_x, bias=None, stride=(1,), padding='same', dilation=(1,))
#         gt_dx = torch.abs(gt_MRI[:, :, :, :-1] - gt_MRI[:, :, :, 1:])
#         gt_dy = torch.abs(gt_MRI[:, :, :-1, :] - gt_MRI[:, :, 1:, :])
#         gen_dx = torch.abs(gen_MRI[:, :, :, :-1] - gen_MRI[:, :, :, 1:])
#         gen_dy = torch.abs(gen_MRI[:, :, :-1, :] - gen_MRI[:, :, 1:, :])

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        gdl = torch.sum((grad_diff_x ** self.alpha + grad_diff_y ** self.alpha))

        return gdl


class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.feature_extractor = Densepyr().cuda()

    def forward(self, realA, realB, fakeB):
        realA = realA.cuda()
        realB = realB.cuda()
        fakeB = fakeB.cuda()
        feat1 = self.feature_extractor(realA)
        feat2 = self.feature_extractor(realB)
        feat3 = self.feature_extractor(fakeB)
        size=feat1.size()
        loss = (1/(size[0]*size[1]*size[2]))*((torch.norm(feat2-feat3)**2))

        return loss

class Densepyr(nn.Module):
    '''
    feature extractor block 
    '''
    def __init__(self):
        super(Densepyr, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0).cuda()
        self.conv_d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same").cuda()
        self.relu = nn.ReLU()
        self.down = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.bnorm1 = nn.BatchNorm2d(num_features=64)
        self.fe = FE_V1().cuda()
        
    def forward(self, x):
        temp0 = self.conv1(x)
        skip0 = self.relu(self.up(self.down(x)))
        attn0 = torch.mul(skip0, temp0)
        x = x + attn0
        x = self.bnorm1(x)
        deep_fe = self.fe(x)
        pur_x = self.relu(self.up(self.down(x)))
        attn2 = torch.mul(pur_x, deep_fe)
        add = attn2 + x
        x = self.bnorm1(add)
        

        return x

class FE_V1(nn.Module):
    '''
    feature extractor block (temporary, will edit if necessary)
    '''
    def __init__(self):
        super(FE_V1, self).__init__()

        # multiscale dilation conv2d
        self.convd1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding="same")
        self.convd2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=3, padding="same")
        self.convd3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=5, padding="same")

        self.reduce = nn.Conv2d(in_channels=64*3, out_channels=64, kernel_size=1, stride=1, padding="same")
        self.relu = nn.ReLU()

        self.bnorm1 = nn.BatchNorm2d(num_features=64)
        
        self.dilran = DILRAN_V1().cuda()

    
    def forward(self, x):

        # dilated convolution
        dilf1 = self.convd1(x)
        dilf2 = self.convd2(x)
        dilf3 = self.convd3(x)

        diltotal = torch.cat((dilf1, dilf2, dilf3), dim = 1)
        diltotal = self.reduce(diltotal)
        diltotal = self.bnorm1(diltotal)

        # single DILRAN
        out = self.dilran(diltotal)
        out = self.bnorm1(out)
        #out = self.relu(out)
        return out
        
class DILRAN_V1(nn.Module):
    '''
    V1: concat the output of three (conv-d,DILRAN) paths channel wise and add the low level feature to the concat output
    temporary, will edit if necessary
    '''
    def __init__(self, cat_first = False, use_leaky = False):
        super(DILRAN_V1, self).__init__()
        # cat_first, whether to perform channel-wise concat before DILRAN
        # convolution in DILRAN, in channel is the channel from the previous block
        if not cat_first:
            self.conv_d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
            self.bnorm = nn.BatchNorm2d(num_features=64)
        else:
            self.conv_d = nn.Conv2d(in_channels=64*3, out_channels=64*3, kernel_size=3, stride=1, padding="same")
            self.bnorm = nn.BatchNorm2d(num_features=64*3)
        
        if not use_leaky:
            self.relu = nn.ReLU()
        else:
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        self.down = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    
    def forward(self, x):
        # pooling -> upsample -> ReLU block
        pur_path = self.relu(self.up(self.down(x)))
        # 3*3, 5*5, 7*7 multiscale addition block
        conv_path = self.conv_d(x) + self.conv_d(self.conv_d(x)) + self.conv_d(self.conv_d(self.conv_d(x)))
        # attention
        attn = torch.mul(pur_path, conv_path)
        # residual + attention
        resid_x = x + attn
        return resid_x

class Encoder_Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect',down_samp=1,gated_fusion=0):
        super(Encoder_Decoder, self).__init__()        
        self.output_nc = output_nc      
        self.encoders=2
        latent_size=16
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        #Encoders
        for ii in range(2):
            model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                     norm_layer(ngf), nn.ReLU(True)]   
            n_downsampling = 2 
            
            ### downsample
            for i in range(n_downsampling):
                mult = 2**i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * mult * 2), nn.ReLU(True)]
            mult = 2**n_downsampling
            for i in range(n_blocks):
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            #model += [nn.ReflectionPad2d(1)]              
            model += [nn.Conv2d(ngf * mult, latent_size, kernel_size=3, padding=1), 
                     norm_layer(latent_size), nn.ReLU(True)]   
            setattr(self, 'model_enc_'+str(ii), nn.Sequential(*model))
        #Decoder
        #model += [nn.ReflectionPad2d(3)] 
        model = [nn.Conv2d(latent_size*2, 256, kernel_size=3, padding=1), 
                 norm_layer(256), nn.ReLU(True)]  
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1,bias=use_bias),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(2), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        setattr(self, 'model_dec', nn.Sequential(*model))

            
            
    def forward(self, input):
        #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        encoded=[]
        for ii in range(self.encoders):
            encoded.append( getattr(self, 'model_enc_'+str(ii))(input[:,ii,:,:]))
        decoded=self.model_dec(torch.cat((encoded[0],encoded[1]),1))
        return decoded
#        else:
#            return self.model(input)







# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect',down_samp=1):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.down_samp=down_samp
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
############################################################################################
############################################################################################
        #skip-connection1
        model = [nn.Conv2d(input_dim, 64, kernel_size=1, padding=0,
                           bias=False),
                 norm_layer(64)]
        setattr(self, 'skip_1', nn.Sequential(*model))
        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_dim, 32, kernel_size=3, padding=0,
                           bias=use_bias),
                 norm_layer(32),
                 nn.ReLU(True),
                 #nn.Dropout(0.3),
                 nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1,
                           bias=use_bias)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        #skip-connection2
        model = [nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=2,
                           bias=False),
                 norm_layer(128)]
        setattr(self, 'skip_2', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        #i = 0
        #mult = 2 ** i
        model = [nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 #nn.Dropout(0.3),
                 nn.Conv2d(64, 96, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(96),
                 nn.ReLU(True),
                 #nn.Dropout(0.3),
                 nn.Conv2d(96, 128, kernel_size=3, padding=1, stride=1,
                           bias=use_bias)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        #skip-connection3
        model = [nn.Conv2d(128, 256, kernel_size=1, padding=0,stride=2,
                           bias=False),
                 norm_layer(256)]
        setattr(self, 'skip_3', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        #i = 1
        #mult = 2 ** i
        model = [nn.BatchNorm2d(128),
                 nn.ReLU(True),
                 #nn.Dropout(0.3),
                 nn.Conv2d(128, 192, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(192),
                 nn.ReLU(True),
                 #nn.Dropout(0.3),
                 nn.Conv2d(192, 256, kernel_size=3, padding=1, stride=1,
                           bias=use_bias)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
############################################################################################
#Layer4-Residual1
        mult = 2**n_downsampling
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_4', nn.Sequential(*model))
############################################################################################
#Layer5-Residual2
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_5', nn.Sequential(*model))
############################################################################################
#Layer6-Residual3
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_6', nn.Sequential(*model))
############################################################################################
#Layer7-Residual4
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_7', nn.Sequential(*model))
############################################################################################
#Layer8-Residual5
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_8', nn.Sequential(*model))
############################################################################################
#Layer9-Residual6
        model = []
        use_dropout = norm_layer == nn.InstanceNorm2d
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_9', nn.Sequential(*model))
############################################################################################
#Layer10-Residual7
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_10', nn.Sequential(*model))
############################################################################################
#Layer11-Residual8
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_11', nn.Sequential(*model))
############################################################################################
#Layer12-Residual9
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_12', nn.Sequential(*model))
############################################################################################
#skip-connection4
        model = [nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2,output_padding=1,
                                    bias=False),
                 norm_layer(128)]
        setattr(self, 'skip_4', nn.Sequential(*model))
        ############################################################################################
        # Layer13-Decoder1
        #n_downsampling = 2
        #i = 0
        #mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(256, 192,
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(192),
                 nn.ReLU(True),
                 #nn.Dropout(0.3),
                 nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=1,
                           bias=use_bias)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        #skip-connection5
        model = [nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2,output_padding=1,
                                    bias=False),
                 norm_layer(64)]
        setattr(self, 'skip_5', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        #i = 1
        #mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.BatchNorm2d(128),
                 nn.ReLU(True),
                 #nn.Dropout(0.3),
                 nn.ConvTranspose2d(128, 96,
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(96),
                 nn.ReLU(True),
                 #nn.Dropout(0.3),
                 nn.Conv2d(96, 64, kernel_size=3, padding=1, stride=1,
                           bias=use_bias)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        #skip-connection6
        model = [nn.Conv2d(64, 1, kernel_size=1, padding=0,
                           bias=False),
                 norm_layer(1)]
        setattr(self, 'skip_6', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 #nn.Dropout(0.3),
                 nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, 32, kernel_size=7, padding=0),
                  norm_layer(32),
                  nn.ReLU(True),
                  #nn.Dropout(0.3),
                  nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1,
                            bias=use_bias)]
        #model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))

############################################################################################


    def forward(self, input):
        #print('input shape:', x.shape) 
        x_input=input    
        x = self.encoder_1(input)
        x_input=self.skip_1(x_input)
        x=x+x_input
        #print('encoder layer 1 output shape:', x.shape)
        x1=x
        x = self.encoder_2(x)
        x1=self.skip_2(x1)
        x=x+x1
        #print('encoder layer 2 output shape:', x.shape)
        x2=x
        x = self.encoder_3(x)
        x2=self.skip_3(x2)
        x=x+x2
        x = nn.BatchNorm2d(256).to(device)(x)
        #print('encoder layer 3 output shape:', x.shape)
        x = self.model_4(x)
        x = self.model_5(x)
        x = self.model_6(x)
        x = self.model_7(x)
        x = self.model_8(x)
        x = self.model_9(x)
        x = self.model_10(x)
        x = self.model_11(x)
        x = self.model_12(x)
        #print('Decoder input shape:', x.shape)
        
        # Decoder
        x3=x
        x = self.decoder_1(x)
        x3 = self.skip_4(x3)
        x=x+x3
        #print('Decoder1 output shape:', x.shape)
        x4=x
        x = self.decoder_2(x)
        x4=self.skip_5(x4)
        x = x+x4
        #print('Decoder2 output shape:', x.shape)
        x5=x
        x = self.decoder_3(x)
        x5 = self.skip_6(x5)
        x=x+x5
        x=nn.Tanh()(x)
        #print('Decoder3 output shape:', x.shape)
        return x
        

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            print(self.model(input).size())
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)
