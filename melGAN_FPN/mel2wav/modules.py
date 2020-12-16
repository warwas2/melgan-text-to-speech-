import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        #model["layer_1"] =  nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        #model["layer_2"] =  nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        #model["layer_3"] =  nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        #model["layer_4"] =  nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 2,
                ),
                nn.LeakyReLU(0.2, True),
            )

        #nf = min(nf * 2, 1024)
        #model["layer_%d" % (n_layers + 1)] = nn.Sequential(
        #    WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
        #    nn.LeakyReLU(0.2, True),
        #)

        #model["layer_%d" % (n_layers + 2)] = WNConv1d(
        #    nf, 1, kernel_size=3, stride=1, padding=1
        #)
        

        self.model = model
        self.toplayer = nn.Conv1d(1024, 1024, kernel_size=1, stride=1, padding=0)
        

        model2 = nn.ModuleDict()
        self.model2 = model2

        
        self.top = nn.Sequential(nn.LeakyReLU(0.2),WNConvTranspose1d(256,256,kernel_size = 21,stride=2,padding=10,output_padding=1,))
        #self.top2 = [nn.LeakyReLU(0.2),WNConvTranspose1d(256,256,kernel_size = 21,stride=2,padding=10,output_padding=1,),]
        #self.top3 = [nn.LeakyReLU(0.2),WNConvTranspose1d(256,256,kernel_size = 21,stride=2,padding=10,output_padding=1,),]
        #self.top4 = [nn.LeakyReLU(0.2),WNConvTranspose1d(256,256,kernel_size = 21,stride=2,padding=10,output_padding=1,),]

        self.final = WNConv1d(256, 1, kernel_size=3, stride=1, padding=1)
		
        # Smooth Layer
        
        #self.smooth1 = WNConv1d(1024, 256, kernel_size=3, stride=1, padding=1)
        #self.smooth2 = WNConv1d(256, 256, kernel_size=3, stride=1, padding=1)
        #self.smooth3 = WNConv1d(64, 64, kernel_size=3, stride=1, padding=1)
        #self.smooth4 = WNConv1d(16, 16, kernel_size=3, stride=1, padding=1)

		#lateral layers

        #self.latlayer1 = WNConv1d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.latlayer_128 = WNConv1d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_64 = WNConv1d( 64, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_32 = WNConv1d( 32, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_16 = WNConv1d( 16, 256, kernel_size=1, stride=1, padding=0)
        #self.latlayer4 = WNConv1d( 16, 16, kernel_size=1, stride=1, padding=0)

        #self.conv1d_0 = WNConvTranspose1d(1024,1024, kernel_size = 5, stride = 1, padding = 2, groups = 1) #change
        #self.conv1d_00 = WNConvTranspose1d(1024,1024, kernel_size = 41, stride = 4, padding = 20, output_padding = 3) #change
        #self.conv1d_1 = WNConvTranspose1d(1024,256, kernel_size = 41, stride = 4, padding = 20, output_padding = 3)
        #self.conv1d_2 = WNConvTranspose1d(256,64, kernel_size = 41, stride = 4, padding = 20, output_padding = 3)
        #self.conv1d_3 = WNConvTranspose1d(64,16, kernel_size = 41, stride = 4, padding = 20, output_padding = 3)
        #self.conv1d_4 = WNConvTranspose1d(16,1, kernel_size = 15, padding = 7)

    def upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        k = F.upsample(x, L) + y
        #print(k.shape)
        return k 

    def forward(self, x):
        bottom_up = []
        top_down = []
        # bottom - up
        
        for key, layer in self.model.items():
            #print('x의 before은 : ',x.shape)
            x = layer(x)
            #print('x의 after은 : ',x.shape)
            bottom_up.append(x)
            
        #print('----------------------------')

        

        result = self.top(bottom_up[4]) + self.latlayer_128(bottom_up[3])
        top_down.append(result)
        
        result = self.top(result) + self.latlayer_64(bottom_up[2])
        top_down.append(result)
        
        result = self.top(result) + self.latlayer_32(bottom_up[1])
        top_down.append(result)

        result = self.top(result) + self.latlayer_16(bottom_up[0])
        top_down.append(result)

        result = self.final(result)    
        top_down.append(result)    
        
        #for i in range(len(top_down)):
            #print(top_down[i].shape)

        return top_down


class Discriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        #print('num D는 ',num_D)
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

        self.p_num = 0
    

    def forward(self, x):
        results = []
        self.p_num = count_parameters(self.model)
        for key, disc in self.model.items():
            
            results.append(disc(x))
            x = self.downsample(x)
            
        return results
