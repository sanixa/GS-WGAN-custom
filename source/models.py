import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import functools

from ops import SpectralNorm, one_hot_embedding, pixel_norm

IMG_W = IMG_H = 28  # image width and height
IMG_C = 1  # image channel

class TemperedSigmoid(nn.Module):
    def __init__(self, s=2, T=2, o=1):
        super().__init__()
        self.s = s
        self.T = T
        self.o = o

    def forward(self, input):
        div = 1 + torch.exp(-1 * self.T *input)
        return self.s / div - self.o

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=nn.ReLU(inplace=False),
                 upsample=functools.partial(F.interpolate, scale_factor=2)):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation = activation
        self.upsample = upsample

        # Conv layers
        self.conv1 = SpectralNorm(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1))
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channels, out_channels,
                                                  kernel_size=1, padding=0))
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = pixel_norm(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(pixel_norm(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GeneratorDCGAN(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Sigmoid()):
        super(GeneratorDCGAN, self).__init__()

        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = nn.Linear(z_dim + num_classes, 4 * 4 * 4 * model_dim)
        deconv1 = nn.ConvTranspose2d(4 * model_dim, 2 * model_dim, 5)
        deconv2 = nn.ConvTranspose2d(2 * model_dim, model_dim, 5)
        deconv3 = nn.ConvTranspose2d(model_dim, IMG_C, 8, stride=2)

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.fc = fc
        self.relu = nn.ReLU()
        self.outact = outact

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, 4 * self.model_dim, 4, 4)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv1(output)
        output = output[:, :, :7, :7]
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv2(output)
        output = self.relu(output).contiguous()
        output = pixel_norm(output)

        output = self.deconv3(output)
        output = self.outact(output)
        return output.view(-1, IMG_W * IMG_H)

class GeneratorDCGAN_TS(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Sigmoid()):
        super(GeneratorDCGAN_TS, self).__init__()

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = nn.Linear(z_dim + num_classes, 4 * 4 * 4 * model_dim)
        deconv1 = nn.ConvTranspose2d(4 * model_dim, 2 * model_dim, 5)
        deconv2 = nn.ConvTranspose2d(2 * model_dim, model_dim, 5)
        deconv3 = nn.ConvTranspose2d(model_dim, IMG_C, 8, stride=2)

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.fc = fc
        self.TS = TemperedSigmoid()
        self.outact = outact

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, 4 * self.model_dim, 4, 4)
        output = self.TS(output)
        output = pixel_norm(output)

        output = self.deconv1(output)
        output = output[:, :, :7, :7]
        output = self.TS(output)
        output = pixel_norm(output)

        output = self.deconv2(output)
        output = self.TS(output).contiguous()
        output = pixel_norm(output)

        output = self.deconv3(output)
        output = self.TS(output)
        return output.view(-1, IMG_W * IMG_H)


class GeneratorResNet(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Sigmoid()):
        super(GeneratorResNet, self).__init__()

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = SpectralNorm(nn.Linear(z_dim + num_classes, 4 * 4 * 4 * model_dim))
        block1 = GBlock(model_dim * 4, model_dim * 4)
        block2 = GBlock(model_dim * 4, model_dim * 4)
        block3 = GBlock(model_dim * 4, model_dim * 4)
        output = SpectralNorm(nn.Conv2d(model_dim * 4, IMG_C, kernel_size=3, padding=0))

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.fc = fc
        self.output = output
        self.relu = nn.ReLU()
        self.outact = outact

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, 4 * self.model_dim, 4, 4)
        output = self.relu(output)
        output = pixel_norm(output)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.outact(self.output(output))
        output = output[:, :, :-2, :-2]
        output = torch.reshape(output, [-1, IMG_H * IMG_W])
        return output

class GeneratorResNet_cifar10(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Sigmoid()):
        super(GeneratorResNet_cifar10, self).__init__()

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.IMG_C = 1
        self.IMG_H, self.IMG_W = 32,32

        fc = SpectralNorm(nn.Linear(z_dim + num_classes, 4 * 4 * 4 * model_dim))
        block1 = GBlock(model_dim * 4, model_dim * 2)
        block2 = GBlock(model_dim * 2, model_dim)
        block3 = GBlock(model_dim, model_dim)
        block4 = GBlock(model_dim, model_dim * 2)
        block5 = GBlock(model_dim * 2, model_dim * 4)
        output1 = SpectralNorm(nn.Conv2d(model_dim * 4,model_dim * 2, kernel_size=3, stride=2, padding=1))
        output2 = SpectralNorm(nn.Conv2d(model_dim * 2, self.IMG_C, kernel_size=3, stride=2, padding=1))

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.block4 = block4
        self.block5 = block5
        self.fc = fc
        self.output1 = output1
        self.output2 = output2
        self.relu = nn.ReLU()
        self.outact = outact

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, 4 * self.model_dim, 4, 4)
        output = self.relu(output)
        output = pixel_norm(output)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.output1(output)
        output = self.relu(output)
        output = self.outact(self.output2(output))
        output = torch.reshape(output, [-1, self.IMG_H * self.IMG_W])
        return output

class DiscriminatorDCGAN(nn.Module):
    def __init__(self, model_dim=64, num_classes=10, if_SN=True):
        super(DiscriminatorDCGAN, self).__init__()

        self.model_dim = model_dim
        self.num_classes = num_classes

        if if_SN:
            self.conv1 = SpectralNorm(nn.Conv2d(1, model_dim, 5, stride=2, padding=2))
            self.conv2 = SpectralNorm(nn.Conv2d(model_dim, model_dim * 2, 5, stride=2, padding=2))
            self.conv3 = SpectralNorm(nn.Conv2d(model_dim * 2, model_dim * 4, 5, stride=2, padding=2))
            self.linear = SpectralNorm(nn.Linear(4 * 4 * 4 * model_dim, 1))
            self.linear_y = SpectralNorm(nn.Embedding(num_classes, 4 * 4 * 4 * model_dim))
        else:
            self.conv1 = nn.Conv2d(1, model_dim, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(model_dim, model_dim * 2, 5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(model_dim * 2, model_dim * 4, 5, stride=2, padding=2)
            self.linear = nn.Linear(4 * 4 * 4 * model_dim, 1)
            self.linear_y = nn.Embedding(num_classes, 4 * 4 * 4 * model_dim)
        self.relu = nn.ReLU()

    def forward(self, input, y):
        input = input.view(-1, 1, IMG_W, IMG_H)
        h = self.relu(self.conv1(input))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = h.view(-1, 4 * 4 * 4 * self.model_dim)
        out = self.linear(h)
        out += torch.sum(self.linear_y(y) * h, dim=1, keepdim=True)
        return out.view(-1)

    def calc_gradient_penalty(self, real_data, fake_data, y, L_gp, device):
        '''
        compute gradient penalty term
        :param real_data:
        :param fake_data:
        :param y:
        :param L_gp:
        :param device:
        :return:
        '''

        batchsize = real_data.shape[0]
        real_data = real_data.to(device)
        fake_data = fake_data.to(device)
        y = y.to(device)
        alpha = torch.rand(batchsize, 1)
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, y)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * L_gp
        return gradient_penalty


class DiscriminatorDCGAN_cifar10(nn.Module):
    def __init__(self, model_dim=64, num_classes=10, if_SN=True):
        super(DiscriminatorDCGAN_cifar10, self).__init__()
        self.IMG_W, self.IMG_H = 32, 32 
        self.IMG_C = 1
        self.model_dim = model_dim
        self.num_classes = num_classes

        if if_SN:
            self.conv1 = SpectralNorm(nn.Conv2d(self.IMG_C, model_dim, 5, stride=2, padding=2))
            self.conv2 = SpectralNorm(nn.Conv2d(model_dim, model_dim * 2, 5, stride=2, padding=2))
            self.conv3 = SpectralNorm(nn.Conv2d(model_dim * 2, model_dim * 4, 5, stride=2, padding=2))
            self.linear = SpectralNorm(nn.Linear(4 * 4 * 4 * model_dim, 1))
            self.linear_y = SpectralNorm(nn.Embedding(num_classes, 4 * 4 * 4 * model_dim))
        else:
            self.conv1 = nn.Conv2d(1, model_dim, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(model_dim, model_dim * 2, 5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(model_dim * 2, model_dim * 4, 5, stride=2, padding=2)
            self.linear = nn.Linear(4 * 4 * 4 * model_dim, 1)
            self.linear_y = nn.Embedding(num_classes, 4 * 4 * 4 * model_dim)
        self.relu = nn.ReLU()

    def forward(self, input, y):
        input = input.view(-1, self.IMG_C, self.IMG_W, self.IMG_H)
        h = self.relu(self.conv1(input))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = h.view(-1, 4 * 4 * 4 * self.model_dim)
        out = self.linear(h)
        out += torch.sum(self.linear_y(y) * h, dim=1, keepdim=True)
        return out.view(-1)

    def calc_gradient_penalty(self, real_data, fake_data, y, L_gp, device):
        '''
        compute gradient penalty term
        :param real_data:
        :param fake_data:
        :param y:
        :param L_gp:
        :param device:
        :return:
        '''

        batchsize = real_data.shape[0]
        real_data = real_data.to(device)
        fake_data = fake_data.to(device)
        y = y.to(device)
        alpha = torch.rand(batchsize, 1)
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, y)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * L_gp
        return gradient_penalty

class GeneratorCustomCNN(nn.Module):
    def __init__(self, z_dim=10, outact=nn.Sigmoid()):
        super(GeneratorCustomCNN, self).__init__()

        self.z_dim = z_dim
        self.num_classes = 100

        fc1 = nn.Linear(self.z_dim +  self.num_classes, 256)
        fc2 = nn.Linear(256, 256)
        fc3 = nn.Linear(256, 2 * 2 * 128)
        deconv1 = nn.ConvTranspose2d(128, 64, kernel_size = 5, stride=2)
        deconv2 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride=2)
        deconv3 = nn.ConvTranspose2d(32, 3, kernel_size = 3, stride=2, output_padding=1)
        BN1 = nn.BatchNorm2d(64)
        BN2 = nn.BatchNorm2d(32)
        BN3 = nn.BatchNorm2d(3)

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3
        self.relu = nn.ReLU()
        #self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.BN1 = BN1
        self.BN2 = BN2
        self.BN3 = BN3
        self.dropout = torch.nn.Dropout(p=0.2)
        #self.outact = outact

    def forward(self, z, y):

        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc1(z_in)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = output.view(-1, 128, 2, 2)

        output = self.dropout(output)

        output = self.deconv1(output)
        output = self.BN1(output)
        output = self.relu(output)
        output = self.deconv2(output)
        output = self.BN2(output)
        output = self.relu(output)
        output = self.deconv3(output)
        output = self.BN3(output)
        output = self.relu(output)

        return output.view(-1, 32 * 32 * 3)


class DiscriminatorCustomCNN(nn.Module):
    def __init__(self):
        super(DiscriminatorCustomCNN, self).__init__()


        self.num_classes = 100
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(2 * 2 * 128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.num_classes)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, input, y):

        input = input.view(-1, 3, 32, 32)
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        #out = TS(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


    def calc_gradient_penalty(self, real_data, fake_data, y, L_gp, device):
        '''
        compute gradient penalty term
        :param real_data:
        :param fake_data:
        :param y:
        :param L_gp:
        :param device:
        :return:
        '''

        batchsize = real_data.shape[0]
        real_data = real_data.to(device)
        fake_data = fake_data.to(device)
        y = y.to(device)
        alpha = torch.rand(batchsize, 1)
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, y)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * L_gp
        return gradient_penalty

class ResNetResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel=3, downsampling=1, conv_shortcut=False, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.kernel, self.downsampling = kernel, downsampling
        self.conv_shortcut= conv_shortcut
        self.activate = nn.ReLU()
        self.shortcut = nn.Conv2d(self.in_channels, filters *4, kernel_size=1,
                      stride=self.downsampling) if self.conv_shortcut else nn.MaxPool2d(kernel_size=1, stride=self.downsampling)

        self.BN_1 = nn.BatchNorm2d(self.in_channels, eps=1.001e-5)
        self.Conv_1 = nn.Conv2d(self.in_channels, filters, kernel_size=1, stride=1, bias=False)
        self.BN_2 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.zeroPad_1 = nn.ZeroPad2d((1,1,1,1))
        self.Conv_2 = nn.Conv2d(filters, filters, kernel_size=self.kernel, stride=self.downsampling, bias=False)
        self.BN_3 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.Conv_3 = nn.Conv2d(filters , filters *4, kernel_size=1)
        
    def forward(self, x):
        x = self.BN_1(x)
        x = self.activate(x)

        residual = self.shortcut(x)

        x = self.Conv_1(x)
        x = self.BN_2(x)
        x = self.activate(x)
        x = self.zeroPad_1(x)
        x = self.Conv_2(x)
        x = self.BN_3(x)
        x = self.activate(x)
        x = self.Conv_3(x)
        x += residual
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)

class Unflatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], 1, 28, 28)

class Unflatten_7(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1, 7, 7)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        '''
        self.label_emb = nn.Sequential(
            nn.Embedding(10, 50),
            nn.Linear(50, 784),
            Unflatten(),
        )

        self.model = nn.Sequential(
            nn.Conv2d(2, 128, 3, 2, 1), #[128, 14, 14]
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, 3, 2, 1), #[128, 7, 7]
            nn.LeakyReLU(),

            Flatten(),

            nn.Dropout(0.4),
            nn.Linear(7*7*128, 1),
            nn.Sigmoid(),
        )
        '''
        self.unflatten = Unflatten()
        self.emb = nn.Embedding(10, 50)
        self.linear_1 = nn.Linear(50, 784)

        self.conv_1 = nn.Conv2d(2, 128, 3, 2, 1)
        self.conv_2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.flatten = Flatten()
        self.linear_2 = nn.Linear(7*7*128, 1)

        self.apply(weights_init)

    def forward(self, imgs, labels):
        '''
        labels = self.label_emb(labels)
        data = torch.cat((imgs, labels), axis=1)
        '''
        labels = self.emb(labels)
        labels = self.linear_1(labels)
        labels = self.unflatten(labels)


        data = torch.cat((imgs, labels), axis=1)
        data = self.conv_1(data)
        data = nn.LeakyReLU()(data)
        data = self.conv_2(data)
        data = nn.LeakyReLU()(data)
        data = self.flatten(data)
        data = nn.Dropout(0.4)(data)
        data = self.linear_2(data)
        data = nn.Sigmoid()(data)
        return data

    def calc_gradient_penalty(self, real_data, fake_data, y, L_gp, device):
        '''
        compute gradient penalty term
        :param real_data:
        :param fake_data:
        :param y:
        :param L_gp:
        :param device:
        :return:
        '''

        batchsize = real_data.shape[0]
        real_data = real_data.to(device)
        fake_data = fake_data.to(device)
        y = y.to(device)
        alpha = torch.rand(batchsize, 1)
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, y)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * L_gp
        return gradient_penalty

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Sequential(
            nn.Embedding(10, 50),
            nn.Linear(50, 49),
            Unflatten_7(),
        )

        self.linear = nn.Sequential(
            nn.Linear(100, 7*7*128),
            nn.LeakyReLU(),
            Unflatten_7(),
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(129, 128, 4, 2, 1), #[128, 14, 14]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), #[128, 28, 28]
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, 7, 1, 3),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, z, labels):
        labels, linear = self.label_emb(labels), self.linear(z)
        data = torch.cat((linear, labels), axis=1)
        return self.model(data)
