#%%
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from utils import condrop
from utils.modules.transformation import TPS_SpatialTransformerNetwork
from utils.modules.feature_extraction import ResNet_FeatureExtractor
from utils.modules.sequence_modeling import BidirectionalLSTM
from utils.modules.prediction import Attention
from utils.condrop import ConcreteDropout

#%%
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class Concrete_conv3x3(nn.Module):
	def __init__(self,w,d, in_channels, v):
		super().__init__()
		self.cd = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
		self.conv = nn.Conv2d(in_channels,v,kernel_size=3, padding=1)
	def forward(self,x):
		return self.cd(x,self.conv)

class Concrete_linear(nn.Module):
	def __init__(self,w,d, in_, out_):
		super().__init__()
		self.cd = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
		self.linear = nn.Linear(in_,out_)

	def forward(self,x):
		return self.cd(x,self.linear)

def make_layers(cfg, w, d, batch_norm=False):

	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = Concrete_conv3x3(w,d,in_channels,v)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v

	return nn.Sequential(*layers)


class VGG(nn.Module):
	def __init__(self, w,d,features):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			Concrete_linear(w,d, 512 * 7 * 7, 4096),
			nn.ReLU(True),
			Concrete_linear(w,d, 4096, 4096),
			nn.ReLU(True),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

class extractor(nn.Module): # feature extractor stem
	def __init__(self, w, d, pretrained):
		super(extractor, self).__init__()

		vgg16_bn = VGG(w=w , d=d, features=make_layers(cfg, w, d, batch_norm=True))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('./utils/pths/vgg16_bn-6c64b313.pth'))
		self.features = vgg16_bn.features
	
	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		return out[1:]


class merge(nn.Module): #FCN
	def __init__(self,w,d):
		super().__init__()

		self.w , self.d = w,d

		self.cd1 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv1 = nn.Conv2d(1024, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.cd2 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.cd3 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.cd4 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.cd5 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv5 = nn.Conv2d(192, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.cd6 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.cd7 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU()
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x): # 논문의 feature merging branch - decoder
		y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True) #2배 업샘플링
		y = torch.cat((y, x[2]), 1)
		y = self.relu1(self.bn1(self.cd1(y,self.conv1)))

		y = self.relu2(self.bn2(self.cd2(y,self.conv2)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[1]), 1)
		y = self.relu3(self.bn3(self.cd3(y,self.conv3)))		
		y = self.relu4(self.bn4(self.cd4(y,self.conv4)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[0]), 1)
		y = self.relu5(self.bn5(self.cd5(y,self.conv5)))		
		y = self.relu6(self.bn6(self.cd6(y,self.conv6)))
		
		y = self.relu7(self.bn7(self.cd7(y,self.conv7)))
		return y

class output(nn.Module):
	def __init__(self, w, d, scope=512):
		super(output, self).__init__()

		self.w, self.d = w,d

		self.cd1 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.cd2 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv2 = nn.Conv2d(32, 4, 1)
		self.sigmoid2 = nn.Sigmoid()
		self.cd3 = ConcreteDropout(weight_regulariser=self.w, dropout_regulariser=self.d)
		self.conv3 = nn.Conv2d(32, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = scope
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		score = self.sigmoid1(self.cd1(x,self.conv1))
		loc   = self.sigmoid2(self.cd2(x,self.conv2)) * self.scope
		angle = (self.sigmoid3(self.cd3(x,self.conv3)) - 0.5) * math.pi #TODO Check
		geo   = torch.cat([loc, angle],axis=1)
		return score, geo
		

@condrop.concrete_regulariser
class EAST(nn.Module):
	def __init__(self, w,d, pretrained):
		super(EAST, self).__init__()
		self.extractor = extractor(w,d,pretrained)
		self.merge     = merge(w,d)
		self.output    = output(w=w,d=d)

	def forward(self, x):
		return self.output(self.merge(self.extractor(x)))


#해당모델은 recognition 파트 의 모델
class recognitionModel(nn.Module):

    def __init__(self, num_class):
        super(recognitionModel, self).__init__()
        self.num_class = num_class
        self.stages = {'Trans': True, 'Feat': True,
                       'Seq': True, 'Pred': True}

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork( # https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=worb1605&logNo=221580830661
            F=20, I_size=(32, 100), I_r_size=(32, 100), I_channel_num=1)

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(1, 512)
        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256
        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, 256, num_class)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=25)

        return prediction


if __name__ == '__main__':
	m = EAST()
	x = torch.randn(1, 3, 256, 256)
	score, geo = m(x)
	print(score.shape)
	print(geo.shape)
