from chainer.functions.array import concat
from chainer.functions.noise import dropout
from chainer.functions.pooling import average_pooling_2d as A
from chainer.functions.pooling import max_pooling_2d as M
from chainer import link
import chainer, cupy
import numpy as np
from chainercv.evaluations import eval_semantic_segmentation
import chainer.functions as F
from chainer.links.connection import convolution_2d as C
from chainer.links.connection import linear
from chainer.links.normalization import batch_normalization as B
from chainer.links.connection import dilated_convolution_2d as D
from chainer import Variable, cuda, initializers
import h5py, os


        
class DilatedConvBN(link.Chain):
    def __init__(self, ich, och, ksize, stride, pad, dilate, init_weights, pool=None):
        super(DilatedConvBN, self).__init__(
            conv = D.DilatedConvolution2D(ich, och, ksize, stride, pad, dilate, nobias=True),
            bn = B.BatchNormalization(och),
            )
        self.pool = pool
        if init_weights:
            f = h5py.File('%s/data/dump/%s.h5' % (os.getcwd(),init_weights),'r')
            self.conv.W.data  = np.array(f['weights']).transpose([3, 2, 0, 1])
            self.bn.beta.data = np.array(f['beta'])
            self.bn.gamma.data = np.array(f['gamma'])
            self.bn.avg_mean = np.array(f['mean'])
            self.bn.avg_var = np.array(f['var'])
    def __call__(self, x):
        if self.pool:
            x = self.pool.apply((x,))[0]
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ConvBN(link.Chain):
    def __init__(self, ich, och, ksize, stride, pad, init_weights, pool=None):
        super(ConvBN, self).__init__(
            conv = C.Convolution2D(ich, och, ksize, stride, pad, nobias=True),
            bn = B.BatchNormalization(och),
            )
        self.pool = pool
        if init_weights:
            f = h5py.File('%s/data/dump/%s.h5' % (os.getcwd(),init_weights),'r')
            self.conv.W.data  = np.array(f['weights']).transpose([3, 2, 0, 1])
            self.bn.beta.data = np.array(f['beta'])
            self.bn.gamma.data = np.array(f['gamma'])
            self.bn.avg_mean = np.array(f['mean'])
            self.bn.avg_var = np.array(f['var'])
    def __call__(self, x):
        if self.pool:
            x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Conv(link.Chain):
    def __init__(self, ich, och, ksize, stride, pad, init_weights, pool=None, nobias=False):
        super(Conv, self).__init__(
            conv = C.Convolution2D(ich, och, ksize, stride, pad, nobias),            
            )
        self.pool = pool
    def __call__(self, x):
        if self.pool:
            x = self.pool(x)
        x = self.conv(x)                
        return x


class Sequential(link.ChainList):
    def __call__(self, x, *args, **kwargs):
        for l in self:
            x = l(x, *args, **kwargs)
        return x


class Inception(link.ChainList):
    def __init__(self, *links, **kw):
        super(Inception, self).__init__(*links)
        self.pool = kw.get('pool', None)
    def __call__(self, x):
        xs = [l(x) for l in self]
        if self.pool:
            xs.append(self.pool(x))
        return concat.concat(xs)


class InceptionV3(link.Chain):
    def __init__(self, args, dilated=True):

        convolution = link.ChainList(ConvBN(3, 32, 3, 2, 0, 'conv'),
            ConvBN(32, 32, 3, 1, 0, 'conv_1'),
            ConvBN(32, 64, 3, 1, 1, 'conv_2'),
            ConvBN(64, 80, 1, 1, 0, 'conv_3'),
            ConvBN(80, 192, 3, 1, 0, 'conv_4'))


        def inception_35(ich, pch, name):
            # 1x1
            s1 = ConvBN(ich, 64, 1, 1, 0, name['1x1'][0])
            # 5x5
            s21 = ConvBN(ich, 48, 1, 1, 0, name['5x5'][0])
            s22 = ConvBN(48, 64, 5, 1, 2, name['5x5'][1])
            s2 = Sequential(s21, s22)
            # double 3x3
            s31 = ConvBN(ich, 64, 1, 1, 0, name['3x3'][0])
            s32 = ConvBN(64, 96, 3, 1, 1, name['3x3'][1])
            s33 = ConvBN(96, 96, 3, 1, 1, name['3x3'][2])
            s3 = Sequential(s31, s32, s33)
            # pool
            s4 = ConvBN(ich, pch, 1, 1, 0, name['pool'][1],
             pool=A.AveragePooling2D(3, 1, 1))
            return Inception(s1, s2, s3, s4)

        inception35_names = ({
        '1x1':['mixed_conv'], 
        '5x5':['mixed_tower_conv','mixed_tower_conv_1'],
        '3x3':['mixed_tower_1_conv','mixed_tower_1_conv_1','mixed_tower_1_conv_2'],
        'pool':['mixed_tower_2_pool','mixed_tower_2_conv']
        },
        {
        '1x1':['mixed_1_conv'], 
        '5x5':['mixed_1_tower_conv','mixed_1_tower_conv_1'],
        '3x3':['mixed_1_tower_1_conv','mixed_1_tower_1_conv_1','mixed_1_tower_1_conv_2'],
        'pool':['mixed_1_tower_2_pool','mixed_1_tower_2_conv']
        },
        {
        '1x1':['mixed_2_conv'], 
        '5x5':['mixed_2_tower_conv','mixed_2_tower_conv_1'],
        '3x3':['mixed_2_tower_1_conv','mixed_2_tower_1_conv_1','mixed_2_tower_1_conv_2'],
        'pool':['mixed_2_tower_2_pool','mixed_2_tower_2_conv']
        })               

        inception35 = Sequential(*[inception_35(ich, pch, name)
                                  for ich, pch, name
                                  in zip([192, 256, 288], [32, 64, 64], inception35_names)])
        
        reduction35 = Inception(
            # strided 3x3
            ConvBN(288, 384, 3, 2, 0, 'mixed_3_conv'), # originally stride-pad: 2-0
            # double 3x3
            Sequential(
                ConvBN(288, 64, 1, 1, 0, 'mixed_3_tower_conv'),
                ConvBN(64, 96, 3, 1, 1, 'mixed_3_tower_conv_1'),
                ConvBN(96, 96, 3, 2, 0, 'mixed_3_tower_conv_2') # originally stride-pad: 2-0
                ),
            # pool
            pool=M.MaxPooling2D(3, 2, 0, cover_all=False)) # originally stride-pad: 2-0


        def inception_17(hidden_channel, name):
            # 1x1
            s1 = ConvBN(768, 192, 1, 1, 0, name['1x1'][0])
            # 7x7
            s21 = ConvBN(768, hidden_channel, 1, 1, 0, name['7x7'][0])
            s22 = ConvBN(hidden_channel, hidden_channel, (1,7), (1,1), (0,3), name['7x7'][1])
            s23 = ConvBN(hidden_channel, 192, (7,1), (1,1), (3,0), name['7x7'][2])
            s2 = Sequential(s21, s22, s23)
            # double 7x7
            s31 = ConvBN(768, hidden_channel, 1, 1, 0, name['double7x7'][0])
            s32 = ConvBN(hidden_channel, hidden_channel, (7,1), (1,1), (3,0), name['double7x7'][1])
            s33 = ConvBN(hidden_channel, hidden_channel, (1,7), (1,1), (0,3), name['double7x7'][2])
            s34 = ConvBN(hidden_channel, hidden_channel, (7,1), (1,1), (3,0), name['double7x7'][3])
            s35 = ConvBN(hidden_channel, 192, (1,7), (1,1), (0,3), name['double7x7'][4])
            s3 = Sequential(s31, s32, s33, s34, s35)
            # pool
            s4 = ConvBN(768, 192, 1, 1, 0, name['pool'][1],
                pool=A.AveragePooling2D(3, 1, 1))
            return Inception(s1, s2, s3, s4)

        inception17_names = ({
        '1x1':['mixed_4_conv'], 
        '7x7':['mixed_4_tower_conv','mixed_4_tower_conv_1','mixed_4_tower_conv_2'],
        'double7x7':['mixed_4_tower_1_conv','mixed_4_tower_1_conv_1',
        'mixed_4_tower_1_conv_2','mixed_4_tower_1_conv_3','mixed_4_tower_1_conv_4'],
        'pool':['mixed_4_tower_2_pool','mixed_4_tower_2_conv']
        },
        {
        '1x1':['mixed_5_conv'], 
        '7x7':['mixed_5_tower_conv','mixed_5_tower_conv_1','mixed_5_tower_conv_2'],
        'double7x7':['mixed_5_tower_1_conv','mixed_5_tower_1_conv_1',
        'mixed_5_tower_1_conv_2','mixed_5_tower_1_conv_3','mixed_5_tower_1_conv_4'],
        'pool':['mixed_5_tower_2_pool','mixed_5_tower_2_conv']
        },        
        {
        '1x1':['mixed_6_conv'], 
        '7x7':['mixed_6_tower_conv','mixed_6_tower_conv_1','mixed_6_tower_conv_2'],
        'double7x7':['mixed_6_tower_1_conv','mixed_6_tower_1_conv_1',
        'mixed_6_tower_1_conv_2','mixed_6_tower_1_conv_3','mixed_6_tower_1_conv_4'],
        'pool':['mixed_6_tower_2_pool','mixed_6_tower_2_conv']
        },                
        {
        '1x1':['mixed_7_conv'], 
        '7x7':['mixed_7_tower_conv','mixed_7_tower_conv_1','mixed_7_tower_conv_2'],
        'double7x7':['mixed_7_tower_1_conv','mixed_7_tower_1_conv_1',
        'mixed_7_tower_1_conv_2','mixed_7_tower_1_conv_3','mixed_7_tower_1_conv_4'],
        'pool':['mixed_7_tower_2_pool','mixed_7_tower_2_conv']
        })               

        inception17 = Sequential(*[inception_17(c, name)
                                  for c, name in zip([128, 160, 160, 192], inception17_names)])



        if dilated:
            # Reduction 17 to 8
            reduction17 = Inception(
                # strided 3x3
                Sequential(
                    ConvBN(768, 192, 1, 1, 0, 'mixed_8_tower_conv'),
                    ConvBN(192, 320, 3, 1, 1, 'mixed_8_tower_conv_1')), # originally stride-pad: 2-0
                # 7x7 and 3x3
                Sequential(
                    ConvBN(768, 192, 1, 1, 0, 'mixed_8_tower_1_conv'),
                    ConvBN(192, 192, (1,7), (1,1), (0,3), 'mixed_8_tower_1_conv_1'),
                    ConvBN(192, 192, (7,1), (1,1), (3,0), 'mixed_8_tower_1_conv_2'),
                    ConvBN(192, 192, 3, 1, 1, 'mixed_8_tower_1_conv_3')), # originally stride-pad: 2-0
                # pool
                pool=M.MaxPooling2D(3, 1, 1, cover_all=False)) # originally stride-pad: 2-0

            def inception_8(input_channel, name):
                # 1x1
                s1 = ConvBN(input_channel, 320, 1, 1, 0, name['1x1'][0])
                # 3x3
                s21 = ConvBN(input_channel, 384, 1, 1, 0, name['3x3'][0])
                s22 = Inception(DilatedConvBN(384, 384, (1,3), (1,1), (0,2), (1,2), name['3x3'][1]),
                                DilatedConvBN(384, 384, (3,1), (1,1), (2,0), (2,1), name['3x3'][2]))
                s2 = Sequential(s21, s22)
                # double 3x3
                s31 = ConvBN(input_channel, 448, 1, 1, 0, name['double3x3'][0])
                s32 = DilatedConvBN(448, 384, 3, 1, 2, 2, name['double3x3'][1])
                s331 = DilatedConvBN(384, 384, (1,3), (1,1), (0,2), (1,2), name['double3x3'][2])
                s332 = DilatedConvBN(384, 384, (3,1), (1,1), (2,0), (2,1), name['double3x3'][3])
                s33 = Inception(s331, s332)
                s3 = Sequential(s31, s32, s33)
                # pool
                s4 = ConvBN(input_channel, 192, 1, 1, 0, name['pool'][1],
                             pool=A.AveragePooling2D(5, 1, 2))
                return Inception(s1, s2, s3, s4)
        else:
            # Reduction 17 to 8
            reduction17 = Inception(
                # strided 3x3
                Sequential(
                    ConvBN(768, 192, 1, 1, 0, 'mixed_8_tower_conv'),
                    ConvBN(192, 320, 3, 2, 0, 'mixed_8_tower_conv_1')), # originally stride-pad: 2-0
                # 7x7 and 3x3
                Sequential(
                    ConvBN(768, 192, 1, 1, 0, 'mixed_8_tower_1_conv'),
                    ConvBN(192, 192, (1,7), (1,1), (0,3), 'mixed_8_tower_1_conv_1'),
                    ConvBN(192, 192, (7,1), (1,1), (3,0), 'mixed_8_tower_1_conv_2'),
                    ConvBN(192, 192, 3, 2, 0, 'mixed_8_tower_1_conv_3')), # originally stride-pad: 2-0
                # pool
                pool=M.MaxPooling2D(3, 2, 0, cover_all=False)) # originally stride-pad: 2-0

            def inception_8(input_channel, name):
                # 1x1
                s1 = ConvBN(input_channel, 320, 1,1,0, name['1x1'][0])
                # 3x3
                s21 = ConvBN(input_channel, 384, 1,1,0, name['3x3'][0])
                s22 = Inception(ConvBN(384, 384, (1, 3),(1,1),(0, 1), name['3x3'][1]),
                                ConvBN(384, 384, (3, 1),(1,1),(1, 0), name['3x3'][2]))
                s2 = Sequential(s21, s22)
                # double 3x3
                s31 = ConvBN(input_channel, 448, 1,1,0, name['double3x3'][0])
                s32 = ConvBN(448, 384, 3, 1,1, name['double3x3'][1])
                s331 = ConvBN(384, 384, (1, 3),(1,1),(0, 1), name['double3x3'][2])
                s332 = ConvBN(384, 384, (3, 1), (1,1),(1, 0), name['double3x3'][3])
                s33 = Inception(s331, s332)
                s3 = Sequential(s31, s32, s33)
                # pool
                s4 = ConvBN(input_channel, 192, 1,1,0, name['pool'][1],
                             pool=A.AveragePooling2D(3, 1, 1))
                return Inception(s1, s2, s3, s4)                        

        inception8_names = ({
        '1x1':['mixed_9_conv'], 
        '3x3':['mixed_9_tower_conv','mixed_9_tower_mixed_conv','mixed_9_tower_mixed_conv_1'],
        'double3x3':['mixed_9_tower_1_conv','mixed_9_tower_1_conv_1',
        'mixed_9_tower_1_mixed_conv','mixed_9_tower_1_mixed_conv_1'],
        'pool':['mixed_9_tower_2_pool','mixed_9_tower_2_conv']
        },        
        {
        '1x1':['mixed_10_conv'], 
        '3x3':['mixed_10_tower_conv','mixed_10_tower_mixed_conv','mixed_10_tower_mixed_conv_1'],
        'double3x3':['mixed_10_tower_1_conv','mixed_10_tower_1_conv_1',
        'mixed_10_tower_1_mixed_conv','mixed_10_tower_1_mixed_conv_1'],
        'pool':['mixed_10_tower_2_pool','mixed_10_tower_2_conv']
        })               

        inception8 = Sequential(*[inception_8(input_channel, name)
                                  for input_channel, name in zip([1280, 2048],inception8_names)])        


        super(InceptionV3, self).__init__(
            convolution=convolution,            
            inception=link.ChainList(inception35, inception17, inception8),
            grid_reduction=link.ChainList(reduction35, reduction17),
            )        

    def __call__(self, x):

        def convolution(x):
            x = self.convolution[0](x)
            x = self.convolution[1](x)
            x = self.convolution[2](x)
            x = M.max_pooling_2d(x, 3, 2)
            x = self.convolution[3](x)
            x = self.convolution[4](x)
            x = M.max_pooling_2d(x, 3, 2)
            return x

        x = convolution(x)
        x = self.inception[0](x)
        x = self.grid_reduction[0](x)
        x = self.inception[1](x)
        x = self.grid_reduction[1](x)
        x = self.inception[2](x)
        return x

class Classifier(link.Chain):
    def __init__(self, label_dim):        
        super(Classifier, self).__init__(
            ASPP=link.ChainList(
                ConvBN(2048, 512, 1, 1, 0, init_weights=None, pool=None),
                ConvBN(2048, 512, 1, 1, 0, init_weights=None, pool=None),
                DilatedConvBN(2048, 256, 3, 1, 3, 3, init_weights=None, pool=None),
                DilatedConvBN(2048, 256, 3, 1, 6, 6, init_weights=None, pool=None),
                DilatedConvBN(2048, 256, 3, 1, 9, 9, init_weights=None, pool=None),
                DilatedConvBN(2048, 256, 3, 1, 12, 12, init_weights=None, pool=None),
                ConvBN(2048, 1024, 1, 1, 0, init_weights=None, pool=None)),
            classifier = link.ChainList(Conv(1024, label_dim, 1, 1, 0, init_weights=None, pool=None, nobias=False)),
        )
    def __call__(self, x):
                                
        def ASPP(x):
            y = [F.tile(self.ASPP[0](F.average_pooling_2d(x, ksize=x.shape[-2:])), x.shape[-2:])]
            y.extend([self.ASPP[i](x) for i in range(1,len(self.ASPP)-1)])
            y = F.concat(y, axis=1)
            y = self.ASPP[-1](y)
            return y

        x = ASPP(x)
        y = self.classifier[0](x)
        return y

class InceptionV3Classifier(link.Chain):
    def __init__(self, predictor, classifiers, args):
        super(InceptionV3Classifier, self).__init__(
            predictor=predictor,
            classifiers=link.ChainList(*classifiers)
            )
        self.args = args


class ReIDClassifier(link.Chain):
    def __init__(self, predictor, label_dim, args):
        super(ReIDClassifier, self).__init__(
            predictor=predictor,
            classifiers=link.ChainList(Conv(2048 * 3, label_dim, 1, 1, 0, init_weights=None, pool=None, nobias=False))
            )
        self.args = args

    def __call__(self, x, t, dataset, train=True):
        
        # Create variables
        x = Variable(x)
        x.to_gpu(self.gpu_id)
        t = Variable(t)
        t.to_gpu(self.gpu_id)

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                xo = self.segmentation.predictor(x)
                y = self.segmentation.classifiers[0](xo)                
                y = F.separate(F.softmax(y), axis=1)
                # foreground, head, torso-hand, lower-body, shoes
                segprob = F.stack((1.0-y[0],
                    y[1]+y[2]+y[4]+y[13],
                    y[5]+y[6]+y[7]+y[11]+y[10]+y[3]+y[14]+y[15],
                    y[9]+y[16]+y[17]+y[12],
                    y[18]+y[19]+y[8]), axis=1)   
        
        # Forward
        with chainer.using_config('train', train):
            with chainer.using_config('enable_backprop', train):
                x = F.resize_images(x,self.args.scales_reid)
                # InceptionV3 backbone                                
                x = self.predictor(x)
                x_a = F.average_pooling_2d(x, x.shape[-2:])
                # Resize to segmentation map resolution
                x = F.resize_images(x,segprob.shape[-2:])     
                # aggregate features at semantic parts
                xl = F.scale(
                    F.batch_matmul(
                    F.reshape(segprob,(segprob.shape[0], segprob.shape[1], -1)),
                    F.reshape(x,(x.shape[0], x.shape[1], -1)), 
                     transb=True),
                    1.0/F.sum(segprob, axis=(2,3)), axis=0)                
                
                xfg, xl = F.split_axis(xl, [1], axis=1)
                xl = F.max(xl, axis=1, keepdims=True)
                x = F.concat((xfg,xl), axis=2)
                # Classifiers                
                x_s = F.reshape(x, (-1, 2*2048, 1, 1))
                x = F.concat((x_s, x_a), axis=1)
                
                if train:
                    self.y_s = self.classifiers[0](x)
                    # Loss                
                    self.loss = F.softmax_cross_entropy(F.squeeze(self.y_s, axis=(2, 3)), t)

                    # Clear grads for uninitialized params
                    self.cleargrads()
                    # Backwards
                    self.loss.backward()

                    # Reporter        
                    self.reporter.update({dataset:{'loss':self.loss.data.tolist()}})        

                else:
                    x = F.squeeze(x)
                    x.to_cpu()
                    self.reporter.update({dataset:{'features':x.data}})
