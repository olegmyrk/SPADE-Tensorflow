import sys
from ops import *
from utils import *
import types
import time
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from tqdm import tqdm
from vgg19_keras import VGGLoss
from unet import unet
from masked_autoregressive import conditional_masked_autoregressive_template

class SPADE(object):
    def __init__(self, args):

        self.model_name = 'SPADE'
        self.train_main = args.train_main
        self.train_nondet = args.train_nondet

        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type
        self.code_gan_type = args.code_gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.code_dist_num_layers = args.code_dist_num_layers
        self.code_num_layers = args.code_num_layers

        self.init_lr = args.lr
        self.TTUR = args.TTUR
        self.ch = args.ch

        self.beta1 = args.beta1
        self.beta2 = args.beta2


        self.num_style = args.num_style
        self.guide_img = args.guide_img


        """ Weight """
        self.adv_weight = args.adv_weight
        self.vgg_weight = args.vgg_weight
        self.feature_weight = args.feature_weight
        self.kl_weight = args.kl_weight
        self.ce_weight = args.ce_weight

        self.ld = args.ld

        """ Generator """
        self.num_upsampling_layers = args.num_upsampling_layers

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale
        self.code_n_critic = args.code_n_critic
        self.n_critic = args.n_critic
        self.sn = args.sn

        self.img_height = args.img_height
        self.img_width = args.img_width

        self.img_ch = args.img_ch
        self.segmap_img_ch = args.segmap_img_ch

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)


        self.dataset_path = os.path.join('./dataset', self.dataset_name)

        self.writer = tf.summary.create_file_writer(self.log_dir + '/' + self.model_dir)

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# TTUR : ", self.TTUR)

        print()

        print("##### Generator #####")
        print("# upsampling_layers : ", self.num_upsampling_layers)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print("# multi-scale : ", self.n_scale)
        print("# the number of critic : ", self.n_critic)
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# kl_weight : ", self.kl_weight)
        print("# vgg_weight : ", self.vgg_weight)
        print("# feature_weight : ", self.feature_weight)
        print("# wgan lambda : ", self.ld)
        print("# beta1 : ", self.beta1)
        print("# beta2 : ", self.beta2)

        print()

    ##################################################################################
    # Generator
    ##################################################################################.

    def encoder_base(self, x_init, channel):
        #x = resize(x_init, self.img_height, self.img_width)
        x = x_init
        
        x = constin_resblock(x, channel, use_bias=True, sn=self.sn, norm=False, scope='preresblock')

        #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv')
        #x = instance_norm(x, scope='ins_norm')
        x = constin_resblock(x, channel*2, use_bias=True, sn=self.sn, scope='resblock')
        x = down_sample_avg(x)

        for i in range(3):
            #x = lrelu(x, 0.2)
            #x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv_' + str(i))
            #x = instance_norm(x, scope='ins_norm_' + str(i))
            x = constin_resblock(x, channel * 4, use_bias=True, sn=self.sn, scope='resblock_' + str(i))
            x = down_sample_avg(x)

            channel = channel * 2

            # 128, 256, 512

        #x = lrelu(x, 0.2)
        #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv_3')
        #x = instance_norm(x, scope='ins_norm_3')
        x = constin_resblock(x, channel*8, use_bias=True, sn=self.sn, scope='resblock_3')
        x = down_sample_avg(x)

        #if self.img_height >= 256 or self.img_width >= 256 :
        if self.img_height >= 256 or self.img_width >= 256 :
            #x = lrelu(x, 0.2)
            #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=False, sn=self.sn, scope='conv_4')
            #x = instance_norm(x, scope='ins_norm_4')
            x = constin_resblock(x, channel*8, use_bias=False, sn=self.sn, scope='resblock_4')
            x = down_sample_avg(x)

        x = lrelu(x, 0.2)

        return x, channel

    def encoder_code(self, x_init, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x, channel = self.encoder_base(x_init, self.ch)

            mean = fully_connected(x, channel // 2, use_bias=True, sn=False, scope='linear_mean')
            var = fully_connected(x, channel // 2, use_bias=True, sn=False, scope='linear_var')
            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def prior_code(self, batch_size, channel_multiplier=4):
        out_channel = self.ch * channel_multiplier
        mean = tf.zeros([batch_size, out_channel])
        var = tf.zeros([batch_size, out_channel])
        return mean, var

    def prior_code_dist(self, code, channel_multiplier=4, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        context = code
        out_channel = self.ch * channel_multiplier
        hidden_channel = self.ch * 64

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            bijectors = []
            for i in range(self.code_dist_num_layers):
                bijectors.append(tfb.MaskedAutoregressiveFlow(
                  #shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                  shift_and_log_scale_fn=conditional_masked_autoregressive_template(code,
                      hidden_layers=[hidden_channel, hidden_channel], name=scope + "/maf_" + str(i))))

                bijectors.append(tfb.BatchNormalization(
                    batchnorm_layer=tf.compat.v1.layers.BatchNormalization(
                                        name=scope + '/batch_norm_' + str(i)),
                    name=scope + '/batch_norm_bijector' + str(i)))

                permutation=tf.compat.v1.get_variable('permutation_'+str(i), dtype=tf.int32, initializer=np.random.permutation(out_channel).astype("int32"), trainable=False)
                bijectors.append(tfb.Permute(permutation))
                
            flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

            mvn_loc = fully_connected(context, units=out_channel, sn=False, scope='mvn_loc')
            mvn_scale_diag = epsilon + tf.math.sigmoid(fully_connected(context, units=out_channel, sn=False, scope='mvn_scale_logdiag'))
            mvn_dist = tfd.MultivariateNormalDiag(mvn_loc, mvn_scale_diag, name=scope + "/MultivariateNormalDiag")

            dist = tfd.TransformedDistribution(
                            distribution=mvn_dist,
                            bijector=flow_bijector
                        )
        return dist

    def encoder_supercode(self, x_init, channel_multiplier=4, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        out_channel = self.ch*channel_multiplier
        hidden_channel = self.ch*64
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x = x_init
            for i in range(self.code_num_layers):
                x = constin_fcblock(x, hidden_channel, scope="fcblock_" + str(i))
                x = lrelu(x, 0.2)

            mean = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_mean')
            var = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_var')

            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def generator_code(self, code, x_init, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        out_channel = self.ch*4
        hidden_channel = self.ch*64
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x = x_init
            for i in range(self.code_num_layers):
                x = adain_fcblock(code, x, hidden_channel, scope="fcblock_" + str(i))
                x = lrelu(x, 0.2)

            mean = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_mean')
            #var = get_trainable_variable("var", [], initializer=tf.compat.v1.constant_initializer(0.0))
            var = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_var')

            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def generator(self, code, z=None, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        context = code

        context_depth = 8
        context_ch = 10*context.get_shape()[-1]
        channel = self.ch * 4 * 4
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            features = []

            #for i in range(context_depth):
            #    context = fully_connected(context, context_ch, use_bias=True, sn=self.sn, scope='linear_context_' + str(i))
            #    context = lrelu(context, 0.2)
            
            x = fully_connected(z, z.get_shape()[-1], use_bias=True, sn=False, scope='linear_noise')

            if self.num_upsampling_layers == 'less':
                num_up_layers = 4
            elif self.num_upsampling_layers == 'normal':
                num_up_layers = 5
            elif self.num_upsampling_layers == 'more':
                num_up_layers = 6
            elif self.num_upsampling_layers == 'most':
                num_up_layers = 7

            z_width = self.img_width // (pow(2, num_up_layers))
            z_height = self.img_height // (pow(2, num_up_layers))

            """
            # If num_up_layers = 5 (normal)
            
            # 64x64 -> 2
            # 128x128 -> 4
            # 256x256 -> 8
            # 512x512 -> 16
            
            """
            x = fully_connected(x, units=z_height * z_width * channel, use_bias=True, sn=False, scope='linear_x')
            x = tf.reshape(x, [-1, z_height, z_width, channel])


            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_0')
            features.append(x)

            x = up_sample(x, scale_factor=2)
            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_1')
            features.append(x)

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_2')
            features.append(x)

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = adain_resblock(context, x, channels=channel//2, use_bias=True, sn=self.sn, scope='resblock_' + str(i))
                features.append(x)

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')
            x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')
            features.append(x)

            x = lrelu(x, 0.2)
            mean = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_mean')
            var = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_var')
            logits = conv(x, channels=self.segmap_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_logits')

            return features, [[tanh(mean), tf.math.log(epsilon + tf.sigmoid(var))], logits]

    def generator_features(self, code, features, z, reuse=False, scope=None):
        context = code
        features = list(reversed(features))

        context_depth = 8
        context_ch = 10*context.get_shape()[-1]
        channel = self.ch * 4 * 4
        with tf.compat.v1.variable_scope(scope, reuse=reuse):

            #for i in range(context_depth):
            #    context = fully_connected(context, context_ch, use_bias=True, sn=False, scope='linear_context_' + str(i))
            #    context = lrelu(context, 0.2)
            
            x = fully_connected(z, z.get_shape()[-1], use_bias=True, sn=False, scope='linear_noise')

            if self.num_upsampling_layers == 'less':
                num_up_layers = 4
            elif self.num_upsampling_layers == 'normal':
                num_up_layers = 5
            elif self.num_upsampling_layers == 'more':
                num_up_layers = 6
            elif self.num_upsampling_layers == 'most':
                num_up_layers = 7

            z_width = self.img_width // (pow(2, num_up_layers))
            z_height = self.img_height // (pow(2, num_up_layers))

            """
            # If num_up_layers = 5 (normal)
            
            # 64x64 -> 2
            # 128x128 -> 4
            # 256x256 -> 8
            # 512x512 -> 16
            
            """

            x = fully_connected(x, units=z_height * z_width * channel, use_bias=True, sn=False, scope='linear_x')
            x = tf.reshape(x, [-1, z_height, z_width, channel])

            x = cprogressive_resblock(context, features.pop(), x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_0')

            x = up_sample(x, scale_factor=2)
            x = cprogressive_resblock(context, features.pop(), x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_1')

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = cprogressive_resblock(context, features.pop(), x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_2')

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = cprogressive_resblock(context, features.pop(), x, channels=channel//2, use_bias=True, sn=self.sn, scope='resblock_' + str(i))

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = cprogressive_resblock(context, features.pop(), x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')
            x = cprogressive_resblock(context, features.pop(), x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')

            x = lrelu(x, 0.2)
            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='logit')
            x = tanh(x)

            return x

    def generator_spatial(self, code, scaffold, z, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        context = code

        context_depth = 8
        context_ch = 10*context.get_shape()[-1]
        channel = self.ch * 4 * 4
        with tf.compat.v1.variable_scope(scope, reuse=reuse):

            #for i in range(context_depth):
            #    context = fully_connected(context, context_ch, use_bias=True, sn=False, scope='linear_context_' + str(i))
            #    context = lrelu(context, 0.2)
            
            x = fully_connected(z, z.get_shape()[-1], use_bias=True, sn=False, scope='linear_noise')

            if self.num_upsampling_layers == 'less':
                num_up_layers = 4
            elif self.num_upsampling_layers == 'normal':
                num_up_layers = 5
            elif self.num_upsampling_layers == 'more':
                num_up_layers = 6
            elif self.num_upsampling_layers == 'most':
                num_up_layers = 7

            z_width = self.img_width // (pow(2, num_up_layers))
            z_height = self.img_height // (pow(2, num_up_layers))

            """
            # If num_up_layers = 5 (normal)
            
            # 64x64 -> 2
            # 128x128 -> 4
            # 256x256 -> 8
            # 512x512 -> 16
            
            """

            x = fully_connected(x, units=z_height * z_width * channel, use_bias=True, sn=False, scope='linear_x')
            x = tf.reshape(x, [-1, z_height, z_width, channel])

            x = cspade_resblock(context, scaffold, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_0')
            #x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_0')

            x = up_sample(x, scale_factor=2)
            x = cspade_resblock(context, scaffold, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_1')
            #x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_1')

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = cspade_resblock(context, scaffold, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_2')
            #x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_2')

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = cspade_resblock(context, scaffold, x, channels=channel//2, use_bias=True, sn=self.sn, scope='resblock_' + str(i))
                #x = adain_resblock(context, x, channels=channel//2, use_bias=True, sn=self.sn, scope='resblock_' + str(i))

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = cspade_resblock(context, scaffold, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')
            x = cspade_resblock(context, scaffold, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')
            #x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')

            x = lrelu(x, 0.2)
            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator_code(self, x, reuse=tf.compat.v1.AUTO_REUSE, scope=None, label=None):
        channel = x.get_shape()[-1]
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x = constin_fcblock(x, channel * 10, use_bias=True, sn=self.sn, norm=False, scope='linear_x_1')

            x = constin_fcblock(x, channel * 10, use_bias=True, sn=self.sn, norm=True, scope='linear_x_2')

            x = constin_fcblock(x, channel * 10, use_bias=True, sn=self.sn, norm=True, scope='linear_x_3')

            x = constin_fcblock(x, channel * 10, use_bias=True, sn=self.sn, norm=True, scope='linear_x_4')

            z = fully_connected(x, 1, sn=self.sn, scope='linear_z')

            return [[z]]

    def full_discriminator(self, x_init, code=None, reuse=tf.compat.v1.AUTO_REUSE, scope=None, label=None):
        channel = self.ch
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            feature_loss = []
            x = x_init
            
            #x = adain_resblock(code, x, channel, use_bias=True, sn=self.sn, norm=False, scope='preresblock')
            x = constin_resblock(x, channel, use_bias=True, sn=self.sn, norm=False, scope='preresblock')

            #x = conv(scaffold, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv')
            #x = instance_norm(x, scope='ins_norm')
            #x = adain_resblock(code, x, channel * 2, use_bias=True, sn=self.sn, norm=True, scope='resblock')
            x = constin_resblock(x, channel * 2, use_bias=True, sn=self.sn, norm=True, scope='resblock')
            x = down_sample_avg(x)
            
            feature_loss.append(x)

            for i in range(3):
                #x = lrelu(x, 0.2)
                #x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='resblock_' + str(i))
                #x = instance_norm(x, scope='ins_norm_' + str(i))
                #x = adain_resblock(code, x, channel * 4, use_bias=True, sn=self.sn, norm=True, scope='resblock_' + str(i))
                x = constin_resblock(x, channel * 4, use_bias=True, sn=self.sn, norm=True, scope='resblock_' + str(i))
                x = down_sample_avg(x)

                feature_loss.append(x)

                channel = channel * 2

                # 128, 256, 512

            #x = lrelu(x, 0.2)
            #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='resblock_3')
            #x = instance_norm(x, scope='ins_norm_3')
            #x = adain_resblock(0*code, x, channel*4, use_bias=True, sn=self.sn, norm=True, scope='resblock_3')
            x = constin_resblock(x, channel*4, use_bias=True, sn=self.sn, norm=True, scope='resblock_3')
            x = down_sample_avg(x)

            #if self.img_height >= 256 or self.img_width >= 256 :
            if self.img_height >= 256 or self.img_width >= 256 :
                #x = lrelu(x, 0.2)
                #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=False, sn=self.sn, scope='resblock_4')
                #x = instance_norm(x, scope='ins_norm_4')
                #x = adain_resblock(0*code, x, channel*8, use_bias=False, sn=self.sn, norm=True, scope='resblock_4')
                x = constin_resblock(x, channel*8, use_bias=False, sn=self.sn, norm=True, scope='resblock_4')
                x = down_sample_avg(x)
                
                feature_loss.append(x)

            x0 = fully_connected(x, channel, use_bias=True, sn=self.sn, scope='linear_x0')
            x0 = lrelu(x0, 0.2)
                
            z0 = fully_connected(x0, 1, sn=self.sn, scope='linear_z0')
            
            z = tf.reshape(z0, [-1, 1, 1, 1])

            D_logit = [feature_loss + [z]]
            return D_logit

    def feature_discriminator(self, x_init, code, reuse=tf.compat.v1.AUTO_REUSE, scope=None, label=None):
        D_logit = []
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            for scale in range(self.n_scale):
                feature_loss = []
                channel = self.ch
                x = x_init
            
                #x = adain_resblock(code, x, channel, use_bias=True, sn=self.sn, norm=False, scope='ms_' + str(scale) + '_preresblock')
                x = constin_resblock(x, channel, use_bias=True, sn=self.sn, norm=False, scope='preresblock')

                #x = conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, sn=False, scope='ms_' + str(scale) + 'conv_0')
                #x = lrelu(x, 0.2)
                #x = adain_resblock(code, x, channel, use_bias=True, sn=self.sn, scope='ms_' + str(scale) + '_resblock')
                x = constin_resblock(x, channel, use_bias=True, sn=self.sn, scope='ms_' + str(scale) + '_resblock')
                x = down_sample_avg(x)

                feature_loss.append(x)

                for i in range(1, self.n_dis):
                    #stride = 1 if i == self.n_dis - 1 else 2
                    #x = conv(x, channel * 2, kernel=4, stride=stride, pad=1, use_bias=True, sn=self.sn, scope='ms_' + str(scale) + 'conv_' + str(i))
                    #x = instance_norm(x, scope='ms_' + str(scale) + 'ins_norm_' + str(i))
                    #x = lrelu(x, 0.2)

                    #x = adain_resblock(code, x, channel*2, use_bias=True, sn=self.sn, scope='ms_' + str(scale) + 'resblock_' + str(i))
                    x = constin_resblock(x, channel*2, use_bias=True, sn=self.sn, scope='ms_' + str(scale) + 'resblock_' + str(i))
                    if i !=  self.n_dis - 1:
                        x = down_sample_avg(x)

                    feature_loss.append(x)

                    channel = min(channel * 2, 512)


                x = conv(x, channels=1, kernel=4, stride=1, pad=1, use_bias=True, sn=self.sn, scope='ms_' + str(scale) + 'D_logit')

                feature_loss.append(x)
                D_logit.append(feature_loss)

                x_init = down_sample_avg(x_init)

            return D_logit

    ##################################################################################
    # Model
    ##################################################################################

    def discriminate_code(self, real_code_img, fake_code_img, reuse=tf.compat.v1.AUTO_REUSE, name=None):
        real_logit = self.discriminator_code(real_code_img, reuse=reuse, scope='discriminator_' + name + '_code', label='real_' + name + '_code')
        fake_logit = self.discriminator_code(fake_code_img, reuse=True, scope='discriminator_' + name + '_code', label='fake_' + name + '_code')

        return real_logit, fake_logit

    def discriminate(self, real_img, fake_img, code=None, reuse=tf.compat.v1.AUTO_REUSE, name=None):
        real_logit = self.discriminator(real_img, code, reuse=reuse, scope='discriminator_' + name + '', label='real_' + name + '')
        fake_logit = self.discriminator(fake_img, code, reuse=True, scope='discriminator_' + name + '', label='fake_' + name + '')

        return real_logit, fake_logit

    def gradient_penalty(self, real, fake, code, discriminator, name=None):
        shape = tf.shape(input=real)
        if self.gan_type == 'dragan':
            eps = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(x=real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            alpha = tf.random.uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else:
            alpha = tf.random.uniform(shape=[shape[0], 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha * real + (1. - alpha) * fake

        logit = discriminator(interpolated, code=code, reuse=True, scope='discriminator_' + name + '', label='interpolated_' + name + '')

        GP = []


        for i in range(len(logit)) :
            grad = tf.gradients(ys=logit[i][-1], xs=interpolated)[0]  # gradient of D(interpolated)
            grad_norm = tf.norm(tensor=flatten(grad), axis=1)  # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp':
                GP.append(self.ld * tf.reduce_mean(input_tensor=tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                GP.append(self.ld * tf.reduce_mean(input_tensor=tf.square(grad_norm - 1.)))

        return tf.reduce_mean(input_tensor=GP)

    def prepare_dataset(self):
        """ Input Image"""
        self.img_class = Image_data(self.img_height, self.img_width, self.img_ch, self.segmap_img_ch, self.dataset_path, self.augment_flag)
        self.img_class.preprocess()
        self.color_value_dict = self.img_class.color_value_dict
        self.segmap_ch = len(self.img_class.color_value_dict)

    def prepare_model(self):
        self.vgg_loss = VGGLoss() 

    def execute_model(self, global_step, real_ctx, real_x, real_x_segmap, real_x_segmap_onehot):
        batch_size = tf.shape(real_ctx)[0]

        """ Define Generator, Discriminator """
        prior_det_code_mean, prior_det_code_logvar = self.prior_code(batch_size)
        x_det_code_mean, x_det_code_logvar = self.encoder_code(real_x, scope='encoder_det_code')
        fake_det_x_code = z_sample(x_det_code_mean, x_det_code_logvar)
       
        supercode_stop_gradient = lambda x: x
        code_stop_gradient = lambda x: x

        x_det_supercode_mean, x_det_supercode_logvar = self.encoder_supercode(code_stop_gradient(fake_det_x_code), channel_multiplier=4, scope='encoder_det_supercode')
        fake_det_x_supercode = z_sample(x_det_supercode_mean, x_det_supercode_logvar)
        x_det_ctxcode_mean, x_det_ctxcode_logvar = self.encoder_code(real_ctx, scope='encoder_det_ctxcode')
        fake_det_x_ctxcode = z_sample(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        x_det_ctxcodex_mean, x_det_ctxcodex_logvar = map(tf.stop_gradient, self.encoder_code(real_x, reuse=True, scope='encoder_det_ctxcode'))
        fake_det_x_ctxcodex = z_sample(x_det_ctxcodex_mean, x_det_ctxcodex_logvar)

        x_det_codectx_mean, x_det_codectx_logvar = map(tf.stop_gradient, self.encoder_code(real_ctx, reuse=True, scope='encoder_det_code'))
        fake_det_x_codectx = z_sample(x_det_codectx_mean, x_det_codectx_logvar)
        fake_det_x_full_ctxcode = tf.concat([fake_det_x_ctxcode, 0*fake_det_x_codectx],-1)
        fake_det_x_code_mean, fake_det_x_code_logvar = self.generator_code(fake_det_x_full_ctxcode, fake_det_x_supercode, scope="generator_det_code")
        
        resample_det_supercode = fake_det_x_supercode#z_sample(*self.prior_code(batch_size))
        prior_det_supercode_mean, prior_det_supercode_logvar = self.prior_code(batch_size, channel_multiplier=4)#self.encoder_code(real_x, scope='prior_det_supercode')
        #random_det_supercode = z_sample(prior_det_supercode_mean, prior_det_supercode_logvar)
        prior_det_supercode_dist = self.prior_code_dist(fake_det_x_full_ctxcode, channel_multiplier=4, scope='prior_det_supercode')
        random_det_supercode = prior_det_supercode_dist.sample()

        prior_det_ctxcode_mean, prior_det_ctxcode_logvar = self.prior_code(batch_size)
        #random_det_ctxcode = z_sample(prior_det_ctxcode_mean, prior_det_ctxcode_logvar)

        resample_det_code_mean, resample_det_code_var = self.generator_code(fake_det_x_full_ctxcode, resample_det_supercode, reuse=True, scope="generator_det_code")
        resample_det_code = z_sample(resample_det_code_mean, resample_det_code_var)
        random_det_code_mean, random_det_code_var = self.generator_code(fake_det_x_full_ctxcode, random_det_supercode, reuse=True, scope="generator_det_code")
        random_det_code = z_sample(random_det_code_mean, random_det_code_var)
        random_gen_det_supercode = z_sample(prior_det_supercode_mean, prior_det_supercode_logvar)
        random_gen_det_code = self.generator_code(tf.stop_gradient(fake_det_x_full_ctxcode), random_gen_det_supercode, scope="generator_gen_det_code")[0]

        random_gaussian_det_code = z_sample(*self.prior_code(batch_size))
        
        prior_nondet_code_mean, prior_nondet_code_logvar = self.prior_code(batch_size)

        x_nondet_code_mean, x_nondet_code_logvar = self.encoder_code(real_x, scope='encoder_nondet_code')
        fake_nondet_x_code = z_sample(x_nondet_code_mean, x_nondet_code_logvar)

        x_nondet_supercode_mean, x_nondet_supercode_logvar = self.encoder_supercode(code_stop_gradient(fake_nondet_x_code), channel_multiplier=4, scope='encoder_nondet_supercode')
        fake_nondet_x_supercode = z_sample(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar = self.encoder_code(real_ctx, scope='encoder_nondet_ctxcode')
        fake_nondet_x_ctxcode = z_sample(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        x_nondet_ctxcodex_mean, x_nondet_ctxcodex_logvar = map(tf.stop_gradient, self.encoder_code(real_x, reuse=True, scope='encoder_nondet_ctxcode'))
        fake_nondet_x_ctxcodex = z_sample(x_nondet_ctxcodex_mean, x_nondet_ctxcodex_logvar)

        x_nondet_codectx_mean, x_nondet_codectx_logvar = map(tf.stop_gradient, self.encoder_code(real_ctx, reuse=True, scope='encoder_nondet_code'))
        fake_nondet_x_codectx = z_sample(x_nondet_codectx_mean, x_nondet_codectx_logvar)
        fake_nondet_x_full_ctxcode = tf.concat([fake_nondet_x_ctxcode, 0*fake_nondet_x_codectx],-1)
        fake_nondet_x_code_mean, fake_nondet_x_code_logvar = self.generator_code(fake_nondet_x_full_ctxcode, fake_nondet_x_supercode, scope="generator_nondet_code")
        
        resample_nondet_supercode = fake_nondet_x_supercode#z_sample(*self.prior_code(batch_size))
        prior_nondet_supercode_mean, prior_nondet_supercode_logvar = self.prior_code(batch_size, channel_multiplier=4)#self.encoder_code(real_x, scope='prior_nondet_supercode')
        #random_nondet_supercode = z_sample(prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        prior_nondet_supercode_dist = self.prior_code_dist(fake_nondet_x_full_ctxcode, channel_multiplier=4, scope='prior_nondet_supercode')
        random_nondet_supercode = prior_nondet_supercode_dist.sample()
        
        prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar = self.prior_code(batch_size)
        #random_nondet_ctxcode = z_sample(prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar)
        
        resample_nondet_code_mean, resample_nondet_code_var = self.generator_code(fake_nondet_x_full_ctxcode, resample_nondet_supercode, reuse=True, scope="generator_nondet_code")
        resample_nondet_code = z_sample(resample_nondet_code_mean, resample_nondet_code_var)
        random_nondet_code_mean, random_nondet_code_var = self.generator_code(fake_nondet_x_full_ctxcode, random_nondet_supercode, reuse=True, scope="generator_nondet_code")
        random_nondet_code = z_sample(random_nondet_code_mean, random_nondet_code_var)
        random_gen_nondet_supercode = z_sample(prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        random_gen_nondet_code = self.generator_code(tf.stop_gradient(fake_nondet_x_full_ctxcode), random_gen_nondet_supercode, scope="generator_gen_nondet_code")[0]

        random_gaussian_nondet_code = z_sample(*self.prior_code(batch_size))

        fake_full_det_x_code = tf.concat([fake_det_x_code, fake_det_x_full_ctxcode],-1)
        fake_full_det_x_z = tf.concat([fake_det_x_code],-1)
        fake_det_x_features, fake_det_x_stats = self.generator(fake_full_det_x_code, z=fake_full_det_x_z, scope="generator_det_x")
        fake_det_x_scaffold = fake_det_x_features#tf.concat([fake_det_x_stats[0][0], fake_det_x_stats[1]], -1)
        fake_det_x_mean, fake_det_x_var = fake_det_x_stats[0]
        fake_det_x_segmap_logits = fake_det_x_stats[1]
        
        fake_full_nondet_x_code = tf.concat([fake_nondet_x_code, fake_nondet_x_full_ctxcode, tf.stop_gradient(fake_full_det_x_code)],-1) 
        fake_full_nondet_x_z = tf.concat([fake_nondet_x_code, tf.stop_gradient(fake_full_det_x_z)],-1) 
        fake_full_nondet_x_discriminator_code = tf.concat([fake_nondet_x_code, fake_nondet_x_full_ctxcode, tf.stop_gradient(fake_det_x_code), tf.stop_gradient(fake_det_x_full_ctxcode)],-1) 
        #fake_nondet_x_output = self.generator_spatial(fake_full_nondet_x_code, tf.stop_gradient(fake_det_x_scaffold), z=fake_full_nondet_x_z, reuse=True, scope="generator_nondet_x")
        fake_nondet_x_output = self.generator_features(fake_full_nondet_x_code, list(map(tf.stop_gradient, fake_det_x_features)), z=fake_full_nondet_x_z, reuse=True, scope="generator_nondet_x")

        random_full_det_x_code = tf.concat([random_det_code, fake_det_x_full_ctxcode], -1)
        random_full_det_x_z = tf.concat([random_det_code], -1)
        random_fake_det_x_features, random_fake_det_x_stats = self.generator(random_full_det_x_code, z=random_full_det_x_z, reuse=True, scope="generator_det_x")
        random_fake_det_x_scaffold = random_fake_det_x_features#tf.concat([random_fake_det_x_stats[0][0], random_fake_det_x_stats[1]], -1)
        random_fake_det_x_mean, random_fake_det_x_var = random_fake_det_x_stats[0]
        random_fake_det_x_segmap_logits = random_fake_det_x_stats[1]

        random_full_nondet_x_code = tf.concat([random_nondet_code, fake_nondet_x_full_ctxcode, random_full_det_x_code], -1) 
        random_full_nondet_x_z = tf.concat([random_nondet_code, random_full_det_x_z], -1) 
        #random_fake_nondet_x_output = self.generator_spatial(random_full_nondet_x_code, random_fake_det_x_scaffold, z=random_full_nondet_x_z, reuse=True, scope="generator_nondet_x")
        random_fake_nondet_x_output = self.generator_features(random_full_nondet_x_code, random_fake_det_x_features, z=random_full_nondet_x_z, reuse=True, scope="generator_nondet_x")

        random_gen_full_det_x_code = tf.concat([random_gen_det_code, fake_det_x_full_ctxcode], -1)
        random_gen_full_det_x_z = tf.concat([random_gen_det_code], -1)
        random_gen_fake_det_x_features, random_gen_fake_det_x_stats = self.generator(random_gen_full_det_x_code, z=random_gen_full_det_x_z, reuse=True, scope="generator_det_x")
        random_gen_fake_det_x_scaffold = random_gen_fake_det_x_features#tf.concat([random_gen_fake_det_x_stats[0][0], random_gen_fake_det_x_stats[1]], -1)
        random_gen_fake_det_x_mean, random_gen_fake_det_x_var = random_gen_fake_det_x_stats[0]
        random_gen_fake_det_x_segmap_logits = random_gen_fake_det_x_stats[1]

        random_gen_full_nondet_x_code = tf.concat([random_gen_nondet_code, fake_nondet_x_full_ctxcode, random_gen_full_det_x_code], -1) 
        random_gen_full_nondet_x_z = tf.concat([random_gen_nondet_code, random_gen_full_det_x_z], -1) 
        #random_gen_fake_nondet_x_output = self.generator_spatial(random_gen_full_nondet_x_code, random_gen_fake_det_x_scaffold, z=random_gen_full_nondet_x_z, reuse=True, scope="generator_nondet_x")
        random_gen_fake_nondet_x_output = self.generator_features(random_gen_full_nondet_x_code, random_gen_fake_det_x_features, z=random_gen_full_nondet_x_z, reuse=True, scope="generator_nondet_x")

        resample_full_det_x_code = tf.concat([resample_det_code, fake_det_x_full_ctxcode], -1)
        resample_full_det_x_z = tf.concat([resample_det_code], -1)
        resample_fake_det_x_features, resample_fake_det_x_stats = self.generator(resample_full_det_x_code, z=resample_full_det_x_z, reuse=True, scope="generator_det_x")
        resample_fake_det_x_scaffold = resample_fake_det_x_features#tf.concat([resample_fake_det_x_stats[0][0], resample_fake_det_x_stats[1]], -1)
        resample_fake_det_x_mean, resample_fake_det_x_var = resample_fake_det_x_stats[0]
        resample_fake_det_x_segmap_logits = resample_fake_det_x_stats[1]

        resample_full_nondet_x_code = tf.concat([resample_nondet_code, fake_nondet_x_full_ctxcode, resample_full_det_x_code], -1)
        resample_full_nondet_x_z = tf.concat([resample_nondet_code, resample_full_det_x_z], -1)
        #resample_fake_nondet_x_output = self.generator_spatial(resample_full_nondet_x_code, resample_fake_det_x_scaffold, z=resample_full_nondet_x_z, reuse=True, scope="generator_nondet_x")
        resample_fake_nondet_x_output = self.generator_features(resample_full_nondet_x_code, resample_fake_det_x_features, z=resample_full_nondet_x_z, reuse=True, scope="generator_nondet_x")

        code_det_prior_real_logit, code_det_prior_fake_logit = self.discriminate_code(real_code_img=tf.concat([tf.stop_gradient(fake_det_x_full_ctxcode), tf.stop_gradient(random_gaussian_det_code)], -1), fake_code_img=tf.concat([tf.stop_gradient(fake_det_x_full_ctxcode), fake_det_x_code], -1), name='det_prior')
        code_nondet_prior_real_logit, code_nondet_prior_fake_logit = self.discriminate_code(real_code_img=tf.concat([tf.stop_gradient(fake_nondet_x_full_ctxcode), tf.stop_gradient(random_gaussian_nondet_code)], -1), fake_code_img=tf.concat([tf.stop_gradient(fake_nondet_x_full_ctxcode), fake_nondet_x_code], -1), name='nondet_prior')

        code_det_gen_real_logit, code_det_gen_fake_logit = self.discriminate_code(real_code_img=tf.concat([tf.stop_gradient(fake_det_x_full_ctxcode), tf.stop_gradient(fake_det_x_code)], -1), fake_code_img=tf.concat([tf.stop_gradient(fake_det_x_full_ctxcode), random_gen_det_code], -1), name='det_gen')
        code_nondet_gen_real_logit, code_nondet_gen_fake_logit = self.discriminate_code(real_code_img=tf.concat([tf.stop_gradient(fake_nondet_x_full_ctxcode), tf.stop_gradient(fake_nondet_x_code)], -1), fake_code_img=tf.concat([tf.stop_gradient(fake_nondet_x_full_ctxcode), random_gen_nondet_code], -1), name='nondet_gen')
  
        discriminator_fun = self.full_discriminator
        nondet_real_logit = discriminator_fun(tf.concat([tf.stop_gradient(tf.nn.softmax(fake_det_x_segmap_logits)), real_x, tf.stop_gradient(fake_det_x_mean)], -1), fake_full_nondet_x_discriminator_code, scope='discriminator_nondet_x', label='real_nondet_x')
        nondet_fake_logit = discriminator_fun(tf.concat([tf.stop_gradient(tf.nn.softmax(fake_det_x_segmap_logits)), fake_nondet_x_output, tf.stop_gradient(fake_det_x_mean)], -1), fake_full_nondet_x_discriminator_code, reuse=True, scope='discriminator_nondet_x', label='fake_nondet_x')
        
        if self.gan_type.__contains__('wgan-') or self.gan_type == 'dragan':
            GP = self.gradient_penalty(real=tf.concat([0*real_ctx, real_x, tf.stop_gradient(fake_det_x_mean)], -1), fake=tf.concat([real_ctx, fake_nondet_x_output, tf.stop_gradient(fake_det_x_mean)],-1), code=fake_full_nondet_x_discriminator_code, discriminator=discriminator_fun, name='nondet_x')
        else:
            GP = 0

        """ Define Loss """
        #g_nondet_ce_loss = L1_loss(real_x, fake_nondet_x_output)
        g_nondet_ce_loss =  gaussian_loss(fake_nondet_x_output, fake_det_x_mean, fake_det_x_var)
        g_nondet_vgg_loss = self.vgg_loss(fake_nondet_x_output, real_x)
        g_nondet_adv_loss = generator_loss(self.gan_type, nondet_fake_logit)
        g_nondet_feature_loss = feature_loss(nondet_real_logit, nondet_fake_logit)
        g_nondet_reg_loss = regularization_loss('generator_nondet_x')

        #g_det_ce_loss = L2_loss(real_x, fake_det_x_mean)
        g_det_ce_loss = gaussian_loss(real_x, fake_det_x_mean, fake_det_x_var)
        g_det_segmapce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_x_segmap_onehot, logits=fake_det_x_stats[1]))
        g_det_vgg_loss = self.vgg_loss(real_x, fake_det_x_stats[0][0])
        g_det_reg_loss = regularization_loss('generator_det_x')

        #g_nondet_code_ce_loss = L2_mean_loss(code_stop_gradient(fake_nondet_x_code), fake_nondet_x_code_mean)
        g_nondet_code_ce_loss = gaussian_loss(code_stop_gradient(fake_nondet_x_code), fake_nondet_x_code_mean, fake_nondet_x_code_logvar)
        e_nondet_code_kl_loss = kl_loss(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        e_nondet_code_prior_loss = gaussian_loss(fake_nondet_x_supercode, prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        e_nondet_code_prior2_loss = -tf.reduce_mean(prior_nondet_supercode_dist.log_prob(supercode_stop_gradient(fake_nondet_x_supercode))) / int(fake_nondet_x_supercode.get_shape()[-1])
        e_nondet_code_negent_loss = negent_loss(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        #e_nondet_code_kl2_loss = kl_loss2(x_nondet_supercode_mean, x_nondet_supercode_logvar, prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        e_nondet_code_kl2_loss = (e_nondet_code_prior2_loss + e_nondet_code_negent_loss)
                
        e_nondet_prior_adv_loss = generator_loss(self.code_gan_type, code_nondet_prior_fake_logit)
        e_nondet_gen_adv_loss = generator_loss(self.code_gan_type, code_nondet_gen_fake_logit)
        e_nondet_kl_loss = kl_loss(x_nondet_code_mean, x_nondet_code_logvar)
        e_nondet_prior_loss = gaussian_loss(fake_nondet_x_code, prior_nondet_code_mean, prior_nondet_code_logvar)
        e_nondet_negent_loss = negent_loss(x_nondet_code_mean, x_nondet_code_logvar)
        e_nondet_prior2_loss = gaussian_loss(tf.stop_gradient(fake_nondet_x_code), prior_nondet_code_mean, prior_nondet_code_logvar)
        e_nondet_crosskl_loss = kl_loss2(x_nondet_codectx_mean, x_nondet_codectx_logvar, x_nondet_code_mean, x_nondet_code_logvar)
        e_nondet_rcrosskl_loss = kl_loss2(x_nondet_code_mean, x_nondet_code_logvar, x_nondet_codectx_mean, x_nondet_codectx_logvar)
        e_nondet_crossws_loss = gaussian_wasserstein2_loss(x_nondet_codectx_mean, x_nondet_codectx_logvar, x_nondet_code_mean, x_nondet_code_logvar)
        e_nondet_kl2_loss = (e_nondet_prior2_loss + e_nondet_negent_loss)
        e_nondet_reg_loss = regularization_loss('encoder_nondet_code')
        
        e_nondet_klctx_loss = kl_loss(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        e_nondet_crossklctx_loss = kl_loss2(x_nondet_ctxcodex_mean, x_nondet_ctxcodex_logvar, x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        e_nondet_rcrossklctx_loss = kl_loss2(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar, x_nondet_ctxcodex_mean, x_nondet_ctxcodex_logvar)
        e_nondet_crosswsctx_loss = gaussian_wasserstein2_loss(x_nondet_ctxcodex_mean, x_nondet_ctxcodex_logvar, x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        e_nondet_priorctx_loss = gaussian_loss(fake_nondet_x_ctxcode, prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar)
        e_nondet_negentctx_loss = negent_loss(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        #e_nondet_klctx2_loss = kl_loss2(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar, prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar)
        e_nondet_klctx2_loss = (e_nondet_priorctx_loss + e_nondet_negentctx_loss)

        #g_det_code_ce_loss = L2_mean_loss(code_stop_gradient(fake_det_x_code), fake_det_x_code_mean)
        g_det_code_ce_loss = gaussian_loss(code_stop_gradient(fake_det_x_code), fake_det_x_code_mean, fake_det_x_code_logvar)
        e_det_code_kl_loss = kl_loss(x_det_supercode_mean, x_det_supercode_logvar)
        e_det_code_prior_loss = gaussian_loss(fake_det_x_supercode, prior_det_supercode_mean, prior_det_supercode_logvar)
        e_det_code_prior2_loss = -tf.reduce_mean(input_tensor=prior_det_supercode_dist.log_prob(supercode_stop_gradient(fake_det_x_supercode))) / int(fake_det_x_supercode.get_shape()[-1])
        e_det_code_negent_loss = negent_loss(x_det_supercode_mean, x_det_supercode_logvar)
        #e_det_code_kl2_loss = kl_loss2(x_det_supercode_mean, x_det_supercode_logvar, prior_det_supercode_mean, prior_det_supercode_logvar)
        e_det_code_kl2_loss = (e_det_code_prior2_loss + e_det_code_negent_loss) 

        e_det_prior_adv_loss = generator_loss(self.code_gan_type, code_det_prior_fake_logit)
        e_det_gen_adv_loss = generator_loss(self.code_gan_type, code_det_gen_fake_logit)
        e_det_kl_loss = kl_loss(x_det_code_mean, x_det_code_logvar)
        e_det_prior_loss = gaussian_loss(fake_det_x_code, prior_det_code_mean, prior_det_code_logvar)
        e_det_negent_loss = negent_loss(x_det_code_mean, x_det_code_logvar)
        e_det_prior2_loss = gaussian_loss(tf.stop_gradient(fake_det_x_code), prior_det_code_mean, prior_det_code_logvar) 
        e_det_crosskl_loss = kl_loss2(x_det_codectx_mean, x_det_codectx_logvar, x_det_code_mean, x_det_code_logvar)
        e_det_rcrosskl_loss = kl_loss2(x_det_code_mean, x_det_code_logvar, x_det_codectx_mean, x_det_codectx_logvar)
        e_det_crossws_loss = gaussian_wasserstein2_loss(x_det_codectx_mean, x_det_codectx_logvar, x_det_code_mean, x_det_code_logvar)
        e_det_kl2_loss = (e_det_prior2_loss + e_det_negent_loss)
        e_det_reg_loss = regularization_loss('encoder_det_code')
        
        e_det_klctx_loss = kl_loss(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        e_det_crossklctx_loss = kl_loss2(x_det_ctxcodex_mean, x_det_ctxcodex_logvar, x_det_ctxcode_mean, x_det_ctxcode_logvar)
        e_det_rcrossklctx_loss = kl_loss2(x_det_ctxcode_mean, x_det_ctxcode_logvar, x_det_ctxcodex_mean, x_det_ctxcodex_logvar)
        e_det_crosswsctx_loss = gaussian_wasserstein2_loss(x_det_ctxcodex_mean, x_det_ctxcodex_logvar, x_det_ctxcode_mean, x_det_ctxcode_logvar)
        e_det_priorctx_loss = gaussian_loss(fake_det_x_ctxcode, prior_det_ctxcode_mean, prior_det_ctxcode_logvar)
        e_det_negentctx_loss = negent_loss(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        #e_det_klctx2_loss = kl_loss2(x_det_ctxcode_mean, x_det_ctxcode_logvar, prior_det_ctxcode_mean, prior_det_ctxcode_logvar)
        e_det_klctx2_loss = (e_det_priorctx_loss + e_det_negentctx_loss)

        d_nondet_adv_loss = discriminator_loss(self.gan_type, nondet_real_logit, nondet_fake_logit)
        d_nondet_score_real, d_nondet_score_fake = discriminator_scores(nondet_real_logit, nondet_fake_logit)
        d_nondet_score_diff = -d_nondet_score_real + d_nondet_score_fake
        d_nondet_reg_loss = GP + regularization_loss('discriminator_nondet_x')

        de_det_prior_adv_loss = discriminator_loss(self.code_gan_type, code_det_prior_real_logit, code_det_prior_fake_logit)
        de_det_prior_reg_loss = regularization_loss('discriminator_det_prior_code')

        de_nondet_prior_adv_loss = discriminator_loss(self.code_gan_type, code_nondet_prior_real_logit, code_nondet_prior_fake_logit)
        de_nondet_prior_reg_loss = regularization_loss('discriminator_nondet_prior_code')

        de_det_gen_adv_loss = discriminator_loss(self.code_gan_type, code_det_gen_real_logit, code_det_gen_fake_logit)
        de_det_gen_reg_loss = regularization_loss('discriminator_det_gen_code')

        de_nondet_gen_adv_loss = discriminator_loss(self.code_gan_type, code_nondet_gen_real_logit, code_nondet_gen_fake_logit)
        de_nondet_gen_reg_loss = regularization_loss('discriminator_nondet_gen_code')

        gp_nondet_loss = (0*e_nondet_prior_loss + e_nondet_prior2_loss + (g_nondet_code_ce_loss + (0*e_nondet_code_prior_loss + e_nondet_code_prior2_loss + e_nondet_code_negent_loss)) + e_nondet_negent_loss) + 0.0001*e_nondet_klctx2_loss
        g_nondet_loss = g_nondet_adv_loss + g_nondet_reg_loss + 10*g_nondet_feature_loss + 0*g_nondet_vgg_loss + tf.zeros_like(g_nondet_ce_loss) + 0*e_nondet_prior_adv_loss + e_nondet_reg_loss + e_nondet_gen_adv_loss + gp_nondet_loss
        de_nondet_loss = de_nondet_prior_adv_loss + de_nondet_prior_reg_loss + de_nondet_gen_adv_loss + de_nondet_gen_reg_loss
        d_nondet_loss = d_nondet_adv_loss + d_nondet_reg_loss

        gp_det_loss = (0*e_det_prior_loss + e_det_prior2_loss + (g_det_code_ce_loss + (0*e_det_code_prior_loss + e_det_code_prior2_loss + e_det_code_negent_loss)) + e_det_negent_loss) + 0.0001*e_det_klctx2_loss
        g_det_loss = 10*g_det_ce_loss + 10*g_det_segmapce_loss + 0*g_det_vgg_loss + g_det_reg_loss + 0*e_det_prior_adv_loss + e_det_reg_loss + e_det_gen_adv_loss + gp_det_loss 
        de_det_loss = de_det_prior_adv_loss + de_det_prior_reg_loss + de_det_gen_adv_loss + de_det_gen_reg_loss

        """ Result Image """
        fake_det_x = fake_det_x_stats[0][0]
        fake_det_x_var = tf.exp(fake_det_x_stats[0][1])
        fake_det_x_segmap = tfd.Categorical(logits=fake_det_x_stats[1]).sample()
        fake_nondet_x = fake_nondet_x_output
        resample_fake_det_x = resample_fake_det_x_stats[0][0]
        resample_fake_det_x_segmap = tfd.Categorical(logits=resample_fake_det_x_stats[1]).sample()
        resample_fake_nondet_x = resample_fake_nondet_x_output
        random_fake_det_x = random_fake_det_x_stats[0][0]
        random_fake_det_x_segmap = tfd.Categorical(logits=random_fake_det_x_stats[1]).sample()
        random_fake_nondet_x = random_fake_nondet_x_output
        random_gen_fake_det_x = random_gen_fake_det_x_stats[0][0]
        random_gen_fake_det_x_segmap = tfd.Categorical(logits=random_gen_fake_det_x_stats[1]).sample()
        random_gen_fake_nondet_x = random_gen_fake_nondet_x_output

        """" Summary """
        def summaries_det():
            summary_global_step = tf.summary.scalar("global_step", global_step, step=global_step)

            summary_g_det_loss = tf.summary.scalar("g_det_loss", g_det_loss, step=global_step)
            summary_gp_det_loss = tf.summary.scalar("gp_det_loss", gp_det_loss, step=global_step)
            summary_de_det_loss = tf.summary.scalar("de_det_loss", de_det_loss, step=global_step)

            summary_g_det_code_ce_loss = tf.summary.scalar("g_det_code_ce_loss", g_det_code_ce_loss, step=global_step)
            summary_e_det_code_kl_loss = tf.summary.scalar("e_det_code_kl_loss", e_det_code_kl_loss, step=global_step)
            summary_e_det_klctx_loss = tf.summary.scalar("e_det_klctx_loss", e_det_klctx_loss, step=global_step)
            summary_e_det_code_kl2_loss = tf.summary.scalar("e_det_code_kl2_loss", e_det_code_kl2_loss, step=global_step)
            summary_e_det_klctx2_loss = tf.summary.scalar("e_det_klctx2_loss", e_det_klctx2_loss, step=global_step)
            summary_e_det_code_prior_loss = tf.summary.scalar("e_det_code_prior_loss", e_det_code_prior_loss, step=global_step)
            summary_e_det_code_prior2_loss = tf.summary.scalar("e_det_code_prior2_loss", e_det_code_prior2_loss, step=global_step)
            summary_e_det_priorctx_loss = tf.summary.scalar("e_det_priorctx_loss", e_det_priorctx_loss, step=global_step)
            summary_e_det_code_negent_loss = tf.summary.scalar("e_det_code_negent_loss", e_det_code_negent_loss, step=global_step)
            summary_e_det_negentctx_loss = tf.summary.scalar("e_det_negentctx_loss", e_det_negentctx_loss, step=global_step)

            summary_e_det_crossklctx_loss = tf.summary.scalar("e_det_crossklctx_loss", e_det_crossklctx_loss, step=global_step)
            summary_e_det_rcrossklctx_loss = tf.summary.scalar("e_det_rcrossklctx_loss", e_det_rcrossklctx_loss, step=global_step)
            summary_e_det_crosswsctx_loss = tf.summary.scalar("e_det_crosswsctx_loss", e_det_crosswsctx_loss, step=global_step)

            summary_e_det_kl_loss = tf.summary.scalar("e_det_kl_loss", e_det_kl_loss, step=global_step)
            summary_e_det_negent_loss = tf.summary.scalar("e_det_negent_loss", e_det_negent_loss, step=global_step)
            summary_e_det_prior_loss = tf.summary.scalar("e_det_prior_loss", e_det_prior_loss, step=global_step)
            summary_e_det_prior2_loss = tf.summary.scalar("e_det_prior2_loss", e_det_prior2_loss, step=global_step)
            summary_e_det_crosskl_loss = tf.summary.scalar("e_det_crosskl_loss", e_det_crosskl_loss, step=global_step)
            summary_e_det_rcrosskl_loss = tf.summary.scalar("e_det_rcrosskl_loss", e_det_rcrosskl_loss, step=global_step)
            summary_e_det_crossws_loss = tf.summary.scalar("e_det_crossws_loss", e_det_crossws_loss, step=global_step)
            summary_e_det_kl2_loss = tf.summary.scalar("e_det_kl2_loss", e_det_kl2_loss, step=global_step)
            summary_e_det_reg_loss = tf.summary.scalar("det_e_reg_loss", e_det_reg_loss, step=global_step)
            summary_e_det_prior_adv_loss = tf.summary.scalar("e_det_prior_adv_loss", e_det_prior_adv_loss, step=global_step)

            summary_g_det_ce_loss = tf.summary.scalar("g_det_ce_loss", g_det_ce_loss, step=global_step)
            summary_g_det_ce_loss = tf.summary.scalar("g_det_segmapce_loss", g_det_segmapce_loss, step=global_step)
            summary_g_det_vgg_loss = tf.summary.scalar("g_det_vgg_loss", g_det_vgg_loss, step=global_step)
            summary_g_det_reg_loss = tf.summary.scalar("g_det_reg_loss", g_det_reg_loss, step=global_step)

            summary_de_det_prior_adv_loss = tf.summary.scalar("de_det_prior_adv_loss", de_det_prior_adv_loss, step=global_step)
            summary_de_det_prior_reg_loss = tf.summary.scalar("de_det_prior_reg_loss", de_det_prior_reg_loss, step=global_step)

            summary_de_det_gen_adv_loss = tf.summary.scalar("de_det_gen_adv_loss", de_det_gen_adv_loss, step=global_step)
            summary_de_det_gen_reg_loss = tf.summary.scalar("de_det_gen_reg_loss", de_det_gen_reg_loss, step=global_step)

        def summaries_nondet():
            summary_g_nondet_loss = tf.summary.scalar("g_nondet_loss", g_nondet_loss, step=global_step)
            summary_gp_nondet_loss = tf.summary.scalar("gp_nondet_loss", gp_nondet_loss, step=global_step)
            summary_de_nondet_loss = tf.summary.scalar("de_nondet_loss", de_nondet_loss, step=global_step)
            summary_d_nondet_loss = tf.summary.scalar("d_nondet_loss", d_nondet_loss, step=global_step)

            summary_g_nondet_code_ce_loss = tf.summary.scalar("g_nondet_code_ce_loss", g_nondet_code_ce_loss, step=global_step)
            summary_e_nondet_code_kl_loss = tf.summary.scalar("e_nondet_code_kl_loss", e_nondet_code_kl_loss, step=global_step)
            summary_e_nondet_klctx_loss = tf.summary.scalar("e_nondet_klctx_loss", e_nondet_klctx_loss, step=global_step)
            summary_e_nondet_code_kl2_loss = tf.summary.scalar("e_nondet_code_kl2_loss", e_nondet_code_kl2_loss, step=global_step)
            summary_e_nondet_klctx2_loss = tf.summary.scalar("e_nondet_klctx2_loss", e_nondet_klctx2_loss, step=global_step)
            summary_e_nondet_code_prior_loss = tf.summary.scalar("e_nondet_code_prior_loss", e_nondet_code_prior_loss, step=global_step)
            summary_e_nondet_code_prior2_loss = tf.summary.scalar("e_nondet_code_prior2_loss", e_nondet_code_prior2_loss, step=global_step)
            summary_e_nondet_priorctx_loss = tf.summary.scalar("e_nondet_priorctx_loss", e_nondet_priorctx_loss, step=global_step)
            summary_e_nondet_code_negent_loss = tf.summary.scalar("e_nondet_code_negent_loss", e_nondet_code_negent_loss, step=global_step)
            summary_e_nondet_negentctx_loss = tf.summary.scalar("e_nondet_negentctx_loss", e_nondet_negentctx_loss, step=global_step)

            summary_e_nondet_crossklctx_loss = tf.summary.scalar("e_nondet_crossklctx_loss", e_nondet_crossklctx_loss, step=global_step)
            summary_e_nondet_rcrossklctx_loss = tf.summary.scalar("e_nondet_rcrossklctx_loss", e_nondet_rcrossklctx_loss, step=global_step)
            summary_e_nondet_crosswsctx_loss = tf.summary.scalar("e_nondet_crosswsctx_loss", e_nondet_crosswsctx_loss, step=global_step)

            summary_e_nondet_kl_loss = tf.summary.scalar("e_nondet_kl_loss", e_nondet_kl_loss, step=global_step)
            summary_e_nondet_negent_loss = tf.summary.scalar("e_nondet_negent_loss", e_nondet_negent_loss, step=global_step)
            summary_e_nondet_prior_loss = tf.summary.scalar("e_nondet_prior_loss", e_nondet_prior_loss, step=global_step)
            summary_e_nondet_prior2_loss = tf.summary.scalar("e_nondet_prior2_loss", e_nondet_prior2_loss, step=global_step)
            summary_e_nondet_crosskl_loss = tf.summary.scalar("e_nondet_crosskl_loss", e_nondet_crosskl_loss, step=global_step)
            summary_e_nondet_rcrosskl_loss = tf.summary.scalar("e_nondet_rcrosskl_loss", e_nondet_rcrosskl_loss, step=global_step)
            summary_e_nondet_crossws_loss = tf.summary.scalar("e_nondet_crossws_loss", e_nondet_crossws_loss, step=global_step)
            summary_e_nondet_kl2_loss = tf.summary.scalar("e_nondet_kl2_loss", e_nondet_kl2_loss, step=global_step)
            summary_e_nondet_reg_loss = tf.summary.scalar("e_nondet_reg_loss", e_nondet_reg_loss, step=global_step)
            summary_e_nondet_prior_adv_loss = tf.summary.scalar("e_nondet_prior_adv_loss", e_nondet_prior_adv_loss, step=global_step)

            summary_g_nondet_ce_loss = tf.summary.scalar("g_nondet_ce_loss", g_nondet_ce_loss, step=global_step)
            summary_g_nondet_vgg_loss = tf.summary.scalar("g_nondet_vgg_loss", g_nondet_vgg_loss, step=global_step)
            summary_g_nondet_feature_loss = tf.summary.scalar("g_nondet_feature_loss", g_nondet_feature_loss, step=global_step)
            summary_g_nondet_reg_loss = tf.summary.scalar("g_nondet_reg_loss", g_nondet_reg_loss, step=global_step)
            summary_g_nondet_adv_loss = tf.summary.scalar("g_nondet_adv_loss", g_nondet_adv_loss, step=global_step)
        
            summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_adv_loss", d_nondet_adv_loss, step=global_step)
            summary_d_nondet_reg_loss = tf.summary.scalar("d_nondet_reg_loss", d_nondet_reg_loss, step=global_step)
            
            summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_score_real", d_nondet_score_real, step=global_step)
            summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_score_fake", d_nondet_score_fake, step=global_step)
            summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_score_diff", d_nondet_score_diff, step=global_step)
            
            summary_de_nondet_prior_adv_loss = tf.summary.scalar("de_nondet_prior_adv_loss", de_nondet_prior_adv_loss, step=global_step)
            summary_de_nondet_prior_reg_loss = tf.summary.scalar("de_nondet_prior_reg_loss", de_nondet_prior_reg_loss, step=global_step)
       
            summary_de_nondet_gen_adv_loss = tf.summary.scalar("de_nondet_gen_adv_loss", de_nondet_gen_adv_loss, step=global_step)
            summary_de_nondet_gen_reg_loss = tf.summary.scalar("de_nondet_gen_reg_loss", de_nondet_gen_reg_loss, step=global_step)
 
        inputs = (real_ctx, real_x, real_x_segmap, real_x_segmap_onehot)
            
        losses_det = types.SimpleNamespace()
        losses_det.g_det_loss = g_det_loss
        losses_det.gp_det_loss = gp_det_loss
        losses_det.de_det_loss = de_det_loss

        losses_nondet = types.SimpleNamespace()
        losses_nondet.g_nondet_loss = g_nondet_loss
        losses_nondet.gp_nondet_loss = gp_nondet_loss
        losses_nondet.de_nondet_loss = de_nondet_loss
        losses_nondet.d_nondet_loss = d_nondet_loss
       
        outputs_det = (fake_det_x, fake_det_x_var, fake_det_x_segmap)

        outputs_resample_det = (resample_fake_det_x, resample_fake_det_x_segmap)
        outputs_random_det = (random_fake_det_x, random_fake_det_x_segmap)
        outputs_random_gen_det = (resample_fake_det_x, random_gen_fake_det_x_segmap)

        outputs_nondet = (fake_nondet_x,)

        outputs_resample_nondet = (resample_fake_nondet_x,)
        outputs_random_nondet = (random_fake_nondet_x,)
        outputs_random_gen_nondet = (random_gen_fake_nondet_x,)

        return inputs, (losses_det, outputs_det, summaries_det), (outputs_resample_det, outputs_random_det, outputs_random_gen_det), (losses_nondet, outputs_nondet, summaries_nondet), (outputs_resample_nondet, outputs_random_nondet, outputs_random_gen_nondet) 

    def build_fake_inputs(self, batch_size):
        real_ctx = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, self.img_height, self.img_width, self.img_ch)))
        real_x = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, self.img_height, self.img_width, self.img_ch)))
        real_x_segmap = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, self.img_height, self.img_width)))
        real_x_segmap_onehot = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, self.img_height, self.img_width, self.segmap_ch)))
        inputs = (real_ctx, real_x, real_x_segmap, real_x_segmap_onehot)
        return inputs
  
    def build_optimizers(self):
        class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_learning_rate, max_step, decay_step, name=None):
                self.initial_learning_rate = initial_learning_rate
                self.max_step = max_step
                self.decay_step = decay_step
                self.name = name

            def __call__(self, step):
                with tf.name_scope(self.name or "LinearDecay") as name:
                    initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
                    dtype = initial_learning_rate.dtype
                    max_step = tf.cast(self.max_step, dtype)
                    decay_step = tf.cast(self.decay_step, dtype)
                    step_recomp = tf.cast(step, dtype)
                    multiplier = tf.cond(
                            tf.math.less(step_recomp, decay_step), 
                            lambda: 1.0,
                            lambda: tf.math.divide(tf.math.subtract(step_recomp, decay_step), tf.math.subtract(max_step, decay_step))
                        )
                    return tf.math.multiply(initial_learning_rate, multiplier, name=name)

        if self.decay_flag:
            decay_fn = lambda lr: LinearDecay(lr, self.epoch*self.iteration, self.decay_epoch*self.iteration)
        else:
            decay_fn = lambda lr: lr
        if self.TTUR :
            beta1 = 0.0
            beta2 = 0.9

            g_nondet_lr = decay_fn(self.init_lr / 2 / float(self.n_critic))
            gp_nondet_lr = decay_fn(self.init_lr / float(self.n_critic))
            d_nondet_lr = decay_fn(self.init_lr * 2)
            de_nondet_lr = decay_fn(self.init_lr * 2)
            g_det_lr = decay_fn(self.init_lr / 2 / float(self.n_critic))
            gp_det_lr = decay_fn(self.init_lr / float(self.n_critic))
            de_det_lr = decay_fn(self.init_lr * 2)

        else :
            beta1 = self.beta1
            beta2 = self.beta2
            g_nondet_lr = decay_fn(self.init_lr / float(self.n_critic))
            gp_nondet_lr = decay_fn(self.init_lr / float(self.n_critic))
            de_nondet_lr = decay_fn(self.init_lr)
            d_nondet_lr = decay_fn(self.init_lr)
            g_det_lr = decay_fn(self.init_lr / float(self.n_critic))
            gp_det_lr = decay_fn(self.init_lr / float(self.n_critic))
            de_det_lr = decay_fn(self.init_lr)

        self.G_nondet_optim = tf.keras.optimizers.Adam(g_nondet_lr, beta_1=beta1, beta_2=beta2)
        self.GP_nondet_optim = tf.keras.optimizers.Adam(gp_nondet_lr, beta_1=beta1, beta_2=beta2)
        self.DE_nondet_optim = tf.keras.optimizers.Adam(de_nondet_lr, beta_1=beta1, beta_2=beta2)

        self.D_nondet_optim = tf.keras.optimizers.Adam(d_nondet_lr, beta_1=beta1, beta_2=beta2)
        self.G_det_optim = tf.keras.optimizers.Adam(g_det_lr, beta_1=beta1, beta_2=beta2)
        self.GP_det_optim = tf.keras.optimizers.Adam(gp_det_lr, beta_1=beta1, beta_2=beta2)
        self.DE_det_optim = tf.keras.optimizers.Adam(de_det_lr, beta_1=beta1, beta_2=beta2)

        t_vars = tf.compat.v1.trainable_variables()
        self.G_nondet_vars = [var for var in t_vars if 'generator_nondet_x' in var.name or 'encoder_nondet_code' in var.name or 'generator_gen_nondet_code' in var.name or 'encoder_nondet_ctxcode' in var.name]
        self.GP_nondet_vars = [var for var in t_vars if 'generator_nondet_code' in var.name or 'prior_nondet_supercode' in var.name or 'encoder_nondet_supercode' in var.name or 'encoder_nondet_ctxcode' in var.name]
        self.DE_nondet_vars = [var for var in t_vars if 'discriminator_nondet_prior_code' in var.name or 'discriminator_nondet_gen_code' in var.name]
        self.D_nondet_vars = [var for var in t_vars if 'discriminator_nondet_x' in var.name]

        self.G_det_vars = [var for var in t_vars if 'generator_det_x' in var.name or 'encoder_det_code' in var.name in var.name or 'generator_gen_det_code' in var.name or 'encoder_det_ctxcode' in var.name]
        self.GP_det_vars = [var for var in t_vars if 'generator_det_code' in var.name or 'prior_det_supercode' in var.name or 'encoder_det_supercode' in var.name or 'encoder_det_ctxcode' in var.name]
        self.DE_det_vars = [var for var in t_vars if 'discriminator_det_prior_code' in var.name or 'discriminator_det_gen_code' in var.name]


    def report_losses_det(self, counter, epoch, idx, duration, g_det_loss, gp_det_loss, de_det_loss):
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f g_det_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, g_det_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f gp_det_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, gp_det_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f de_det_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, de_det_loss))
        sys.stdout.flush()

    def report_losses_nondet(self, counter, epoch, idx, duration, g_nondet_loss, gp_nondet_loss, de_nondet_loss, d_nondet_loss):
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f g_nondet_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, g_nondet_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f gp_nondet_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, gp_nondet_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f de_nondet_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, de_nondet_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f d_nondet_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, d_nondet_loss))
        sys.stdout.flush()

    def report_inputs(self, epoch, idx, real_ctx, real_x, real_x_segmap, real_x_segmap_onehot):
        total_batch_size = real_ctx.shape[0]

        save_images(real_ctx, [total_batch_size, 1],
                   './{}/real_ctximage_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

        imsave(real_x_segmap, [total_batch_size, 1],
                    './{}/real_segmap_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

        save_images(real_x, [total_batch_size, 1],
                   './{}/real_image_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

    def report_outputs_det(self, epoch, idx, fake_det_x, fake_det_x_var, fake_det_x_segmap):
        total_batch_size = fake_det_x.shape[0]

        save_images(fake_det_x, [total_batch_size, 1],
                    './{}/fake_det_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
        imsave(image_to_uint8(fake_det_x_var), [total_batch_size, 1],
                    './{}/fake_det_var_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
        save_segmaps(fake_det_x_segmap, self.color_value_dict, [total_batch_size, 1],
                     './{}/fake_det_segmap_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

    def report_outputs_random_det(self, epoch, idx, random_fake_det_x, random_fake_det_x_segmap, prefix=None):
        total_batch_size = random_fake_det_x.shape[0]

        save_images(random_fake_det_x, [total_batch_size, 1],
                    './{}/{}_fake_det_{:03d}_{:05d}.png'.format(self.sample_dir, prefix, epoch, idx + 1))
 
        save_segmaps(random_fake_det_x_segmap, self.color_value_dict, [total_batch_size, 1],
                     './{}/{}_fake_det_segmap_{:03d}_{:05d}.png'.format(self.sample_dir, prefix, epoch, idx+1))

    def report_outputs_nondet(self, epoch, idx, fake_nondet_x):
        total_batch_size = fake_nondet_x.shape[0]

        save_images(fake_nondet_x, [total_batch_size, 1],
                    './{}/fake_nondet_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

    def report_outputs_random_nondet(self, epoch, idx, random_fake_nondet_x, prefix=None):
        total_batch_size = random_fake_nondet_x.shape[0]

        save_images(random_fake_nondet_x, [total_batch_size, 1],
                    './{}/{}_fake_nondet_{:03d}_{:05d}.png'.format(self.sample_dir, prefix, epoch, idx + 1))

    def train(self):
        distribute_strategy = tf.distribute.MirroredStrategy()
        distributed_batch_size = distribute_strategy.num_replicas_in_sync * self.batch_size
        print("Distributed replicas: %s" % (distribute_strategy.num_replicas_in_sync,))
        print("Distributed batch size: %s" % (distributed_batch_size,))

        dataset = tf.data.Dataset.from_tensor_slices((self.img_class.ctximage, self.img_class.image, self.img_class.segmap))
        dataset = dataset.shuffle(len(self.img_class.image), reshuffle_each_iteration=True).repeat(None)
        dataset = dataset.map(self.img_class.image_processing, num_parallel_calls=64*distributed_batch_size).batch(distributed_batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = distribute_strategy.experimental_distribute_dataset(dataset)

        with distribute_strategy.scope():
            # build global step
            global_step = tf.Variable(0, dtype=tf.int64, name="global_step", aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA, trainable=False)

            # prepare model
            print("> Preparing model", time.time())
            self.prepare_model()

            # build model
            print("> Building model", time.time())
            @tf.function
            def build():
                def build_fn(global_step):
                    fake_batch_size = 1
                    (real_ctx, real_x, real_x_segmap, real_x_segmap_onehot) = self.build_fake_inputs(fake_batch_size)
                    return self.execute_model(global_step, real_ctx, real_x, real_x_segmap, real_x_segmap_onehot)
                distribute_strategy.experimental_run_v2(build_fn, args=(0,))
            build()
            del build

            # show network architecture
            show_all_variables()

            # build optimizers
            print("> Building optimizers", time.time())
            self.build_optimizers()

            # saver to save model
            print("> Loading checkpoint", time.time())
            checkpoint = tf.train.Checkpoint(global_step=global_step, **dict([(var.name, var) for var in tf.compat.v1.global_variables()]))
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(self.checkpoint_dir, self.model_dir), max_to_keep=1000)

            # restore check-point if it exits
            if checkpoint_manager.latest_checkpoint:
                checkpoint_vars = tf.train.list_variables(checkpoint_manager.latest_checkpoint)
                for (checkpoint_var_name, checkpoint_var_shape) in checkpoint_vars:
                    print("Checkpoint variable: %s%s" % (checkpoint_var_name, checkpoint_var_shape))
                checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            # record start time
            print("> Training", time.time())
            start_time = time.time()

            def train_loop():
                def train_det_grad(global_step, train_main, discriminate, *inputs):
                    def train_fn(global_step, *inputs):
                        with tf.GradientTape(persistent=True) as tape:
                            inputs, (losses_det, outputs_det, summaries_det), (outputs_resample_det, outputs_random_det, outputs_random_gen_det), _, _ = self.execute_model(global_step, *inputs)
                        result_losses_det = (losses_det.g_det_loss, losses_det.gp_det_loss, losses_det.de_det_loss)
                        if discriminate != False:
                            if train_main:
                                self.DE_det_optim.apply_gradients(zip(tape.gradient(losses_det.de_det_loss, self.DE_det_vars), self.DE_det_vars))
                        if discriminate != True:
                            self.GP_det_optim.apply_gradients(zip(tape.gradient(losses_det.gp_det_loss, self.GP_det_vars), self.GP_det_vars))
                            if train_main:
                                self.G_det_optim.apply_gradients(zip(tape.gradient(losses_det.g_det_loss, self.G_det_vars), self.G_det_vars))
                            summaries_det()
                            global_step.assign_add(1)
                        return tf.convert_to_tensor(global_step), inputs, result_losses_det, outputs_det, outputs_resample_det, outputs_random_det, outputs_random_gen_det

                    result_counter, result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, result_outputs_random_gen_det = distribute_strategy.experimental_run_v2(train_fn, args=(global_step, *inputs))

                    converted_counter = tf.reduce_mean(distribute_strategy.experimental_local_results(result_counter))

                    converted_inputs = list(map(lambda result_input: tf.concat(distribute_strategy.experimental_local_results(result_input), axis=0), result_inputs))

                    converted_losses_det = list(map(lambda result_loss: tf.reduce_mean(distribute_strategy.experimental_local_results(result_loss)), result_losses_det))

                    def convert_outputs(result_outputs):
                        return list(map(lambda result_output: tf.concat(distribute_strategy.experimental_local_results(result_output), axis=0), result_outputs))

                    converted_outputs_det = convert_outputs(result_outputs_det)
                    converted_outputs_resample_det = convert_outputs(result_outputs_resample_det)
                    converted_outputs_random_det = convert_outputs(result_outputs_random_det)
                    converted_outputs_random_gen_det = convert_outputs(result_outputs_random_gen_det)

                    return converted_counter, converted_inputs, converted_losses_det, converted_outputs_det, converted_outputs_resample_det, converted_outputs_random_det, converted_outputs_random_gen_det 

                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def train_det_grad_both(global_step, train_main, *inputs):
                    return train_det_grad(global_step, train_main, None, *inputs)

                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def train_det_grad_discriminate(global_step, train_main, *inputs):
                    return train_det_grad(global_step, train_main, True, *inputs)
                
                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def train_det_grad_generate(global_step, train_main, *inputs):
                    return train_det_grad(global_step, train_main, False, *inputs)

                def train_nondet_grad(global_step, train_main, discriminate, *inputs):
                    def train_fn(global_step, *inputs):
                        with tf.GradientTape(persistent=True) as tape:
                            inputs, (losses_det, outputs_det, summaries_det), (outputs_resample_det, outputs_random_det, outputs_random_gen_det), (losses_nondet, outputs_nondet, summaries_nondet), (outputs_resample_nondet, outputs_random_nondet, outputs_random_gen_nondet) = self.execute_model(global_step, *inputs)
                        result_losses_det = (losses_det.g_det_loss, losses_det.gp_det_loss, losses_det.de_det_loss)
                        result_losses_nondet = (losses_nondet.g_nondet_loss, losses_nondet.gp_nondet_loss, losses_nondet.de_nondet_loss, losses_nondet.d_nondet_loss)
                        if discriminate != False:
                            if train_main:
                                self.D_nondet_optim.apply_gradients(zip(tape.gradient(losses_nondet.d_nondet_loss, self.D_nondet_vars), self.D_nondet_vars))
                                self.DE_nondet_optim.apply_gradients(zip(tape.gradient(losses_nondet.de_nondet_loss, self.DE_nondet_vars), self.DE_nondet_vars))
                        if discriminate != True:
                            self.GP_nondet_optim.apply_gradients(zip(tape.gradient(losses_nondet.gp_nondet_loss, self.GP_nondet_vars), self.GP_nondet_vars))
                            if train_main:
                                self.G_nondet_optim.apply_gradients(zip(tape.gradient(losses_nondet.g_nondet_loss, self.G_nondet_vars), self.G_nondet_vars))
                            summaries_det()
                            summaries_nondet()
                            global_step.assign_add(1)
                        return tf.convert_to_tensor(global_step), inputs, result_losses_det, outputs_det, outputs_resample_det, outputs_random_det, outputs_random_gen_det, result_losses_nondet, outputs_nondet, outputs_resample_nondet, outputs_random_nondet, outputs_random_gen_nondet

                    result_counter, result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, result_outputs_random_gen_det, result_losses_nondet, result_outputs_nondet, result_outputs_resample_nondet, result_outputs_random_nondet, result_outputs_random_gen_nondet = distribute_strategy.experimental_run_v2(train_fn, args=(global_step, *inputs))
                   
                    converted_counter = tf.reduce_mean(distribute_strategy.experimental_local_results(result_counter))

                    converted_inputs = list(map(lambda result_input: tf.concat(distribute_strategy.experimental_local_results(result_input), axis=0), result_inputs))

                    converted_losses_det = list(map(lambda result_loss: tf.reduce_mean(distribute_strategy.experimental_local_results(result_loss)), result_losses_det))
                    converted_losses_nondet = list(map(lambda result_loss: tf.reduce_mean(distribute_strategy.experimental_local_results(result_loss)), result_losses_nondet))

                    def convert_outputs(result_outputs):
                        return list(map(lambda result_output: tf.concat(distribute_strategy.experimental_local_results(result_output), axis=0), result_outputs))

                    converted_outputs_det = convert_outputs(result_outputs_det)
                    converted_outputs_resample_det = convert_outputs(result_outputs_resample_det)
                    converted_outputs_random_det = convert_outputs(result_outputs_random_det)
                    converted_outputs_random_gen_det = convert_outputs(result_outputs_random_gen_det)

                    converted_outputs_nondet = convert_outputs(result_outputs_nondet)
                    converted_outputs_resample_nondet = convert_outputs(result_outputs_resample_nondet)
                    converted_outputs_random_nondet = convert_outputs(result_outputs_random_nondet)
                    converted_outputs_random_gen_nondet = convert_outputs(result_outputs_random_gen_nondet)

                    return converted_counter, converted_inputs, converted_losses_det, converted_outputs_det, converted_outputs_resample_det, converted_outputs_random_det, converted_outputs_random_gen_det, converted_losses_nondet, converted_outputs_nondet, converted_outputs_resample_nondet, converted_outputs_random_nondet, converted_outputs_random_gen_nondet

                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def train_nondet_grad_both(global_step, train_main, *inputs):
                    return train_nondet_grad(global_step, train_main, None, *inputs)
 
                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def train_nondet_grad_discriminate(global_step, train_main, *inputs):
                    return train_nondet_grad(global_step, train_main, True, *inputs)
                
                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def train_nondet_grad_generate(global_step, train_main, *inputs):
                    return train_nondet_grad(global_step, train_main, False, *inputs)

                # training loop
                with self.writer.as_default():
                    for inputs_idx, inputs in enumerate(dataset):
                        print("> Training step %s" % (inputs_idx,), time.time())

                        print("L1", time.time())
                        if not self.train_nondet:
                            #if self.train_main:
                            #    print("L2DET:D", time.time())
                            #    train_det_grad_discriminate(global_step, self.train_main, *inputs)
                            #print("L2DET:G", time.time())
                            #counter, result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, result_outputs_random_gen_det = train_det_grad_generate(global_step, self.train_main, *inputs)
                            print("L2DET", time.time())
                            counter, result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, result_outputs_random_gen_det = train_det_grad_both(global_step, self.train_main, *inputs)
                        else:
                            #if self.train_main:
                            #    print("L2NONDET:D", time.time())
                            #    train_nondet_grad_discriminate(global_step, self.train_main, *inputs)
                            #print("L2NONDET:G", time.time())
                            #counter, result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, result_outputs_random_gen_det, result_losses_nondet, result_outputs_nondet, result_outputs_resample_nondet, result_outputs_random_nondet, result_outputs_random_gen_nondet = train_nondet_grad_generate(global_step, self.train_main, *inputs)
                            print("L2NONDET", time.time())
                            counter, result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, result_outputs_random_gen_det, result_losses_nondet, result_outputs_nondet, result_outputs_resample_nondet, result_outputs_random_nondet, result_outputs_random_gen_nondet = train_nondet_grad_both(global_step, self.train_main, *inputs)

                        print("L4",  time.time())
                        epoch = (counter-1) // self.iteration 
                        idx = (counter-1) % self.iteration

                        print("L5",  time.time())
                        self.report_losses_det(counter, epoch, idx, time.time() - start_time, *result_losses_det)
                        if self.train_nondet:
                            self.report_losses_nondet(counter, epoch, idx, time.time() - start_time, *result_losses_nondet)

                        def report_tensors(f, tensors, **kwargs):
                            f(epoch, idx, *map(lambda tensor: tensor.numpy(), tensors), **kwargs)

                        if (idx+1) % self.print_freq == 0:
                            print("L6",  time.time())
                            report_tensors(self.report_inputs, result_inputs)
                            report_tensors(self.report_outputs_det, result_outputs_det)
                            report_tensors(self.report_outputs_random_det, result_outputs_resample_det, prefix="resample")
                            report_tensors(self.report_outputs_random_det, result_outputs_random_det, prefix="random")
                            report_tensors(self.report_outputs_random_det, result_outputs_random_gen_det, prefix="random_gen")
                            if self.train_nondet:
                                report_tensors(self.report_outputs_nondet, result_outputs_nondet)
                                report_tensors(self.report_outputs_random_nondet, result_outputs_resample_nondet, prefix="resample")
                                report_tensors(self.report_outputs_random_nondet, result_outputs_random_nondet, prefix="random")
                                report_tensors(self.report_outputs_random_nondet, result_outputs_random_gen_nondet, prefix="random_gen")

                        if counter-1 > 0 and (counter-1) % self.save_freq == 0:
                            print("L7",  time.time())
                            checkpoint_manager.save()
                            #tf.py_function(checkpoint_manager.save, [], [tf.string])

                    print("L3",  time.time())
                    self.writer.flush()
                    
            train_loop()

    @property
    def model_dir(self):
        return "{}_dataset={}".format(self.model_name, self.dataset_name)

        n_dis = str(self.n_scale) + 'multi_' + str(self.n_dis) + 'dis'


        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        if self.TTUR :
            TTUR = '_TTUR'
        else :
            TTUR = ''


        return "{}_dataset={}__gan{}__n_dis={}__n_critic={}__adv_weight={}__vgg_weight={}__feature_weight={}__kl_weight={}__{}{}__ch={}__num_upsampling_layers={}__ch={}__num_upsampling_layers={}".format(self.model_name, self.dataset_name,
                                                                   self.gan_type, n_dis, self.n_critic,
                                                                   self.adv_weight, self.vgg_weight, self.feature_weight,
                                                                   self.kl_weight,
                                                                   sn, TTUR, self.ch, self.num_upsampling_layers, self.ch, self.num_upsampling_layers)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint_reader = tf.compat.v1.train.NewCheckpointReader(ckpt.model_checkpoint_path)
            tensor_shapes = checkpoint_reader.get_variable_to_shape_map()
            variables_to_restore = {}
            for known_variable in tf.compat.v1.global_variables():
                tensor_name = known_variable.name.split(':')[0]
                if checkpoint_reader.has_tensor(tensor_name) and known_variable.shape == tensor_shapes[tensor_name]:
                    print("Variable restored: %s Shape: %s" % (known_variable.name, known_variable.shape))
                    variables_to_restore[tensor_name] = known_variable
                else:
                    print("Variable NOT restored: %s Shape: %s OriginalShape: %s" % (known_variable.name, known_variable.shape, tensor_shapes.get(tensor_name)))

            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver = tf.compat.v1.train.Saver(variables_to_restore)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def random_test(self):
        tf.compat.v1.global_variables_initializer().run()

        files = glob('./dataset/{}/{}/*.*'.format(self.dataset_name, 'test'))

        self.saver = tf.compat.v1.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in tqdm(files) :
            sample_image = load(self.dataset_path, sample_file, self.img_width, self.img_height, self.img_ch)
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_style) :
                image_path = os.path.join(self.result_dir, '{}_style{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.random_test_fake_x, feed_dict={self.test_image : sample_image})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                            '../..' + os.path.sep + sample_file), self.img_width, self.img_height))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                            '../..' + os.path.sep + image_path), self.img_width, self.img_height))
                index.write("</tr>")

        index.close()

    def guide_test(self):
        tf.compat.v1.global_variables_initializer().run()

        files = glob('./dataset/{}/{}/*.*'.format(self.dataset_name, 'test'))

        style_image = load_style_image(self.guide_img, self.img_width, self.img_height, self.img_ch)

        self.saver = tf.compat.v1.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir, 'guide')
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>style</th><th>input</th><th>output</th></tr>")

        for sample_file in tqdm(files):
            sample_image = load(self.dataset_path, sample_file, self.img_width, self.img_height, self.img_ch)
            image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.guide_test_fake_x, feed_dict={self.test_image : sample_image, self.test_guide_image : style_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (self.guide_img if os.path.isabs(self.guide_img) else (
                        '../../..' + os.path.sep + self.guide_img), self.img_width, self.img_height))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_width, self.img_height))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_width, self.img_height))
            index.write("</tr>")

        index.close()
