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
        self.train_det = args.train_det
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
        self.validate_freq = args.validate_freq
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
        self.sn_det = args.sn_det
        self.sn_nondet = args.sn_nondet

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
        print("# spectral normalization det: ", self.sn_det)
        print("# spectral normalization nondet: ", self.sn_nondet)

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

    def encoder_base(self, x_init, channel, sn):
        #x = resize(x_init, self.img_height, self.img_width)
        x = x_init
        
        x = constin_resblock(x, channel, use_bias=True, sn=sn, norm=False, scope='preresblock')

        #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, scope='conv')
        #x = instance_norm(x, scope='ins_norm')
        x = constin_resblock(x, channel*2, use_bias=True, sn=sn, scope='resblock')
        x = down_sample_avg(x)

        for i in range(3):
            #x = lrelu(x, 0.2)
            #x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, scope='conv_' + str(i))
            #x = instance_norm(x, scope='ins_norm_' + str(i))
            x = constin_resblock(x, channel * 4, use_bias=True, sn=sn, scope='resblock_' + str(i))
            x = down_sample_avg(x)

            channel = channel * 2

            # 128, 256, 512

        #x = lrelu(x, 0.2)
        #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, scope='conv_3')
        #x = instance_norm(x, scope='ins_norm_3')
        x = constin_resblock(x, channel*8, use_bias=True, sn=sn, scope='resblock_3')
        x = down_sample_avg(x)

        #if self.img_height >= 256 or self.img_width >= 256 :
        if self.img_height >= 256 or self.img_width >= 256 :
            #x = lrelu(x, 0.2)
            #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=False, sn=sn, scope='conv_4')
            #x = instance_norm(x, scope='ins_norm_4')
            x = constin_resblock(x, channel*8, use_bias=False, sn=sn, scope='resblock_4')
            x = down_sample_avg(x)

        x = lrelu(x, 0.2)

        return x, channel

    def encoder_code(self, x_init, sn, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x, channel = self.encoder_base(x_init, self.ch, sn=sn)

            mean = fully_connected(x, channel // 2, use_bias=True, sn=False, scope='linear_mean')
            var = fully_connected(x, channel // 2, use_bias=True, sn=False, scope='linear_var')
            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def prior_code(self, batch_size, sn, channel_multiplier=4):
        out_channel = self.ch * channel_multiplier
        mean = tf.zeros([batch_size, out_channel])
        var = tf.zeros([batch_size, out_channel])
        return mean, var

    def prior_code_dist(self, code, sn, channel_multiplier=4, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
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

    def encoder_supercode(self, x_init, sn, channel_multiplier=4, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        out_channel = self.ch*channel_multiplier
        hidden_channel = self.ch*64
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x = x_init
            for i in range(self.code_num_layers):
                x = constin_fcblock(x, hidden_channel, sn=sn, scope="fcblock_" + str(i))
                x = lrelu(x, 0.2)

            mean = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_mean')
            var = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_var')

            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def generator_code(self, code, x_init, sn, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        out_channel = self.ch*4
        hidden_channel = self.ch*64
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x = x_init
            for i in range(self.code_num_layers):
                x = adain_fcblock(code, x, hidden_channel, sn=sn, scope="fcblock_" + str(i))
                x = lrelu(x, 0.2)

            mean = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_mean')
            var = get_trainable_variable("var", [], initializer=tf.compat.v1.constant_initializer(0.0))
            #var = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_var')

            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def decoder(self, code, z, sn, epsilon=1e-8, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        context = code

        context_depth = 8
        context_ch = 10*context.get_shape()[-1]
        channel = self.ch * 4 * 4
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            features = []

            #for i in range(context_depth):
            #    context = fully_connected(context, context_ch, use_bias=True, sn=sn, scope='linear_context_' + str(i))
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


            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_0')
            features.append(x)

            x = up_sample(x, scale_factor=2)
            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_1')
            features.append(x)

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_2')
            features.append(x)

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = adain_resblock(context, x, channels=channel//2, use_bias=True, sn=sn, scope='resblock_' + str(i))
                features.append(x)

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=sn, scope='resblock_4')
            x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=sn, scope='resblock_4')
            features.append(x)

            x = lrelu(x, 0.2)
            mean = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_mean')
            var = get_trainable_variable("var", [], initializer=tf.compat.v1.constant_initializer(0.0)) * tf.ones_like(mean)
            #var = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_var')
            logits = conv(tf.stop_gradient(x), channels=self.segmap_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_logits')

            return features, [[tanh(mean), tf.math.log(epsilon + tf.sigmoid(var))], logits]

    def decoder_features(self, code, features, z, sn, reuse=False, scope=None):
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

            x = cprogressive_resblock(context, features.pop(), x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_0')

            x = up_sample(x, scale_factor=2)
            x = cprogressive_resblock(context, features.pop(), x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_1')

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = cprogressive_resblock(context, features.pop(), x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_2')

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = cprogressive_resblock(context, features.pop(), x, channels=channel//2, use_bias=True, sn=sn, scope='resblock_' + str(i))

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = cprogressive_resblock(context, features.pop(), x, channels=channel // 2, use_bias=True, sn=sn, scope='resblock_4')
            x = cprogressive_resblock(context, features.pop(), x, channels=channel // 2, use_bias=True, sn=sn, scope='resblock_4')

            x = lrelu(x, 0.2)
            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='logit')
            x = tanh(x)

            return x

    def decoder_spatial(self, code, scaffold, z, sn, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
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

            x = cspade_resblock(context, scaffold, x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_0')
            #x = adain_resblock(context, x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_0')

            x = up_sample(x, scale_factor=2)
            x = cspade_resblock(context, scaffold, x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_1')
            #x = adain_resblock(context, x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_1')

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = cspade_resblock(context, scaffold, x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_2')
            #x = adain_resblock(context, x, channels=channel, use_bias=True, sn=sn, scope='resblock_fix_2')

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = cspade_resblock(context, scaffold, x, channels=channel//2, use_bias=True, sn=sn, scope='resblock_' + str(i))
                #x = adain_resblock(context, x, channels=channel//2, use_bias=True, sn=sn, scope='resblock_' + str(i))

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = cspade_resblock(context, scaffold, x, channels=channel // 2, use_bias=True, sn=sn, scope='resblock_4')
            x = cspade_resblock(context, scaffold, x, channels=channel // 2, use_bias=True, sn=sn, scope='resblock_4')
            #x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=sn, scope='resblock_4')

            x = lrelu(x, 0.2)
            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator_code(self, x, sn, reuse=tf.compat.v1.AUTO_REUSE, scope=None, label=None):
        channel = x.get_shape()[-1]
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x = constin_fcblock(x, channel * 10, use_bias=True, sn=sn, norm=False, scope='linear_x_1')

            x = constin_fcblock(x, channel * 10, use_bias=True, sn=sn, norm=True, scope='linear_x_2')

            x = constin_fcblock(x, channel * 10, use_bias=True, sn=sn, norm=True, scope='linear_x_3')

            x = constin_fcblock(x, channel * 10, use_bias=True, sn=sn, norm=True, scope='linear_x_4')

            z = fully_connected(x, 1, sn=sn, scope='linear_z')

            return [[z]]

    def full_discriminator(self, x_init, code, sn, reuse=tf.compat.v1.AUTO_REUSE, scope=None, label=None):
        channel = self.ch
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            feature_loss = []
            x = x_init
            
            x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, scope='preconv')
            x = lrelu(x, 0.2)
            #x = adain_resblock(code, x, channel, use_bias=True, sn=sn, norm=False, scope='preresblock')
            #x = constin_resblock(x, channel, use_bias=True, sn=sn, norm=False, scope='preresblock')

            x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = lrelu(x, 0.2)
            #x = adain_resblock(code, x, channel * 2, use_bias=True, sn=sn, norm=True, scope='resblock')
            #x = constin_resblock(x, channel * 2, use_bias=True, sn=sn, norm=True, scope='resblock')
            
            feature_loss.append(x)

            for i in range(3):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, scope='conv_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = lrelu(x, 0.2)
                #x = adain_resblock(code, x, channel * 4, use_bias=True, sn=sn, norm=True, scope='resblock_' + str(i))
                #x = constin_resblock(x, channel * 4, use_bias=True, sn=sn, norm=True, scope='resblock_' + str(i))
                #x = down_sample_avg(x)

                feature_loss.append(x)

                channel = channel * 2

                # 128, 256, 512

            x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, scope='conv_3')
            x = instance_norm(x, scope='ins_norm_3')
            x = lrelu(x, 0.2)
            #x = adain_resblock(0*code, x, channel*4, use_bias=True, sn=sn, norm=True, scope='resblock_3')
            #x = constin_resblock(x, channel*4, use_bias=True, sn=sn, norm=True, scope='resblock_3')
            #x = down_sample_avg(x)

            #if self.img_height >= 256 or self.img_width >= 256 :
            if self.img_height >= 256 or self.img_width >= 256 :
                x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=False, sn=sn, scope='conv_4')
                x = instance_norm(x, scope='ins_norm_4')
                x = lrelu(x, 0.2)
                #x = adain_resblock(0*code, x, channel*8, use_bias=False, sn=sn, norm=True, scope='resblock_4')
                #x = constin_resblock(x, channel*8, use_bias=False, sn=sn, norm=True, scope='resblock_4')
                #x = down_sample_avg(x)
                
                feature_loss.append(x)

            x = fully_connected(x, 1, use_bias=True, sn=sn, scope='linear_x')

            D_logit = [feature_loss + [x]]
            return D_logit

    def feature_discriminator(self, x_init, code, sn, reuse=tf.compat.v1.AUTO_REUSE, scope=None, label=None):
        D_logit = []
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            for scale in range(self.n_scale):
                feature_loss = []
                channel = self.ch
                x = x_init

                x = conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, sn=False, scope='ms_' + str(scale) + 'conv_0')
                x = lrelu(x, 0.2)
                #x = adain_resblock(code, x, channel, use_bias=True, sn=sn, scope='ms_' + str(scale) + '_resblock')
                #x = constin_resblock(x, channel, use_bias=True, sn=sn, scope='ms_' + str(scale) + '_resblock')

                feature_loss.append(x)

                for i in range(1, self.n_dis):
                    stride = 1 if i == self.n_dis - 1 else 2

                    x = conv(x, channel * 2, kernel=4, stride=stride, pad=1, use_bias=True, sn=sn, scope='ms_' + str(scale) + 'conv_' + str(i))
                    x = instance_norm(x, scope='ms_' + str(scale) + 'ins_norm_' + str(i))
                    x = lrelu(x, 0.2)

                    #x = adain_resblock(code, x, channel*2, use_bias=True, sn=sn, scope='ms_' + str(scale) + 'resblock_' + str(i))
                    #x = constin_resblock(x, channel*2, use_bias=True, sn=sn, scope='ms_' + str(scale) + 'resblock_' + str(i))
                    #if i !=  self.n_dis - 1:
                    #    x = down_sample_avg(x)

                    feature_loss.append(x)

                    channel = min(channel * 2, 512)


                x = conv(x, channels=1, kernel=4, stride=1, pad=1, use_bias=True, sn=sn, scope='ms_' + str(scale) + 'D_logit')

                feature_loss.append(x)
                D_logit.append(feature_loss)

                x_init = down_sample_avg(x_init)

            return D_logit

    ##################################################################################
    # Model
    ##################################################################################

    def discriminate_code(self, real_code_img, fake_code_img, sn, reuse=tf.compat.v1.AUTO_REUSE, name=None):
        real_logit = self.discriminator_code(real_code_img, sn=sn, reuse=reuse, scope='discriminator_' + name + '_code', label='real_' + name + '_code')
        fake_logit = self.discriminator_code(fake_code_img, sn=sn, reuse=True, scope='discriminator_' + name + '_code', label='fake_' + name + '_code')

        return real_logit, fake_logit

    def discriminate(self, real_img, fake_img, code, sn, reuse=tf.compat.v1.AUTO_REUSE, name=None):
        real_logit = self.discriminator(real_img, code, sn=sn, reuse=reuse, scope='discriminator_' + name + '', label='real_' + name + '')
        fake_logit = self.discriminator(fake_img, code, sn=sn, reuse=True, scope='discriminator_' + name + '', label='fake_' + name + '')

        return real_logit, fake_logit

    def gradient_penalty(self, real, fake, code, discriminator, sn, name=None):
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

        logit = discriminator(interpolated, code=code, sn=sn, reuse=True, scope='discriminator_' + name + '', label='interpolated_' + name + '')

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
        self.img_class_train = Image_data(self.img_height, self.img_width, self.img_ch, self.segmap_img_ch, self.dataset_path, self.augment_flag)
        self.img_class_train.preprocess(is_train=True)
        
        self.color_value_dict = self.img_class_train.color_value_dict
        self.segmap_ch = len(self.img_class_train.color_value_dict)

        self.img_class_validate = Image_data(self.img_height, self.img_width, self.img_ch, self.segmap_img_ch, self.dataset_path, self.augment_flag)
        self.img_class_validate.preprocess(is_train=False)

    def prepare_model(self):
        self.vgg_loss = VGGLoss() 

    def execute_model(self, real_ctx, real_ctx_pose, real_x, real_x_pose, real_x_segmap, real_x_segmap_onehot):
        batch_size = tf.shape(real_ctx)[0]

        """ Define Generator, Discriminator """
        prior_det_code_mean, prior_det_code_logvar = self.prior_code(batch_size, sn=self.sn_det)
        x_det_code_mean, x_det_code_logvar = self.encoder_code(real_x, sn=self.sn_det, scope='encoder_det_code')
        fake_det_x_code = z_sample(x_det_code_mean, x_det_code_logvar)
        x_det_codectx_mean, x_det_codectx_logvar = self.encoder_code(real_ctx, sn=self.sn_det, reuse=True, scope='encoder_det_code')
        fake_det_x_codectx = z_sample(x_det_codectx_mean, x_det_codectx_logvar)
       
        supercode_stop_gradient = lambda x: x
        code_stop_gradient = lambda x: x

        x_det_supercode_mean, x_det_supercode_logvar = self.encoder_supercode(code_stop_gradient(fake_det_x_code), sn=self.sn_det, scope='encoder_det_supercode')
        fake_det_x_supercode = z_sample(x_det_supercode_mean, x_det_supercode_logvar)
        x_det_supercodectx_mean, x_det_supercodectx_logvar = self.encoder_supercode(code_stop_gradient(fake_det_x_codectx), sn=self.sn_det, reuse=True, scope='encoder_det_supercode')
        fake_det_x_supercodectx = z_sample(x_det_supercodectx_mean, x_det_supercodectx_logvar)

        x_det_ctxcode_mean, x_det_ctxcode_logvar = self.encoder_code(real_ctx, sn=self.sn_det, scope='encoder_det_ctxcode')
        fake_det_x_ctxcode = z_sample(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        x_det_ctxcodex_mean, x_det_ctxcodex_logvar = self.encoder_code(real_x, sn=self.sn_det, reuse=True, scope='encoder_det_ctxcode')
        fake_det_x_ctxcodex = z_sample(x_det_ctxcodex_mean, x_det_ctxcodex_logvar)

        fake_det_x_full_ctxcode = tf.concat([fake_det_x_supercode, fake_det_x_ctxcode, real_x_pose],-1)
        fake_det_x_full_supercode = tf.concat([fake_det_x_supercode, fake_det_x_ctxcode],-1)
        fake_det_x_code_mean, fake_det_x_code_logvar = self.generator_code(fake_det_x_full_ctxcode, fake_det_x_full_supercode, sn=self.sn_det, scope="generator_det_code")

        fake_det_x_full_ctxcodectx = tf.concat([fake_det_x_supercodectx, fake_det_x_ctxcodex, real_ctx_pose],-1)
        fake_det_x_full_supercodectx = tf.concat([fake_det_x_supercodectx, fake_det_x_ctxcodex],-1)
        fake_det_x_codectx_mean, fake_det_x_codectx_logvar = self.generator_code(fake_det_x_full_ctxcodectx, fake_det_x_full_supercodectx, sn=self.sn_det, scope="generator_det_code")
        
        resample_det_full_ctxcode = fake_det_x_full_ctxcode
        resample_det_full_supercode = fake_det_x_full_supercode
        prior_det_supercode_mean, prior_det_supercode_logvar = self.prior_code(batch_size, sn=self.sn_det, channel_multiplier=4)#self.encoder_code(real_x, sn=self.sn_det, scope='prior_det_supercode')
        #random_det_supercode = z_sample(prior_det_supercode_mean, prior_det_supercode_logvar)
        prior_det_supercode_dist = self.prior_code_dist(fake_det_x_ctxcode, sn=self.sn_det, channel_multiplier=4, scope='prior_det_supercode')
        random_det_supercode = prior_det_supercode_dist.sample()
        random_det_full_ctxcode = tf.concat([random_det_supercode, fake_det_x_ctxcode, real_x_pose],-1)
        random_det_full_supercode = tf.concat([random_det_supercode, fake_det_x_ctxcode],-1)

        prior_det_ctxcode_mean, prior_det_ctxcode_logvar = self.prior_code(batch_size, sn=self.sn_det)
        #random_det_ctxcode = z_sample(prior_det_ctxcode_mean, prior_det_ctxcode_logvar)

        resample_det_code_mean, resample_det_code_var = self.generator_code(resample_det_full_ctxcode, resample_det_full_supercode, sn=self.sn_det, reuse=True, scope="generator_det_code")
        resample_det_code = z_sample(resample_det_code_mean, resample_det_code_var)
        random_det_code_mean, random_det_code_var = self.generator_code(random_det_full_ctxcode, random_det_full_supercode, sn=self.sn_det, reuse=True, scope="generator_det_code")
        random_det_code = z_sample(random_det_code_mean, random_det_code_var)

        random_gaussian_det_code = z_sample(*self.prior_code(batch_size, sn=self.sn_det))
        
        prior_nondet_code_mean, prior_nondet_code_logvar = self.prior_code(batch_size, sn=self.sn_nondet)

        x_nondet_code_mean, x_nondet_code_logvar = self.encoder_code(real_x, sn=self.sn_nondet, scope='encoder_nondet_code')
        fake_nondet_x_code = z_sample(x_nondet_code_mean, x_nondet_code_logvar)
        x_nondet_codectx_mean, x_nondet_codectx_logvar = self.encoder_code(real_ctx, sn=self.sn_nondet, reuse=True, scope='encoder_nondet_code')
        fake_nondet_x_codectx = z_sample(x_nondet_codectx_mean, x_nondet_codectx_logvar)

        x_nondet_supercode_mean, x_nondet_supercode_logvar = self.encoder_supercode(code_stop_gradient(fake_nondet_x_code), sn=self.sn_nondet, scope='encoder_nondet_supercode')
        fake_nondet_x_supercode = z_sample(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        x_nondet_supercodectx_mean, x_nondet_supercodectx_logvar = self.encoder_supercode(code_stop_gradient(fake_nondet_x_codectx), sn=self.sn_nondet, scope='encoder_nondet_supercode')
        fake_nondet_x_supercodectx = z_sample(x_nondet_supercodectx_mean, x_nondet_supercodectx_logvar)

        x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar = self.encoder_code(real_ctx, sn=self.sn_nondet, scope='encoder_nondet_ctxcode')
        fake_nondet_x_ctxcode = z_sample(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        x_nondet_ctxcodex_mean, x_nondet_ctxcodex_logvar = self.encoder_code(real_x, sn=self.sn_nondet, reuse=True, scope='encoder_nondet_ctxcode')
        fake_nondet_x_ctxcodex = z_sample(x_nondet_ctxcodex_mean, x_nondet_ctxcodex_logvar)

        fake_nondet_x_full_ctxcode = tf.concat([fake_nondet_x_supercode, fake_nondet_x_ctxcode, real_x_pose],-1)
        fake_nondet_x_full_supercode = tf.concat([fake_nondet_x_supercode, fake_nondet_x_ctxcode],-1)
        fake_nondet_x_code_mean, fake_nondet_x_code_logvar = self.generator_code(fake_nondet_x_full_ctxcode, fake_nondet_x_full_supercode, sn=self.sn_nondet, scope="generator_nondet_code")

        fake_nondet_x_full_ctxcodectx = tf.concat([fake_nondet_x_supercodectx, fake_nondet_x_ctxcodex, real_ctx_pose],-1)
        fake_nondet_x_full_supercodectx = tf.concat([fake_nondet_x_supercodectx, fake_nondet_x_ctxcodex],-1)
        fake_nondet_x_codectx_mean, fake_nondet_x_codectx_logvar = self.generator_code(fake_nondet_x_full_ctxcodectx, fake_nondet_x_full_supercodectx, sn=self.sn_nondet, reuse=True, scope="generator_nondet_code")
        
        resample_nondet_full_ctxcode = fake_nondet_x_full_ctxcode
        resample_nondet_full_supercode = fake_nondet_x_full_supercode
        prior_nondet_supercode_mean, prior_nondet_supercode_logvar = self.prior_code(batch_size, sn=self.sn_nondet, channel_multiplier=4)#self.encoder_code(real_x, sn=self.sn_nondet, scope='prior_nondet_supercode')
        #random_nondet_supercode = z_sample(prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        prior_nondet_supercode_dist = self.prior_code_dist(fake_nondet_x_ctxcode, sn=self.sn_nondet, channel_multiplier=4, scope='prior_nondet_supercode')
        random_nondet_supercode = prior_nondet_supercode_dist.sample()
        random_nondet_full_ctxcode = tf.concat([random_nondet_supercode, fake_nondet_x_ctxcode, real_x_pose],-1)
        random_nondet_full_supercode = tf.concat([random_nondet_supercode, fake_nondet_x_ctxcode],-1)
        
        prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar = self.prior_code(batch_size, sn=self.sn_nondet)
        #random_nondet_ctxcode = z_sample(prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar)
        
        resample_nondet_code_mean, resample_nondet_code_var = self.generator_code(resample_nondet_full_ctxcode, resample_nondet_full_supercode, sn=self.sn_nondet, reuse=True, scope="generator_nondet_code")
        resample_nondet_code = z_sample(resample_nondet_code_mean, resample_nondet_code_var)
        random_nondet_code_mean, random_nondet_code_var = self.generator_code(random_nondet_full_ctxcode, random_nondet_full_supercode, sn=self.sn_nondet, reuse=True, scope="generator_nondet_code")
        random_nondet_code = z_sample(random_nondet_code_mean, random_nondet_code_var)

        random_gaussian_nondet_code = z_sample(*self.prior_code(batch_size, sn=self.sn_nondet))

        fake_full_det_x_code = tf.concat([fake_det_x_code, 0*fake_det_x_full_ctxcode, real_x_pose],-1)
        fake_full_det_x_z = tf.concat([fake_det_x_code, 0*fake_det_x_full_ctxcode],-1)
        fake_full_det_x_discriminator_code = tf.concat([fake_det_x_code, fake_det_x_full_ctxcode],-1) 
        fake_det_x_features, fake_det_x_stats = self.decoder(fake_full_det_x_code, z=fake_full_det_x_z, sn=self.sn_det, scope="generator_det_x")
        fake_det_x_scaffold = fake_det_x_features#tf.concat([fake_det_x_stats[0][0], fake_det_x_stats[1]], -1)
        fake_det_x_mean, fake_det_x_var = fake_det_x_stats[0]
        fake_det_x_segmap_logits = fake_det_x_stats[1]
        
        fake_full_nondet_x_code = tf.concat([fake_nondet_x_code, 0*fake_nondet_x_full_ctxcode, tf.stop_gradient(fake_full_det_x_code)],-1) 
        fake_full_nondet_x_z = tf.concat([fake_nondet_x_code, 0*fake_nondet_x_full_ctxcode, tf.stop_gradient(fake_full_det_x_z)],-1) 
        fake_full_nondet_x_discriminator_code = tf.concat([fake_nondet_x_code, fake_nondet_x_full_ctxcode, tf.stop_gradient(fake_det_x_code), tf.stop_gradient(fake_det_x_full_ctxcode)],-1) 
        #fake_nondet_x_output = self.decoder_spatial(fake_full_nondet_x_code, tf.stop_gradient(fake_det_x_scaffold), z=fake_full_nondet_x_z, sn=self.sn_nondet, reuse=True, scope="generator_nondet_x")
        fake_nondet_x_output = self.decoder_features(fake_full_nondet_x_code, list(map(tf.stop_gradient, fake_det_x_features)), z=fake_full_nondet_x_z, sn=self.sn_nondet, reuse=True, scope="generator_nondet_x")

        random_full_det_x_code = tf.concat([random_det_code, 0*fake_det_x_full_ctxcode, real_x_pose], -1)
        random_full_det_x_z = tf.concat([random_det_code, 0*fake_det_x_full_ctxcode], -1)
        random_fake_det_x_features, random_fake_det_x_stats = self.decoder(random_full_det_x_code, z=random_full_det_x_z, sn=self.sn_det, reuse=True, scope="generator_det_x")
        random_fake_det_x_scaffold = random_fake_det_x_features#tf.concat([random_fake_det_x_stats[0][0], random_fake_det_x_stats[1]], -1)
        random_fake_det_x_mean, random_fake_det_x_var = random_fake_det_x_stats[0]
        random_fake_det_x_segmap_logits = random_fake_det_x_stats[1]

        random_full_nondet_x_code = tf.concat([random_nondet_code, 0*fake_nondet_x_full_ctxcode, random_full_det_x_code], -1) 
        random_full_nondet_x_z = tf.concat([random_nondet_code, 0*fake_nondet_x_full_ctxcode, random_full_det_x_z], -1) 
        #random_fake_nondet_x_output = self.decoder_spatial(random_full_nondet_x_code, random_fake_det_x_scaffold, z=random_full_nondet_x_z, sn=self.sn_nondet, reuse=True, scope="generator_nondet_x")
        random_fake_nondet_x_output = self.decoder_features(random_full_nondet_x_code, random_fake_det_x_features, z=random_full_nondet_x_z, sn=self.sn_nondet, reuse=True, scope="generator_nondet_x")

        resample_full_det_x_code = tf.concat([resample_det_code, 0*fake_det_x_full_ctxcode, real_x_pose], -1)
        resample_full_det_x_z = tf.concat([resample_det_code, 0*fake_det_x_full_ctxcode], -1)
        resample_fake_det_x_features, resample_fake_det_x_stats = self.decoder(resample_full_det_x_code, z=resample_full_det_x_z, sn=self.sn_det, reuse=True, scope="generator_det_x")
        resample_fake_det_x_scaffold = resample_fake_det_x_features#tf.concat([resample_fake_det_x_stats[0][0], resample_fake_det_x_stats[1]], -1)
        resample_fake_det_x_mean, resample_fake_det_x_var = resample_fake_det_x_stats[0]
        resample_fake_det_x_segmap_logits = resample_fake_det_x_stats[1]

        resample_full_nondet_x_code = tf.concat([resample_nondet_code, 0*fake_nondet_x_full_ctxcode, resample_full_det_x_code], -1)
        resample_full_nondet_x_z = tf.concat([resample_nondet_code, 0*fake_nondet_x_full_ctxcode, resample_full_det_x_z], -1)
        #resample_fake_nondet_x_output = self.decoder_spatial(resample_full_nondet_x_code, resample_fake_det_x_scaffold, z=resample_full_nondet_x_z, sn=self.sn_nondet, reuse=True, scope="generator_nondet_x")
        resample_fake_nondet_x_output = self.decoder_features(resample_full_nondet_x_code, resample_fake_det_x_features, z=resample_full_nondet_x_z, sn=self.sn_nondet, reuse=True, scope="generator_nondet_x")

        code_det_prior_real_logit, code_det_prior_fake_logit = self.discriminate_code(real_code_img=tf.concat([tf.stop_gradient(fake_det_x_full_ctxcode), tf.stop_gradient(random_gaussian_det_code)], -1), fake_code_img=tf.concat([tf.stop_gradient(fake_det_x_full_ctxcode), fake_det_x_code], -1), sn=self.sn_det, name='det_prior')
        code_nondet_prior_real_logit, code_nondet_prior_fake_logit = self.discriminate_code(real_code_img=tf.concat([tf.stop_gradient(fake_nondet_x_full_ctxcode), tf.stop_gradient(random_gaussian_nondet_code)], -1), fake_code_img=tf.concat([tf.stop_gradient(fake_nondet_x_full_ctxcode), fake_nondet_x_code], -1), sn=self.sn_nondet, name='nondet_prior')

        discriminator_fun = self.full_discriminator
        nondet_real_logit = discriminator_fun(tf.concat([real_ctx, real_x, 0*tf.stop_gradient(fake_det_x_mean)], -1), fake_full_nondet_x_discriminator_code, sn=self.sn_nondet, scope='discriminator_nondet_x', label='real_nondet_x')
        nondet_fake_logit = discriminator_fun(tf.concat([real_ctx, fake_nondet_x_output, 0*tf.stop_gradient(fake_det_x_mean)], -1), fake_full_nondet_x_discriminator_code, sn=self.sn_nondet, reuse=True, scope='discriminator_nondet_x', label='fake_nondet_x')
        
        if self.gan_type.__contains__('wgan-') or self.gan_type == 'dragan':
            GP = self.gradient_penalty(real=tf.concat([0*real_ctx, real_x, tf.stop_gradient(fake_det_x_mean)], -1), fake=tf.concat([0*real_ctx, fake_nondet_x_output, tf.stop_gradient(fake_det_x_mean)],-1), code=fake_full_nondet_x_discriminator_code, discriminator=discriminator_fun, sn=self.sn_nondet, name='nondet_x')
        else:
            GP = 0

        discriminator_fun = self.feature_discriminator
        det_real_logit = discriminator_fun(tf.concat([0*real_ctx, real_x], -1), fake_full_det_x_discriminator_code, sn=self.sn_det, scope='discriminator_det_x', label='real_det_x')
        det_fake_logit = discriminator_fun(tf.concat([0*real_ctx, fake_det_x_mean], -1), fake_full_det_x_discriminator_code, sn=self.sn_det, reuse=True, scope='discriminator_det_x', label='fake_det_x')
        
        if self.gan_type.__contains__('wgan-') or self.gan_type == 'dragan':
            GP = self.gradient_penalty(real=tf.concat([0*real_ctx, real_x, tf.stop_gradient(fake_det_x_mean)], -1), fake=tf.concat([0*real_ctx, fake_det_x_output, tf.stop_gradient(fake_det_x_mean)],-1), code=fake_full_det_x_discriminator_code, discriminator=discriminator_fun, sn=self.sn_det, name='det_x')
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
        g_det_vgg_loss = self.vgg_loss(real_x, fake_det_x_mean)
        g_det_adv_loss = generator_loss(self.gan_type, det_fake_logit)
        g_det_feature_loss = feature_loss(det_real_logit, det_fake_logit)
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

        d_det_adv_loss = discriminator_loss(self.gan_type, det_real_logit, det_fake_logit)
        d_det_score_real, d_det_score_fake = discriminator_scores(det_real_logit, det_fake_logit)
        d_det_score_diff = -d_det_score_real + d_det_score_fake
        d_det_reg_loss = GP + regularization_loss('discriminator_det_x')

        d_nondet_adv_loss = discriminator_loss(self.gan_type, nondet_real_logit, nondet_fake_logit)
        d_nondet_score_real, d_nondet_score_fake = discriminator_scores(nondet_real_logit, nondet_fake_logit)
        d_nondet_score_diff = -d_nondet_score_real + d_nondet_score_fake
        d_nondet_reg_loss = GP + regularization_loss('discriminator_nondet_x')

        de_det_prior_adv_loss = discriminator_loss(self.code_gan_type, code_det_prior_real_logit, code_det_prior_fake_logit)
        de_det_prior_reg_loss = regularization_loss('discriminator_det_prior_code')

        de_nondet_prior_adv_loss = discriminator_loss(self.code_gan_type, code_nondet_prior_real_logit, code_nondet_prior_fake_logit)
        de_nondet_prior_reg_loss = regularization_loss('discriminator_nondet_prior_code')

        gp_nondet_loss = 1.0*(0*e_nondet_prior_loss + e_nondet_prior2_loss + (g_nondet_code_ce_loss + 1.0*(0*e_nondet_code_prior_loss + e_nondet_code_prior2_loss + e_nondet_code_negent_loss)) + e_nondet_negent_loss) + 1e-6*e_nondet_klctx2_loss
        g_nondet_loss = g_nondet_adv_loss + 0*g_nondet_feature_loss + 0.1*g_nondet_vgg_loss + g_nondet_reg_loss + tf.zeros_like(g_nondet_ce_loss) + 0*e_nondet_prior_adv_loss + e_nondet_reg_loss + gp_nondet_loss
        de_nondet_loss = de_nondet_prior_adv_loss + de_nondet_prior_reg_loss
        d_nondet_loss = d_nondet_adv_loss + d_nondet_reg_loss

        gp_det_loss = 1.0*(0*e_det_prior_loss + e_det_prior2_loss + (g_det_code_ce_loss + 1.0*(0*e_det_code_prior_loss + e_det_code_prior2_loss + e_det_code_negent_loss)) + e_det_negent_loss) + 1e-6*e_det_klctx2_loss
        g_det_loss = g_det_adv_loss + 0*g_det_feature_loss + 10*g_det_ce_loss + 10*g_det_segmapce_loss + 0.1*g_det_vgg_loss + g_det_reg_loss + 0*e_det_prior_adv_loss + e_det_reg_loss + gp_det_loss 
        de_det_loss = de_det_prior_adv_loss + de_det_prior_reg_loss
        d_det_loss = d_det_adv_loss + d_det_reg_loss

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

        """" Summary """
        def summaries_det(global_step, mode):
            summary_global_step = tf.summary.scalar("global_step/" + mode, global_step, step=global_step)

            summary_g_det_loss = tf.summary.scalar("g_det_loss/" + mode, g_det_loss, step=global_step)
            summary_gp_det_loss = tf.summary.scalar("gp_det_loss/" + mode, gp_det_loss, step=global_step)
            summary_de_det_loss = tf.summary.scalar("de_det_loss/" + mode, de_det_loss, step=global_step)
            summary_d_det_loss = tf.summary.scalar("d_det_loss/" + mode, d_det_loss, step=global_step)

            summary_g_det_code_ce_loss = tf.summary.scalar("g_det_code_ce_loss/" + mode, g_det_code_ce_loss, step=global_step)
            summary_e_det_code_kl_loss = tf.summary.scalar("e_det_code_kl_loss/" + mode, e_det_code_kl_loss, step=global_step)
            summary_e_det_klctx_loss = tf.summary.scalar("e_det_klctx_loss/" + mode, e_det_klctx_loss, step=global_step)
            summary_e_det_code_kl2_loss = tf.summary.scalar("e_det_code_kl2_loss/" + mode, e_det_code_kl2_loss, step=global_step)
            summary_e_det_klctx2_loss = tf.summary.scalar("e_det_klctx2_loss/" + mode, e_det_klctx2_loss, step=global_step)
            summary_e_det_code_prior_loss = tf.summary.scalar("e_det_code_prior_loss/" + mode, e_det_code_prior_loss, step=global_step)
            summary_e_det_code_prior2_loss = tf.summary.scalar("e_det_code_prior2_loss/" + mode, e_det_code_prior2_loss, step=global_step)
            summary_e_det_priorctx_loss = tf.summary.scalar("e_det_priorctx_loss/" + mode, e_det_priorctx_loss, step=global_step)
            summary_e_det_code_negent_loss = tf.summary.scalar("e_det_code_negent_loss/" + mode, e_det_code_negent_loss, step=global_step)
            summary_e_det_negentctx_loss = tf.summary.scalar("e_det_negentctx_loss/" + mode, e_det_negentctx_loss, step=global_step)

            summary_e_det_crossklctx_loss = tf.summary.scalar("e_det_crossklctx_loss/" + mode, e_det_crossklctx_loss, step=global_step)
            summary_e_det_rcrossklctx_loss = tf.summary.scalar("e_det_rcrossklctx_loss/" + mode, e_det_rcrossklctx_loss, step=global_step)
            summary_e_det_crosswsctx_loss = tf.summary.scalar("e_det_crosswsctx_loss/" + mode, e_det_crosswsctx_loss, step=global_step)

            summary_e_det_kl_loss = tf.summary.scalar("e_det_kl_loss/" + mode, e_det_kl_loss, step=global_step)
            summary_e_det_negent_loss = tf.summary.scalar("e_det_negent_loss/" + mode, e_det_negent_loss, step=global_step)
            summary_e_det_prior_loss = tf.summary.scalar("e_det_prior_loss/" + mode, e_det_prior_loss, step=global_step)
            summary_e_det_prior2_loss = tf.summary.scalar("e_det_prior2_loss/" + mode, e_det_prior2_loss, step=global_step)
            summary_e_det_crosskl_loss = tf.summary.scalar("e_det_crosskl_loss/" + mode, e_det_crosskl_loss, step=global_step)
            summary_e_det_rcrosskl_loss = tf.summary.scalar("e_det_rcrosskl_loss/" + mode, e_det_rcrosskl_loss, step=global_step)
            summary_e_det_crossws_loss = tf.summary.scalar("e_det_crossws_loss/" + mode, e_det_crossws_loss, step=global_step)
            summary_e_det_kl2_loss = tf.summary.scalar("e_det_kl2_loss/" + mode, e_det_kl2_loss, step=global_step)
            summary_e_det_reg_loss = tf.summary.scalar("det_e_reg_loss/" + mode, e_det_reg_loss, step=global_step)
            summary_e_det_prior_adv_loss = tf.summary.scalar("e_det_prior_adv_loss/" + mode, e_det_prior_adv_loss, step=global_step)

            summary_g_det_ce_loss = tf.summary.scalar("g_det_ce_loss/" + mode, g_det_ce_loss, step=global_step)
            summary_g_det_ce_loss = tf.summary.scalar("g_det_segmapce_loss/" + mode, g_det_segmapce_loss, step=global_step)
            summary_g_det_vgg_loss = tf.summary.scalar("g_det_vgg_loss/" + mode, g_det_vgg_loss, step=global_step)
            summary_g_det_adv_loss = tf.summary.scalar("g_det_adv_loss/" + mode, g_det_adv_loss, step=global_step)
            summary_g_det_feature_loss = tf.summary.scalar("g_det_feature_loss/" + mode, g_det_feature_loss, step=global_step)
            summary_g_det_reg_loss = tf.summary.scalar("g_det_reg_loss/" + mode, g_det_reg_loss, step=global_step)

            summary_de_det_prior_adv_loss = tf.summary.scalar("de_det_prior_adv_loss/" + mode, de_det_prior_adv_loss, step=global_step)
            summary_de_det_prior_reg_loss = tf.summary.scalar("de_det_prior_reg_loss/" + mode, de_det_prior_reg_loss, step=global_step)
        
            summary_d_det_adv_loss = tf.summary.scalar("d_det_adv_loss/" + mode, d_det_adv_loss, step=global_step)
            summary_d_det_reg_loss = tf.summary.scalar("d_det_reg_loss/" + mode, d_det_reg_loss, step=global_step)
            
            summary_d_det_adv_loss = tf.summary.scalar("d_det_score_real/" + mode, d_det_score_real, step=global_step)
            summary_d_det_adv_loss = tf.summary.scalar("d_det_score_fake/" + mode, d_det_score_fake, step=global_step)
            summary_d_det_adv_loss = tf.summary.scalar("d_det_score_diff/" + mode, d_det_score_diff, step=global_step)

        def summaries_nondet(global_step, mode):
            summary_g_nondet_loss = tf.summary.scalar("g_nondet_loss/" + mode, g_nondet_loss, step=global_step)
            summary_gp_nondet_loss = tf.summary.scalar("gp_nondet_loss/" + mode, gp_nondet_loss, step=global_step)
            summary_de_nondet_loss = tf.summary.scalar("de_nondet_loss/" + mode, de_nondet_loss, step=global_step)
            summary_d_nondet_loss = tf.summary.scalar("d_nondet_loss/" + mode, d_nondet_loss, step=global_step)

            summary_g_nondet_code_ce_loss = tf.summary.scalar("g_nondet_code_ce_loss/" + mode, g_nondet_code_ce_loss, step=global_step)
            summary_e_nondet_code_kl_loss = tf.summary.scalar("e_nondet_code_kl_loss/" + mode, e_nondet_code_kl_loss, step=global_step)
            summary_e_nondet_klctx_loss = tf.summary.scalar("e_nondet_klctx_loss/" + mode, e_nondet_klctx_loss, step=global_step)
            summary_e_nondet_code_kl2_loss = tf.summary.scalar("e_nondet_code_kl2_loss/" + mode, e_nondet_code_kl2_loss, step=global_step)
            summary_e_nondet_klctx2_loss = tf.summary.scalar("e_nondet_klctx2_loss/" + mode, e_nondet_klctx2_loss, step=global_step)
            summary_e_nondet_code_prior_loss = tf.summary.scalar("e_nondet_code_prior_loss/" + mode, e_nondet_code_prior_loss, step=global_step)
            summary_e_nondet_code_prior2_loss = tf.summary.scalar("e_nondet_code_prior2_loss/" + mode, e_nondet_code_prior2_loss, step=global_step)
            summary_e_nondet_priorctx_loss = tf.summary.scalar("e_nondet_priorctx_loss/" + mode, e_nondet_priorctx_loss, step=global_step)
            summary_e_nondet_code_negent_loss = tf.summary.scalar("e_nondet_code_negent_loss/" + mode, e_nondet_code_negent_loss, step=global_step)
            summary_e_nondet_negentctx_loss = tf.summary.scalar("e_nondet_negentctx_loss/" + mode, e_nondet_negentctx_loss, step=global_step)

            summary_e_nondet_crossklctx_loss = tf.summary.scalar("e_nondet_crossklctx_loss/" + mode, e_nondet_crossklctx_loss, step=global_step)
            summary_e_nondet_rcrossklctx_loss = tf.summary.scalar("e_nondet_rcrossklctx_loss/" + mode, e_nondet_rcrossklctx_loss, step=global_step)
            summary_e_nondet_crosswsctx_loss = tf.summary.scalar("e_nondet_crosswsctx_loss/" + mode, e_nondet_crosswsctx_loss, step=global_step)

            summary_e_nondet_kl_loss = tf.summary.scalar("e_nondet_kl_loss/" + mode, e_nondet_kl_loss, step=global_step)
            summary_e_nondet_negent_loss = tf.summary.scalar("e_nondet_negent_loss/" + mode, e_nondet_negent_loss, step=global_step)
            summary_e_nondet_prior_loss = tf.summary.scalar("e_nondet_prior_loss/" + mode, e_nondet_prior_loss, step=global_step)
            summary_e_nondet_prior2_loss = tf.summary.scalar("e_nondet_prior2_loss/" + mode, e_nondet_prior2_loss, step=global_step)
            summary_e_nondet_crosskl_loss = tf.summary.scalar("e_nondet_crosskl_loss/" + mode, e_nondet_crosskl_loss, step=global_step)
            summary_e_nondet_rcrosskl_loss = tf.summary.scalar("e_nondet_rcrosskl_loss/" + mode, e_nondet_rcrosskl_loss, step=global_step)
            summary_e_nondet_crossws_loss = tf.summary.scalar("e_nondet_crossws_loss/" + mode, e_nondet_crossws_loss, step=global_step)
            summary_e_nondet_kl2_loss = tf.summary.scalar("e_nondet_kl2_loss/" + mode, e_nondet_kl2_loss, step=global_step)
            summary_e_nondet_reg_loss = tf.summary.scalar("e_nondet_reg_loss/" + mode, e_nondet_reg_loss, step=global_step)
            summary_e_nondet_prior_adv_loss = tf.summary.scalar("e_nondet_prior_adv_loss/" + mode, e_nondet_prior_adv_loss, step=global_step)

            summary_g_nondet_ce_loss = tf.summary.scalar("g_nondet_ce_loss/" + mode, g_nondet_ce_loss, step=global_step)
            summary_g_nondet_vgg_loss = tf.summary.scalar("g_nondet_vgg_loss/" + mode, g_nondet_vgg_loss, step=global_step)
            summary_g_nondet_adv_loss = tf.summary.scalar("g_nondet_adv_loss/" + mode, g_nondet_adv_loss, step=global_step)
            summary_g_nondet_feature_loss = tf.summary.scalar("g_nondet_feature_loss/" + mode, g_nondet_feature_loss, step=global_step)
            summary_g_nondet_reg_loss = tf.summary.scalar("g_nondet_reg_loss/" + mode, g_nondet_reg_loss, step=global_step)
        
            summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_adv_loss/" + mode, d_nondet_adv_loss, step=global_step)
            summary_d_nondet_reg_loss = tf.summary.scalar("d_nondet_reg_loss/" + mode, d_nondet_reg_loss, step=global_step)
            
            summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_score_real/" + mode, d_nondet_score_real, step=global_step)
            summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_score_fake/" + mode, d_nondet_score_fake, step=global_step)
            summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_score_diff/" + mode, d_nondet_score_diff, step=global_step)
            
            summary_de_nondet_prior_adv_loss = tf.summary.scalar("de_nondet_prior_adv_loss/" + mode, de_nondet_prior_adv_loss, step=global_step)
            summary_de_nondet_prior_reg_loss = tf.summary.scalar("de_nondet_prior_reg_loss/" + mode, de_nondet_prior_reg_loss, step=global_step)
       
        inputs = (real_ctx, real_x, real_x_pose, real_x_segmap, real_x_segmap_onehot)
            
        losses_det = types.SimpleNamespace()
        losses_det.g_det_loss = g_det_loss
        losses_det.gp_det_loss = gp_det_loss
        losses_det.de_det_loss = de_det_loss
        losses_det.d_det_loss = d_det_loss

        losses_nondet = types.SimpleNamespace()
        losses_nondet.g_nondet_loss = g_nondet_loss
        losses_nondet.gp_nondet_loss = gp_nondet_loss
        losses_nondet.de_nondet_loss = de_nondet_loss
        losses_nondet.d_nondet_loss = d_nondet_loss
       
        outputs_det = (fake_det_x, fake_det_x_var, fake_det_x_segmap)

        outputs_resample_det = (resample_fake_det_x, resample_fake_det_x_segmap)
        outputs_random_det = (random_fake_det_x, random_fake_det_x_segmap)

        outputs_nondet = (fake_nondet_x,)

        outputs_resample_nondet = (resample_fake_nondet_x,)
        outputs_random_nondet = (random_fake_nondet_x,)

        return inputs, (losses_det, outputs_det, summaries_det), (outputs_resample_det, outputs_random_det), (losses_nondet, outputs_nondet, summaries_nondet), (outputs_resample_nondet, outputs_random_nondet) 

    def build_fake_inputs(self, batch_size):
        real_ctx = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, self.img_height, self.img_width, self.img_ch)))
        real_ctx_pose = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, 3)))
        real_x = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, self.img_height, self.img_width, self.img_ch)))
        real_x_pose = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, 3)))
        real_x_segmap = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, self.img_height, self.img_width)))
        real_x_segmap_onehot = tf.convert_to_tensor(np.zeros(dtype=np.float32, shape=(batch_size, self.img_height, self.img_width, self.segmap_ch)))
        inputs = (real_ctx, real_ctx_pose, real_x, real_x_pose, real_x_segmap, real_x_segmap_onehot)
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
            d_det_lr = decay_fn(self.init_lr * 2)

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
            d_det_lr = decay_fn(self.init_lr)

        self.G_nondet_optim = tf.keras.optimizers.Adam(g_nondet_lr, beta_1=beta1, beta_2=beta2)
        self.GP_nondet_optim = tf.keras.optimizers.Adam(gp_nondet_lr, beta_1=beta1, beta_2=beta2)
        self.DE_nondet_optim = tf.keras.optimizers.Adam(de_nondet_lr, beta_1=beta1, beta_2=beta2)
        self.D_nondet_optim = tf.keras.optimizers.Adam(d_nondet_lr, beta_1=beta1, beta_2=beta2)

        self.G_det_optim = tf.keras.optimizers.Adam(g_det_lr, beta_1=beta1, beta_2=beta2)
        self.GP_det_optim = tf.keras.optimizers.Adam(gp_det_lr, beta_1=beta1, beta_2=beta2)
        self.DE_det_optim = tf.keras.optimizers.Adam(de_det_lr, beta_1=beta1, beta_2=beta2)
        self.D_det_optim = tf.keras.optimizers.Adam(d_det_lr, beta_1=beta1, beta_2=beta2)

        t_vars = tf.compat.v1.trainable_variables()
        self.G_nondet_vars = [var for var in t_vars if 'generator_nondet_x' in var.name or 'encoder_nondet_code' in var.name or 'encoder_nondet_ctxcode' in var.name]
        self.GP_nondet_vars = [var for var in t_vars if 'generator_nondet_code' in var.name or 'prior_nondet_supercode' in var.name or 'encoder_nondet_supercode' in var.name or 'encoder_nondet_ctxcode' in var.name]
        self.DE_nondet_vars = [var for var in t_vars if 'discriminator_nondet_prior_code' in var.name]
        self.D_nondet_vars = [var for var in t_vars if 'discriminator_nondet_x' in var.name]

        self.G_det_vars = [var for var in t_vars if 'generator_det_x' in var.name or 'encoder_det_code' in var.name in var.name in var.name or 'encoder_det_ctxcode' in var.name]
        self.GP_det_vars = [var for var in t_vars if 'generator_det_code' in var.name or 'prior_det_supercode' in var.name or 'encoder_det_supercode' in var.name or 'encoder_det_ctxcode' in var.name]
        self.DE_det_vars = [var for var in t_vars if 'discriminator_det_prior_code' in var.name]
        self.D_det_vars = [var for var in t_vars if 'discriminator_det_x' in var.name]


    def report_losses_det(self, counter, epoch, idx, duration, g_det_loss, gp_det_loss, de_det_loss, mode=None):
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f %s g_det_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, mode, g_det_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f %s gp_det_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, mode, gp_det_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f %s de_det_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, mode, de_det_loss))
        sys.stdout.flush()

    def report_losses_nondet(self, counter, epoch, idx, duration, g_nondet_loss, gp_nondet_loss, de_nondet_loss, d_nondet_loss, mode=None):
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f %s g_nondet_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, mode, g_nondet_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f %s gp_nondet_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, mode, gp_nondet_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f %s de_nondet_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, mode, de_nondet_loss))
        print("Counter: [%2d]: Epoch: [%2d] [%5d/%5d] time: %4.4f %s d_nondet_loss: %.8f" % (
            counter, epoch, idx, self.iteration, duration, mode, d_nondet_loss))
        sys.stdout.flush()

    def report_inputs(self, epoch, idx, real_ctx, real_ctx_pose, real_x, real_x_pose, real_x_segmap, real_x_segmap_onehot, mode=None):
        total_batch_size = real_ctx.shape[0]

        save_images(real_ctx, [total_batch_size, 1],
                   './{}/{}_real_ctximage_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx+1))

        imsave(real_x_segmap, [total_batch_size, 1],
                    './{}/{}_real_segmap_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx+1))

        save_images(real_x, [total_batch_size, 1],
                   './{}/{}_real_image_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx+1))

    def report_outputs_det(self, epoch, idx, fake_det_x, fake_det_x_var, fake_det_x_segmap, mode=None):
        total_batch_size = fake_det_x.shape[0]

        save_images(fake_det_x, [total_batch_size, 1],
                    './{}/{}_fake_det_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx+1))
        imsave(image_to_uint8(fake_det_x_var), [total_batch_size, 1],
                    './{}/{}_fake_det_var_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx+1))
        save_segmaps(fake_det_x_segmap, self.color_value_dict, [total_batch_size, 1],
                     './{}/{}_fake_det_segmap_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx+1))

    def report_outputs_random_det(self, epoch, idx, random_fake_det_x, random_fake_det_x_segmap, mode=None):
        total_batch_size = random_fake_det_x.shape[0]

        save_images(random_fake_det_x, [total_batch_size, 1],
                    './{}/{}_fake_det_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx + 1))
 
        save_segmaps(random_fake_det_x_segmap, self.color_value_dict, [total_batch_size, 1],
                     './{}/{}_fake_det_segmap_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx+1))

    def report_outputs_nondet(self, epoch, idx, fake_nondet_x, mode=None):
        total_batch_size = fake_nondet_x.shape[0]

        save_images(fake_nondet_x, [total_batch_size, 1],
                    './{}/{}_fake_nondet_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx+1))

    def report_outputs_random_nondet(self, epoch, idx, random_fake_nondet_x, mode=None):
        total_batch_size = random_fake_nondet_x.shape[0]

        save_images(random_fake_nondet_x, [total_batch_size, 1],
                    './{}/{}_fake_nondet_{:03d}_{:05d}.png'.format(self.sample_dir, mode, epoch, idx + 1))

    def report_tensors(self, epoch, idx, f, tensors, **kwargs):
        f(epoch, idx, *map(lambda tensor: tensor.numpy(), tensors), **kwargs)

    def report_all_outputs_det(self, epoch, idx, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, mode=None):
        self.report_tensors(epoch, idx, self.report_outputs_det, result_outputs_det, mode=mode)
        self.report_tensors(epoch, idx, self.report_outputs_random_det, result_outputs_resample_det, mode=mode + "_resample")
        self.report_tensors(epoch, idx, self.report_outputs_random_det, result_outputs_random_det, mode=mode + "_random")

    def report_all_outputs_nondet(self, epoch, idx, result_outputs_nondet, result_outputs_resample_nondet, result_outputs_random_nondet, mode=None):
        self.report_tensors(epoch, idx, self.report_outputs_nondet, result_outputs_nondet, mode=mode)
        self.report_tensors(epoch, idx, self.report_outputs_random_nondet, result_outputs_resample_nondet, mode=mode + "_resample")
        self.report_tensors(epoch, idx, self.report_outputs_random_nondet, result_outputs_random_nondet, mode=mode + "_random")

    def train(self):
        distribute_strategy = tf.distribute.MirroredStrategy()
        distributed_batch_size = distribute_strategy.num_replicas_in_sync * self.batch_size
        print("Distributed replicas: %s" % (distribute_strategy.num_replicas_in_sync,))
        print("Distributed batch size: %s" % (distributed_batch_size,))

        def build_dataset(img_class):
            dataset = tf.data.Dataset.from_tensor_slices((img_class.ctximage, img_class.ctxpose, img_class.image, img_class.pose, img_class.segmap))
            dataset = dataset.shuffle(len(img_class.image), reshuffle_each_iteration=True).repeat(None)
            dataset = dataset.map(img_class.image_processing, num_parallel_calls=64*distributed_batch_size).batch(distributed_batch_size)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            dataset = distribute_strategy.experimental_distribute_dataset(dataset)
            return dataset

        train_dataset = iter(build_dataset(self.img_class_train))
        validate_dataset = iter(build_dataset(self.img_class_validate))

        with distribute_strategy.scope():
            # prepare model
            print("> Preparing model", time.time())
            self.prepare_model()

            # build model
            print("> Building model", time.time())
            @tf.function
            def build():
                def build_fn():
                    global_step = tf.compat.v1.train.get_or_create_global_step()
                    fake_batch_size = 1
                    (real_ctx, real_ctx_pose, real_x, real_x_pose, real_x_segmap, real_x_segmap_onehot) = self.build_fake_inputs(fake_batch_size)
                    return self.execute_model(real_ctx, real_ctx_pose, real_x, real_x_pose, real_x_segmap, real_x_segmap_onehot)
                distribute_strategy.experimental_run_v2(build_fn, args=())
            build()
            del build

            # show network architecture
            show_all_variables()

            # build optimizers
            print("> Building optimizers", time.time())
            self.build_optimizers()

            # saver to save model
            print("> Loading checkpoint", time.time())
            checkpoint_manager = self.load(self.checkpoint_dir)

            # record start time
            print("> Training", time.time())
            start_time = time.time()

            def train_loop():
                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def train_det(*inputs):
                    def work_fn(*inputs):
                        global_step = tf.compat.v1.train.get_or_create_global_step()
                        with tf.GradientTape(persistent=True) as tape:
                            inputs, (losses_det, outputs_det, summaries_det), (outputs_resample_det, outputs_random_det), _, _ = self.execute_model(*inputs)
                        result_losses_det = (losses_det.g_det_loss, losses_det.gp_det_loss, losses_det.de_det_loss)
                        self.D_det_optim.apply_gradients(zip(tape.gradient(losses_det.d_det_loss, self.D_det_vars), self.D_det_vars))
                        self.DE_det_optim.apply_gradients(zip(tape.gradient(losses_det.de_det_loss, self.DE_det_vars), self.DE_det_vars))
                        self.GP_det_optim.apply_gradients(zip(tape.gradient(losses_det.gp_det_loss, self.GP_det_vars), self.GP_det_vars))
                        self.G_det_optim.apply_gradients(zip(tape.gradient(losses_det.g_det_loss, self.G_det_vars), self.G_det_vars))
                        summaries_det(global_step, "train")
                        global_step.assign_add(1)
                        return tf.convert_to_tensor(global_step), result_losses_det

                    result_counter, result_losses_det = distribute_strategy.experimental_run_v2(work_fn, args=inputs)

                    converted_counter = tf.reduce_mean(distribute_strategy.experimental_local_results(result_counter))
                    converted_losses_det = list(map(lambda result_loss: tf.reduce_mean(distribute_strategy.experimental_local_results(result_loss)), result_losses_det))

                    return converted_counter, converted_losses_det

                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def eval_det(*inputs, summary_mode):
                    def work_fn(*inputs):
                        global_step = tf.compat.v1.train.get_or_create_global_step()
                        inputs, (losses_det, outputs_det, summaries_det), (outputs_resample_det, outputs_random_det), _, _ = self.execute_model(*inputs)
                        result_losses_det = (losses_det.g_det_loss, losses_det.gp_det_loss, losses_det.de_det_loss)
                        if summary_mode:
                            summaries_det(global_step, summary_mode)
                        return inputs, result_losses_det, outputs_det, outputs_resample_det, outputs_random_det

                    result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det = distribute_strategy.experimental_run_v2(work_fn, args=inputs)

                    converted_inputs = list(map(lambda result_input: tf.concat(distribute_strategy.experimental_local_results(result_input), axis=0), result_inputs))
                    converted_losses_det = list(map(lambda result_loss: tf.reduce_mean(distribute_strategy.experimental_local_results(result_loss)), result_losses_det))

                    def convert_outputs(result_outputs):
                        return list(map(lambda result_output: tf.concat(distribute_strategy.experimental_local_results(result_output), axis=0), result_outputs))

                    converted_outputs_det = convert_outputs(result_outputs_det)
                    converted_outputs_resample_det = convert_outputs(result_outputs_resample_det)
                    converted_outputs_random_det = convert_outputs(result_outputs_random_det)

                    return converted_inputs, converted_losses_det, converted_outputs_det, converted_outputs_resample_det, converted_outputs_random_det

                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def train_nondet(*inputs):
                    def work_fn(*inputs):
                        global_step = tf.compat.v1.train.get_or_create_global_step()
                        with tf.GradientTape(persistent=True) as tape:
                            inputs, (losses_det, outputs_det, summaries_det), (outputs_resample_det, outputs_random_det), (losses_nondet, outputs_nondet, summaries_nondet), (outputs_resample_nondet, outputs_random_nondet) = self.execute_model(*inputs)
                        result_losses_det = (losses_det.g_det_loss, losses_det.gp_det_loss, losses_det.de_det_loss)
                        result_losses_nondet = (losses_nondet.g_nondet_loss, losses_nondet.gp_nondet_loss, losses_nondet.de_nondet_loss, losses_nondet.d_nondet_loss)
                        self.D_nondet_optim.apply_gradients(zip(tape.gradient(losses_nondet.d_nondet_loss, self.D_nondet_vars), self.D_nondet_vars))
                        self.DE_nondet_optim.apply_gradients(zip(tape.gradient(losses_nondet.de_nondet_loss, self.DE_nondet_vars), self.DE_nondet_vars))
                        self.GP_nondet_optim.apply_gradients(zip(tape.gradient(losses_nondet.gp_nondet_loss, self.GP_nondet_vars), self.GP_nondet_vars))
                        self.G_nondet_optim.apply_gradients(zip(tape.gradient(losses_nondet.g_nondet_loss, self.G_nondet_vars), self.G_nondet_vars))
                        summaries_det(global_step, "train")
                        summaries_nondet(global_step, "train")
                        global_step.assign_add(1)
                        return tf.convert_to_tensor(global_step), result_losses_det, result_losses_nondet

                    result_counter, result_losses_det, result_losses_nondet = distribute_strategy.experimental_run_v2(workfn, args=inputs)
                   
                    converted_counter = tf.reduce_mean(distribute_strategy.experimental_local_results(result_counter))
                    converted_losses_det = list(map(lambda result_loss: tf.reduce_mean(distribute_strategy.experimental_local_results(result_loss)), result_losses_det))
                    converted_losses_nondet = list(map(lambda result_loss: tf.reduce_mean(distribute_strategy.experimental_local_results(result_loss)), result_losses_nondet))

                    return converted_counter, converted_losses_det, converted_losses_nondet

                @tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS,tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
                def eval_nondet(*inputs, summary_mode):
                    def work_fn(*inputs):
                        global_step = tf.compat.v1.train.get_or_create_global_step()
                        inputs, (losses_det, outputs_det, summaries_det), (outputs_resample_det, outputs_random_det), (losses_nondet, outputs_nondet, summaries_nondet), (outputs_resample_nondet, outputs_random_nondet) = self.execute_model(*inputs)
                        result_losses_det = (losses_det.g_det_loss, losses_det.gp_det_loss, losses_det.de_det_loss)
                        result_losses_nondet = (losses_nondet.g_nondet_loss, losses_nondet.gp_nondet_loss, losses_nondet.de_nondet_loss, losses_nondet.d_nondet_loss)
                        if summary_mode:
                            summaries_det(global_step, summary_mode)
                            summaries_nondet(global_step, summary_mode)
                        return inputs, result_losses_det, outputs_det, outputs_resample_det, outputs_random_det, result_losses_nondet, outputs_nondet, outputs_resample_nondet, outputs_random_nondet

                    result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, result_losses_nondet, result_outputs_nondet, result_outputs_resample_nondet, result_outputs_random_nondet = distribute_strategy.experimental_run_v2(workfn, args=inputs)
                   
                    converted_inputs = list(map(lambda result_input: tf.concat(distribute_strategy.experimental_local_results(result_input), axis=0), result_inputs))

                    converted_losses_det = list(map(lambda result_loss: tf.reduce_mean(distribute_strategy.experimental_local_results(result_loss)), result_losses_det))
                    converted_losses_nondet = list(map(lambda result_loss: tf.reduce_mean(distribute_strategy.experimental_local_results(result_loss)), result_losses_nondet))

                    def convert_outputs(result_outputs):
                        return list(map(lambda result_output: tf.concat(distribute_strategy.experimental_local_results(result_output), axis=0), result_outputs))

                    converted_outputs_det = convert_outputs(result_outputs_det)
                    converted_outputs_resample_det = convert_outputs(result_outputs_resample_det)
                    converted_outputs_random_det = convert_outputs(result_outputs_random_det)

                    converted_outputs_nondet = convert_outputs(result_outputs_nondet)
                    converted_outputs_resample_nondet = convert_outputs(result_outputs_resample_nondet)
                    converted_outputs_random_nondet = convert_outputs(result_outputs_random_nondet)

                    return converted_inputs, converted_losses_det, converted_outputs_det, converted_outputs_resample_det, converted_outputs_random_det, converted_losses_nondet, converted_outputs_nondet, converted_outputs_resample_nondet, converted_outputs_random_nondet

                # training loop
                with self.writer.as_default():
                    while True:
                        print("> Training step %s" % time.time())

                        print("L1", time.time())
                        if self.train_det:
                            print("L2DET", time.time())
                            inputs = next(train_dataset)
                            counter, result_losses_det = train_det(*inputs)
                        if self.train_nondet:
                            print("L2NONDET", time.time())
                            inputs = next(train_dataset)
                            counter, result_losses_det, result_losses_nondet = train_nondet(*inputs)

                        print("L4",  time.time())
                        epoch = (counter-1) // self.iteration 
                        idx = (counter-1) % self.iteration

                        print("L5",  time.time())
                        self.report_losses_det(counter, epoch, idx, time.time() - start_time, *result_losses_det, mode="train")
                        if self.train_nondet:
                            self.report_losses_nondet(counter, epoch, idx, time.time() - start_time, *result_losses_nondet, mode="train")

                        if (idx+1) % self.print_freq == 0:
                            inputs = next(train_dataset)
                            if not self.train_nondet:
                                print("L6DET", time.time())
                                result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det = eval_det(*inputs, summary_mode=None)
                            else:
                                print("L7NONDET", time.time())
                                result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, result_losses_nondet, result_outputs_nondet, result_outputs_resample_nondet, result_outputs_random_nondet = eval_nondet(*inputs, summary_mode=None)

                            print("L6",  time.time())
                            self.report_tensors(epoch, idx, self.report_inputs, result_inputs, mode="train")
                            self.report_all_outputs_det(epoch, idx, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, mode="train")
                            if self.train_nondet:
                                self.report_all_outputs_nondet(epoch, idx, result_outputs_nondet, result_outputs_resample_nondet, result_outputs_random_nondet, mode="train")

                        if (idx+1) % self.validate_freq == 0: 
                            inputs = next(validate_dataset)
                            if not self.train_nondet:
                                print("L6DETV", time.time())
                                result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det = eval_det(*inputs, summary_mode="validate")
                            else:
                                print("L7NONDETV", time.time())
                                result_inputs, result_losses_det, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, result_losses_nondet, result_outputs_nondet, result_outputs_resample_nondet, result_outputs_random_nondet = eval_nondet(*inputs, summary_mode="validate")

                            print("L8V",  time.time())
                            self.report_losses_det(counter, epoch, idx, time.time() - start_time, *result_losses_det, mode="validate")
                            if self.train_nondet:
                                self.report_losses_nondet(counter, epoch, idx, time.time() - start_time, *result_losses_nondet, mode="validate")

                            if (idx+1) % self.print_freq == 0:
                                print("L9V",  time.time())
                                self.report_tensors(epoch, idx, self.report_inputs, result_inputs, mode="validate")
                                self.report_all_outputs_det(epoch, idx, result_outputs_det, result_outputs_resample_det, result_outputs_random_det, mode="validate")
                                if self.train_nondet:
                                    self.report_all_outputs_nondet(epoch, idx, result_outputs_nondet, result_outputs_resample_nondet, result_outputs_random_nondet, mode="validate")


                        if counter-1 > 0 and (counter-1) % self.save_freq == 0:
                            print("L10",  time.time())
                            checkpoint_manager.save()

                    print("L11",  time.time())
                    self.writer.flush()
                    
            train_loop()

    @property
    def model_dir(self):
        return "{}_dataset={}".format(self.model_name, self.dataset_name)

        n_dis = str(self.n_scale) + 'multi_' + str(self.n_dis) + 'dis'


        sn = ('_sndet' if self.sn_det else '') + ('_snnondet' if self.sn_nondet else '')

        if self.TTUR :
            TTUR = '_TTUR'
        else :
            TTUR = ''


        return "{}_dataset={}__gan{}__n_dis={}__n_critic={}__adv_weight={}__vgg_weight={}__feature_weight={}__kl_weight={}__{}{}__ch={}__num_upsampling_layers={}__ch={}__num_upsampling_layers={}".format(self.model_name, self.dataset_name,
                                                                   self.gan_type, n_dis, self.n_critic,
                                                                   self.adv_weight, self.vgg_weight, self.feature_weight,
                                                                   self.kl_weight,
                                                                   sn, TTUR, self.ch, self.num_upsampling_layers, self.ch, self.num_upsampling_layers)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        full_vars = dict([(var.name, var) for var in tf.compat.v1.global_variables()])
        full_checkpoint = tf.train.Checkpoint(**full_vars)
        checkpoint_manager = tf.train.CheckpointManager(full_checkpoint, checkpoint_dir, max_to_keep=1000)

        # restore check-point if it exits
        if checkpoint_manager.latest_checkpoint:
            checkpoint_vars = dict(tf.train.list_variables(checkpoint_manager.latest_checkpoint))
            for (checkpoint_var_name, checkpoint_var_shape) in checkpoint_vars.items():
                print("Checkpoint variable: %s%s" % (checkpoint_var_name, checkpoint_var_shape))

            filtered_vars = {}
            for full_var_name in full_vars:
                full_var_shape = full_vars[full_var_name].shape
                checkpoint_var_name = full_var_name.replace("/", ".S") + "/.ATTRIBUTES/VARIABLE_VALUE"
                if checkpoint_var_name in checkpoint_vars and checkpoint_vars[checkpoint_var_name] == full_var_shape:
                    print("Restoring variable: %s%s -> %s" % (full_var_name, full_var_shape, checkpoint_var_name))
                    filtered_vars[full_var_name] = full_vars[full_var_name]
                else:
                    print("NOT restoring variable: %s%s -> %s" % (full_var_name, full_var_shape, checkpoint_var_name))
            filtered_vars["global_step"] = full_vars["global_step:0"]
            filtered_checkpoint = tf.train.Checkpoint(**filtered_vars)
            filtered_checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(" [*] Success to read {}".format(checkpoint_manager.latest_checkpoint))
        else:
            print(" [*] Failed to find a checkpoint")
        return checkpoint_manager

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
