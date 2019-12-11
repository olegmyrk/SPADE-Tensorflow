import sys
from ops import *
from utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from tqdm import tqdm
from vgg19_keras import VGGLoss
from unet import unet
from masked_autoregressive import conditional_masked_autoregressive_template

class SPADE(object):
    def __init__(self, sess, args):

        self.model_name = 'SPADE'

        self.sess = sess
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

    def encoder_code(self, x_init, epsilon=1e-8, reuse=False, scope=None):
        with tf.variable_scope(scope, reuse=reuse):
            x, channel = self.encoder_base(x_init, self.ch)

            mean = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_mean')
            var = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_var')
            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def prior_code(self, batch_size, channel_multiplier=4):
        out_channel = self.ch * channel_multiplier
        mean = tf.zeros([batch_size, out_channel])
        var = tf.zeros([batch_size, out_channel])
        return mean, var

    def prior_code_dist(self, code, channel_multiplier=4, epsilon=1e-8, reuse=False, scope=None):
        context = code
        out_channel = self.ch * channel_multiplier
        hidden_channel = self.ch * 64

        with tf.variable_scope(scope, reuse=reuse):
            bijectors = []
            for i in range(self.code_dist_num_layers):
                bijectors.append(tfb.MaskedAutoregressiveFlow(
                  shift_and_log_scale_fn=conditional_masked_autoregressive_template(
                      code, hidden_layers=[hidden_channel, hidden_channel], name=scope + "/maf_" + str(i))))

                #context_gamma = fully_connected(context, units=out_channel, scope='linear_gamma_' + str(i))
                #context_beta = fully_connected(context, units=out_channel, scope='linear_beta_' + str(i))
                bijectors.append(tfb.BatchNormalization(
                    batchnorm_layer=tf.layers.BatchNormalization(
                                        #gamma_constraint=lambda x:tf.ones(shape=[out_channel]), beta_constraint=lambda x:tf.zeros(shape=[out_channel]),
                                        name=scope + '/batch_norm_' + str(i)),
                    name=scope + '/batch_norm_bijector' + str(i)))
                #bijectors.append(tfb.AffineLinearOperator(context_beta, tf.linalg.LinearOperatorDiag(context_gamma)))

                permutation=tf.get_variable('permutation_'+str(i), initializer=np.random.permutation(out_channel).astype("int32"), trainable=False)
                bijectors.append(tfb.Permute(permutation))
                
            flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

            mvn_loc = fully_connected(context, units=out_channel, scope='mvn_loc')
            mvn_scale_diag = epsilon + tf.math.sigmoid(fully_connected(context, units=out_channel, scope='mvn_scale_logdiag'))

            #import correlation_cholesky
            #mvn_corr_bijector = correlation_cholesky.CorrelationCholesky(name=scope + "/CorrelationCholesky")
            #mvn_corr = mvn_corr_bijector.forward(fully_connected(context, units=(out_channel-1)*out_channel/2, scope='mvn_corr_seed'))
            #mvn_scale = tf.linalg.transpose(mvn_corr * tf.expand_dims(mvn_scale_diag,2)) * tf.expand_dims(mvn_scale_diag,2)
            #mvn_dist = tfd.MultivariateNormalFullCovariance(mvn_loc, mvn_scale, name=scope + "/MultivariateNormalFullCovariance")

            #_, mvn_scale_u, _ = tf.linalg.svd(tf.reshape(fully_connected(context, units=out_channel*out_channel, scope='mvn_scale_seed'), [-1, out_channel, out_channel]), full_matrices=True)
            #mvn_scale = tf.linalg.matmul(tf.matmul(mvn_scale_u, mvn_scale_diag), tf.linalg.transpose(mvn_scale_u))
            #mvn_dist = tfd.MultivariateNormalFullCovariance(mvn_loc, mvn_scale, name=scope + "/MultivariateNormalFullCovariance")

            mvn_dist = tfd.MultivariateNormalDiag(mvn_loc, mvn_scale_diag, name=scope + "/MultivariateNormalDiag")

            dist = tfd.TransformedDistribution(
                            distribution=mvn_dist,
                            bijector=flow_bijector
                        )
        return dist

    def encoder_supercode(self, x_init, channel_multiplier=4, epsilon=1e-8, reuse=False, scope=None):
        out_channel = self.ch*channel_multiplier
        hidden_channel = self.ch*64
        with tf.variable_scope(scope, reuse=reuse):
            x = x_init
            for i in range(self.code_num_layers):
                x = constin_fcblock(x, hidden_channel, scope="fcblock_" + str(i))
                x = lrelu(x, 0.2)

            mean = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_mean')
            var = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_var')

            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def generator_code(self, code, x_init, epsilon=1e-8, reuse=False, scope=None):
        out_channel = self.ch*4
        hidden_channel = self.ch*64
        with tf.variable_scope(scope, reuse=reuse):
            x = x_init
            for i in range(self.code_num_layers):
                x = adain_fcblock(code, x, hidden_channel, scope="fcblock_" + str(i))
                x = lrelu(x, 0.2)

            mean = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_mean')
            #var = tf.get_variable("var", [], initializer=tf.constant_initializer(0.0))
            var = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_var')

            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def generator(self, code, z=None, epsilon=1e-8, reuse=False, scope="generator"):
        context = code

        context_depth = 8
        context_ch = 10*context.get_shape()[-1]
        channel = self.ch * 4 * 4
        with tf.variable_scope(scope, reuse=reuse):
            features = []

            #for i in range(context_depth):
            #    context = fully_connected(context, context_ch, use_bias=True, sn=self.sn, scope='linear_context_' + str(i))
            #    context = lrelu(context, 0.2)
            
            x = fully_connected(z, z.get_shape()[-1], use_bias=True, sn=self.sn, scope='linear_noise')

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
            mean = conv(x, channels=self.out_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_mean')
            var = conv(x, channels=self.out_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_var')
            logits = conv(x, channels=self.segmap_out_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_logits')

            return features, [[tanh(mean), tf.math.log(epsilon + tf.sigmoid(var))], logits]

    def generator_features(self, code, features, z, reuse=False, scope=None):
        context = code
        features = list(reversed(features))

        context_depth = 8
        context_ch = 10*context.get_shape()[-1]
        channel = self.ch * 4 * 4
        with tf.variable_scope(scope, reuse=reuse):

            #for i in range(context_depth):
            #    context = fully_connected(context, context_ch, use_bias=True, sn=self.sn, scope='linear_context_' + str(i))
            #    context = lrelu(context, 0.2)
            
            x = fully_connected(z, z.get_shape()[-1], use_bias=True, sn=self.sn, scope='linear_noise')

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
            x = x + features.pop()

            x = up_sample(x, scale_factor=2)
            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_1')
            x = x + features.pop()

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_2')
            x = x + features.pop()

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = adain_resblock(context, x, channels=channel//2, use_bias=True, sn=self.sn, scope='resblock_' + str(i))
                x = x + features.pop()

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')
            x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')
            x = x + features.pop()

            x = lrelu(x, 0.2)
            x = conv(x, channels=self.out_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='logit')
            x = tanh(x)

            return x

    def generator_spatial(self, code, scaffold, z, reuse=False, scope=None):
        context = code

        context_depth = 8
        context_ch = 10*context.get_shape()[-1]
        channel = self.ch * 4 * 4
        with tf.variable_scope(scope, reuse=reuse):

            #for i in range(context_depth):
            #    context = fully_connected(context, context_ch, use_bias=True, sn=self.sn, scope='linear_context_' + str(i))
            #    context = lrelu(context, 0.2)
            
            x = fully_connected(z, z.get_shape()[-1], use_bias=True, sn=self.sn, scope='linear_noise')

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
            x = conv(x, channels=self.out_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator_code(self, x, reuse=False, scope=None, label=None):
        channel = x.get_shape()[-1]
        with tf.variable_scope(scope, reuse=reuse):
            x = fully_connected(x, channel * 10, use_bias=True, sn=self.sn, scope='linear_x_1')
            x = lrelu(x, 0.2)

            x = fully_connected(x, channel * 10, use_bias=True, sn=self.sn, scope='linear_x_2')
            x = lrelu(x, 0.2)

            x = fully_connected(x, channel * 10, use_bias=True, sn=self.sn, scope='linear_x_3')
            x = lrelu(x, 0.2)

            x = fully_connected(x, channel * 10, use_bias=True, sn=self.sn, scope='linear_x_4')
            x = lrelu(x, 0.2)

            z = fully_connected(x, 1, sn=self.sn, scope='linear_z')

            z_summary = [tf.summary.scalar(label + ".logit", tf.reduce_sum(z))]

            return [[z]], z_summary

    def full_discriminator(self, x_init, code=None, reuse=False, scope=None, label=None):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
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
            #z0 = tf.reduce_mean(x0*fixed_code, 1, keep_dims=True)
            #z0 = -tf.reduce_mean(tf.math.square(x0-fixed_code), 1, keep_dims=True)
            #z0 = -tf.reduce_mean(tf.math.abs(x0-fixed_code), 1, keep_dims=True)
            
            z0_summary = tf.summary.scalar(label + ".logit0", tf.reduce_mean(z0))
            z = z0
            D_summary = [z0_summary]

            #if code is not None:
            #   x1 = fully_connected(x, channel, use_bias=True, sn=self.sn, scope='linear_x1')
            #   x1 = lrelu(x1, 0.2)
            #   z1 = -tf.reduce_sum(tf.math.abs(x1-code), 1, keep_dims=True)
            #   z1_summary = tf.summary.scalar(label + ".logit1", tf.reduce_mean(z1))
            #   D_summary = D_summary + [z1_summary]

            z = tf.reshape(z, [z.get_shape()[0], 1, 1, 1])

            D_logit = [feature_loss + [z]]
            return D_logit, D_summary

    def feature_discriminator(self, x_init, code, reuse=False, scope=None, label=None):
        D_logit = []
        D_summary = []
        with tf.variable_scope(scope, reuse=reuse):
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

                feature_summary = tf.summary.scalar(label + ".logit_" + str(scale) , tf.reduce_mean(x))
                D_summary.append(feature_summary)

                x_init = down_sample_avg(x_init)

            return D_logit, D_summary

    ##################################################################################
    # Model
    ##################################################################################

    def discriminate_code(self, real_code_img, fake_code_img, name=None):
        real_logit = self.discriminator_code(real_code_img, scope='discriminator_' + name + '_code', label='real_' + name + '_code')
        fake_logit = self.discriminator_code(fake_code_img, reuse=True, scope='discriminator_' + name + '_code', label='fake_' + name + '_code')

        return real_logit, fake_logit

    def discriminate(self, real_img, fake_img, code=None, name=None):
        real_logit = self.discriminator(real_img, code, scope='discriminator_' + name + '', label='real_' + name + '')
        fake_logit = self.discriminator(fake_img, code, reuse=True, scope='discriminator_' + name + '', label='fake_' + name + '')

        return real_logit, fake_logit

    def gradient_penalty(self, real, fake, code, discriminator, name=None):
        shape = tf.shape(real)
        if self.gan_type == 'dragan':
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else:
            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha * real + (1. - alpha) * fake

        logit, logit_summary = discriminator(interpolated, code=code, reuse=True, scope='discriminator_' + name + '', label='interpolated_' + name + '')

        GP = []


        for i in range(len(logit)) :
            grad = tf.gradients(logit[i][-1], interpolated)[0]  # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp':
                GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))

        return tf.reduce_mean(GP)


    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image"""
        img_class = Image_data(self.img_height, self.img_width, self.img_ch, self.segmap_img_ch, self.dataset_path, self.augment_flag)
        img_class.preprocess()
        self.color_value_dict = img_class.color_value_dict
        self.out_ch = self.img_ch
        self.segmap_out_ch = len(img_class.color_value_dict)

        self.dataset_num = len(img_class.image)


        ctximg_and_img_and_segmap = tf.data.Dataset.from_tensor_slices((img_class.ctximage, img_class.image, img_class.segmap))


        gpu_device = '/gpu:0'
        ctximg_and_img_and_segmap = ctximg_and_img_and_segmap.apply(shuffle_and_repeat(self.dataset_num)).apply(
            map_and_batch(img_class.image_processing, self.batch_size, num_parallel_batches=16,
                          drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))

        ctximg_and_img_and_segmap_iterator = ctximg_and_img_and_segmap.make_one_shot_iterator()

        self.real_ctx, self.real_x, self.real_x_segmap, self.real_x_segmap_onehot = ctximg_and_img_and_segmap_iterator.get_next()

        self.global_step = tf.train.create_global_step()

        batch_size = self.batch_size

        """ Define Generator, Discriminator """
        prior_det_code_mean, prior_det_code_logvar = self.prior_code(batch_size)
        x_det_code_mean, x_det_code_logvar = self.encoder_code(self.real_x, scope='encoder_det_code')
        fake_det_x_code = z_sample(x_det_code_mean, x_det_code_logvar)
       
        supercode_stop_gradient = lambda x: x
        code_stop_gradient = lambda x: x

        x_det_supercode_mean, x_det_supercode_logvar = self.encoder_supercode(code_stop_gradient(fake_det_x_code), channel_multiplier=4, scope='encoder_det_supercode')
        fake_det_x_supercode = z_sample(x_det_supercode_mean, x_det_supercode_logvar)
        x_det_ctxcode_mean, x_det_ctxcode_logvar = self.encoder_code(self.real_ctx, scope='encoder_det_ctxcode')
        fake_det_x_ctxcode = z_sample(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        x_det_codectx_mean, x_det_codectx_logvar = self.encoder_code(self.real_ctx, reuse=True, scope='encoder_det_code')
        fake_det_x_codectx = z_sample(x_det_codectx_mean, x_det_codectx_logvar)
        fake_det_x_full_ctxcode = tf.concat([fake_det_x_ctxcode, tf.stop_gradient(fake_det_x_codectx)],-1)
        fake_det_x_code_mean, fake_det_x_code_logvar = self.generator_code(fake_det_x_full_ctxcode, fake_det_x_supercode, scope="generator_det_code")
        
        prior_det_supercode_mean, prior_det_supercode_logvar = self.prior_code(batch_size, channel_multiplier=4)#self.encoder_code(self.real_x, scope='prior_det_supercode')
        #random_det_supercode = z_sample(prior_det_supercode_mean, prior_det_supercode_logvar)
        prior_det_supercode_dist = self.prior_code_dist(fake_det_x_full_ctxcode, channel_multiplier=4, scope='prior_det_supercode')
        random_det_supercode = prior_det_supercode_dist.sample()
        prior_det_ctxcode_mean, prior_det_ctxcode_logvar = self.prior_code(batch_size)
        #random_det_ctxcode = z_sample(prior_det_ctxcode_mean, prior_det_ctxcode_logvar)
        random_det_code_mean, random_det_code_var = self.generator_code(fake_det_x_full_ctxcode, random_det_supercode, reuse=True, scope="generator_det_code")
        random_det_code = z_sample(random_det_code_mean, random_det_code_var)
        random_simple_det_code = z_sample(*self.prior_code(batch_size))
        
        prior_nondet_code_mean, prior_nondet_code_logvar = self.prior_code(batch_size)

        x_nondet_code_mean, x_nondet_code_logvar = self.encoder_code(self.real_x, scope='encoder_nondet_code')
        fake_nondet_x_code = z_sample(x_nondet_code_mean, x_nondet_code_logvar)

        x_nondet_supercode_mean, x_nondet_supercode_logvar = self.encoder_supercode(code_stop_gradient(fake_nondet_x_code), channel_multiplier=4, scope='encoder_nondet_supercode')
        fake_nondet_x_supercode = z_sample(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar = self.encoder_code(self.real_ctx, scope='encoder_nondet_ctxcode')
        fake_nondet_x_ctxcode = z_sample(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        x_nondet_codectx_mean, x_nondet_codectx_logvar = self.encoder_code(self.real_ctx, reuse=True, scope='encoder_nondet_code')
        fake_nondet_x_codectx = z_sample(x_nondet_codectx_mean, x_nondet_codectx_logvar)
        fake_nondet_x_full_ctxcode = tf.concat([fake_nondet_x_ctxcode, tf.stop_gradient(fake_nondet_x_codectx)],-1)
        fake_nondet_x_code_mean, fake_nondet_x_code_logvar = self.generator_code(fake_nondet_x_full_ctxcode, fake_nondet_x_supercode, scope="generator_nondet_code")
        
        prior_nondet_supercode_mean, prior_nondet_supercode_logvar = self.prior_code(batch_size, channel_multiplier=4)#self.encoder_code(self.real_x, scope='prior_nondet_supercode')
        #random_nondet_supercode = z_sample(prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        prior_nondet_supercode_dist = self.prior_code_dist(fake_nondet_x_full_ctxcode, channel_multiplier=4, scope='prior_nondet_supercode')
        random_nondet_supercode = prior_nondet_supercode_dist.sample()
        prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar = self.prior_code(batch_size)
        #random_nondet_ctxcode = z_sample(prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar)
        random_nondet_code_mean, random_nondet_code_var = self.generator_code(fake_nondet_x_full_ctxcode, random_nondet_supercode, reuse=True, scope="generator_nondet_code")
        random_nondet_code = z_sample(random_nondet_code_mean, random_nondet_code_var)
        random_simple_nondet_code = z_sample(*self.prior_code(batch_size))

        fake_full_det_x_code = tf.concat([fake_det_x_full_ctxcode],-1)
        fake_full_det_x_z = tf.concat([fake_det_x_code],-1)
        fake_det_x_features, fake_det_x_stats = self.generator(fake_full_det_x_code, z=fake_full_det_x_z, scope="generator_det")
        fake_det_x_scaffold = fake_det_x_features#tf.concat([fake_det_x_stats[0][0], fake_det_x_stats[1]], -1)
        fake_det_x_mean, fake_det_x_var = fake_det_x_stats[0]
        
        fake_full_nondet_x_code = tf.concat([fake_nondet_x_full_ctxcode, tf.stop_gradient(fake_det_x_full_ctxcode)],-1) 
        fake_full_nondet_x_z = tf.concat([fake_nondet_x_code, tf.stop_gradient(fake_det_x_code)],-1) 
        fake_full_nondet_x_discriminator_code = tf.concat([fake_nondet_x_code, fake_nondet_x_full_ctxcode, tf.stop_gradient(fake_det_x_code), tf.stop_gradient(fake_det_x_full_ctxcode)],-1) 
        #fake_nondet_x_output = self.generator_spatial(fake_full_nondet_x_code, tf.stop_gradient(fake_det_x_scaffold), z=fake_full_nondet_x_z, reuse=False, scope="generator_nondet")
        fake_nondet_x_output = self.generator_features(fake_full_nondet_x_code, map(tf.stop_gradient, fake_det_x_features), z=fake_full_nondet_x_z, reuse=False, scope="generator_nondet")

        random_full_det_x_code = tf.concat([fake_det_x_full_ctxcode], -1)
        random_full_det_x_z = tf.concat([random_det_code], -1)
        random_fake_det_x_features, random_fake_det_x_stats = self.generator(random_full_det_x_code, z=random_full_det_x_z, reuse=True, scope="generator_det")
        random_fake_det_x_scaffold = random_fake_det_x_features#tf.concat([random_fake_det_x_stats[0][0], random_fake_det_x_stats[1]], -1)
        random_fake_det_x_mean, random_fake_det_x_var = random_fake_det_x_stats[0]

        random_full_nondet_x_code = tf.concat([fake_nondet_x_full_ctxcode, fake_det_x_full_ctxcode], -1) 
        random_full_nondet_x_z = tf.concat([random_nondet_code, random_det_code], -1) 
        #random_fake_nondet_x_output = self.generator_spatial(random_full_nondet_x_code, random_fake_det_x_scaffold, z=random_full_nondet_x_z, reuse=True, scope="generator_nondet")
        random_fake_nondet_x_output = self.generator_features(random_full_nondet_x_code, random_fake_det_x_features, z=random_full_nondet_x_z, reuse=True, scope="generator_nondet")

        [code_det_prior_real_logit, code_det_prior_real_summary], [code_det_prior_fake_logit, code_det_prior_fake_summary] = self.discriminate_code(real_code_img=tf.concat([tf.stop_gradient(fake_det_x_full_ctxcode), tf.stop_gradient(random_simple_det_code)], -1), fake_code_img=tf.concat([tf.stop_gradient(fake_det_x_full_ctxcode), fake_det_x_code], -1), name='det_prior')
        [code_nondet_prior_real_logit, code_nondet_prior_real_summary], [code_nondet_prior_fake_logit, code_nondet_prior_fake_summary] = self.discriminate_code(real_code_img=tf.concat([tf.stop_gradient(fake_nondet_x_full_ctxcode), tf.stop_gradient(random_simple_nondet_code)], -1), fake_code_img=tf.concat([tf.stop_gradient(fake_nondet_x_full_ctxcode), fake_nondet_x_code], -1), name='nondet_prior')

        discriminator_fun = self.full_discriminator
        [nondet_real_logit, real_nondet_summary] = discriminator_fun(tf.concat([self.real_ctx, self.real_x, tf.stop_gradient(fake_det_x_mean)], -1), fake_full_nondet_x_discriminator_code, scope='discriminator_nondet', label='real_nondet')
        [nondet_fake_logit, fake_nondet_summary] = discriminator_fun(tf.concat([self.real_ctx, fake_nondet_x_output, tf.stop_gradient(fake_det_x_mean)], -1), fake_full_nondet_x_discriminator_code, reuse=True, scope='discriminator_nondet', label='fake_nondet')
        
        if self.gan_type.__contains__('wgan-') or self.gan_type == 'dragan':
            GP = self.gradient_penalty(real=tf.concat([self.real_ctx, self.real_x, tf.stop_gradient(fake_det_x_mean)], -1), fake=tf.concat([self.real_ctx, fake_nondet_x_output, tf.stop_gradient(fake_det_x_mean)],-1), code=fake_full_nondet_x_discriminator_code, discriminator=discriminator_fun, name='nondet')
        else:
            GP = 0

        """ Define Loss """
        g_nondet_ce_loss = L1_loss(self.real_x, fake_nondet_x_output)
        g_nondet_vgg_loss = VGGLoss()(self.real_x, fake_nondet_x_output)
        g_nondet_adv_loss = generator_loss(self.gan_type, nondet_fake_logit)
        g_nondet_feature_loss = self.feature_weight * feature_loss(nondet_real_logit, nondet_fake_logit)
        g_nondet_reg_loss = regularization_loss('generator_nondet')

        #g_det_ce_loss = L2_loss(self.real_x, fake_det_x_mean)
        g_det_ce_loss = gaussian_loss(self.real_x, fake_det_x_mean, fake_det_x_var)
        g_det_segmapce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.real_x_segmap_onehot, logits=fake_det_x_stats[1]))
        g_det_vgg_loss = VGGLoss()(self.real_x, fake_det_x_stats[0][0])
        g_det_reg_loss = regularization_loss('generator_det')

        #g_nondet_code_ce_loss = L2_mean_loss(code_stop_gradient(fake_nondet_x_code), fake_nondet_x_code_mean)
        g_nondet_code_ce_loss = gaussian_loss(code_stop_gradient(fake_nondet_x_code), fake_nondet_x_code_mean, fake_nondet_x_code_logvar)
        e_nondet_code_kl_loss = kl_loss(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        e_nondet_code_prior_loss = gaussian_loss(fake_nondet_x_supercode, prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        e_nondet_code_prior2_loss = -tf.reduce_mean(prior_nondet_supercode_dist.log_prob(supercode_stop_gradient(fake_nondet_x_supercode))) / int(fake_nondet_x_supercode.get_shape()[-1])
        e_nondet_code_negent_loss = negent_loss(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        #e_nondet_code_kl2_loss = kl_loss2(x_nondet_supercode_mean, x_nondet_supercode_logvar, prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        e_nondet_code_kl2_loss = (e_nondet_code_prior2_loss + e_nondet_code_negent_loss)
        self.nondet_code_vae = tf.stack([g_nondet_code_ce_loss, e_nondet_code_prior2_loss, e_nondet_code_negent_loss])
                
        e_nondet_prior_adv_loss = generator_loss(self.code_gan_type, code_nondet_prior_fake_logit)
        e_nondet_kl_loss = kl_loss(x_nondet_code_mean, x_nondet_code_logvar)
        e_nondet_prior_loss = gaussian_loss(fake_nondet_x_code, prior_nondet_code_mean, prior_nondet_code_logvar)
        e_nondet_negent_loss = negent_loss(x_nondet_code_mean, x_nondet_code_logvar)
        e_nondet_prior2_loss = e_nondet_prior_loss 
        e_nondet_kl2_loss = (e_nondet_prior2_loss + e_nondet_negent_loss)
        e_nondet_reg_loss = regularization_loss('encoder_nondet_code')
        
        e_nondet_klctx_loss = kl_loss(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        e_nondet_priorctx_loss = gaussian_loss(fake_nondet_x_ctxcode, prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar)
        e_nondet_negentctx_loss = negent_loss(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        #e_nondet_klctx2_loss = kl_loss2(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar, prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar)
        e_nondet_klctx2_loss = (e_nondet_priorctx_loss + e_nondet_negentctx_loss)

        #g_det_code_ce_loss = L2_mean_loss(code_stop_gradient(fake_det_x_code), fake_det_x_code_mean)
        g_det_code_ce_loss = gaussian_loss(code_stop_gradient(fake_det_x_code), fake_det_x_code_mean, fake_det_x_code_logvar)
        e_det_code_kl_loss = kl_loss(x_det_supercode_mean, x_det_supercode_logvar)
        e_det_code_prior_loss = gaussian_loss(fake_det_x_supercode, prior_det_supercode_mean, prior_det_supercode_logvar)
        e_det_code_prior2_loss = -tf.reduce_mean(prior_det_supercode_dist.log_prob(supercode_stop_gradient(fake_det_x_supercode))) / int(fake_det_x_supercode.get_shape()[-1])
        e_det_code_negent_loss = negent_loss(x_det_supercode_mean, x_det_supercode_logvar)
        #e_det_code_kl2_loss = kl_loss2(x_det_supercode_mean, x_det_supercode_logvar, prior_det_supercode_mean, prior_det_supercode_logvar)
        e_det_code_kl2_loss = (e_det_code_prior2_loss + e_det_code_negent_loss) 
        self.det_code_vae = tf.stack([g_det_code_ce_loss, e_det_code_prior2_loss, e_det_code_negent_loss])

        e_det_prior_adv_loss = generator_loss(self.code_gan_type, code_det_prior_fake_logit)
        e_det_kl_loss = kl_loss(x_det_code_mean, x_det_code_logvar)
        e_det_prior_loss = gaussian_loss(fake_det_x_code, prior_det_code_mean, prior_det_code_logvar)
        e_det_negent_loss = negent_loss(x_det_code_mean, x_det_code_logvar)
        e_det_prior2_loss = e_det_prior_loss 
        e_det_kl2_loss = (e_det_prior2_loss + e_det_negent_loss)
        e_det_reg_loss = regularization_loss('encoder_det_code')
        
        e_det_klctx_loss = kl_loss(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        e_det_priorctx_loss = gaussian_loss(fake_det_x_ctxcode, prior_det_ctxcode_mean, prior_det_ctxcode_logvar)
        e_det_negentctx_loss = negent_loss(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        #e_det_klctx2_loss = kl_loss2(x_det_ctxcode_mean, x_det_ctxcode_logvar, prior_det_ctxcode_mean, prior_det_ctxcode_logvar)
        e_det_klctx2_loss = (e_det_priorctx_loss + e_det_negentctx_loss)

        d_nondet_adv_loss = discriminator_loss(self.gan_type, nondet_real_logit, nondet_fake_logit)
        d_nondet_score_real, d_nondet_score_fake = discriminator_scores(nondet_real_logit, nondet_fake_logit)
        d_nondet_score_diff = -d_nondet_score_real + d_nondet_score_fake
        d_nondet_reg_loss = GP + regularization_loss('discriminator_nondet')

        de_det_prior_adv_loss = discriminator_loss(self.code_gan_type, code_det_prior_real_logit, code_det_prior_fake_logit)
        de_det_prior_reg_loss = regularization_loss('discriminator_det_prior_code')

        de_nondet_prior_adv_loss = discriminator_loss(self.code_gan_type, code_nondet_prior_real_logit, code_nondet_prior_fake_logit)
        de_nondet_prior_reg_loss = regularization_loss('discriminator_nondet_prior_code')

        ema = tf.train.ExponentialMovingAverage(decay=0.9, zero_debias=True)
        e_det_kl_loss_truncated = tf.minimum(100.0,tf.abs(e_det_kl_loss))
        e_nondet_kl_loss_truncated = tf.minimum(100.0,tf.abs(e_nondet_kl_loss))
        with tf.control_dependencies([ema.apply([e_det_kl_loss_truncated, e_nondet_kl_loss_truncated])]):
            e_det_kl_loss_ema = ema.average(e_det_kl_loss_truncated)
            e_det_kl_loss_weight = tf.maximum(0.0,e_det_kl_loss_ema - 1.0)/1.0
            e_det_kl_loss_adjusted = e_det_kl_loss_weight*e_det_kl_loss

            e_nondet_kl_loss_ema = ema.average(e_nondet_kl_loss_truncated)
            e_nondet_kl_loss_weight = tf.maximum(0.0,e_nondet_kl_loss_ema - 1.0)/1.0
            e_nondet_kl_loss_adjusted = e_nondet_kl_loss_weight*e_nondet_kl_loss

            self.g_nondet_loss = g_nondet_adv_loss + g_nondet_reg_loss + 10*g_nondet_feature_loss + 10*g_nondet_vgg_loss + 0*g_nondet_ce_loss + e_nondet_prior_adv_loss + e_nondet_reg_loss + 0.05*(0*e_nondet_prior_loss + e_nondet_prior2_loss + (g_nondet_code_ce_loss + 0.01*(0*e_nondet_code_prior_loss + e_nondet_code_prior2_loss + e_nondet_code_negent_loss)) + e_nondet_negent_loss) + 0.0001*e_nondet_klctx2_loss
            self.g_det_loss = 10*g_det_ce_loss + 10*g_det_segmapce_loss + g_det_reg_loss + e_det_prior_adv_loss + e_det_reg_loss + 0.05*(0*e_det_prior_loss + e_det_prior2_loss + (g_det_code_ce_loss + 0.01*(0*e_det_code_prior_loss + e_det_code_prior2_loss + e_det_code_negent_loss)) + e_det_negent_loss) + 0.0001*e_det_klctx2_loss
            self.de_loss = de_det_prior_adv_loss + de_det_prior_reg_loss + de_nondet_prior_adv_loss + de_nondet_prior_reg_loss
            self.d_loss = d_nondet_adv_loss + d_nondet_reg_loss

        """ Result Image """
        self.fake_det_x = fake_det_x_stats[0][0]
        self.fake_det_x_var = tf.exp(fake_det_x_stats[0][1])
        self.fake_det_x_segmap = tfd.Categorical(logits=fake_det_x_stats[1]).sample()
        self.fake_nondet_x = fake_nondet_x_output
        self.random_fake_det_x = random_fake_det_x_stats[0][0]
        self.random_fake_det_x_segmap = tfd.Categorical(logits=random_fake_det_x_stats[1]).sample()
        self.random_fake_nondet_x = random_fake_nondet_x_output

        """ Test """
        self.test_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, len(img_class.color_value_dict)])
        self.test_guide_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch])


        """ Training """
        t_vars = tf.trainable_variables()
        G_nondet_vars = [var for var in t_vars if 'generator_nondet' in var.name or 'encoder_nondet_code' in var.name or 'generator_nondet_code' in var.name or 'prior_nondet_supercode' in var.name or 'encoder_nondet_supercode' in var.name or 'encoder_nondet_ctxcode' in var.name]
        G_det_vars = [var for var in t_vars if 'generator_det' in var.name or 'encoder_det_code' in var.name in var.name or 'generator_det_code' in var.name or 'prior_det_supercode' in var.name or 'encoder_det_supercode' in var.name or 'encoder_det_ctxcode' in var.name]
        DE_vars = [var for var in t_vars if 'discriminator_det_prior_code' in var.name or 'discriminator_nondet_prior_code' in var.name]
        D_vars = [var for var in t_vars if 'discriminator_nondet' in var.name]

        if self.TTUR :
            beta1 = 0.0
            beta2 = 0.9

            g_lr = self.lr / 2
            d_lr = self.lr * 2

        else :
            beta1 = self.beta1
            beta2 = self.beta2
            g_lr = self.lr
            d_lr = self.lr

        self.G_nondet_optim = tf.train.AdamOptimizer(g_lr, beta1=beta1, beta2=beta2).minimize(self.g_nondet_loss, var_list=G_nondet_vars)
        self.G_det_optim = tf.train.AdamOptimizer(d_lr, beta1=beta1, beta2=beta2).minimize(self.g_det_loss, var_list=G_det_vars)
        self.DE_optim = tf.train.AdamOptimizer(d_lr, beta1=beta1, beta2=beta2).minimize(self.de_loss, var_list=DE_vars)
        self.D_optim = tf.train.AdamOptimizer(d_lr, beta1=beta1, beta2=beta2).minimize(self.d_loss, var_list=D_vars, global_step = self.global_step)

        """" Summary """
        self.summary_global_step = tf.summary.scalar("global_step", self.global_step)

        self.summary_g_nondet_loss = tf.summary.scalar("g_nondet_loss", self.g_nondet_loss)
        self.summary_g_det_loss = tf.summary.scalar("g_det_loss", self.g_det_loss)
        self.summary_de_loss = tf.summary.scalar("de_loss", self.de_loss)
        self.summary_d_loss = tf.summary.scalar("d_loss", self.d_loss)

        self.summary_g_det_code_ce_loss = tf.summary.scalar("g_det_code_ce_loss", g_det_code_ce_loss)
        self.summary_e_det_code_kl_loss = tf.summary.scalar("e_det_code_kl_loss", e_det_code_kl_loss)
        self.summary_e_det_klctx_loss = tf.summary.scalar("e_det_klctx_loss", e_det_klctx_loss)
        self.summary_e_det_code_kl2_loss = tf.summary.scalar("e_det_code_kl2_loss", e_det_code_kl2_loss)
        self.summary_e_det_klctx2_loss = tf.summary.scalar("e_det_klctx2_loss", e_det_klctx2_loss)
        self.summary_e_det_code_prior_loss = tf.summary.scalar("e_det_code_prior_loss", e_det_code_prior_loss)
        self.summary_e_det_code_prior2_loss = tf.summary.scalar("e_det_code_prior2_loss", e_det_code_prior2_loss)
        self.summary_e_det_priorctx_loss = tf.summary.scalar("e_det_priorctx_loss", e_det_priorctx_loss)
        self.summary_e_det_code_negent_loss = tf.summary.scalar("e_det_code_negent_loss", e_det_code_negent_loss)
        self.summary_e_det_negentctx_loss = tf.summary.scalar("e_det_negentctx_loss", e_det_negentctx_loss)

        self.summary_e_det_kl_loss = tf.summary.scalar("e_det_kl_loss", e_det_kl_loss)
        self.summary_e_det_kl_loss_ema = tf.summary.scalar("e_det_kl_loss_ema", e_det_kl_loss_ema)
        self.summary_e_det_kl_loss_weight = tf.summary.scalar("e_det_kl_loss_weight", e_det_kl_loss_weight)
        self.summary_e_det_negent_loss = tf.summary.scalar("e_det_negent_loss", e_det_negent_loss)
        self.summary_e_det_prior_loss = tf.summary.scalar("e_det_prior_loss", e_det_prior_loss)
        self.summary_e_det_prior2_loss = tf.summary.scalar("e_det_prior2_loss", e_det_prior2_loss)
        self.summary_e_det_kl2_loss = tf.summary.scalar("e_det_kl2_loss", e_det_kl2_loss)
        self.summary_e_det_reg_loss = tf.summary.scalar("det_e_reg_loss", e_det_reg_loss)
        self.summary_e_det_prior_adv_loss = tf.summary.scalar("e_det_prior_adv_loss", e_det_prior_adv_loss)

        self.summary_g_nondet_code_ce_loss = tf.summary.scalar("g_nondet_code_ce_loss", g_nondet_code_ce_loss)
        self.summary_e_nondet_code_kl_loss = tf.summary.scalar("e_nondet_code_kl_loss", e_nondet_code_kl_loss)
        self.summary_e_nondet_klctx_loss = tf.summary.scalar("e_nondet_klctx_loss", e_nondet_klctx_loss)
        self.summary_e_nondet_code_kl2_loss = tf.summary.scalar("e_nondet_code_kl2_loss", e_nondet_code_kl2_loss)
        self.summary_e_nondet_klctx2_loss = tf.summary.scalar("e_nondet_klctx2_loss", e_nondet_klctx2_loss)
        self.summary_e_nondet_code_prior_loss = tf.summary.scalar("e_nondet_code_prior_loss", e_nondet_code_prior_loss)
        self.summary_e_nondet_code_prior2_loss = tf.summary.scalar("e_nondet_code_prior2_loss", e_nondet_code_prior2_loss)
        self.summary_e_nondet_priorctx_loss = tf.summary.scalar("e_nondet_priorctx_loss", e_nondet_priorctx_loss)
        self.summary_e_nondet_code_negent_loss = tf.summary.scalar("e_nondet_code_negent_loss", e_nondet_code_negent_loss)
        self.summary_e_nondet_negentctx_loss = tf.summary.scalar("e_nondet_negentctx_loss", e_nondet_negentctx_loss)
        
        self.summary_e_nondet_kl_loss = tf.summary.scalar("e_nondet_kl_loss", e_nondet_kl_loss)
        self.summary_e_nondet_kl_loss_ema = tf.summary.scalar("e_nondet_kl_loss_ema", e_nondet_kl_loss_ema)
        self.summary_e_nondet_kl_loss_weight = tf.summary.scalar("e_nondet_kl_loss_weight", e_nondet_kl_loss_weight)
        self.summary_e_nondet_negent_loss = tf.summary.scalar("e_nondet_negent_loss", e_nondet_negent_loss)
        self.summary_e_nondet_prior_loss = tf.summary.scalar("e_nondet_prior_loss", e_nondet_prior_loss)
        self.summary_e_nondet_prior2_loss = tf.summary.scalar("e_nondet_prior2_loss", e_nondet_prior2_loss)
        self.summary_e_nondet_kl2_loss = tf.summary.scalar("e_nondet_kl2_loss", e_nondet_kl2_loss)
        self.summary_e_nondet_reg_loss = tf.summary.scalar("e_nondet_reg_loss", e_nondet_reg_loss)
        self.summary_e_nondet_prior_adv_loss = tf.summary.scalar("e_nondet_prior_adv_loss", e_nondet_prior_adv_loss)

        self.summary_g_det_ce_loss = tf.summary.scalar("g_det_ce_loss", g_det_ce_loss)
        self.summary_g_det_segmapce_loss = tf.summary.scalar("g_det_segmapce_loss", g_det_segmapce_loss)
        self.summary_g_det_vgg_loss = tf.summary.scalar("g_det_vgg_loss", g_det_vgg_loss)
        self.summary_g_det_reg_loss = tf.summary.scalar("g_det_reg_loss", g_det_reg_loss)

        self.summary_g_nondet_ce_loss = tf.summary.scalar("g_nondet_ce_loss", g_nondet_ce_loss)
        self.summary_g_nondet_vgg_loss = tf.summary.scalar("g_nondet_vgg_loss", g_nondet_vgg_loss)
        self.summary_g_nondet_feature_loss = tf.summary.scalar("g_nondet_feature_loss", g_nondet_feature_loss)
        self.summary_g_nondet_reg_loss = tf.summary.scalar("g_nondet_reg_loss", g_nondet_reg_loss)
        self.summary_g_nondet_adv_loss = tf.summary.scalar("g_nondet_adv_loss", g_nondet_adv_loss)
        
        self.summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_adv_loss", d_nondet_adv_loss)
        self.summary_d_nondet_score_real = tf.summary.scalar("d_nondet_score_real", d_nondet_score_real)
        self.summary_d_nondet_score_fake = tf.summary.scalar("d_nondet_score_fake", d_nondet_score_fake)
        self.summary_d_nondet_score_diff = tf.summary.scalar("d_nondet_score_diff", d_nondet_score_diff)
        self.summary_d_nondet_reg_loss = tf.summary.scalar("d_nondet_reg_loss", d_nondet_reg_loss)

        self.summary_de_det_prior_adv_loss = tf.summary.scalar("de_det_prior_adv_loss", de_det_prior_adv_loss)
        self.summary_de_det_prior_reg_loss = tf.summary.scalar("de_det_prior_reg_loss", de_det_prior_reg_loss)
        
        self.summary_de_nondet_prior_adv_loss = tf.summary.scalar("de_nondet_prior_adv_loss", de_nondet_prior_adv_loss)
        self.summary_de_nondet_prior_reg_loss = tf.summary.scalar("de_nondet_prior_reg_loss", de_nondet_prior_reg_loss)

        g_nondet_summary_list = [self.summary_g_nondet_loss, self.summary_g_nondet_adv_loss, self.summary_g_nondet_reg_loss, self.summary_g_nondet_ce_loss, self.summary_g_nondet_vgg_loss, self.summary_g_nondet_feature_loss, self.summary_e_nondet_kl_loss, self.summary_e_nondet_kl2_loss, self.summary_e_nondet_kl_loss_ema, self.summary_e_nondet_kl_loss_weight, self.summary_e_nondet_prior_adv_loss, self.summary_e_nondet_reg_loss, self.summary_g_nondet_code_ce_loss, self.summary_e_nondet_code_kl_loss, self.summary_e_nondet_klctx_loss, self.summary_e_nondet_code_kl2_loss, self.summary_e_nondet_klctx2_loss, self.summary_e_nondet_code_prior_loss, self.summary_e_nondet_code_prior2_loss, self.summary_e_nondet_priorctx_loss, self.summary_e_nondet_code_negent_loss, self.summary_e_nondet_negentctx_loss, self.summary_e_nondet_prior_loss, self.summary_e_nondet_prior2_loss, self.summary_e_nondet_negent_loss]
        g_det_summary_list = [self.summary_g_det_ce_loss, self.summary_g_det_segmapce_loss, self.summary_g_det_vgg_loss, self.summary_g_det_reg_loss, self.summary_g_det_loss, self.summary_e_det_kl_loss, self.summary_e_det_kl2_loss, self.summary_e_det_kl_loss_ema, self.summary_e_det_kl_loss_weight, self.summary_e_det_prior_adv_loss, self.summary_e_det_reg_loss, self.summary_g_det_code_ce_loss, self.summary_e_det_code_kl_loss, self.summary_e_det_klctx_loss, self.summary_e_det_klctx2_loss, self.summary_e_det_code_kl2_loss, self.summary_e_det_code_prior_loss, self.summary_e_det_code_prior2_loss, self.summary_e_det_priorctx_loss, self.summary_e_det_code_negent_loss, self.summary_e_det_negentctx_loss, self.summary_e_det_prior_loss, self.summary_e_det_prior2_loss, self.summary_e_det_negent_loss]
        d_summary_list = [self.summary_global_step, self.summary_d_loss, self.summary_d_nondet_adv_loss, self.summary_d_nondet_score_real, self.summary_d_nondet_score_fake, self.summary_d_nondet_score_diff, self.summary_d_nondet_reg_loss] + real_nondet_summary + fake_nondet_summary
        de_summary_list = [self.summary_de_loss, self.summary_de_det_prior_adv_loss, self.summary_de_det_prior_reg_loss, self.summary_de_nondet_prior_adv_loss, self.summary_de_nondet_prior_reg_loss] + code_det_prior_real_summary + code_det_prior_fake_summary + code_nondet_prior_real_summary + code_nondet_prior_fake_summary

        self.G_nondet_loss = tf.summary.merge(g_nondet_summary_list)
        self.G_det_loss = tf.summary.merge(g_det_summary_list)
        self.D_loss = tf.summary.merge(d_summary_list)
        self.DE_loss = tf.summary.merge(de_summary_list)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=1000)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_g_nondet_loss = -1.
        past_g_det_loss = -1.
        lr = self.init_lr

        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag:
                # lr = self.init_lr * pow(0.5, epoch // self.decay_epoch)
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)
            for idx in range(start_batch_id, self.iteration):
                train_feed_dict = {
                    self.lr: lr
                }

                # Update D
                _, d_loss, d_summary_str = self.sess.run([self.D_optim, self.d_loss, self.D_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(d_summary_str, counter)

                g_nondet_loss = None
                g_det_loss = None
                
                if (counter - 1) % self.n_critic == 0:
                    # Update DE
                    _, de_loss, de_summary_str = self.sess.run([self.DE_optim, self.de_loss, self.DE_loss], feed_dict=train_feed_dict)
                    self.writer.add_summary(de_summary_str, counter)

                    if (counter - 1) % (self.n_critic*self.code_n_critic) == 0:
                        # Update E
                        det_code_vae, _, g_det_loss, g_det_summary_str = self.sess.run([self.det_code_vae, self.G_det_optim, self.g_det_loss, self.G_det_loss], feed_dict=train_feed_dict)
                        self.writer.add_summary(g_det_summary_str, counter)
                        past_g_det_loss = g_det_loss
                        print("det_code_vae:", det_code_vae) 

                        # Update G
                        nondet_code_vae, _, g_nondet_loss, g_nondet_summary_str = self.sess.run(
                            [self.nondet_code_vae,
                             self.G_nondet_optim,
                             self.g_nondet_loss, self.G_nondet_loss], feed_dict=train_feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                        print("nondet_code_vae:", nondet_code_vae) 

                        self.writer.add_summary(g_nondet_summary_str, counter)
                        past_g_nondet_loss = g_nondet_loss

                # display training status
                counter += 1
                if g_nondet_loss == None:
                    g_nondet_loss = past_g_nondet_loss
                if g_det_loss == None:
                    g_det_loss = past_g_det_loss
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f g_nondet_loss: %.8f" % (
                    epoch, idx, self.iteration, time.time() - start_time, g_nondet_loss))
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f g_det_loss: %.8f" % (
                    epoch, idx, self.iteration, time.time() - start_time, g_det_loss))
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f de_loss: %.8f" % (
                    epoch, idx, self.iteration, time.time() - start_time, de_loss))
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f d_loss: %.8f" % (
                    epoch, idx, self.iteration, time.time() - start_time, d_loss))
                sys.stdout.flush()

                if np.mod(idx + 1, self.print_freq) == 0:
                    real_ctx_images, real_x_images, real_x_segmap, fake_det_x, fake_det_x_var, fake_det_x_segmap, fake_nondet_x, random_fake_det_x, random_fake_det_x_segmap, random_fake_nondet_x = self.sess.run(
                        [self.real_ctx, self.real_x, self.real_x_segmap, self.fake_det_x, self.fake_det_x_var, self.fake_det_x_segmap, self.fake_nondet_x, self.random_fake_det_x, self.random_fake_det_x_segmap, self.random_fake_nondet_x], feed_dict=train_feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

                    save_images(real_ctx_images, [self.batch_size, 1],
                               './{}/real_ctximage_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    save_images(real_x_images, [self.batch_size, 1],
                               './{}/real_image_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(real_x_segmap, [self.batch_size, 1],
                                './{}/real_segmap_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    save_images(fake_det_x, [self.batch_size, 1],
                                './{}/fake_det_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    imsave(fake_det_x_var, [self.batch_size, 1],
                                './{}/fake_det_var_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_segmaps(fake_det_x_segmap, self.color_value_dict, [self.batch_size, 1],
                                 './{}/fake_det_segmap_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_nondet_x, [self.batch_size, 1],
                                './{}/fake_nondet_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    save_images(random_fake_det_x, [self.batch_size, 1],
                                './{}/random_fake_det_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))
                    save_segmaps(random_fake_det_x_segmap, self.color_value_dict, [self.batch_size, 1],
                                 './{}/random_fake_det_segmap_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(random_fake_nondet_x, [self.batch_size, 1],
                                './{}/random_fake_nondet_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))

                if np.mod(counter - 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

        # save model for final step
        self.save(self.checkpoint_dir, counter)

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
            checkpoint_reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
            tensor_shapes = checkpoint_reader.get_variable_to_shape_map()
            variables_to_restore = {}
            for known_variable in tf.global_variables():
                tensor_name = known_variable.name.split(':')[0]
                if checkpoint_reader.has_tensor(tensor_name) and known_variable.shape == tensor_shapes[tensor_name]:
                    print("Variable restored: %s Shape: %s" % (known_variable.name, known_variable.shape))
                    variables_to_restore[tensor_name] = known_variable
                else:
                    print("Variable NOT restored: %s Shape: %s OriginalShape: %s" % (known_variable.name, known_variable.shape, tensor_shapes.get(tensor_name)))

            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def random_test(self):
        tf.global_variables_initializer().run()

        files = glob('./dataset/{}/{}/*.*'.format(self.dataset_name, 'test'))

        self.saver = tf.train.Saver()
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
        tf.global_variables_initializer().run()

        files = glob('./dataset/{}/{}/*.*'.format(self.dataset_name, 'test'))

        style_image = load_style_image(self.guide_img, self.img_width, self.img_height, self.img_ch)

        self.saver = tf.train.Saver()
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
