import sys
from ops import *
from utils import *
import time
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from tqdm import tqdm
from vgg19_keras import VGGLoss
from unet import unet

class SPADE(object):
    def __init__(self, args):

        self.model_name = 'SPADE'

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

    def encoder_code(self, x_init, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x, channel = self.encoder_base(x_init, self.ch)

            mean = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_mean')
            var = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_var')
            return mean, var

    def prior_code(self):
        batch_size = self.batch_size
        out_channel = self.ch * 4
        mean = tf.zeros([batch_size, out_channel])
        var = tf.zeros([batch_size, out_channel])
        return mean, var

    def prior_code_dist(self, code, epsilon=1e-3, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        context = code
        batch_size = self.batch_size
        out_channel = self.ch * 4
        hidden_channel = self.ch * 64

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            bijectors = []
            for i in range(self.code_dist_num_layers):
                bijectors.append(tfb.MaskedAutoregressiveFlow(
                  shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                  hidden_layers=[hidden_channel, hidden_channel], name=scope + "/masked_autoregressive_default_template_" + str(i))))

                context_gamma = fully_connected(context, units=out_channel, scope='linear_gamma_' + str(i))
                context_beta = fully_connected(context, units=out_channel, scope='linear_beta_' + str(i))
                bijectors.append(tfb.BatchNormalization(
                    batchnorm_layer=tf.compat.v1.layers.BatchNormalization(
                                        gamma_constraint=lambda x:tf.ones(shape=[out_channel]), beta_constraint=lambda x:tf.zeros(shape=[out_channel]),
                                        name=scope + '/batch_norm_' + str(i)),
                    name=scope + '/batch_norm_bijector' + str(i)))
                bijectors.append(tfb.AffineLinearOperator(context_beta, tf.linalg.LinearOperatorDiag(context_gamma)))

                permutation=tf.compat.v1.get_variable('permutation_'+str(i), initializer=np.random.permutation(out_channel).astype("int32"), trainable=False)
                bijectors.append(tfb.Permute(permutation))
                
            flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

            mvn_loc = fully_connected(context, units=out_channel, scope='mvn_loc')
            mvn_scale_diag = epsilon + tf.math.sigmoid(fully_connected(context, units=out_channel, scope='mvn_scale_logdiag'))

            #import correlation_cholesky
            #mvn_corr_bijector = correlation_cholesky.CorrelationCholesky(name=scope + "/CorrelationCholesky")
            #mvn_corr = mvn_corr_bijector.forward(fully_connected(context, units=(out_channel-1)*out_channel/2, scope='mvn_corr_seed'))
            #mvn_scale = tf.linalg.transpose(mvn_corr * tf.expand_dims(mvn_scale_diag,2)) * tf.expand_dims(mvn_scale_diag,2)
            #mvn_dist = tfd.MultivariateNormalFullCovariance(mvn_loc, mvn_scale, name=scope + "/MultivariateNormalFullCovariance")

            #_, mvn_scale_u, _ = tf.linalg.svd(tf.reshape(fully_connected(context, units=out_channel*out_channel, scope='mvn_scale_seed'), [batch_size, out_channel, out_channel]), full_matrices=True)
            #mvn_scale = tf.linalg.matmul(tf.matmul(mvn_scale_u, mvn_scale_diag), tf.linalg.transpose(mvn_scale_u))
            #mvn_dist = tfd.MultivariateNormalFullCovariance(mvn_loc, mvn_scale, name=scope + "/MultivariateNormalFullCovariance")

            mvn_dist = tfd.MultivariateNormalDiag(mvn_loc, mvn_scale_diag, name=scope + "/MultivariateNormalDiag")

            dist = tfd.TransformedDistribution(
                            distribution=mvn_dist,
                            bijector=flow_bijector
                        )
        return dist

    def encoder_supercode(self, x_init, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        out_channel = self.ch*4
        hidden_channel = self.ch*64
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            xs = [x_init]
            x = x_init
            for i in range(self.code_num_layers):
                x = fully_connected(x, hidden_channel, use_bias=True, sn=False, scope='linear_' + str(i))
                x = batch_norm(x, hidden_channel, scope="batch_norm_" + str(i))
                x = lrelu(x, 0.2)
                xs.append(x)
            x = tf.concat(xs,-1)

            mean = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_mean')
            var = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_var')

            return mean, var

    def generator_code(self, code, x_init, epsilon=1e-3, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        out_channel = self.ch*4
        hidden_channel = self.ch*64
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            xs = [x_init]
            x = x_init
            for i in range(self.code_num_layers):
                x = fully_connected(x, hidden_channel, use_bias=True, sn=False, scope='linear_' + str(i))
                x = adain_vector(code, x, hidden_channel, scope="adain_" + str(i))
                x = lrelu(x, 0.2)
                xs.append(x)
            x = tf.concat(xs,-1)

            mean = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_mean')
            var = get_trainable_variable("var", [], initializer=tf.compat.v1.constant_initializer(0.0))
            #var = fully_connected(x, out_channel, use_bias=True, sn=False, scope='linear_var')

            return mean, tf.math.log(epsilon + tf.math.sigmoid(var))

    def generator(self, code, z=None, epsilon=1e-3, reuse=tf.compat.v1.AUTO_REUSE, scope="generator"):
        context = code

        context_depth = 8
        context_ch = 10*context.get_shape()[-1]
        channel = self.ch * 4 * 4
        batch_size = self.batch_size
        with tf.compat.v1.variable_scope(scope, reuse=reuse):

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
            x = tf.reshape(x, [batch_size, z_height, z_width, channel])


            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_0')

            x = up_sample(x, scale_factor=2)
            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_1')

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='resblock_fix_2')

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = adain_resblock(context, x, channels=channel//2, use_bias=True, sn=self.sn, scope='resblock_' + str(i))

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')
            x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='resblock_4')

            x = lrelu(x, 0.2)
            mean = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_mean')
            var = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='linear_var')

            return x, [tanh(mean), tf.math.log(epsilon + tf.sigmoid(var))]

    def generator_spatial(self, code, scaffold, z, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
        context = code

        context_depth = 8
        context_ch = 10*context.get_shape()[-1]
        channel = self.ch * 4 * 4
        batch_size = self.batch_size
        _, h, w, _ = scaffold.get_shape().as_list()
        with tf.compat.v1.variable_scope(scope, reuse=reuse):

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
            x = tf.reshape(x, [batch_size, z_height, z_width, channel])

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
            x = fully_connected(x, channel * 10, use_bias=True, sn=self.sn, scope='linear_x_1')
            x = lrelu(x, 0.2)

            x = fully_connected(x, channel * 10, use_bias=True, sn=self.sn, scope='linear_x_2')
            x = lrelu(x, 0.2)

            x = fully_connected(x, channel * 10, use_bias=True, sn=self.sn, scope='linear_x_3')
            x = lrelu(x, 0.2)

            x = fully_connected(x, channel * 10, use_bias=True, sn=self.sn, scope='linear_x_4')
            x = lrelu(x, 0.2)

            z = fully_connected(x, 1, sn=self.sn, scope='linear_z')

            return [[z]]

    def full_discriminator(self, x_init, code=None, reuse=tf.compat.v1.AUTO_REUSE, scope=None, label=None):
        channel = self.ch
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            feature_loss = []
            x = x_init
            
            x = adain_resblock(code, x, channel, use_bias=True, sn=self.sn, norm=False, scope='preresblock')

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
            
            z = tf.reshape(z0, [z0.get_shape()[0], 1, 1, 1])

            D_logit = [feature_loss + [z]]
            return D_logit

    def feature_discriminator(self, x_init, code, reuse=tf.compat.v1.AUTO_REUSE, scope=None, label=None):
        D_logit = []
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            for scale in range(self.n_scale):
                feature_loss = []
                channel = self.ch
                x = x_init
            
                x = adain_resblock(code, x, channel, use_bias=True, sn=self.sn, norm=False, scope='ms_' + str(scale) + '_preresblock')

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
        if self.gan_type == 'dragan':
            shape = tf.shape(input=real)
            eps = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(x=real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            alpha = tf.random.uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else:
            alpha = tf.random.uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
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

    def execute_model(self, global_step, real_ctx, real_x, real_x_segmap, real_x_segmap_onehot, summary):
        """ Define Generator, Discriminator """
        prior_det_code_mean, prior_det_code_logvar = self.prior_code()
        x_det_code_mean, x_det_code_logvar = self.encoder_code(real_x, scope='encoder_det_code')
        fake_det_x_code = z_sample(x_det_code_mean, x_det_code_logvar)
       
        supercode_stop_gradient = lambda x: x
        code_stop_gradient = lambda x: x 

        x_det_supercode_mean, x_det_supercode_logvar = self.encoder_supercode(code_stop_gradient(fake_det_x_code), scope='encoder_det_supercode')
        fake_det_x_supercode = z_sample(x_det_supercode_mean, x_det_supercode_logvar)
        x_det_ctxcode_mean, x_det_ctxcode_logvar = self.encoder_code(real_ctx, scope='encoder_det_ctxcode')
        #x_det_ctxcode_mean, x_det_ctxcode_logvar = self.encoder_code(real_ctx, reuse=True, scope='encoder_det_code')
        fake_det_x_ctxcode = z_sample(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        fake_det_x_code_mean, fake_det_x_code_logvar = self.generator_code(fake_det_x_ctxcode, fake_det_x_supercode, scope="generator_det_code")
        
        prior_det_supercode_mean, prior_det_supercode_logvar = self.prior_code()#self.encoder_code(real_x, scope='prior_det_supercode')
        #random_det_supercode = z_sample(prior_det_supercode_mean, prior_det_supercode_logvar)
        prior_det_supercode_dist = self.prior_code_dist(fake_det_x_ctxcode, scope='prior_det_supercode')
        random_det_supercode = prior_det_supercode_dist.sample()
        prior_det_ctxcode_mean, prior_det_ctxcode_logvar = self.prior_code()
        #random_det_ctxcode = z_sample(prior_det_ctxcode_mean, prior_det_ctxcode_logvar)
        random_det_code_mean, random_det_code_var = self.generator_code(fake_det_x_ctxcode, random_det_supercode, reuse=True, scope="generator_det_code")
        random_det_code = z_sample(random_det_code_mean, random_det_code_var)
        random_simple_det_code = z_sample(*self.prior_code())

        prior_dist_det_code_dist = self.prior_code_dist(tf.stop_gradient(fake_det_x_ctxcode), scope='prior_dist_det_code')
        random_dist_det_code = prior_dist_det_code_dist.sample()
        
        prior_nondet_code_mean, prior_nondet_code_logvar = self.prior_code()

        x_nondet_code_mean, x_nondet_code_logvar = self.encoder_code(real_x, scope='encoder_nondet_code')
        fake_nondet_x_code = z_sample(x_nondet_code_mean, x_nondet_code_logvar)

        x_nondet_supercode_mean, x_nondet_supercode_logvar = self.encoder_supercode(code_stop_gradient(fake_nondet_x_code), scope='encoder_nondet_supercode')
        fake_nondet_x_supercode = z_sample(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar = self.encoder_code(real_ctx, scope='encoder_nondet_ctxcode')
        #x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar = self.encoder_code(real_ctx, reuse=True, scope='encoder_nondet_code')
        fake_nondet_x_ctxcode = z_sample(x_nondet_ctxcode_mean, x_nondet_ctxcode_logvar)
        fake_nondet_x_code_mean, fake_nondet_x_code_logvar = self.generator_code(fake_nondet_x_ctxcode, fake_nondet_x_supercode, scope="generator_nondet_code")
        
        prior_nondet_supercode_mean, prior_nondet_supercode_logvar = self.prior_code()#self.encoder_code(real_x, scope='prior_nondet_supercode')
        #random_nondet_supercode = z_sample(prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        prior_nondet_supercode_dist = self.prior_code_dist(fake_nondet_x_ctxcode, scope='prior_nondet_supercode')
        random_nondet_supercode = prior_nondet_supercode_dist.sample()
        prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar = self.prior_code()
        #random_nondet_ctxcode = z_sample(prior_nondet_ctxcode_mean, prior_nondet_ctxcode_logvar)
        random_nondet_code_mean, random_nondet_code_var = self.generator_code(fake_nondet_x_ctxcode, random_nondet_supercode, reuse=True, scope="generator_nondet_code")
        random_nondet_code = z_sample(random_nondet_code_mean, random_nondet_code_var)
        random_simple_nondet_code = z_sample(*self.prior_code())

        prior_dist_nondet_code_dist = self.prior_code_dist(tf.stop_gradient(fake_nondet_x_ctxcode), scope='prior_dist_nondet_code')
        random_dist_nondet_code = prior_dist_nondet_code_dist.sample()

        fake_full_det_x_code = tf.concat([fake_det_x_ctxcode],-1)
        fake_full_det_x_z = tf.concat([fake_det_x_code],-1)
        fake_det_x_features, fake_det_x_stats = self.generator(fake_full_det_x_code, z=fake_full_det_x_z, scope="generator_det")
        fake_det_x_scaffold = fake_det_x_stats[0]
        
        fake_full_nondet_x_code = tf.concat([fake_nondet_x_ctxcode, tf.stop_gradient(fake_det_x_ctxcode)],-1) 
        fake_full_nondet_x_z = tf.concat([fake_nondet_x_code, tf.stop_gradient(fake_det_x_code)],-1) 
        fake_full_nondet_x_discriminator_code = tf.concat([fake_nondet_x_code, fake_nondet_x_ctxcode, tf.stop_gradient(fake_det_x_code), tf.stop_gradient(fake_det_x_ctxcode)],-1) 
        fake_nondet_x_output = self.generator_spatial(fake_full_nondet_x_code, tf.stop_gradient(fake_det_x_scaffold), z=fake_full_nondet_x_z, scope="generator_nondet")

        random_full_det_x_code = tf.concat([fake_det_x_ctxcode], -1)
        random_full_det_x_z = tf.concat([random_det_code], -1)
        random_fake_det_x_features, random_fake_det_x_stats = self.generator(random_full_det_x_code, z=random_full_det_x_z, reuse=True, scope="generator_det")
        random_fake_det_x_scaffold = random_fake_det_x_stats[0]

        random_full_nondet_x_code = tf.concat([fake_nondet_x_ctxcode, fake_det_x_ctxcode], -1) 
        random_full_nondet_x_z = tf.concat([random_nondet_code, random_det_code], -1) 
        random_fake_nondet_x_output = self.generator_spatial(random_full_nondet_x_code, random_fake_det_x_scaffold, z=random_full_nondet_x_z, reuse=True, scope="generator_nondet")

        random_dist_full_det_x_code = tf.concat([fake_det_x_ctxcode], -1) 
        random_dist_full_det_x_z = tf.concat([random_dist_det_code], -1) 
        random_dist_fake_det_x_features, random_dist_fake_det_x_stats = self.generator(random_dist_full_det_x_code, z=random_dist_full_det_x_z, reuse=True, scope="generator_det")
        random_dist_fake_det_x_scaffold = random_dist_fake_det_x_stats[0]

        random_dist_full_nondet_x_code = tf.concat([fake_nondet_x_ctxcode, fake_det_x_ctxcode], -1) 
        random_dist_full_nondet_x_z = tf.concat([random_dist_nondet_code, random_dist_det_code], -1) 
        random_dist_fake_nondet_x_output = self.generator_spatial(random_dist_full_nondet_x_code, random_dist_fake_det_x_scaffold, z=random_dist_full_nondet_x_z, reuse=True, scope="generator_nondet")

        code_det_real_logit, code_det_fake_logit = self.discriminate_code(real_code_img=random_simple_det_code, fake_code_img=fake_det_x_code, name='det')
        code_nondet_real_logit, code_nondet_fake_logit = self.discriminate_code(real_code_img=random_simple_nondet_code, fake_code_img=fake_nondet_x_code, name='nondet')

        discriminator_fun = self.full_discriminator
        nondet_real_logit = discriminator_fun(tf.concat([real_ctx, real_x, tf.stop_gradient(fake_det_x_scaffold)], -1), fake_full_nondet_x_discriminator_code, scope='discriminator_nondet_x', label='real_nondet_x')
        fake_nondet_logit = discriminator_fun(tf.concat([real_ctx, fake_nondet_x_output, tf.stop_gradient(fake_det_x_scaffold)], -1), fake_full_nondet_x_discriminator_code, reuse=True, scope='discriminator_nondet_x', label='fake_nondet_x')
        
        if self.gan_type.__contains__('wgan-') or self.gan_type == 'dragan':
            GP = self.gradient_penalty(real=tf.concat([real_ctx, real_x, tf.stop_gradient(fake_det_x_scaffold)], -1), fake=tf.concat([real_ctx, fake_nondet_x_output, tf.stop_gradient(fake_det_x_scaffold)],-1), code=fake_full_nondet_x_discriminator_code, discriminator=discriminator_fun, name='nondet_x')
        else:
            GP = 0

        """ Define Loss """
        g_nondet_ce_loss = L1_loss(real_x, fake_nondet_x_output)
        #g_nondet_vgg_loss = self.vgg_loss(fake_nondet_x_output, real_x)
        g_nondet_adv_loss = generator_loss(self.gan_type, fake_nondet_logit)
        g_nondet_feature_loss = feature_loss(nondet_real_logit, fake_nondet_logit)
        g_nondet_reg_loss = regularization_loss('generator_nondet')

        #g_det_ce_loss = L2_loss(real_x, fake_det_x_stats[0])
        g_det_ce_loss = gaussian_loss(real_x, *fake_det_x_stats)
        #g_det_vgg_loss = self.vgg_loss(fake_det_x_stats[0], real_x)
        g_det_reg_loss = regularization_loss('generator_det')

        #g_nondet_code_ce_loss = L2_mean_loss(code_stop_gradient(fake_nondet_x_code), fake_nondet_x_code_mean)
        g_nondet_code_ce_loss = gaussian_loss(code_stop_gradient(fake_nondet_x_code), fake_nondet_x_code_mean, fake_nondet_x_code_logvar)
        e_nondet_code_kl_loss = kl_loss(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        e_nondet_code_prior_loss = gaussian_loss(fake_nondet_x_supercode, prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        e_nondet_code_prior2_loss = -tf.reduce_mean(input_tensor=prior_nondet_supercode_dist.log_prob(supercode_stop_gradient(fake_nondet_x_supercode))) / int(fake_nondet_x_supercode.get_shape()[-1])
        e_nondet_code_negent_loss = negent_loss(x_nondet_supercode_mean, x_nondet_supercode_logvar)
        #e_nondet_code_kl2_loss = kl_loss2(x_nondet_supercode_mean, x_nondet_supercode_logvar, prior_nondet_supercode_mean, prior_nondet_supercode_logvar)
        e_nondet_code_kl2_loss = (e_nondet_code_prior2_loss + e_nondet_code_negent_loss)
                
        e_nondet_adv_loss = generator_loss(self.code_gan_type, code_nondet_fake_logit)
        e_nondet_kl_loss = kl_loss(x_nondet_code_mean, x_nondet_code_logvar)
        e_nondet_prior_loss = gaussian_loss(fake_nondet_x_code, prior_nondet_code_mean, prior_nondet_code_logvar)
        e_nondet_negent_loss = negent_loss(x_nondet_code_mean, x_nondet_code_logvar)
        e_nondet_prior2_loss = -tf.reduce_mean(input_tensor=prior_dist_nondet_code_dist.log_prob(tf.stop_gradient(fake_nondet_x_code))) / int(fake_nondet_x_code.get_shape()[-1])
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
        e_det_code_prior2_loss = -tf.reduce_mean(input_tensor=prior_det_supercode_dist.log_prob(supercode_stop_gradient(fake_det_x_supercode))) / int(fake_det_x_supercode.get_shape()[-1])
        e_det_code_negent_loss = negent_loss(x_det_supercode_mean, x_det_supercode_logvar)
        #e_det_code_kl2_loss = kl_loss2(x_det_supercode_mean, x_det_supercode_logvar, prior_det_supercode_mean, prior_det_supercode_logvar)
        e_det_code_kl2_loss = (e_det_code_prior2_loss + e_det_code_negent_loss) 

        e_det_adv_loss = generator_loss(self.code_gan_type, code_det_fake_logit)
        e_det_kl_loss = kl_loss(x_det_code_mean, x_det_code_logvar)
        e_det_prior_loss = gaussian_loss(fake_det_x_code, prior_det_code_mean, prior_det_code_logvar)
        e_det_negent_loss = negent_loss(x_det_code_mean, x_det_code_logvar)
        e_det_prior2_loss = -tf.reduce_mean(input_tensor=prior_dist_det_code_dist.log_prob(tf.stop_gradient(fake_det_x_code))) / int(fake_det_x_code.get_shape()[-1])
        e_det_kl2_loss = (e_det_prior2_loss + e_det_negent_loss)
        e_det_reg_loss = regularization_loss('encoder_det_code')
        
        e_det_klctx_loss = kl_loss(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        e_det_priorctx_loss = gaussian_loss(fake_det_x_ctxcode, prior_det_ctxcode_mean, prior_det_ctxcode_logvar)
        e_det_negentctx_loss = negent_loss(x_det_ctxcode_mean, x_det_ctxcode_logvar)
        #e_det_klctx2_loss = kl_loss2(x_det_ctxcode_mean, x_det_ctxcode_logvar, prior_det_ctxcode_mean, prior_det_ctxcode_logvar)
        e_det_klctx2_loss = (e_det_priorctx_loss + e_det_negentctx_loss)

        d_nondet_adv_loss = discriminator_loss(self.gan_type, nondet_real_logit, fake_nondet_logit)
        d_nondet_reg_loss = GP + regularization_loss('discriminator_nondet_x')

        de_det_adv_loss = discriminator_loss(self.code_gan_type, code_det_real_logit, code_det_fake_logit)
        de_det_reg_loss = regularization_loss('discriminator_det_code')

        de_nondet_adv_loss = discriminator_loss(self.code_gan_type, code_nondet_real_logit, code_nondet_fake_logit)
        de_nondet_reg_loss = regularization_loss('discriminator_nondet_code')

        g_loss = g_nondet_adv_loss + g_nondet_reg_loss + 10*g_nondet_feature_loss + 0*g_nondet_ce_loss + e_nondet_adv_loss + e_nondet_reg_loss + 0.05*(0*e_nondet_prior_loss + e_nondet_prior2_loss + (g_nondet_code_ce_loss + (0*e_nondet_code_prior_loss + e_nondet_code_prior2_loss + e_nondet_code_negent_loss)) + e_nondet_negent_loss) + 0.001*e_nondet_klctx2_loss
        e_loss = 10*g_det_ce_loss + g_det_reg_loss + 0*e_det_adv_loss + e_det_reg_loss + 0.05*(0*e_det_prior_loss + e_det_prior2_loss + (g_det_code_ce_loss + (0*e_det_code_prior_loss + e_det_code_prior2_loss + e_det_code_negent_loss)) + e_det_negent_loss) + 0.001*e_det_klctx2_loss
        de_loss = de_nondet_adv_loss + de_nondet_reg_loss + de_det_adv_loss + de_det_reg_loss
        d_loss = d_nondet_adv_loss + d_nondet_reg_loss

        """ Result Image """
        fake_det_x = fake_det_x_stats[0]
        fake_det_x_var = tf.exp(fake_det_x_stats[1])
        fake_nondet_x = fake_nondet_x_output
        random_fake_det_x = random_fake_det_x_stats[0]
        random_fake_nondet_x = random_fake_nondet_x_output
        random_dist_fake_det_x = random_dist_fake_det_x_stats[0]
        random_dist_fake_nondet_x = random_dist_fake_nondet_x_output

        """" Summary """
        if summary:
            summary_global_step = tf.summary.scalar("global_step", global_step, step=global_step)

            summary_g_loss = tf.summary.scalar("g_loss", g_loss, step=global_step)
            summary_e_loss = tf.summary.scalar("e_loss", e_loss, step=global_step)
            summary_de_loss = tf.summary.scalar("de_loss", de_loss, step=global_step)
            summary_d_loss = tf.summary.scalar("d_loss", d_loss, step=global_step)

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

            summary_e_det_kl_loss = tf.summary.scalar("e_det_kl_loss", e_det_kl_loss, step=global_step)
            summary_e_det_negent_loss = tf.summary.scalar("e_det_negent_loss", e_det_negent_loss, step=global_step)
            summary_e_det_prior_loss = tf.summary.scalar("e_det_prior_loss", e_det_prior_loss, step=global_step)
            summary_e_det_prior2_loss = tf.summary.scalar("e_det_prior2_loss", e_det_prior2_loss, step=global_step)
            summary_e_det_kl2_loss = tf.summary.scalar("e_det_kl2_loss", e_det_kl2_loss, step=global_step)
            summary_e_det_reg_loss = tf.summary.scalar("det_e_reg_loss", e_det_reg_loss, step=global_step)
            summary_e_det_adv_loss = tf.summary.scalar("e_det_adv_loss", e_det_adv_loss, step=global_step)

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
            
            summary_e_nondet_kl_loss = tf.summary.scalar("e_nondet_kl_loss", e_nondet_kl_loss, step=global_step)
            summary_e_nondet_negent_loss = tf.summary.scalar("e_nondet_negent_loss", e_nondet_negent_loss, step=global_step)
            summary_e_nondet_prior_loss = tf.summary.scalar("e_nondet_prior_loss", e_nondet_prior_loss, step=global_step)
            summary_e_nondet_prior2_loss = tf.summary.scalar("e_nondet_prior2_loss", e_nondet_prior2_loss, step=global_step)
            summary_e_nondet_kl2_loss = tf.summary.scalar("e_nondet_kl2_loss", e_nondet_kl2_loss, step=global_step)
            summary_e_nondet_reg_loss = tf.summary.scalar("e_nondet_reg_loss", e_nondet_reg_loss, step=global_step)
            summary_e_nondet_adv_loss = tf.summary.scalar("e_nondet_adv_loss", e_nondet_adv_loss, step=global_step)

            summary_g_det_ce_loss = tf.summary.scalar("g_det_ce_loss", g_det_ce_loss, step=global_step)
            #summary_g_det_vgg_loss = tf.summary.scalar("g_det_vgg_loss", g_det_vgg_loss, step=global_step)
            summary_g_det_reg_loss = tf.summary.scalar("g_det_reg_loss", g_det_reg_loss, step=global_step)

            summary_g_nondet_ce_loss = tf.summary.scalar("g_nondet_ce_loss", g_nondet_ce_loss, step=global_step)
            #summary_g_nondet_vgg_loss = tf.summary.scalar("g_nondet_vgg_loss", g_nondet_vgg_loss, step=global_step)
            summary_g_nondet_feature_loss = tf.summary.scalar("g_nondet_feature_loss", g_nondet_feature_loss, step=global_step)
            summary_g_nondet_reg_loss = tf.summary.scalar("g_nondet_reg_loss", g_nondet_reg_loss, step=global_step)
            summary_g_nondet_adv_loss = tf.summary.scalar("g_nondet_adv_loss", g_nondet_adv_loss, step=global_step)
            
            summary_d_nondet_adv_loss = tf.summary.scalar("d_nondet_adv_loss", d_nondet_adv_loss, step=global_step)
            summary_d_nondet_reg_loss = tf.summary.scalar("d_nondet_reg_loss", d_nondet_reg_loss, step=global_step)

            summary_de_det_adv_loss = tf.summary.scalar("de_det_adv_loss", de_det_adv_loss, step=global_step)
            summary_de_det_reg_loss = tf.summary.scalar("de_det_reg_loss", de_det_reg_loss, step=global_step)
            
            summary_de_nondet_adv_loss = tf.summary.scalar("de_nondet_adv_loss", de_nondet_adv_loss, step=global_step)
            summary_de_nondet_reg_loss = tf.summary.scalar("de_nondet_reg_loss", de_nondet_reg_loss, step=global_step)

        losses = (g_loss, e_loss, de_loss, d_loss)
        outputs = (real_ctx, real_x, real_x_segmap, real_x_segmap_onehot, fake_det_x, fake_det_x_var, fake_nondet_x, random_fake_det_x, random_fake_nondet_x, random_dist_fake_det_x, random_dist_fake_nondet_x)

        return losses, outputs

    @tf.function
    def build_model(self):
        with tf.compat.v1.variable_scope("inputs", reuse=tf.compat.v1.AUTO_REUSE):
            global_step = tf.compat.v1.get_variable('global_step', shape=[], initializer=tf.compat.v1.constant_initializer(0), trainable=False)
            real_ctx = tf.compat.v1.get_variable('real_ctx', shape=[self.batch_size, self.img_height, self.img_width, self.img_ch], initializer=tf.compat.v1.constant_initializer(0.0), trainable=False)
            real_x = tf.compat.v1.get_variable('real_x', shape=[self.batch_size, self.img_height, self.img_width, self.img_ch], initializer=tf.compat.v1.constant_initializer(0.0), trainable=False)
            real_x_segmap = tf.compat.v1.get_variable('real_x_segmap', shape=[self.batch_size, self.img_height, self.img_width], initializer=tf.compat.v1.constant_initializer(0), trainable=False)
            real_x_segmap_onehot = tf.compat.v1.get_variable('real_x_segmap_onehot', shape=[self.batch_size, self.img_height, self.img_width, self.segmap_ch], initializer=tf.compat.v1.constant_initializer(0), trainable=False)
        
        return self.execute_model(global_step, real_ctx, real_x, real_x_segmap, real_x_segmap_onehot, summary=False)
  
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

            g_lr = decay_fn(self.init_lr / 2 / float(self.n_critic))
            e_lr = decay_fn(self.init_lr / 2 / float(self.n_critic))
            de_lr = decay_fn(self.init_lr * 2)
            d_lr = decay_fn(self.init_lr * 2)

        else :
            beta1 = self.beta1
            beta2 = self.beta2
            g_lr = decay_fn(self.init_lr / float(self.n_critic))
            e_lr = decay_fn(self.init_lr / float(self.n_critic))
            de_lr = decay_fn(self.init_lr)
            d_lr = decay_fn(self.init_lr)

        self.G_optim = tf.keras.optimizers.Adam(g_lr, beta_1=beta1, beta_2=beta2)
        self.E_optim = tf.keras.optimizers.Adam(e_lr, beta_1=beta1, beta_2=beta2)
        self.DE_optim = tf.keras.optimizers.Adam(de_lr, beta_1=beta1, beta_2=beta2)
        self.D_optim = tf.keras.optimizers.Adam(d_lr, beta_1=beta1, beta_2=beta2)

        t_vars = tf.compat.v1.trainable_variables()
        self.G_vars = [var for var in t_vars if 'generator_nondet' in var.name or 'encoder_nondet_code' in var.name or 'generator_nondet_code' in var.name or 'prior_nondet_supercode' in var.name or 'encoder_nondet_supercode' in var.name or 'encoder_nondet_ctxcode' in var.name or 'prior_dist_nondet_code' in var.name]
        self.E_vars = [var for var in t_vars if 'generator_det' in var.name or 'encoder_det_code' in var.name in var.name or 'generator_det_code' in var.name or 'prior_det_supercode' in var.name or 'encoder_det_supercode' in var.name or 'encoder_det_ctxcode' in var.name or 'prior_dist_det_code' in var.name]
        self.DE_vars = [var for var in t_vars if 'discriminator_det_code' in var.name or 'discriminator_nondet_code' in var.name]
        self.D_vars = [var for var in t_vars if 'discriminator_nondet_x' in var.name]

    def report_losses(self, epoch, idx, duration, losses):
        [g_loss, e_loss, de_loss, d_loss] = losses
        print("Epoch: [%2d] [%5d/%5d] time: %4.4f g_loss: %.8f" % (
            epoch, idx, self.iteration, duration, g_loss))
        print("Epoch: [%2d] [%5d/%5d] time: %4.4f e_loss: %.8f" % (
            epoch, idx, self.iteration, duration, e_loss))
        print("Epoch: [%2d] [%5d/%5d] time: %4.4f de_loss: %.8f" % (
            epoch, idx, self.iteration, duration, de_loss))
        print("Epoch: [%2d] [%5d/%5d] time: %4.4f d_loss: %.8f" % (
            epoch, idx, self.iteration, duration, d_loss))
        sys.stdout.flush()

    def report_outputs(self, epoch, idx, outputs):
        print("O1", time.time())
        [real_ctx, real_x, real_x_segmap, real_x_segmap_onehot, fake_det_x, fake_det_x_var, fake_nondet_x, random_fake_det_x, random_fake_nondet_x, random_dist_fake_det_x, random_dist_fake_nondet_x] = outputs

        print("O2", time.time())
        save_images(real_ctx.numpy(), [self.batch_size, 1],
                   './{}/real_ctximage_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

        print("O3", time.time())
        save_images(real_x.numpy(), [self.batch_size, 1],
                   './{}/real_image_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

        print("O4", time.time())
        save_images(fake_det_x.numpy(), [self.batch_size, 1],
                    './{}/fake_det_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
        print("O5", time.time())
        imsave(image_to_uint8(fake_det_x_var.numpy()), [self.batch_size, 1],
                    './{}/fake_det_var_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

        print("O6", time.time())
        save_images(fake_nondet_x.numpy(), [self.batch_size, 1],
                    './{}/fake_nondet_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

        print("O7", time.time())
        save_images(random_fake_det_x.numpy(), [self.batch_size, 1],
                    './{}/random_fake_det_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))
        print("O8", time.time())
        save_images(random_fake_nondet_x.numpy(), [self.batch_size, 1],
                    './{}/random_fake_nondet_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))

        print("O9", time.time())
        save_images(random_dist_fake_det_x.numpy(), [self.batch_size, 1],
                    './{}/random_dist_fake_det_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))
        print("O10", time.time())
        save_images(random_dist_fake_nondet_x.numpy(), [self.batch_size, 1],
                    './{}/random_dist_fake_nondet_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))
        print("O11", time.time())

    def train(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.img_class.ctximage, self.img_class.image, self.img_class.segmap))
        dataset = dataset.shuffle(len(self.img_class.image), reshuffle_each_iteration=True).repeat(None)
        dataset = dataset.map(self.img_class.image_processing, num_parallel_calls=16*self.batch_size).batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.batch_size)
        
        distribute_strategy = tf.distribute.MirroredStrategy()
        dataset = distribute_strategy.experimental_distribute_dataset(dataset)
        with distribute_strategy.scope():
            # build global step
            global_step = tf.Variable(0, dtype=tf.int64, name="global_step", aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA) 

            # prepare model
            self.prepare_model()

            # build model
            self.build_model()

            # show network architecture
            show_all_variables()

            # build optimizers
            self.build_optimizers()

            # saver to save model
            checkpoint = tf.train.Checkpoint(**dict([(var.name, var) for var in tf.compat.v1.trainable_variables()]))
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(self.checkpoint_dir, self.model_dir), max_to_keep=1000)

            # restore check-point if it exits
            if checkpoint_manager.latest_checkpoint:
                checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            # record start time
            start_time = time.time()

            @tf.function
            def train_step(global_step, inputs):
                def step_fn(global_step, inputs):
                    [real_ctx, real_x, real_x_segmap, real_x_segmap_onehot] = inputs
                    with tf.GradientTape(persistent=True) as tape:
                        losses, outputs = self.execute_model(global_step, real_ctx, real_x, real_x_segmap, real_x_segmap_onehot, summary=True)

                    [g_loss, e_loss, de_loss, d_loss] = losses
                   
                    g_gradients = tape.gradient(g_loss, self.G_vars)
                    e_gradients = tape.gradient(e_loss, self.E_vars)
                    de_gradients = tape.gradient(de_loss, self.DE_vars)
                    d_gradients = tape.gradient(d_loss, self.D_vars)

                    self.G_optim.apply_gradients(zip(g_gradients, self.G_vars))
                    self.E_optim.apply_gradients(zip(e_gradients, self.E_vars))
                    self.DE_optim.apply_gradients(zip(de_gradients, self.DE_vars))
                    self.D_optim.apply_gradients(zip(d_gradients, self.D_vars))

                    global_step.assign_add(1)

                    return global_step, losses, outputs 
                #return step_fn(global_step, inputs)
                return distribute_strategy.experimental_run_v2(step_fn, args=(global_step, inputs))

            # training loop
            for inputs in dataset:
                print("L1", time.time())
                with self.writer.as_default():
                    result = train_step(global_step, inputs)

                print("L2", time.time())
                self.writer.flush()

                print("L3", time.time())
                [counter, losses, outputs] = result
                
                print("L4", time.time())
                epoch = (counter-1) // self.iteration 
                idx = (counter-1) % self.iteration

                print("L5", time.time())
                self.report_losses(epoch, idx, time.time() - start_time, losses)

                print("L6", time.time())
                if np.mod(idx+1, self.print_freq) == 0:
                    self.report_outputs(epoch, idx, outputs)

                print("L7", time.time())
                if counter-1 > 0 and np.mod(counter-1, self.save_freq) == 0:
                    checkpoint_manager.save()
                
                print("L8", time.time())

            # save model for final step
            checkpoint_manager.save()

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
