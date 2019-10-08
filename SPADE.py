import sys
from ops import *
from utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
from tqdm import tqdm
from vgg19_keras import VGGLoss
from unet import unet

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

        self.init_lr = args.lr
        self.TTUR = args.TTUR
        self.ch = args.ch
        self.segmap_ch = args.segmap_ch

        self.beta1 = args.beta1
        self.beta2 = args.beta2


        self.num_style = args.num_style
        self.guide_img = args.guide_img


        """ Weight """
        self.unet_kl_weight = args.unet_kl_weight
        self.unet_ce_weight = args.unet_ce_weight

        self.segmap_adv_weight = args.segmap_adv_weight
        self.segmap_kl_weight = args.segmap_kl_weight
        self.segmap_ce_weight = args.segmap_ce_weight
        self.segmap_vgg_weight = args.segmap_vgg_weight
        self.segmap_feature_weight = args.segmap_feature_weight

        self.adv_weight = args.adv_weight
        self.vgg_weight = args.vgg_weight
        self.feature_weight = args.feature_weight
        self.kl_weight = args.kl_weight
        self.ce_weight = args.ce_weight

        self.ld = args.ld

        """ Generator """
        self.segmap_num_upsampling_layers = args.segmap_num_upsampling_layers
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

    def image_encoder_base(self, x_init, channel):
        #x = resize(x_init, self.img_height, self.img_width)
        x = x_init

        #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv')
        #x = instance_norm(x, scope='ins_norm')
        x = constin_resblock(x, channel, use_bias=True, sn=self.sn, scope='conv')
        x = down_sample_avg(x)
        print(x)

        for i in range(3):
            #x = lrelu(x, 0.2)
            #x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv_' + str(i))
            #x = instance_norm(x, scope='ins_norm_' + str(i))
            x = constin_resblock(x, channel * 2, use_bias=True, sn=self.sn, scope='conv_' + str(i))
            x = down_sample_avg(x)
            print(x)

            channel = channel * 2

            # 128, 256, 512

        #x = lrelu(x, 0.2)
        #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv_3')
        #x = instance_norm(x, scope='ins_norm_3')
        x = constin_resblock(x, channel, use_bias=True, sn=self.sn, scope='conv_3')
        x = down_sample_avg(x)
        print(x)

        #if self.img_height >= 256 or self.img_width >= 256 :
        if self.img_height >= 256 or self.img_width >= 256 :
            #x = lrelu(x, 0.2)
            #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=False, sn=self.sn, scope='conv_4')
            #x = instance_norm(x, scope='ins_norm_4')
            x = constin_resblock(x, channel, use_bias=False, sn=self.sn, scope='conv_4')
            x = down_sample_avg(x)
            print(x)

        x = lrelu(x, 0.2)
        print(x)

        return x, channel

    def image_encoder_segmap_code(self, x_init, reuse=False, scope='encoder_segmap_code'):
        with tf.variable_scope(scope, reuse=reuse):
            x, channel = self.image_encoder_base(x_init, self.segmap_ch)

            mean = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_mean')
            var = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_var')
            return mean, var

    def image_prior_segmap_code(self):
        batch_size = self.batch_size
        channel = self.segmap_ch * 4
        mean = tf.zeros([batch_size, channel])
        var = tf.ones([batch_size, channel])
        return mean, var

    def image_encoder(self, x_init, reuse=False, scope='encoder'):
        with tf.variable_scope(scope, reuse=reuse):
            x, channel = self.image_encoder_base(x_init, self.ch)

            mean = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_mean')
            var = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_var')
            return mean, var

    def generator_segmap(self, z, deterministic=False, reuse=False, scope="generator_segmap"):
        context_depth = 8
        channel = self.segmap_ch * 4 * 4
        batch_size = self.batch_size
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.random_normal(shape=[batch_size, self.segmap_ch * 4])
            if deterministic:
                x = 0*x 
            context = z

            #for i in range(context_depth):
            #    context = fully_connected(context, context.get_shape()[-1], use_bias=True, sn=self.sn, scope='linear_context_' + str(i))

            if self.segmap_num_upsampling_layers == 'less':
                num_up_layers = 4
            elif self.segmap_num_upsampling_layers == 'normal':
                num_up_layers = 5
            elif self.segmap_num_upsampling_layers == 'more':
                num_up_layers = 6
            elif self.segmap_num_upsampling_layers == 'most':
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


            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='adain_resblock_fix_0')

            x = up_sample(x, scale_factor=2)
            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='adain_resblock_fix_1')

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = adain_resblock(context, x, channels=channel, use_bias=True, sn=self.sn, scope='adain_resblock_fix_2')

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = adain_resblock(context, x, channels=channel//2, use_bias=True, sn=self.sn, scope='adain_resblock_' + str(i))

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='adain_resblock_4')
            x = adain_resblock(context, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='adain_resblock_4')

            x = lrelu(x, 0.2)
            x = conv(x, channels=self.segmap_out_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='logit')

            return x

    def generator(self, segmap, x_mean, x_var, random_style=False, reuse=False, scope="generator"):
        channel = self.ch * 4 * 4
        with tf.variable_scope(scope, reuse=reuse):
            batch_size = segmap.get_shape().as_list()[0]
            if random_style :
                x = tf.random_normal(shape=[batch_size, self.ch * 4])
            else :
                x = z_sample(x_mean, x_var)

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


            x = spade_resblock(segmap, x, channels=channel, use_bias=True, sn=self.sn, scope='spade_resblock_fix_0')

            x = up_sample(x, scale_factor=2)
            x = spade_resblock(segmap, x, channels=channel, use_bias=True, sn=self.sn, scope='spade_resblock_fix_1')

            if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)

            x = spade_resblock(segmap, x, channels=channel, use_bias=True, sn=self.sn, scope='spade_resblock_fix_2')

            for i in range(4) :
                x = up_sample(x, scale_factor=2)
                x = spade_resblock(segmap, x, channels=channel//2, use_bias=True, sn=self.sn, scope='spade_resblock_' + str(i))

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64

            if self.num_upsampling_layers == 'most':
                x = up_sample(x, scale_factor=2)
            #    x = spade_resblock(segmap, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='spade_resblock_4')
            x = spade_resblock(segmap, x, channels=channel // 2, use_bias=True, sn=self.sn, scope='spade_resblock_4')

            x = lrelu(x, 0.2)
            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator_segmap_code(self, x, reuse=False, scope="discriminator_segmap_code", label=None):
        channel = x.get_shape()[-1]
        with tf.variable_scope(scope, reuse=reuse):
            x = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_x_1')
            print(x)
            x = lrelu(x, 0.2)
            print(x)

            x = fully_connected(x, channel // 4, use_bias=True, sn=self.sn, scope='linear_x_2')
            print(x)
            x = lrelu(x, 0.2)
            print(x)

            z = fully_connected(x, 1, sn=self.sn, scope='linear_z')
            print(z)

            z_summary = [tf.summary.scalar(label + ".logit", tf.reduce_sum(z))]

            return [[z]], z_summary

    def full_discriminator_segmap(self, segmap, segmap_code=None, reuse=False, scope="discriminator_segmap", label=None):
        channel = self.segmap_ch
        segmap_code = segmap_code
        with tf.variable_scope(scope, reuse=reuse):
            #x = conv(segmap, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv')
            #x = instance_norm(x, scope='ins_norm')
            x = adain_resblock(segmap_code, segmap, channel, use_bias=True, sn=self.sn, scope='conv')
            x = down_sample_avg(x)
            print(x)

            for i in range(3):
                #x = lrelu(x, 0.2)
                #x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv_' + str(i))
                #x = instance_norm(x, scope='ins_norm_' + str(i))
                x = adain_resblock(segmap_code, x, channel * 2, use_bias=True, sn=self.sn, scope='conv_' + str(i))
                x = down_sample_avg(x)
                print(x)

                channel = channel * 2

                # 128, 256, 512

            #x = lrelu(x, 0.2)
            #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=self.sn, scope='conv_3')
            #x = instance_norm(x, scope='ins_norm_3')
            x = adain_resblock(segmap_code, x, channel, use_bias=True, sn=self.sn, scope='conv_3')
            x = down_sample_avg(x)
            print(x)

            #if self.img_height >= 256 or self.img_width >= 256 :
            if self.img_height >= 256 or self.img_width >= 256 :
                #x = lrelu(x, 0.2)
                #x = conv(x, channel, kernel=3, stride=2, pad=1, use_bias=False, sn=self.sn, scope='conv_4')
                #x = instance_norm(x, scope='ins_norm_4')
                x = adain_resblock(segmap_code, x, channel, use_bias=False, sn=self.sn, scope='conv_4')
                x = down_sample_avg(x)
                print(x)

            x = lrelu(x, 0.2)
            print(x)

            x0 = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_x0')
            print(x0)
            x0 = lrelu(x0, 0.2)
            print(x0)

            z0 = fully_connected(x0, 1, sn=self.sn, scope='linear_z0')
            #z0 = tf.reduce_mean(x0*fixed_segmap_code, 1, keep_dims=True)
            #z0 = -tf.reduce_mean(tf.math.square(x0-fixed_segmap_code), 1, keep_dims=True)
            #z0 = -tf.reduce_mean(tf.math.abs(x0-fixed_segmap_code), 1, keep_dims=True)
            print(z0)

            z0_summary = tf.summary.scalar(label + ".logit0", tf.reduce_mean(z0))
            z = z0
            D_summary = [z0_summary]

            if segmap_code is not None:
                x1 = fully_connected(x, channel // 2, use_bias=True, sn=self.sn, scope='linear_x1')
                print(x1)
                x1 = lrelu(x1, 0.2)
                print(x1)
                #z1 = tf.reduce_mean(x1*segmap_code, 1, keep_dims=True)
                #z1 = tf.reduce_mean(tf.math.square(x1-segmap_code), 1, keep_dims=True)
                z1 = -tf.reduce_sum(tf.math.abs(x1-segmap_code), 1, keep_dims=True)
                print(z1)
                z1_summary = tf.summary.scalar(label + ".logit1", tf.reduce_mean(z1))
                #z = z + z1
                #D_summary = D_summary + [z1_summary]

            z = tf.reshape(z, [z.get_shape()[0], 1, 1, 1])

            D_logit = [[x, z]]
            return D_logit, D_summary

    def discriminator_segmap(self, x_init, segmap_code, reuse=False, scope="discriminator_segmap", label=None):
        D_logit = []
        D_summary = []
        with tf.variable_scope(scope, reuse=reuse):
            for scale in range(self.n_scale):
                feature_loss = []
                channel = self.segmap_ch
                x = x_init

                #x = conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, sn=False, scope='ms_' + str(scale) + 'conv_0')
                #x = lrelu(x, 0.2)
                x = adain_resblock(segmap_code, x, channel, use_bias=True, sn=self.sn, scope='ms_' + str(scale) + 'adain_0')
                x = down_sample_avg(x)

                feature_loss.append(x)

                for i in range(1, self.n_dis):
                    #stride = 1 if i == self.n_dis - 1 else 2
                    #x = conv(x, channel * 2, kernel=4, stride=stride, pad=1, use_bias=True, sn=self.sn, scope='ms_' + str(scale) + 'conv_' + str(i))
                    #x = instance_norm(x, scope='ms_' + str(scale) + 'ins_norm_' + str(i))
                    #x = lrelu(x, 0.2)

                    x = adain_resblock(segmap_code, x, channel*2, use_bias=True, sn=self.sn, scope='ms_' + str(scale) + 'adain_' + str(i))
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

    def image_translate(self, segmap_img, x_img=None, random_style=False, reuse=False):

        if random_style :
            x_mean, x_var = None, None
        else :
            x_mean, x_var = self.image_encoder(x_img, reuse=reuse, scope='encoder')

        x = self.generator(segmap_img, x_mean, x_var, random_style, reuse=reuse, scope='generator')

        return x, x_mean, x_var

    def image_discriminate_segmap_code(self, real_segmap_code_img, fake_segmap_code_img):
        real_logit = self.discriminator_segmap_code(real_segmap_code_img, scope='discriminator_segmap_code', label='real_segmap_code')
        fake_logit = self.discriminator_segmap_code(fake_segmap_code_img, reuse=True, scope='discriminator_segmap_code', label='fake_segmap_code')

        return real_logit, fake_logit

    def image_discriminate_segmap(self, real_segmap_img, fake_segmap_img, segmap_code=None):
        real_logit = self.discriminator_segmap(real_segmap_img, segmap_code, scope='discriminator_segmap', label='real_segmap')
        fake_logit = self.discriminator_segmap(fake_segmap_img, segmap_code, reuse=True, scope='discriminator_segmap', label='fake_segmap')

        return real_logit, fake_logit

    def image_discriminate(self, real_segmap_img, real_img, fake_segmap_img, fake_img):
        real_logit = self.discriminator(real_segmap_img, real_img, scope='discriminator')
        fake_logit = self.discriminator(fake_segmap_img, fake_img, reuse=True, scope='discriminator')

        return real_logit, fake_logit

    def gradient_penalty_segmap_code(self, real_segmap_code, fake_segmap_code):
        if self.gan_type == 'dragan':
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            alpha = tf.random_uniform(shape=[shape[0], 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else:
            alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
            interpolated_segmap_code = alpha * real_segmap_code + (1. - alpha) * fake_segmap_code

        logit, logit_summary = self.discriminator_segmap_code(interpolated_segmap_code, reuse=True, scope='discriminator_segmap_code', label='interpolated_segmap_code')

        GP = []


        for i in range(len(logit)) :
            grad = tf.gradients(logit[i][-1], interpolated_segmap_code)[0]  # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp':
                GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))

        return tf.reduce_mean(GP)

    def gradient_penalty_segmap(self, real_segmap, fake_segmap, segmap_code):
        if self.gan_type == 'dragan':
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else:
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated_segmap = alpha * real_segmap + (1. - alpha) * fake_segmap

        logit, logit_summary = self.discriminator_segmap(interpolated_segmap, segmap_code=segmap_code, reuse=True, scope='discriminator_segmap', label='interpolated_segmap')

        GP = []


        for i in range(len(logit)) :
            grad = tf.gradients(logit[i][-1], interpolated_segmap)[0]  # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp':
                GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))

        return tf.reduce_mean(GP)

    def gradient_penalty(self, real_segmap, real, fake_segmap, fake):
        if self.gan_type == 'dragan':
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else:
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated_segmap = alpha * real_segmap + (1. - alpha) * fake_segmap
            interpolated = alpha * real + (1. - alpha) * fake

        logit = self.discriminator(interpolated_segmap, interpolated, reuse=True, scope='discriminator')

        GP = []


        for i in range(self.n_scale) :
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
        #self.segmap_out_ch = len(img_class.color_value_dict)
        self.segmap_out_ch = 3

        self.dataset_num = len(img_class.image)
        self.test_dataset_num = len(img_class.segmap_test)


        img_and_segmap = tf.data.Dataset.from_tensor_slices((img_class.image, img_class.segmap))
        segmap_test = tf.data.Dataset.from_tensor_slices(img_class.segmap_test)


        gpu_device = '/gpu:0'
        img_and_segmap = img_and_segmap.apply(shuffle_and_repeat(self.dataset_num)).apply(
            map_and_batch(img_class.image_processing, self.batch_size, num_parallel_batches=16,
                          drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))

        segmap_test = segmap_test.apply(shuffle_and_repeat(self.dataset_num)).apply(
            map_and_batch(img_class.test_image_processing, batch_size=self.batch_size, num_parallel_batches=16,
                          drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))


        img_and_segmap_iterator = img_and_segmap.make_one_shot_iterator()
        segmap_test_iterator = segmap_test.make_one_shot_iterator()

        self.real_x, self.real_x_segmap, self.real_x_segmap_onehot = img_and_segmap_iterator.get_next()
        self.real_x_segmap_test, self.real_x_segmap_test_onehot = segmap_test_iterator.get_next()

        self.global_step = tf.train.create_global_step()

        """ Define Generator, Discriminator """
        ###UNET###unet_x_logits = unet(self.real_x, segmap_out_ch = self.segmap_out_ch)
        x_segmap_mean, x_segmap_var = self.image_encoder_segmap_code(self.real_x)
        fake_x_segmap_code = z_sample(x_segmap_mean, x_segmap_var)
        random_segmap_mean, random_segmap_var = self.image_prior_segmap_code()
        random_segmap_code = z_sample(random_segmap_mean, random_segmap_var)
        fake_x_segmap_logits = self.generator_segmap(fake_x_segmap_code, deterministic=True)
        random_fake_x_segmap_logits = self.generator_segmap(random_segmap_code, deterministic=True, reuse=True)
        #fake_x_segmap_logits, x_segmap_mean, x_segmap_var = self.image_translate_segmap(x_img=softmax(unet_x_logits))
        #fake_x_segmap_logits, x_segmap_mean, x_segmap_var = self.image_translate_segmap(random_segmap_code=True)
        ###GENIMG###fake_x, x_mean, x_var = self.image_translate(segmap_img=tf.nn.softmax(fake_x_segmap_logits), x_img=self.real_x)
        
        [segmap_code_real_logit, segmap_code_real_summary], [segmap_code_fake_logit, segmap_code_fake_summary] = self.image_discriminate_segmap_code(real_segmap_code_img=random_segmap_code, fake_segmap_code_img=fake_x_segmap_code)

        real_x_segmap_img = self.real_x
        fake_x_segmap_img = fake_x_segmap_logits 
        #segmap_real_logit, segmap_fake_logit = self.image_discriminate_segmap(real_segmap_img=self.real_x_segmap_onehot, fake_segmap_img=softmax(fake_x_segmap_logits), segmap_code=None)
        [segmap_real_logit, segmap_real_summary], [segmap_fake_logit, segmap_fake_summary] = self.image_discriminate_segmap(real_segmap_img=real_x_segmap_img, fake_segmap_img=fake_x_segmap_img, segmap_code=fake_x_segmap_code)
        ###GAN###real_logit, fake_logit = self.image_discriminate(real_segmap_img=self.real_x_segmap_onehot, real_img=self.real_x, fake_segmap_img=fake_x_segmap, fake_img=fake_x)
        
        if self.gan_type.__contains__('wgan-') or self.gan_type == 'dragan':
            segmap_GP = self.gradient_penalty_segmap(real_segmap=real_x_segmap_img, fake_segmap=fake_x_segmap_img, segmap_code=fake_x_segmap_code)
        else:
            segmap_GP = 0

        if self.code_gan_type.__contains__('wgan-') or self.code_gan_type == 'dragan':
            segmap_code_GP = self.gradient_penalty_segmap_code(real_segmap_code=random_segmap_code, fake_segmap_code=fake_x_segmap_code)
        else:
            segmap_code_GP = 0

        ###GAN###if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
        ###GAN###    GP = self.gradient_penalty(real=self.real_x, real_segmap=self.real_x_segmap_onehot, fake=fake_x, fake_segmap=fake_x_segmap)
        ###GAN###else:
        ###GAN###    GP = 0

        """ Define Loss """
        ###UNET###unet_g_kl_loss = 0.0 
        ###UNET###unet_g_ce_loss = self.unet_ce_weight * ce_loss(self.real_x_segmap_onehot, unet_x_logits)
        ###UNET###unet_g_reg_loss = regularization_loss('unet')
        
        ###UNET###self.unet_g_loss = unet_g_kl_loss + unet_g_ce_loss + unet_g_reg_loss

        segmap_g_adv_loss = self.segmap_adv_weight * generator_loss(self.gan_type, segmap_fake_logit)
        #segmap_g_ce_loss = self.segmap_ce_weight * ce_loss(self.real_x_segmap_onehot, fake_x_segmap_logits)
        #segmap_g_ce_loss = self.segmap_ce_weight * L2_loss(real_x_segmap_img, fake_x_segmap_img)
        segmap_g_ce_loss = self.segmap_ce_weight * L2_loss(real_x_segmap_img, fake_x_segmap_img)
        segmap_g_vgg_loss = self.segmap_vgg_weight * VGGLoss()(real_x_segmap_img, fake_x_segmap_img)
        #segmap_g_ce_loss = self.segmap_ce_weight * tf.nn.l2_loss(softmax(unet_x_logits) - softmax(fake_x_segmap_logits))
        segmap_g_feature_loss = self.segmap_feature_weight * feature_loss(segmap_real_logit, segmap_fake_logit)
        segmap_g_reg_loss = regularization_loss('generator_segmap')
        
        segmap_e_adv_loss = self.segmap_adv_weight * generator_loss(self.code_gan_type, segmap_code_fake_logit)
        segmap_e_kl_loss = self.segmap_kl_weight * kl_loss(x_segmap_mean, x_segmap_var)
        segmap_e_reg_loss = regularization_loss('encoder_segmap_code')

        segmap_d_adv_loss = self.segmap_adv_weight * discriminator_loss(self.gan_type, segmap_real_logit, segmap_fake_logit)
        segmap_d_reg_loss = self.segmap_adv_weight * segmap_GP + regularization_loss('discriminator_segmap')

        segmap_de_adv_loss = self.segmap_adv_weight * discriminator_loss(self.code_gan_type, segmap_code_real_logit, segmap_code_fake_logit)
        segmap_de_reg_loss = self.segmap_adv_weight * segmap_code_GP + regularization_loss('discriminator_segmap_code')

        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        segmap_e_kl_loss_truncated = tf.minimum(1000.0,tf.abs(segmap_e_kl_loss))
        with tf.control_dependencies([ema.apply([segmap_e_kl_loss_truncated])]):
            segmap_e_kl_loss_ema = ema.average(segmap_e_kl_loss_truncated)
            segmap_e_kl_loss_weight = tf.maximum(0.0,segmap_e_kl_loss_ema - 100.0)/100.0
            segmap_e_kl_loss_adjusted = 0.0#segmap_e_kl_loss_weight*segmap_e_kl_loss + (1-segmap_e_kl_loss_weight)*segmap_e_kl_loss_ema

            #self.segmap_g_loss = segmap_g_ce_loss + segmap_g_reg_loss + segmap_e_kl_loss + segmap_e_reg_loss
            #self.segmap_g_loss = segmap_g_adv_loss + segmap_g_reg_loss + segmap_e_kl_loss + segmap_e_reg_loss
            #self.segmap_g_loss = segmap_g_adv_loss + segmap_g_ce_loss + segmap_g_reg_loss + segmap_e_kl_loss + segmap_e_reg_loss
            #self.segmap_g_loss = segmap_g_adv_loss + segmap_g_ce_loss + segmap_g_reg_loss + segmap_e_kl_loss + segmap_e_reg_loss
            self.segmap_g_loss = segmap_g_adv_loss + segmap_g_reg_loss + segmap_g_ce_loss + segmap_g_vgg_loss + 0*segmap_g_feature_loss
            #self.segmap_d_loss = segmap_d_adv_loss + segmap_d_reg_loss + segmap_e_kl_loss + segmap_e_reg_loss
            self.segmap_e_loss = segmap_g_adv_loss + segmap_g_ce_loss + segmap_g_vgg_loss + 0*segmap_g_feature_loss + segmap_e_kl_loss_adjusted + segmap_e_adv_loss + segmap_e_reg_loss
            self.segmap_de_loss = segmap_de_adv_loss + segmap_de_reg_loss
            self.segmap_d_loss = segmap_d_adv_loss + segmap_d_reg_loss - 0*segmap_g_feature_loss

        ###GAN###g_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_logit)
        ###GENIMG###g_kl_loss = self.kl_weight * kl_loss(x_mean, x_var)
        ###GENIMG###g_ce_loss = self.ce_weight * tf.nn.l2_loss(self.real_x - fake_x)
        ###GAN###g_vgg_loss = self.vgg_weight * VGGLoss()(self.real_x, fake_x)
        ###GAN###g_feature_loss = self.feature_weight * feature_loss(real_logit, fake_logit)
        ###GENIMG###g_reg_loss = regularization_loss('generator') + regularization_loss('encoder')

        ###GAN###d_adv_loss = self.adv_weight * (discriminator_loss(self.gan_type, real_logit, fake_logit) + GP)
        ###GAN###d_reg_loss = regularization_loss('discriminator')

        ###GAN###self.g_loss = g_adv_loss + g_kl_loss + g_vgg_loss + g_feature_loss + g_reg_loss
        ###GENIMG###self.g_loss = g_kl_loss + g_ce_loss + g_reg_loss
        ###GAN###self.d_loss = d_adv_loss + d_reg_loss

        """ Result Image """
        ###UNET###self.unet_x_segmap = tf.distributions.Categorical(logits=unet_x_logits).sample()
        #self.fake_x_segmap = tf.distributions.Categorical(logits=fake_x_segmap_logits).sample()
        self.fake_x_segmap = fake_x_segmap_logits
        ###GENIMG###self.fake_x = fake_x
        #self.random_fake_x_segmap = tf.distributions.Categorical(logits=random_fake_x_segmap_logits).sample()
        self.random_fake_x_segmap = random_fake_x_segmap_logits
        ###GENIMG###self.random_fake_x, _, _ = self.image_translate(segmap_img=tf.nn.softmax(random_fake_x_segmap_logits), random_style=True, reuse=True)

        """ Test """
        self.test_segmap_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, len(img_class.color_value_dict)])
        ###GENIMG###self.random_test_fake_x, _, _ = self.image_translate(segmap_img=self.test_segmap_image, random_style=True, reuse=True)

        self.test_guide_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch])
        ###GENIMG###self.guide_test_fake_x, _, _ = self.image_translate(segmap_img=self.test_segmap_image, x_img=self.test_guide_image, reuse=True)


        """ Training """
        t_vars = tf.trainable_variables()
        ###UNET###unet_G_vars = [var for var in t_vars if 'unet' in var.name]
        #segmap_G_vars = [var for var in t_vars if 'encoder_segmap_code' in var.name or 'generator_segmap' in var.name]
        segmap_G_vars = [var for var in t_vars if 'generator_segmap' in var.name]
        segmap_E_vars = [var for var in t_vars if 'encoder_segmap_code' in var.name in var.name]
        segmap_DE_vars = [var for var in t_vars if 'discriminator_segmap_code' in var.name]
        segmap_D_vars = [var for var in t_vars if 'discriminator_segmap' in var.name]
        ###GENIMG###G_vars = [var for var in t_vars if 'encoder' in var.name or 'generator' in var.name]
        ###GAN###D_vars = [var for var in t_vars if 'discriminator' in var.name]

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

        ###UNET###self.unet_G_optim = tf.train.AdamOptimizer(g_lr, beta1=beta1, beta2=beta2).minimize(self.unet_g_loss, var_list=unet_G_vars)
        self.segmap_G_optim = tf.train.AdamOptimizer(g_lr, beta1=beta1, beta2=beta2).minimize(self.segmap_g_loss, var_list=segmap_G_vars)
        ###GENIMG###self.G_optim = tf.train.AdamOptimizer(g_lr, beta1=beta1, beta2=beta2).minimize(self.g_loss, var_list=G_vars)
        ###UNET+GENSEGM+IMGVAE###self.G_optim = tf.train.AdamOptimizer(g_lr, beta1=beta1, beta2=beta2).minimize(self.unet_g_loss+self.segmap_g_loss+self.g_loss, var_list=unet_G_vars+segmap_G_vars+G_vars)
        self.segmap_E_optim = tf.train.AdamOptimizer(d_lr, beta1=beta1, beta2=beta2).minimize(self.segmap_e_loss, var_list=segmap_E_vars)
        self.segmap_DE_optim = tf.train.AdamOptimizer(d_lr, beta1=beta1, beta2=beta2).minimize(self.segmap_de_loss, var_list=segmap_DE_vars)
        self.segmap_D_optim = tf.train.AdamOptimizer(d_lr, beta1=beta1, beta2=beta2).minimize(self.segmap_d_loss, var_list=segmap_D_vars, global_step = self.global_step)
        ###GAN###self.D_optim = tf.train.AdamOptimizer(d_lr, beta1=beta1, beta2=beta2).minimize(self.d_loss, var_list=D_vars)

        """" Summary """
        self.summary_global_step = tf.summary.scalar("global_step", self.global_step)

        ###UNET###self.summary_unet_g_loss = tf.summary.scalar("unet_g_loss", self.unet_g_loss)
        self.summary_segmap_g_loss = tf.summary.scalar("segmap_g_loss", self.segmap_g_loss)
        self.summary_segmap_e_loss = tf.summary.scalar("segmap_e_loss", self.segmap_e_loss)
        self.summary_segmap_de_loss = tf.summary.scalar("segmap_de_loss", self.segmap_de_loss)
        self.summary_segmap_d_loss = tf.summary.scalar("segmap_d_loss", self.segmap_d_loss)
        ###GENIMG###self.summary_g_loss = tf.summary.scalar("g_loss", self.g_loss)
        ###GAN###self.summary_d_loss = tf.summary.scalar("d_loss", self.d_loss)

        ###UNET###self.summary_unet_g_kl_loss = tf.summary.scalar("unet_g_kl_loss", unet_g_kl_loss)
        ###UNET###self.summary_unet_g_ce_loss = tf.summary.scalar("unet_g_ce_loss", unet_g_ce_loss)

        self.summary_segmap_e_adv_loss = tf.summary.scalar("segmap_e_adv_loss", segmap_e_adv_loss)
        self.summary_segmap_e_kl_loss = tf.summary.scalar("segmap_e_kl_loss", segmap_e_kl_loss)
        self.summary_segmap_e_kl_loss_ema = tf.summary.scalar("segmap_e_kl_loss_ema", segmap_e_kl_loss_ema)
        self.summary_segmap_e_kl_loss_weight = tf.summary.scalar("segmap_e_kl_loss_weight", segmap_e_kl_loss_weight)
        self.summary_segmap_e_reg_loss = tf.summary.scalar("segmap_e_reg_loss", segmap_e_reg_loss)

        self.summary_segmap_g_ce_loss = tf.summary.scalar("segmap_g_ce_loss", segmap_g_ce_loss)
        self.summary_segmap_g_vgg_loss = tf.summary.scalar("segmap_g_vgg_loss", segmap_g_vgg_loss)
        self.summary_segmap_g_feature_loss = tf.summary.scalar("segmap_g_feature_loss", segmap_g_feature_loss)
        self.summary_segmap_g_reg_loss = tf.summary.scalar("segmap_g_reg_loss", segmap_g_reg_loss)

        self.summary_segmap_g_adv_loss = tf.summary.scalar("segmap_g_adv_loss", segmap_g_adv_loss)
        self.summary_segmap_d_adv_loss = tf.summary.scalar("segmap_d_adv_loss", segmap_d_adv_loss)
        self.summary_segmap_d_reg_loss = tf.summary.scalar("segmap_d_reg_loss", segmap_d_reg_loss)
        self.summary_segmap_de_adv_loss = tf.summary.scalar("segmap_de_adv_loss", segmap_de_adv_loss)
        self.summary_segmap_de_reg_loss = tf.summary.scalar("segmap_de_reg_loss", segmap_de_reg_loss)

        ###GAN###self.summary_g_adv_loss = tf.summary.scalar("g_adv_loss", g_adv_loss)
        ###GENIMG###self.summary_g_kl_loss = tf.summary.scalar("g_kl_loss", g_kl_loss)
        ###GENIMG###self.summary_g_ce_loss = tf.summary.scalar("g_ce_loss", g_ce_loss)
        ###GAN###self.summary_g_vgg_loss = tf.summary.scalar("g_vgg_loss", g_vgg_loss)
        ###GAN###self.summary_g_feature_loss = tf.summary.scalar("g_feature_loss", g_feature_loss)

        ###UNET###unet_g_summary_list = [self.summary_unet_g_loss, self.summary_unet_g_kl_loss, self.summary_unet_g_ce_loss]
        #segmap_g_summary_list = [self.summary_segmap_g_loss, self.summary_segmap_e_kl_loss, self.summary_segmap_g_ce_loss]
        #segmap_g_summary_list = [self.summary_segmap_g_loss, self.summary_segmap_g_adv_loss, self.summary_segmap_e_kl_loss]
        segmap_g_summary_list = [self.summary_segmap_g_loss, self.summary_segmap_g_adv_loss, self.summary_segmap_g_feature_loss, self.summary_segmap_g_ce_loss, self.summary_segmap_g_vgg_loss, self.summary_segmap_g_reg_loss]
        segmap_e_summary_list = [self.summary_segmap_e_loss, self.summary_segmap_e_adv_loss, self.summary_segmap_e_kl_loss, self.summary_segmap_e_kl_loss_ema, self.summary_segmap_e_kl_loss_weight, self.summary_segmap_e_reg_loss]
        segmap_d_summary_list = [self.summary_global_step, self.summary_segmap_d_loss, self.summary_segmap_d_adv_loss, self.summary_segmap_d_reg_loss] + segmap_real_summary + segmap_fake_summary
        segmap_de_summary_list = [self.summary_segmap_de_loss, self.summary_segmap_de_adv_loss, self.summary_segmap_de_reg_loss] + segmap_code_real_summary + segmap_code_fake_summary
        ###GENIMG###g_summary_list = [self.summary_g_loss, self.summary_g_kl_loss, self.summary_g_ce_loss]
        ###GAN###g_summary_list = [self.summary_g_loss, self.summary_g_adv_loss, self.summary_g_kl_loss, self.summary_g_vgg_loss, self.summary_g_feature_loss]
        ###GAN###d_summary_list = [self.summary_d_loss]

        ###UNET###self.unet_G_loss = tf.summary.merge(unet_g_summary_list)
        self.segmap_G_loss = tf.summary.merge(segmap_g_summary_list)
        self.segmap_E_loss = tf.summary.merge(segmap_e_summary_list)
        self.segmap_D_loss = tf.summary.merge(segmap_d_summary_list)
        self.segmap_DE_loss = tf.summary.merge(segmap_de_summary_list)
        ###GENIMG###self.G_loss = tf.summary.merge(g_summary_list)
        ###GAN###self.D_loss = tf.summary.merge(d_summary_list)

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
        past_segmap_g_loss = -1.
        past_segmap_e_loss = -1.
        ###GENIMG###past_g_loss = -1.
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
                _, segmap_d_loss, segmap_d_summary_str = self.sess.run([self.segmap_D_optim, self.segmap_d_loss, self.segmap_D_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(segmap_d_summary_str, counter)

                ###GAN###_, d_loss, summary_str = self.sess.run([self.D_optim, self.d_loss, self.D_loss], feed_dict=train_feed_dict)
                ###GAN###self.writer.add_summary(summary_str, counter)
                
                segmap_e_loss = None
                segmap_g_loss = None
                
                if (counter - 1) % self.n_critic == 0:
                    # Update DE
                    _, segmap_de_loss, segmap_de_summary_str = self.sess.run([self.segmap_DE_optim, self.segmap_de_loss, self.segmap_DE_loss], feed_dict=train_feed_dict)
                    self.writer.add_summary(segmap_de_summary_str, counter)

                    if (counter - 1) % (self.n_critic*self.code_n_critic) == 0:
                        # Update E
                        _, segmap_e_loss, segmap_e_summary_str = self.sess.run([self.segmap_E_optim, self.segmap_e_loss, self.segmap_E_loss], feed_dict=train_feed_dict)
                        self.writer.add_summary(segmap_e_summary_str, counter)
                        past_segmap_e_loss = segmap_e_loss

                        # Update G
                        ###UNET###unet_g_loss = None
                        ###GENIMG###g_loss = None
                        ###UNET###real_x_images, real_x_segmap, unet_x_segmap, _, unet_g_loss, unet_summary_str = self.sess.run(
                        ###UNET###    [self.real_x, self.real_x_segmap, self.unet_x_segmap,
                        ###UNET###     self.unet_G_optim,
                        ###UNET###     self.unet_g_loss, self.unet_G_loss], feed_dict=train_feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

                        real_x_images, real_x_segmap, fake_x_segmap, random_fake_x_segmap, _, segmap_g_loss, segmap_summary_str = self.sess.run(
                            [self.real_x, self.real_x_segmap, self.fake_x_segmap, self.random_fake_x_segmap,
                             self.segmap_G_optim,
                             self.segmap_g_loss, self.segmap_G_loss], feed_dict=train_feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

                        ###UNET+GENSEGM###real_x_images, real_x_segmap, unet_x_segmap, fake_x_segmap, random_fake_x_segmap, _, unet_g_loss, unet_summary_str, _, segmap_g_loss, segmap_summary_str = self.sess.run(
                        ###UNET+GENSEGM###    [self.real_x, self.real_x_segmap, self.unet_x_segmap, self.fake_x_segmap, self.random_fake_x_segmap,
                        ###UNET+GENSEGM###     self.unet_G_optim,
                        ###UNET+GENSEGM###     self.unet_g_loss, self.unet_G_loss,
                        ###UNET+GENSEGM###     self.segmap_G_optim,
                        ###UNET+GENSEGM###     self.segmap_g_loss, self.segmap_G_loss], feed_dict=train_feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

                        ###GAN###real_x_images, real_x_segmap, fake_x_images, fake_x_segmap, random_fake_x_images, random_fake_x_segmap, _, g_loss, summary_str = self.sess.run(
                        ###GAN###    [self.real_x, self.real_x_segmap, self.fake_x, self.fake_x_segmap, self.random_fake_x, self.random_fake_x_segmap,
                        ###GAN###     self.G_optim,
                        ###GAN###     self.g_loss, self.G_loss], feed_dict=train_feed_dict)

                        ###GENIMG###real_x_images, real_x_segmap, fake_x_images, fake_x_segmap, random_fake_x_images, random_fake_x_segmap, _, segmap_g_loss, segmap_summary_str, _, g_loss, summary_str = self.sess.run(
                        ###GENIMG###    [self.real_x, self.real_x_segmap, self.fake_x, self.fake_x_segmap, self.random_fake_x, self.random_fake_x_segmap,
                        ###GENIMG###     self.segmap_G_optim,
                        ###GENIMG###     self.segmap_g_loss, self.segmap_G_loss,
                        ###GENIMG###     self.G_optim,
                        ###GENIMG###     self.g_loss, self.G_loss], feed_dict=train_feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

                        ###UNET+GENSEGM+IMGVAE###real_x_images, real_x_segmap, fake_x_images, unet_x_segmap, fake_x_segmap, random_fake_x_images, random_fake_x_segmap, _, unet_g_loss, unet_summary_str, segmap_g_loss, segmap_summary_str, g_loss, summary_str = self.sess.run(
                        ###UNET+GENSEGM+IMGVAE###    [self.real_x, self.real_x_segmap, self.fake_x, self.unet_x_segmap, self.fake_x_segmap, self.random_fake_x, self.random_fake_x_segmap,
                        ###UNET+GENSEGM+IMGVAE###     self.G_optim,
                        ###UNET+GENSEGM+IMGVAE###     #self.unet_G_optim,
                        ###UNET+GENSEGM+IMGVAE###     self.unet_g_loss, self.unet_G_loss,
                        ###UNET+GENSEGM+IMGVAE###     #self.segmap_G_optim,
                        ###UNET+GENSEGM+IMGVAE###     self.segmap_g_loss, self.segmap_G_loss,
                        ###UNET+GENSEGM+IMGVAE###     #self.G_optim,
                        ###UNET+GENSEGM+IMGVAE###     self.g_loss, self.G_loss], feed_dict=train_feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
     
                        ###UNET###self.writer.add_summary(unet_summary_str, counter)
                        self.writer.add_summary(segmap_summary_str, counter)
                        ###GENIMG###self.writer.add_summary(summary_str, counter)
                        ###UNET###past_unet_g_loss = unet_g_loss
                        past_segmap_g_loss = segmap_g_loss
                        ###GENIMG###past_g_loss = g_loss

                # display training status
                counter += 1
                ###UNET###if unet_g_loss == None:
                ###UNET###    unet_g_loss = past_unet_g_loss
                ###UNET###print("Epoch: [%2d] [%5d/%5d] time: %4.4f unet_g_loss: %.8f" % (
                ###UNET###    epoch, idx, self.iteration, time.time() - start_time, unet_g_loss))
                if segmap_e_loss == None:
                    segmap_e_loss = past_segmap_e_loss
                if segmap_g_loss == None:
                    segmap_g_loss = past_segmap_g_loss
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f segmap_g_loss: %.8f" % (
                    epoch, idx, self.iteration, time.time() - start_time, segmap_g_loss))
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f segmap_e_loss: %.8f" % (
                    epoch, idx, self.iteration, time.time() - start_time, segmap_e_loss))
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f segmap_de_loss: %.8f" % (
                    epoch, idx, self.iteration, time.time() - start_time, segmap_de_loss))
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f segmap_d_loss: %.8f" % (
                    epoch, idx, self.iteration, time.time() - start_time, segmap_d_loss))
                ###GENIMG###if g_loss == None:
                ###GENIMG###    g_loss = past_g_loss
                ###GENIMG###print("Epoch: [%2d] [%5d/%5d] time: %4.4f g_loss: %.8f" % (
                ###GENIMG###    epoch, idx, self.iteration, time.time() - start_time, g_loss))
                ###GAN###print("Epoch: [%2d] [%5d/%5d] time: %4.4f d_loss: %.8f" % (
                ###GAN###    epoch, idx, self.iteration, time.time() - start_time, d_loss))
                sys.stdout.flush()

                if np.mod(idx + 1, self.print_freq) == 0:

                    save_images(real_x_images, [self.batch_size, 1],
                               './{}/image_real_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    save_images(real_x_segmap, [self.batch_size, 1],
                                './{}/segmap_real_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))

                    ###GENIMG###save_images(fake_x_images, [self.batch_size, 1],
                    ###GENIMG###            './{}/image_fake_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    ###UNET###save_segmaps(unet_x_segmap, self.color_value_dict, [self.batch_size, 1],
                    ###UNET###            './{}/segmap_unet_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    #save_segmaps(fake_x_segmap, self.color_value_dict, [self.batch_size, 1],
                    save_images(fake_x_segmap, [self.batch_size, 1],
                                './{}/segmap_fake_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    ###GENIMG###save_images(random_fake_x_images, [self.batch_size, 1],
                    ###GENIMG###            './{}/random_image_fake_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))

                    #save_segmaps(random_fake_x_segmap, self.color_value_dict, [self.batch_size, 1],
                    save_images(random_fake_x_segmap, [self.batch_size, 1],
                                './{}/random_segmap_fake_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))

                if np.mod(counter - 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):

        n_dis = str(self.n_scale) + 'multi_' + str(self.n_dis) + 'dis'


        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        if self.TTUR :
            TTUR = '_TTUR'
        else :
            TTUR = ''


        return "{}_dataset={}__gan{}__n_dis={}__n_critic={}__adv_weight={}__vgg_weight={}__feature_weight={}__kl_weight={}__{}{}__segmap_ch={}__segmap_num_upsampling_layers={}__ch={}__num_upsampling_layers={}".format(self.model_name, self.dataset_name,
                                                                   self.gan_type, n_dis, self.n_critic,
                                                                   self.adv_weight, self.vgg_weight, self.feature_weight,
                                                                   self.kl_weight,
                                                                   sn, TTUR, self.segmap_ch, self.num_upsampling_layers, self.ch, self.segmap_num_upsampling_layers)

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
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def random_test(self):
        tf.global_variables_initializer().run()

        segmap_files = glob('./dataset/{}/{}/*.*'.format(self.dataset_name, 'segmap_test'))

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

        for sample_file in tqdm(segmap_files) :
            sample_image = load_segmap(self.dataset_path, sample_file, self.img_width, self.img_height, self.segmap_img_ch)
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_style) :
                image_path = os.path.join(self.result_dir, '{}_style{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.random_test_fake_x, feed_dict={self.test_segmap_image : sample_image})
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

        segmap_files = glob('./dataset/{}/{}/*.*'.format(self.dataset_name, 'segmap_test'))

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

        for sample_file in tqdm(segmap_files):
            sample_image = load_segmap(self.dataset_path, sample_file, self.img_width, self.img_height, self.segmap_img_ch)
            image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.guide_test_fake_x, feed_dict={self.test_segmap_image : sample_image, self.test_guide_image : style_image})
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
