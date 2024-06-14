import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

class DCGAN:
    def __init__(self, file_name, batch_size=64, noise_dimension=100, 
                 optimizer_lr=0.0002, optimizer_betas=(0.5, 0.999), weight_init_stddev=0.02, 
                 print_stats_after_batch=500, num_epochs=1):
        
        self.file_name = file_name
        self.batch_size = batch_size
        self.noise_dimension = noise_dimension
        self.optimizer_lr = optimizer_lr
        self.optimizer_betas = optimizer_betas
        self.weight_init_stddev = weight_init_stddev
        self.print_stats_after_batch = print_stats_after_batch
        self.num_epochs = num_epochs

        # Initialize loss function, init schema, and optimizers
        self.cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.weight_init = tf.keras.initializers.RandomNormal(stddev=self.weight_init_stddev)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.optimizer_lr, beta_1=self.optimizer_betas[0], beta_2=self.optimizer_betas[1])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.optimizer_lr, beta_1=self.optimizer_betas[0], beta_2=self.optimizer_betas[1])

    def make_directory_for_run(self):
        """ Make a directory for this training run. """
        print(f'Preparing training run {self.file_name}')
        if not os.path.exists('./runs'):
            os.mkdir('./runs')
        if not os.path.exists(f'./runs/{self.file_name}'):
            os.mkdir(f'./runs/{self.file_name}')
        if not os.path.exists(f'./runs/{self.file_name}/images'):
            os.mkdir(f'./runs/{self.file_name}/images')
        if not os.path.exists(f'./runs/{self.file_name}/generator'):
            os.mkdir(f'./runs/{self.file_name}/generator')
        if not os.path.exists(f'./runs/{self.file_name}/discriminator'):
            os.mkdir(f'./runs/{self.file_name}/discriminator')
    
    def load_data_mnist(self):
        """ Load mnist data """
        (images, _), (_, _) = tf.keras.datasets.mnist.load_data()
        images = images.reshape(images.shape[0], 28, 28, 1)
        images = images.astype('float32')
        images = (images - 127.5) / 127.5
        return tf.data.Dataset.from_tensor_slices(images).batch(self.batch_size)

    def create_generator(self):
        """ Create Generator """
        generator = tf.keras.Sequential()
        # Input block
        generator.add(layers.Dense(7*7*256, use_bias=False, input_shape=(self.noise_dimension,), kernel_initializer=self.weight_init))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        # Reshape 1D Tensor into 3D
        generator.add(layers.Reshape((7, 7, 256)))
        # First upsampling block
        generator.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.weight_init))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        # Second upsampling block
        generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=self.weight_init))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        # Third upsampling block: note tanh, specific for DCGAN
        generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh', kernel_initializer=self.weight_init))
        return generator

    def create_discriminator(self):
        """ Create Discriminator """
        discriminator = tf.keras.Sequential()
        # First Convolutional block
        discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1], kernel_initializer=self.weight_init))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.3))
        # Second Convolutional block
        discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=self.weight_init))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.3))
        # Flatten and generate output prediction
        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(1, kernel_initializer=self.weight_init, activation='sigmoid'))
        return discriminator

    def compute_generator_loss(self, predicted_fake):
        """ Compute cross entropy loss for the generator """
        return self.cross_entropy_loss(tf.ones_like(predicted_fake), predicted_fake)

    def compute_discriminator_loss(self, predicted_real, predicted_fake):
        """ Compute discriminator loss """
        loss_on_reals = self.cross_entropy_loss(tf.ones_like(predicted_real), predicted_real)
        loss_on_fakes = self.cross_entropy_loss(tf.zeros_like(predicted_fake), predicted_fake)
        return loss_on_reals + loss_on_fakes

    def generate_noise(self, number_of_images=1):
        """ Generate noise for number_of_images images, with a specific noise_dimension """
        return tf.random.normal([number_of_images, self.noise_dimension])

    def generate_image(self, generator, epoch=0, batch=0):
        """ Generate subplots with generated examples. """
        noise = self.generate_noise(self.batch_size)
        images = generator(noise, training=False)
        plt.figure(figsize=(10, 10))
        for i in range(16):
            # Get image and reshape
            image = images[i]
            image = np.reshape(image, (28, 28))
            # Plot
            plt.subplot(4, 4, i + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.savefig(f'./runs/{self.file_name}/images/epoch{epoch}_batch{batch}.jpg')

    def save_models(self, generator, discriminator, epoch):
        """ Save models at specific point in time. """
        tf.keras.models.save_model(generator, f'./runs/{self.file_name}/generator/generator_{epoch}.model', overwrite=True, include_optimizer=True)
        tf.keras.models.save_model(discriminator, f'./runs/{self.file_name}/discriminator/discriminator_{epoch}.model', overwrite=True, include_optimizer=True)

    def print_training_progress(self, batch, generator_loss, discriminator_loss):
        """ Print training progress. """
        print('Losses after mini-batch %5d: generator %e, discriminator %e' % (batch, generator_loss, discriminator_loss))

    @tf.function
    def perform_train_step(self, real_images, generator, discriminator):
        """ Perform one training step with Gradient Tapes """
        noise = self.generate_noise(self.batch_size)
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            generated_images = generator(noise, training=True)
            discriminated_generated_images = discriminator(generated_images, training=True)
            discriminated_real_images = discriminator(real_images, training=True)
            generator_loss = self.compute_generator_loss(discriminated_generated_images)
            discriminator_loss = self.compute_discriminator_loss(discriminated_real_images, discriminated_generated_images)
            
        generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        return generator_loss, discriminator_loss

    def train_gan(self, data):
        """ Train the GAN """
        generator = self.create_generator()
        discriminator = self.create_discriminator()
        for epoch_no in range(self.num_epochs):
            print(f'Starting epoch {epoch_no + 1}...')
            batch_no = 0
            for batch in data:
                generator_loss, discriminator_loss = self.perform_train_step(batch, generator, discriminator)
                batch_no += 1
                if batch_no % self.print_stats_after_batch == 0:
                    self.print_training_progress(batch_no, generator_loss, discriminator_loss)
                    self.generate_image(generator, epoch_no, batch_no)
            self.save_models(generator, discriminator, epoch_no)
        print(f'Finished unique run {self.file_name}')

    def run(self):
        # Make directory for the run
        self.make_directory_for_run()

        tf.random.set_seed(42)

        data = self.load_data_mnist()
        print('Training GAN ...')
        self.train_gan(data)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
    dcgan = DCGAN(file_name='mnist', num_epochs=100)
    dcgan.run()
