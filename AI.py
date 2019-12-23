from tensorflow_core.python.keras import models, layers, optimizers, losses, backend


inputs = [
    layers.Input(shape=(1000, 1000, 3)), # Velocity Field
    layers.Input(shape=(1000, 1000, 3)), # External Acceleration
    layers.Input(shape=(1000, 1000, 1)), # Pressure Field
    layers.Input(shape=(1000, 1000, 1)), # Mass Density
]

# Mathematically-Based Model
mass_density_grad = layers.Conv2D(3, 3, padding="same")(inputs[3])
neg_pressure_field_grad = layers.Conv2D(3, 3, padding="same")(inputs[2])
mu_laplacian_velocity = layers.Conv2D(3, 3, padding="same")(inputs[0])
neg_vel_dot_grad_vel = layers.Conv2D(3, 3, padding="same")(inputs[0])
inv_mass_density = layers.Conv2D(3, 3, padding="same")(inputs[3])
mu_third_grad_div_velocity = layers.Conv2D(3, 3, padding="same")(layers.Conv2D(1, 3, padding="same")(inputs[0]))
output = layers.Add()([inputs[0], layers.Conv2D(3, 3, padding="same")(layers.Add()([
  layers.Multiply()([
      layers.Add()([
          neg_pressure_field_grad,
          mu_laplacian_velocity,
          mu_third_grad_div_velocity,
          layers.Multiply()([inputs[1], layers.Concatenate()([inputs[3], inputs[3], inputs[3]])])
      ]), 
      inv_mass_density
  ]), 
  neg_vel_dot_grad_vel]))
])

mathematical = models.Model(inputs = inputs, outputs = [output])

mathematical.summary()

# Residual Block Adder

def residual_block(layer, output_channels):
    conv1 = layers.LeakyReLU(alpha=0.4)(layers.Conv2D(output_channels, 1)(layer))
    bn1 = layers.BatchNormalization()(conv1)
    conv2 = layers.LeakyReLU(alpha=0.4)(layers.Conv2D(output_channels, 3, padding="same")(bn1))
    bn2 = layers.BatchNormalization()(conv2)
    conv3 = layers.LeakyReLU(alpha=0.4)(layers.Conv2D(output_channels, 3, padding="same")(bn2))
    bn3 = layers.BatchNormalization()(conv3)
    conv4 = layers.LeakyReLU(alpha=0.4)(layers.Conv2D(output_channels, 3, padding="same")(bn3))
    bn4 = layers.BatchNormalization()(conv4)
    lconv1 = layers.LeakyReLU(alpha=0.4)(layers.Conv2D(output_channels, 1)(layer))
    lbn1 = layers.BatchNormalization()(lconv1)
    add = layers.Add()([lbn1, bn4])
    small = layers.AveragePooling2D()(add)
    return small

def residual_block_big(layer, output_channels):
    conv1 = layers.LeakyReLU(alpha=0.4)(layers.Conv2DTranspose(output_channels, 1)(layer))
    bn1 = layers.BatchNormalization()(conv1)
    conv2 = layers.LeakyReLU(alpha=0.4)(layers.Conv2DTranspose(output_channels, 3, padding="same")(bn1))
    bn2 = layers.BatchNormalization()(conv2)
    conv3 = layers.LeakyReLU(alpha=0.4)(layers.Conv2DTranspose(output_channels, 3, padding="same")(bn2))
    bn3 = layers.BatchNormalization()(conv3)
    conv4 = layers.LeakyReLU(alpha=0.4)(layers.Conv2DTranspose(output_channels, 3, padding="same")(bn3))
    bn4 = layers.BatchNormalization()(conv4)
    lconv1 = layers.LeakyReLU(alpha=0.4)(layers.Conv2DTranspose(output_channels, 1)(layer))
    lbn1 = layers.BatchNormalization()(lconv1)
    add = layers.Add()([lbn1, bn4])
    big = layers.UpSampling2D()(add)
    return big
# Autoencoder-Based Model
LATENT_SIZE = 2048

data = layers.Input(shape=(1000, 1000, 8))

compress1 = layers.LeakyReLU(alpha=0.4)(layers.Conv2D(32, 3, padding="same")(data))
compress2 = residual_block(compress1, 64)
compress3 = residual_block(compress2, 128)
compress4 = residual_block(compress3, 256)
compress5 = layers.LeakyReLU(alpha=0.4)(layers.Conv2D(256, 6)(compress4))
compress6 = residual_block(compress5, 512)
compress7 = residual_block(compress6, 256)
compress8 = residual_block(compress7, 128)
compress9 = layers.LeakyReLU(alpha=0.4)(layers.Conv2D(64, 2)(compress8))
compress10 = residual_block(compress9, 32)
flat = layers.Flatten()(compress10)
flat2 = layers.Dense(LATENT_SIZE)(flat)


encoder = models.Model(inputs = data, outputs = flat2)
encoder.summary()

latent_layer = layers.Input(shape=(2048,))

flat = layers.Dense(1568)(latent_layer)
twod1 = layers.Reshape((7, 7, 32))(flat)
decompress1 = residual_block_big(twod1, 32)
decompress2 = layers.LeakyReLU(alpha=0.4)(layers.Conv2DTranspose(32, 2)(decompress1))
decompress3 = residual_block_big(decompress2, 64)
decompress4 = residual_block_big(decompress3, 128)
decompress5 = residual_block_big(decompress4, 256)
decompress6 = layers.LeakyReLU(alpha=0.4)(layers.Conv2DTranspose(256, 6)(decompress5))
decompress7 = residual_block_big(decompress6, 128)
decompress8 = residual_block_big(decompress7, 64)
decompress9 = residual_block_big(decompress8, 32)
decompress10 = layers.LeakyReLU(alpha=0.4)(layers.Conv2DTranspose(8, 1)(decompress9))

decoder = models.Model(inputs = latent_layer, outputs = decompress10)
decoder.summary()

full_model = models.Model(inputs = data, outputs = decoder(encoder(data)))
full_model.summary()