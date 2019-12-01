from tensorflow.keras import models, layers, optimizers, losses, backend

inputs = [
    layers.Input(shape=(1000, 1000, 3)), # Velocity Field
    layers.Input(shape=(1000, 1000, 3)), # External Acceleration
    layers.Input(shape=(1000, 1000, 1)), # Pressure Field
    layers.Input(shape=(1000, 1000, 1)), # Mass Density
]

mass_density_grad = layers.Conv2D(3, 3, padding="same")(inputs[3])
neg_pressure_field_grad = layers.Conv2D(3, 3, padding="same")(inputs[2])
mu_laplacian_velocity = layers.Conv2D(3, 3, padding="same")(inputs[0])
neg_vel_dot_grad_vel = layers.Conv2D(3, 3, padding="same")(inputs[0])
inv_mass_density = layers.Conv2D(3, 3, padding="same")(inputs[3])
mu_third_grad_div_velocity = layers.Conv2D(3, 3, padding="same")(layers.Conv2D(1, 3, padding="same")(inputs[0]))
output = layers.Add()([inputs[0], layers.Conv2D(3, 3, padding="same")(layers.Add()([layers.Multiply()([layers.Add()([neg_pressure_field_grad, mu_laplacian_velocity, mu_third_grad_div_velocity, layers.Multiply()([inputs[1], layers.Concatenate()([inputs[3], inputs[3], inputs[3]])])]), inv_mass_density]), neg_vel_dot_grad_vel]))])

model = models.Model(inputs = inputs, outputs = [output])

model.summary()