import numpy as np
import matplotlib.pyplot as plt

# Assuming the weights are ordered such that each neuron's weights come sequentially,
# and the last weight for each neuron is the bias for that neuron.
# The given network has 3 input neurons, 4 neurons in each hidden layer, and 1 output neuron.

# Given weights and biases (assuming the last in each segment is a bias)
params = [
    0.751422278895, 0.992276309544, -0.370254443802, -0.41576721662,
    -0.270355879413, 0.95756719395, -0.334038998793, 0.969248052327,
    -0.366431430154, -0.239892987989, 0.865229332423, 0.954567012953,
    0.885822632415, -0.333025273147, 0.93616865497, -0.330658671224,
    -0.252887616715, 0.853033055712, -0.220978853104, 0.968214216579, 0.903943885408,
    0.825726986182, -0.308093341405, 0.316282346418, -0.940841996916, -0.452774706779,
    -0.636337264886, 0.671651747534, -0.898467136639, -0.771418594937, 0.452763913498,
    -0.386062284379, 0.694204146805, -0.316922982361, 0.880958069115, 0.832613181878,
    -0.528983509507, 0.907268250811, 1.3867600867, -0.464189662002, 0.737743255704
]

# Reshape parameters into weight matrices and bias vectors for each layer
# Input to hidden layer 1
W1 = np.array(params[0:12]).reshape(4, 3)
b1 = np.array(params[12:16]).reshape(4, 1)

# Hidden layer 1 to hidden layer 2
W2 = np.array(params[16:32]).reshape(4, 4)
b2 = np.array(params[32:36]).reshape(4, 1)

# Hidden layer 2 to output layer
W3 = np.array(params[36:40]).reshape(1, 4)
b3 = np.array(params[40]).reshape(1, 1)

# Activation function
def tanh(x):
    return np.tanh(x)


# Adjust the forward pass to properly handle the input for 3D plotting
def forward_pass_3d(X, Y):
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            input_vector = np.array([[X[i, j]], [Y[i, j]], [0.0]])  # Third input fixed at 0
            H1 = tanh(np.dot(W1, input_vector) + b1)
            H2 = tanh(np.dot(W2, H1) + b2)
            Z[i, j] = tanh(np.dot(W3, H2) + b3)
    return Z

input_range = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(input_range, input_range)
Z = np.zeros(X.shape)

# Use the function to compute the output Z for the 3D plot
Z = forward_pass_3d(X, Y)

# Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Set labels
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')

# Title and color bar
plt.title('3D Visualization of the Learned Function (Fixed input 3)')


# Show plot
plt.show()
