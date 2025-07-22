import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Could not set memory growth: {e}")
else:
    print("No GPU detected. Running on CPU.")

# Run a simple computation to test GPU usage
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("Result of matrix multiplication:")
    print(c.numpy())
    print(f"Operation executed on: {'GPU' if gpus else 'CPU'}")
