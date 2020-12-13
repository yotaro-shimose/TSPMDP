# GPU setting module.
# Tensorflow allocates all of memory in all visible devices which results in Out Of Memory
# when secondly loaded process which don't share its memory with other processes
# tries to allocate gpu memory.
import tensorflow as tf

# configure tf to allocate memory dynamically.
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
