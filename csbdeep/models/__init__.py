from __future__ import absolute_import, print_function
import warnings

# checks
try:
    import tensorflow
    del tensorflow
except ModuleNotFoundError as e:
    from six import raise_from
    raise_from(RuntimeError('Please install TensorFlow: https://www.tensorflow.org/install/'), e)


# for now, tensorflow >= 2.0 is not supported
import tensorflow as tf
from distutils.version import LooseVersion
_tf_version = LooseVersion(tf.__version__)
# print(_tf_version)
if  _tf_version >= LooseVersion("2.0.0"):
    tf.compat.v1.disable_v2_behavior()
    warnings.warn("csbdeep only supports tensorflow 1 behavior for now. We disable v2 behavior (installed tensorflow version: %s)" % _tf_version)
del tf



try:
    from tensorflow import keras
    del keras
except ModuleNotFoundError as e:
    if e.name in {'theano','cntk'}:
        from six import raise_from
        raise_from(RuntimeError(
            "Keras is configured to use the '%s' backend, which is not installed. "
            "Please change it to use 'tensorflow' instead: "
            "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % e.name
        ), e)
    else:
        raise e

from tensorflow.keras import backend as K
if K.backend() != 'tensorflow':
    raise NotImplementedError(
            "Keras is configured to use the '%s' backend, which is currently not supported. "
            "Please configure Keras to use 'tensorflow' instead: "
            "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.backend()
        )
if K.image_data_format() != 'channels_last':
    raise NotImplementedError(
        "Keras is configured to use the '%s' image data format, which is currently not supported. "
        "Please change it to use 'channels_last' instead: "
        "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.image_data_format()
    )
del K


# imports
from .config import BaseConfig, Config
from .base_model import BaseModel
from .care_standard import CARE
from .care_upsampling import UpsamplingCARE
from .care_isotropic import IsotropicCARE
from .care_projection import ProjectionConfig, ProjectionCARE
