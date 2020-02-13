#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

from model import Model


class ImagenetModel(Model):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.batch_size = 128 
        self.epochs     = 200

        self.weight_decay = 0.0001

        Model.__init__(self, args)

    def dataset(self):
        """
        TODO: Write Comment
        """

        import os

        import tensorflow as tf

        def parse_record(raw_record, is_training, exclusion=None):
            """
            TODO: Write Comment
            """

            feature_map = {
                    'image/encoded':     tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
                    'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                    'image/class/text':  tf.io.FixedLenFeature([], dtype=tf.string,  default_value=''),
                    'image/filename':    tf.io.FixedLenFeature([], dtype=tf.string, default_value='')
            }

            sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
            feature_map.update({k: sparse_float32 for k in ['image/object/bbox/xmin', 'image/object/bbox/ymin', 'image/object/bbox/xmax', 'image/object/bbox/ymax']})

            features = tf.io.parse_single_example(serialized=raw_record, features=feature_map)
            bbox     = tf.transpose(a=tf.expand_dims(tf.concat([ tf.expand_dims(features['image/object/bbox/ymin'].values, 0), tf.expand_dims(features['image/object/bbox/xmin'].values, 0), tf.expand_dims(features['image/object/bbox/ymax'].values, 0), tf.expand_dims(features['image/object/bbox/xmax'].values, 0)], 0), 0), perm=[0, 2, 1])
            
            if is_training:

                bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
                    tf.image.extract_jpeg_shape(features['image/encoded']),
                    bounding_boxes=bbox,
                    min_object_covered=0.1,
                    aspect_ratio_range=[0.75, 1.33],
                    area_range=[0.05, 1.0],
                    max_attempts=100,
                    use_image_if_no_bounding_boxes=True)

                offset_y, offset_x, _          = tf.unstack(bbox_begin)
                target_height, target_width, _ = tf.unstack(bbox_size)

                image = tf.compat.v1.image.resize(tf.image.random_flip_left_right(tf.image.decode_and_crop_jpeg(features['image/encoded'], tf.stack([offset_y, offset_x, target_height, target_width]), channels=self.img_channels)), [self.img_rows, self.img_cols], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

            else:

                image = tf.image.decode_jpeg(features['image/encoded'], channels=self.img_channels)

                shape = tf.shape(input=image)
                height, width = shape[0], shape[1]

                height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
                scale_ratio   = tf.cast(256, tf.float32) / tf.minimum(height, width)

                image = tf.compat.v1.image.resize(image, [tf.cast(height * scale_ratio, tf.int32), tf.cast(width * scale_ratio, tf.int32)], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

                shape = tf.shape(input=image)
                height, width = shape[0], shape[1]

                image = tf.slice(image, [(height - self.img_rows) // 2, (width - self.img_cols) // 2, 0], [self.img_rows, self.img_cols, -1])
            
            image.set_shape([self.img_rows, self.img_cols, self.img_channels])

            label = tf.cast(features['image/class/label'], dtype=tf.int32) 
            
            if self.args.use_dataset == 0: label = label-1 

            return tf.cast(features['image/filename'], dtype=tf.string), tf.cast(image, dtype=tf.float32), tf.cast(label,dtype=tf.int64)

        def parser(filename, image, label): 
            """
            TODO: Write Comment
            """

            return image - tf.broadcast_to(self.mean, tf.shape(image)), tf.one_hot(label, self.num_classes)
        
        def create_dataset(is_training, exclusion=None):
            """
            TODO: Write Comment
            """

            options = tf.data.Options()
            options.experimental_threading.max_intra_op_parallelism = 1
            
            if is_training:
                filenames = [os.path.join(f"./{self.data_dir}/{self.dataset_name}_img_train/",             'train-%05d-of-01024' % i) for i in range(1024)]
            else:
                filenames = [os.path.join(f"./{self.data_dir}/{self.dataset_name.split('_')[0]}_img_val/", 'val-%05d-of-00128' % i)   for i in range(128)]
        
            raw_dataset = tf.data.Dataset.from_tensor_slices(filenames)
            raw_dataset = raw_dataset.shuffle(buffer_size= 1024 if is_training else 128)
            raw_dataset = raw_dataset.interleave(tf.data.TFRecordDataset, cycle_length= 12, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if is_training:
                raw_dataset = raw_dataset.shuffle(buffer_size=10000)
                raw_dataset = raw_dataset.repeat(count=self.epochs)
            
            raw_dataset = raw_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            raw_dataset = raw_dataset.map(lambda value: parse_record(value, is_training, exclusion), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            raw_dataset = raw_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                
            raw_dataset = raw_dataset.with_options(options)

            processed_dataset = raw_dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            processed_dataset = processed_dataset.batch(self.batch_size)
            # if not is_training: processed_dataset  = processed_dataset.cache()
            processed_dataset = processed_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            return raw_dataset, processed_dataset

        self.img_rows, self.img_cols, self.img_channels = self.size, self.size, 3
        self.data_dir = 'imagenet_files'

        self.mean = [123.68, 116.78, 103.94]
        self.std  = [1., 1., 1.]

        if self.args.use_dataset == 0:     
            
            self.dataset_name = 'Imagenet'
            self.num_images   = {'train': 1281167, 'test': 50000}
            self.num_classes  = 1000
            with open(f"./{self.data_dir}/labels-imagenet.txt") as f: self.class_names = f.read().splitlines()

        elif self.args.use_dataset == 1:     

            self.dataset_name = 'RestrictedImagenet'
            self.num_images   = {'train': 129359, 'test': 5000}
            self.num_classes  = 10
            self.class_names  = ['Automobile', 'Ball', 'Bird', 'Dog', 'Feline', 'Fruit', 'Insect', 'Snake', 'Primate', 'Vegetable']

        self.raw_train_dataset, self.processed_train_dataset = create_dataset(True)
        self.raw_test_dataset,  self.processed_test_dataset  = create_dataset(False)

        self.iterations_train = (self.num_images['train'] // self.batch_size) + 1   
        self.iterations_test  = (self.num_images['test']  // self.batch_size) + 1   

    def build_model(self):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, models, optimizers, regularizers, utils

        base_model = self.model_class(weights=None, include_top=False, input_shape=(self.img_rows, self.img_cols, self.img_channels))
        x = base_model.output

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.num_classes, name='Output', activation='softmax', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        
        final_model = models.Model(inputs=base_model.input, outputs=x)
        for layer in final_model.layers[:]: layer.trainable = True

        return final_model