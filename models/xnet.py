import torch
import torch.nn as nn

def basic_block(in_ch, out_ch, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, (kernel_size, kernel_size), padding="same"),
        nn.BatchNorm2d(),
        nn.ReLU(),
    )



def model(input_shape=(64, 64, 3), classes=3, kernel_size=3, filter_depth=(64, 128, 256, 512, 0)):
    x = torch.zeros(1, 1, 100, 100)

    # Encoder
    bl1 = basic_block(1, filter_depth[0])
    max_pool = nn.MaxPool2d(kernel_size=2)

    # 100x100

    bl2 = basic_block(filter_depth[0], filter_depth[1])
    # 50x50

    bl3 = basic_block(filter_depth[1], filter_depth[2])
    # 25x25

    # Flat
    conv4 = nn.Conv2d(filter_depth[2], filter_depth[3], (kernel_size, kernel_size), padding="same")
    batch4 = nn.BatchNorm2d()
    act4 = nn.ReLU()
    # 25x25

    conv5 = nn.Conv2d(filter_depth[3], filter_depth[3], (kernel_size, kernel_size), padding="same")
    batch5 = nn.BatchNorm2d()
    act5 = nn.ReLU()
    # 25x25

    # Up
    up6 = UpSampling2D(size=(2, 2))(act5)
    conv6 = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(up6)
    batch6 = BatchNormalization()(conv6)
    act6 = Activation("relu")(batch6)
    concat6 = Concatenate()([act3, act6])
    # 50x50

    up7 = UpSampling2D(size=(2, 2))(concat6)
    conv7 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(up7)
    batch7 = BatchNormalization()(conv7)
    act7 = Activation("relu")(batch7)
    concat7 = Concatenate()([act2, act7])
    # 100x100

    # Down
    conv8 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(concat7)
    batch8 = BatchNormalization()(conv8)
    act8 = Activation("relu")(batch8)
    pool8 = MaxPooling2D(pool_size=(2, 2))(act8)
    # 50x50

    conv9 = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(pool8)
    batch9 = BatchNormalization()(conv9)
    act9 = Activation("relu")(batch9)
    pool9 = MaxPooling2D(pool_size=(2, 2))(act9)

    # 25x25

    # Flat
    conv10 = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(pool9)
    batch10 = BatchNormalization()(conv10)
    act10 = Activation("relu")(batch10)
    # 25x25

    conv11 = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(act10)
    batch11 = BatchNormalization()(conv11)
    act11 = Activation("relu")(batch11)
    # 25x25

    # Encoder
    up12 = UpSampling2D(size=(2, 2))(act11)
    conv12 = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(up12)
    batch12 = BatchNormalization()(conv12)
    act12 = Activation("relu")(batch12)
    concat12 = Concatenate()([act9, act12])
    # 50x50

    up13 = UpSampling2D(size=(2, 2))(concat12)
    conv13 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(up13)
    batch13 = BatchNormalization()(conv13)
    act13 = Activation("relu")(batch13)
    concat13 = Concatenate()([act8, act13])
    # 100x100

    up14 = UpSampling2D(size=(2, 2))(concat13)
    conv14 = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(up14)
    batch14 = BatchNormalization()(conv14)
    act14 = Activation("relu")(batch14)
    concat14 = Concatenate()([act1, act14])
    # 200x200

    conv15 = Conv2D(classes, (1, 1), padding="valid")(concat14)

    reshape15 = Reshape((input_shape[0] * input_shape[1], classes))(conv15)
    act15 = Activation("softmax")(reshape15)

    model = Model(img_input, act15)

    return model
