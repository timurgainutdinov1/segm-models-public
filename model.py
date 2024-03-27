import segmentation_models_pytorch as smp


def get_model(model_name: str, encoder_name: str, encoder_weights: str, activation: str,
              classes=1, in_channels=3):
    # model name should be unique!
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    if model_name == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    return model
