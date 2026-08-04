# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
bit_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',    # mmrotate/models/data_preprocessors/data_preprocessor.py
    mean=[0.5 * 255] * 6,
    std=[0.5 * 255] * 6,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,          # todo: right ?
    seg_pad_val=255,    # todo: right ?
    test_cfg=dict(size_divisor=32))
model = dict(
    type='SiamEncoderDecoder',      # mmrotate/models/detectors/siamencoder_decoder.py
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmseg.ResNet',
        depth=18,
        stem_channels=64,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 1, 1),
        out_indices=(2,),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        deep_stem=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    ),
    neck=dict(
        type='FeatureFusionNeck',   # mmrotate/models/necks/feature_fusion.py
        policy='concat',
        out_indices=(0,)),
    decode_head=dict(
        type='BITHead',             # projects/bit/bit/bit_head.py
        in_channels=256,
        channels=32,
        encoder_head_dim=64,
        decoder_head_dim=8,
        enc_depth=1,
        enc_with_pos=True,
        dec_depth=8,
        num_heads=8,
        drop_rate=0,
        use_tokenizer=True,
        token_len=4,
        num_classes=2,
        norm_cfg=bit_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)