# Copyright (c) AI4RS. All rights reserved.
import torch
from collections import OrderedDict
import re


src_path = 'your_path/best_ckpt.pt'
new_path = 'your_path/best_ckpt_converted.pth'
src_model = torch.load(src_path)
for name, param in src_model.items():
    print(name)
print(type(src_model['model_G_state_dict']))
print(len(src_model['model_G_state_dict']))
dst_state_dict = OrderedDict()
for k, v in src_model['model_G_state_dict'].items():    # type=<class 'collections.OrderedDict'>
    old_key = k
    new_key = None

    # 1. Backbone
    if k.startswith('resnet.'):
        if 'fc' in k: continue
        new_key = k.replace('resnet.', 'backbone.')

    # 2. Tokenizer
    elif k.startswith('conv_a.'):
        new_key = k.replace('conv_a.', 'decode_head.conv_a.conv.')

    # 3. Feature Projection (conv_pred -> pre_process)
    elif k.startswith('conv_pred.'):
        new_key = k.replace('conv_pred.', 'decode_head.pre_process.1.conv.')

    # 4. Position Embedding
    elif k == 'pos_embedding':
        new_key = 'decode_head.enc_pos_embedding'

    # 5. Transformer Encoder (Self Attention with to_qkv)
    elif k.startswith('transformer.layers.'):
        match = re.match(r'transformer\.layers\.(\d+)\.(.*)', k)
        if match:
            idx = match.group(1)
            sub_path = match.group(2)

            if sub_path.startswith('0.fn.fn.to_qkv'):
                new_key = f'decode_head.encoder.0.attn.to_qkv.{sub_path.replace("0.fn.fn.to_qkv.", "")}'
            elif sub_path.startswith('0.fn.norm'):
                new_key = f'decode_head.encoder.0.norm1.{sub_path.replace("0.fn.norm.", "")}'
            elif sub_path.startswith('0.fn.fn.to_out'):
                new_key = f'decode_head.encoder.0.attn.{sub_path.replace("0.fn.fn.", "")}'
            elif sub_path.startswith('1.fn.norm'):
                new_key = f'decode_head.encoder.0.norm2.{sub_path.replace("1.fn.norm.", "")}'
            elif sub_path.startswith('1.fn.fn.net.0'):
                new_key = f'decode_head.encoder.0.ff.0.{sub_path.split(".")[-1]}'
            elif sub_path.startswith('1.fn.fn.net.3'):
                new_key = f'decode_head.encoder.0.ff.3.{sub_path.split(".")[-1]}'

    # 6. Transformer Decoder (Cross Attention)
    elif k.startswith('transformer_decoder.layers.'):
        match = re.match(r'transformer_decoder\.layers\.(\d+)\.(.*)', k)
        if match:
            idx = match.group(1)
            sub_path = match.group(2)

            if sub_path.startswith('0.fn.fn.to_q'):
                new_key = f'decode_head.decoder.{idx}.attn.{sub_path.replace("0.fn.fn.", "")}'
            elif sub_path.startswith('0.fn.fn.to_k'):
                new_key = f'decode_head.decoder.{idx}.attn.{sub_path.replace("0.fn.fn.", "")}'
            elif sub_path.startswith('0.fn.fn.to_v'):
                new_key = f'decode_head.decoder.{idx}.attn.{sub_path.replace("0.fn.fn.", "")}'
            elif sub_path.startswith('0.fn.fn.to_out'):
                new_key = f'decode_head.decoder.{idx}.attn.{sub_path.replace("0.fn.fn.", "")}'
            elif sub_path.startswith('0.fn.norm'):
                new_key = f'decode_head.decoder.{idx}.norm1.{sub_path.split(".")[-1]}'
            elif sub_path.startswith('1.fn.norm'):
                new_key = f'decode_head.decoder.{idx}.norm2.{sub_path.split(".")[-1]}'
            elif sub_path.startswith('1.fn.fn.net.0'):
                new_key = f'decode_head.decoder.{idx}.ff.0.{sub_path.split(".")[-1]}'
            elif sub_path.startswith('1.fn.fn.net.3'):
                new_key = f'decode_head.decoder.{idx}.ff.3.{sub_path.split(".")[-1]}'

    # 7. Classifier (Final Layer)
    elif k.startswith('classifier'):
        new_key = k.replace('classifier', 'decode_head.classifier')

    else:
        print(f'{k} is not converted !!!!')
        continue

    if new_key:
        dst_state_dict[new_key] = v
        print(f'{new_key} -> {new_key}')


new_model = dict(meta={},
                 state_dict=dst_state_dict,
                 message_hub={},
                 optimizer={},
                 param_schedulers={})
torch.save(new_model, new_path)