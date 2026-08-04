_base_ = [
    './bit_r18.py',
    './standard_256x256_40k_levircd.py']

custom_imports = dict(
    imports=['projects.bit.bit'], allow_failed_imports=False
)

find_unused_parameters=True