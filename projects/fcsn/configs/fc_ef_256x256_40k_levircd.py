_base_ = [
    './fc_ef.py',
    './standard_256x256_40k_levircd.py']

custom_imports = dict(
    imports=['projects.fcsn.fcsn'], allow_failed_imports=False)