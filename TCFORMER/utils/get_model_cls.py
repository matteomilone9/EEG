from models import (
    TCFormer,
    TCFormerGAF,
    # ATCNet,
    # ATCNet_2_0,
    # BaseNet,
    # SST_DPN,
    # EEGConformer,
    # EEGNet,
    # EEGTCNet,
    # ShallowNet,
    # TSSEFFNet,
    # CTNet,
    # MSCFormer,
    # EEGDeformer,
)


model_dict = dict(
    TCFormer=TCFormer,
    TCFormerGAF=TCFormerGAF,        # ← aggiunto
    # SST_DPN=SST_DPN,
    # ATCNet=ATCNet,
    # ATCNet_2_0 = ATCNet_2_0,
    # BaseNet=BaseNet,
    # EEGConformer=EEGConformer,
    # EEGNet=EEGNet,
    # EEGTCNet=EEGTCNet,
    # ShallowNet=ShallowNet,
    # TSSEFFNet=TSSEFFNet,
    # CTNet = CTNet,
    # MSCFormer = MSCFormer,
    # EEGDeformer=EEGDeformer,
)


def get_model_cls(model_name):
    return model_dict[model_name]
