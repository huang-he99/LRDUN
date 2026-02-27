import torch
from model.simu.lrdun import LRDUN
from ptflops import get_model_complexity_info
import warnings

warnings.filterwarnings("ignore")

def prepare_input(resolution):
    mask = torch.randn(1, 256, 256).cuda()
    Phi_s = torch.randn(1, 256, 310).cuda()
    Phi = torch.randn(1, 28, 256, 310).cuda()
    meas = torch.randn(1, 256, 310).cuda()

    inputs = {
        "CASSI_measure": meas,
        "CASSI_mask": mask,
        "Phi_s": Phi_s,
        "CASSI_mask_3d_shift": Phi,
    }
    return dict(inputs=inputs)

for stage in [3,6,9]:
    model = LRDUN(stage=stage, bands=28, rank=11, dim=16).cuda()
    flops, params = get_model_complexity_info(
        model,
        input_res=(1, 256, 256),
        input_constructor=prepare_input,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"************ LRDUN with {stage} stages ******")
    print("{:<30}  {:<8}".format("Computational complexity: ", flops))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
