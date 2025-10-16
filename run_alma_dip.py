# from_another_script.py
import numpy as np
from alma_dip import DIPConfig, reconstruct_dip, pick_device

# Load your data (u, v, Re, Im, weight):
data = np.loadtxt("DiscoMisterioso.txt", skiprows=1)
uv = data[:, :2].astype(np.float32)
vis = data[:, 2:4].astype(np.float32)
w   = data[:, 4].astype(np.float32)

cfg = DIPConfig(
    num_iters=2000,
    tv_weight=1e-2,
    cell_size_arcsec=0.0075,
    nu=3.0,
    learn_sigma=False,
)

device = str(pick_device("mps"))  # or "cuda" / "cpu"
image, dirty, beam = reconstruct_dip(
    uv=uv, vis=vis, weight=w,
    img_size=(540, 540),
    cfg=cfg,
    device=device
)

import astropy.io.fits as pyfits
pyfits.writeto('alma_dip_1e-2.fits', image, overwrite=True)

np.save("recon.npy", image)
