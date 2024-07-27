import torch
import pado
import numpy as np
from utils.optics_utils import *

camera_resolution = 512
camera_pitch = 5.4e-6
senser_size = camera_resolution * camera_pitch

R = 1600

DOE_sample_ratio = 1
DOE_pitch = camera_pitch/DOE_sample_ratio
DOE_material = 'FUSED_SILICA'
material = pado.Material(DOE_material)

aperture_shape='circle'
wvls = [656e-9, 589e-9, 486e-9] # camera RGB wavelength
DOE_wvl = 550e-9 # wavelength used to set DOE

sensor_dist = 50e-3
object_dist = 3.5
focal_length = 1/(1/sensor_dist + 1/object_dist)
optics_gap = 0

aperture_diamter = R * DOE_pitch
F_number = focal_length / aperture_diamter

target_PSF_R = camera_resolution

Nyquist_focal_length = R * DOE_pitch**2 / np.min(np.array(wvls))
max_diffraction_pxl = max_diffraction_pxl(np.min(np.array(wvls)), DOE_pitch, focal_length, camera_pitch)
if focal_length < Nyquist_focal_length:
    print("Warning: Nyquist Focal Length", focal_length, Nyquist_focal_length)
if max_diffraction_pxl < camera_resolution / 2:
    print("Warning: Maximum diffraction angle cannot reach edge", int(camera_resolution / 2), max_diffraction_pxl)

aperture_mask = circle_mask(int(R/2),int(R/2))
mask_l, mask_r = half_aperture_mask(int(R/2), R-int(R/2),R)
mask_half_aperture = mask_l

rotational_design = True
DOE_phase_init = torch.zeros(int(1 + np.sqrt(2*(R/2)**2))) 

lens1 = pado.RefractiveLens(R, R, DOE_pitch, focal_length, DOE_wvl,'cpu')
lens2 = pado.RefractiveLens(R, R, DOE_pitch, 1/(1/sensor_dist + 1/5), DOE_wvl,'cpu')
phase_modulation = lens2.phase_change - lens1.phase_change
DOE_phase_init[:int(R/2)] = phase_modulation[0,0,int(R/2),int(R/2):]

depth = np.arange(1,5.1,0.25)
radius = -5 * depth + 25