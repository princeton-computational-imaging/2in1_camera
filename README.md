# Split-Aperture 2-in-1 Computational Cameras
### [Project Page](https://light.princeton.edu/publication/2in1-camera/) | [Paper](https://dl.acm.org/doi/10.1145/3658225)

[Zheng Shi](https://zheng-shi.github.io/), [Ilya Chugunov](https://ilyac.info/), [Mario Bijelic](http://www.mariobijelic.de/), [Geoffroi Côté](https://scholar.google.ca/citations?user=7lWpsmYAAAAJ&hl=en), [Jiwoon Yeom](https://jiwoonyeom.wordpress.com/),[Qiang Fu](https://cemse.kaust.edu.sa/vcc/people/person/qiang-fu), [Hadi Amata ](https://cemse.kaust.edu.sa/people/person/hadi-amata),  [Wolfgang Heidrich](https://vccimaging.org/People/heidriw/), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

If you find our work useful in your research, please cite:
```
@article{splitaperturecameras,
author = {Shi, Zheng and Chugunov, Ilya and Bijelic, Mario and C\^{o}t\'{e}, Geoffroi and Yeom, Jiwoon and Fu, Qiang and Amata, Hadi and Heidrich, Wolfgang and Heide, Felix},
title = {Split-Aperture 2-in-1 Computational Cameras},
year = {2024},
issue_date = {July 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {43},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3658225},
doi = {10.1145/3658225},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {141},
numpages = {19},
keywords = {computational imaging, co-designed optics, dual-pixel sensor, HDR imaging, hyperspectral imaging, monocular depth estimation}
}
```
## Requirements
This code is developed using Pytorch on Linux machine. Full frozen environment can be found in 'env.yml', note some of these libraries are not necessary to run this code. Other than the packages installed in the environment, our image formation model is built based on package [pado](https://github.com/shwbaek/pado) to simulate wave optics. 

## Training and Evaluation
We include all training and eval bash scripts under 'bash_scripts/' folder. Please refer to 'config/' for optics and sensor specs, and 'utils/dataloader' for data processing. 

## Pre-trained Models and Optimized DOE Designs
Optimzed DOE Designs and pre-trained models are available under 'ckpts/' folder available at http://2in1_camera.cs.princeton.edu. Please refer to the supplemental documents for fabrication details.

## Inference
We include a sample script that demonstrates the reconstruction process using an experimental capture in outdoor setting, including the left-right calibration, reconstruction and test-time refinement. You can download the example capture and calibration under 'captures/' folder available at http://2in1_camera.cs.princeton.edu and run the 'inference.ipynb' notebook in Jupyter Notebook.

## License
Our code is licensed under BSL-1. By downloading the software, you agree to the terms of this License. 

## Questions
If there is anything unclear, please feel free to reach out to me using the latest email on my [personal website](https://zheng-shi.github.io/).