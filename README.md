# SPH2D-Taichi

Simple 2D SPH fluid simulation implemented using [**taichi**](https://taichi.readthedocs.io/en/stable/) and python.
Weakly compressible SPH (WCSPH) and Predictive-Corrective Incompressible SPH (PCISPH) are implemented in the project.

## Tested Environment

+ x64 windows
+ python 3
+ taichi==0.6.15
+ numpy==1.15.4
+ either GPU or CPU

## Results

### WCSPH
![WCSPH.gif](https://raw.githubusercontent.com/MmmmHeee/SPH-Taichi/master/result/wcsph.gif)

### PCISPH
![WCSPH.gif](https://raw.githubusercontent.com/MmmmHeee/SPH-Taichi/master/result/pcisph.gif)

## References
1. Weakly compressible SPH for free surface flows. Author:Markus Becker(2007), Matthias Teschner
2. Predictive-Corrective Incompressible SPH(2009). Author: B. Solenthaler  R. Pajarola
3. https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py. Author: Ye Kuang (k-ye)
4. https://github.com/erizmr/SPH_Taichi. Author: erizmr, mRay_Zhang
