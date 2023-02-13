# Update

### Set Python virtual environment

```
> cd 'folder of AeroBenchVVPython'
> python -m venv .venv
> .venv\Scripts\activate.bat
(.venv) > python.exe -m pip install --upgrade pip
(.venv) > pip install -r requirements.txt
(.venv) > DO SOMETHING...
```


<p align="center"> <img src="anim3d.gif"/> </p>

Note: This is the v2 branch of the code, which is now a python3 project and includes more modularity and general simulation capabilities. For the original benchmark paper version see the v1 branch.

# AeroBenchVVPython Overview
This project contains a python version of models and controllers that test automated aircraft maneuvers by performing simulations. The hope is to provide a benchmark to motivate better verification and analysis methods, working beyond models based on Dubins car dynamics, towards the sorts of models used in aerospace engineering. Roughly speaking, the dynamics are nonlinear, have about 10-20 dimensions (continuous state variables), and hybrid in the sense of discontinuous ODEs, but not with jumps in the state. 

This is a python port of the original matlab version, which can can see for
more information: https://github.com/pheidlauf/AeroBenchVV

# Citation

For citation purposes, please use: "Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers", P. Heidlauf, A. Collins, M. Bolender, S. Bak, 5th International Workshop on Applied Verification for Continuous and Hybrid Systems (ARCH 2018)

# 基于遗传算法的离线航迹规划

![GA_path7](https://user-images.githubusercontent.com/68150454/211138603-c6683e96-47dd-439f-a485-28b67012d704.jpg)
![GA_offline7](https://user-images.githubusercontent.com/68150454/211138644-b32cd47f-8259-44fa-af7d-b7ecc3ecf487.gif)
