# torcs_scripts
Python scripts for controlling cars in TORCS.

TORCS SCRC Patch installation instructions:
* Download torcs-1.3.4: https://sourceforge.net/projects/torcs/files/all-in-one/1.3.4/
* Untar into ~/torcs-1.3.4
* Download SCR patch: https://sourceforge.net/projects/cig/files/SCR%20Championship/Server%20Linux/
* Untar into ~/torcs-1.3.4
* sh scr-patch/do_patch.sh
* ./configure
* Edit src/modules/simu/simuv2/simu.cpp
  * Line 70: 
  * if (isnan(float(car->ctrl->gear)) || isinf(float(car->ctrl->gear))) car->ctrl->gear = 0;
* make
* make install
* sudo make datainstall
* torcs

Try to configure a race, if you get segfault when you hit accept:
Delete ~/.torcs and /usr/local/lib/torcs and /usr/local/share/games/torcs. Then I recompiled and reinstalled. That worked.

