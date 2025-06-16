# RL-4WS

This repository contains reinforcement-learning and MPC examples for a four-wheel steering vehicle model.

## DLL Path

Training and evaluation scripts now accept a `--dll` argument specifying the path to `vehiclemodel_public_0326_win64.dll`. If omitted, the DLL is loaded from `vehiclemodel_public_0326_win64.dll` in the repository root.

Example usage:

```bash
python rl_ddpg.py --dll path/to/vehiclemodel_public_0326_win64.dll
python plot_ddpg.py --dll path/to/vehiclemodel_public_0326_win64.dll
python rl_sac_silver.py --dll path/to/vehiclemodel_public_0326_win64.dll
python sort_ddpg.py --dll path/to/vehiclemodel_public_0326_win64.dll
python test_mpc.py --dll path/to/vehiclemodel_public_0326_win64.dll
```
