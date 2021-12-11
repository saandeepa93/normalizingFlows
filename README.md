# **Classification using NFs**

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Gaussianize the moon dataset
```bash
python train.py
```

### Check the `plots/sample` directory for gaussinization and inverse function plots.

## 3. Sample output
### Gaussianized
<video width="320" height="240" controls>
  <source src="./artifacts/x.mp4" type="video/mp4">
</video>

<!-- ![](./artifacts/x.mp4) -->

### Reconstruction
<video width="320" height="240" controls>
  <source src="./artifacts/x_recon.mp4" type="video/mp4">
</video>
<!-- ![](./artifacts/x_recon.mp4) -->


## 3. Papers
@INPROCEEDINGS{6050038,
  author={Lee, Ken Yoong and Bretschneider, Timo Rolf},
  booktitle={2011 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Derivation of separability measures based on central complex Gaussian and Wishart distributions}, 
  year={2011},
  volume={},
  number={},
  pages={3740-3743},
  doi={10.1109/IGARSS.2011.6050038}}