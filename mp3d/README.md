### Prepare Matterport3D Data

1. Download Matterport3D data [HERE](https://niessner.github.io/Matterport/). Move/link the `scans` folder such that under `scans` there should be folders with names such as `scene0001_01`.

2. Matterport3D follows similar data format with ScanNet V2. Please use the provided script "process_matterport.py" to extract point clouds and annotations (semantic seg, instance seg etc.). Please change the path accordingly.
