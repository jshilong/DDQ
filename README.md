## [Group R-CNN for Point-based Weakly Semi-supervised Object Detection](https://arxiv.org/abs/2205.05920)
By Shilong Zhang*, Xinjiang Wang*, Jiaqi Wang, Jiangmiao Pang, Kai Chen

#### DDQ greatly surpasses the current record on both object detection and instance segmentation tasks on different datasets(COCO,  LVIS V1.0, CrowdHuman) with less epochs.

### Abstract:
End-to-end object detection is rapidly progressed after the emergence of DETR.
DETRs use a set of sparse queries that replace the dense candidate boxes in most traditional detectors.
In comparison, the sparse queries cannot guarantee a high recall as dense priors. However, making queries dense is not trivial in current frameworks.
It not only suffers from a heavy computational cost but also an increase in optimization difficulty.
As both sparse and dense queries are problematic, then \emph{what are expected queries in end-to-end object detection}?
This paper shows that the expected queries should be Dense Distinct Queries (DDQ).
Concretely, we introduce dense priors back to the framework to generate dense queries.
A duplicate query removal pre-process is applied to these queries so that they are distinguishable from each other.
The dense distinct queries are then iterative to be the final sparse outputs.
We show that DDQ is stronger, converges faster, and is more robust.
It obtains 44.5 AP on the MS COCO detection dataset with standard 1x schedule. DDQ is also robust as it greatly surpasses the current record on both object detection and instance segmentation tasks on different datasets.
DDQ blends advantages from the traditional dense priors and the recent end-to-end detectors.
We hope it can serve as a new baseline and inspires researchers to think about the complementarity between traditional methods and end-to-end detectors.


![](./figs/ddq.png)


## Results & Checkpoints

All results is with R-50 as backbone.

### COCO Detection & Instance Segmentation
All with 680-800 multi-scale augmentations

| Model |  Epochs | Bbox AP | Mask AP |
| :----: | :------: | :-----: | :----: |
| Cascade Mask |   36e  |  44.5 | 38.6 |
| QueryInst | 36e | 45.60 | 40.6 |
| DDQ | 12e | 47.2 | 41.5 |

###LVIS Detection & Instance Segmentation

NOTE: QueryInst seems cannot adapt to the long-tailed dataset LVIS v1.0, the same phenomenon was observed on Sparse R-CNN.

| Model |  Epochs | Bbox AP | Mask AP |
| :----: | :------: | :-----: | :----: |
| Cascade Mask |   12e  |  22.5 | 21.6 |
| QueryInst | 12e | 23.4 | 21.4 |
| QueryInst | 36e | 22.5 | 22.8 |
| DDQ | 12e | 29.6 | 26.6 |

###CrowdHuman Detection

NOTE: * indicate multi-scale input size of range 480-800, otherwise it is normal 1x setting.

| Model |  Epochs | AP@0.5 | mMR | Recall |
| :----: | :------: | :-----: | :----: | :----:
| ATSS |   12e  |  87 | 49.4 | 95.1 |
| TOOD | 12e | 88.7 | 46.5 | 95.5 |
| Cascade R-CNN | 12e | 83.8 | 46.5 | 97.5 |
| DDQ | 12e | 91.1 | 46.1 | 97.5 |
| Sparse R-CNN* | 50e | 89.2 | 48.3 | 95.9 |
| DDQ* | 36e | 93.2 | 40.5 | 98.2 |


### COCO Detection
With DETR Augmentation

| Model |  Epochs | Bbox AP |  Latency(ms)|
| :----: | :------: | :-----: | :-----: |
| Sparse R-CNN | 36e | 45 |  16.4 |
|Deform DETR | 50e | 43.8 | 21.7 |
| DDQ-Lite | 36e | 47.2 | 15.6 |
| DDQ | 36e | 47.7 |  17.7 |
