base VAE: sd1.5 d8c4
base diffusion: SiT ImageNet 80 epochs

| Method | gFID | wall clock time |
|--------|------|------|
| SDVAE + REPA cfg| 16.36 | 36h |
| SDVAE + REPA-E | 6.74 | > 72h |
| SDVAE + REPA-E2E flow layer 1 ttur 40 | 8.20 | 36h |
| SDVAE + REPA-E2E flow layer 6 ttur 40 | 8.34 | 36h |
| SDVAE + REPA-E2E flow layer 6 better loss | 9.36 | 36h |
| REPA-E2E-VAE-flow-6-ttur-40 + REPA | 

## cfg 1.5
| Method | gFID | wall clock time |
|--------|------|------|
| REPA | 6.41 | 28h |
| REPA-E | 3.12 | 104h |
| REPA-E2E flow layer 6 better loss (Ours) | 2.63 | 28h |

| LDM |------|------|
| LDM-E2E flow layer 6 |------|------|

## Tuning cfg

SiT
* sit-xl-dinov2-b-enc8-ldm-sdvae-0.0-0.0-400k_0400000_cfg1.5-0.0-1.0
Inception Score: 152.29364013671875
FID: 9.571880291921332
sFID: 15.193813439596738
Precision: 0.7479
Recall: 0.4622

SDVAE + REPA
* sit-xl-dinov2-b-enc8-repae-ldm-sdvae-0.5-1.5-400k_0400000_cfg1.5-0.0-1.0
Inception Score: 230.641357421875
FID: 6.4162902725665845
sFID: 15.37103229345962
Precision: 0.77154
Recall: 0.4947


SDVAE + REPA-E cfg 1.0
* sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k_0400000_cfg1.0-0.0-1.0
* 6.74 

SDVAE + REPA-E cfg 1.5
* sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k_0400000_cfg1.5-0.0-1.0
Inception Score: 280.94
FID: 3.13
sFID: 4.55
Precision: 0.86
Recall: 0.52

SDVAE + REPA-E cfg 2.0
* sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k_0400000_cfg1.5-0.0-1.0
* 7.13

SDVAE + REPA-E cfg 4.0 
* sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k_0400000_cfg4.0-0.0-1.0
* 18.45

SDVAE + REPA-E2E flow layer 6 1.5 better loss cfg 1.0 
* 8.30

SDVAE + REPA-E2E flow layer 6 1.5 better loss cfg 1.5 
* sit-xl-dinov2-b-enc8-repae-sdvae-flow-6r-1.5-1500k_0400000_cfg1.5-0.0-1.0
Inception Score: 287.63
FID: 2.64
sFID: 4.48
Precision: 0.84
Recall: 0.55

SDVAE + REPA-E2E flow layer 6 1.5 better loss cfg 2.0 
sit-xl-dinov2-b-enc8-repae-sdvae-flow-6r-1.5-1500k_0400000_cfg2.0-0.0-1.0
* 6.40
x
SDVAE + REPA-E2E flow layer 6 1.5 better loss cfg 4.0 
* sit-xl-dinov2-b-enc8-repae-sdvae-flow-6r-1.5-1500k_0400000_cfg4.0-0.0-1.0
* 17.31

SDVAE + REPA-E2E flow layer 10 4.5 better loss cfg 1.5 
* 2.61

SDVAE + REPA-E2E flow layer 1 1.5 better loss cfg 1.5 
* sit-xl-dinov2-b-enc8-repae-sdvae-flow-1r-1.5-400k_0400000_cfg1.5-0.0-1.0

SDVAE + REPA-E2E flow layer 10 1.5 better loss cfg 1.5 
* sit-xl-dinov2-b-enc8-repae-sdvae-flow-10r-1.5-400k_0400000_cfg1.5-0.0-1.0

250k
* REPA E: 8.82
* REPA flow 0.5: 11.03
* REPA flow 1.5: 9.76
* REPA flow 4.5

VAVAE + LDM
* sit-xl-dinov2-b-enc8-ldm-vavae-0.0-400k_0400000_cfg1.5-0.0-1.0
Inception Score: 297.3164978027344
FID: 4.170955887909372
sFID: 5.5646472488908785
Precision: 0.87852
Recall: 0.4789

* sit-xl-dinov2-b-enc8-ldm-vavae-0.0-400k_0400000_cfg1.2-0.0-1.0

Inception Score: 201.09664916992188
FID: 3.1942743499186577
sFID: 5.447456437801975
Precision: 0.8182
Recall: 0.5599

VAVAE + LDM + SJO 
* sit-xl-dinov2-b-enc8-repae-vavae-flow-6r-0.0-400k_0400000_cfg1.5-0.0-1.0
Inception Score: 296.11822509765625
FID: 3.6305298010981915
sFID: 6.252062253687768
Precision: 0.86744
Recall: 0.4929

* sit-xl-dinov2-b-enc8-repae-vavae-flow-6r-0.0-400k_0400000_cfg1.2-0.0-1.0
Inception Score: 203.3240966796875
FID: 3.166589893947048
sFID: 6.337009148545349
Precision: 0.80302
Recall: 0.5662

* sit-xl-dinov2-b-enc8-repae-vavae-flow-6r-0.0-z-16.0-400k_0400000_cfg1.2-0.0-1.0
Inception Score: 209.7203369140625
FID: 2.895547881778782
sFID: 5.016219436623373
Precision: 0.81666
Recall: 0.5707

VAVAE + REPA
* sit-xl-dinov2-b-enc8-ldm-vavae-0.5-400k_0400000_cfg1.5-0.0-1.0
* sit-xl-dinov2-b-enc8-ldm-vavae-0.5-400k_0400000_cfg1.2-0.0-1.0

Inception Score: 215.90707397460938
FID: 2.894008480054424
sFID: 5.465893993253417
Precision: 0.8199
Recall: 0.5642


VAVAE + REPA + SJO
* sit-xl-dinov2-b-enc8-repae-vavae-flow-6r-1.5-400k_0400000_cfg1.5-0.0-1.0
* sit-xl-dinov2-b-enc8-repae-vavae-flow-6r-1.5-400k_0400000_cfg1.2-0.0-1.0
Inception Score: 234.2647247314453
FID: 2.5622383407151688
sFID: 5.064308290095823
Precision: 0.82032
Recall: 0.5787

FLUX + LDM 
* sit-xl-dinov2-b-enc8-ldm-flux-0.0-400k_0400000_cfg2.0-0.0-1.0
Inception Score: 198.59878540039062
FID: 6.733654085519731
sFID: 5.549151377155795
Precision: 0.86952
Recall: 0.3745

FLUX + LDM + SJO
* sit-xl-dinov2-b-enc8-repae-flux-flow-6r-0.0-z-4.0-gp-1-lm-400k_0400000_cfg2.0-0.0-1.0
Inception Score: 196.90359497070312
FID: 6.60526383499905
sFID: 5.441091770598291
Precision: 0.87006
Recall: 0.3805

sit-xl-dinov2-b-enc8-repae-flux-flow-6r-0.0-z-1.0-gp-1-lm-400k_0400000_cfg2.0-0.0-1.0.npz
fid= 7.89

FLUX + REPA
fid= 2.83

FLUX + REPA + SJO
* sit-xl-dinov2-b-enc8-repae-flux-flow-6r-1.5-z-4.0-gp-1-lm-400k_0400000_cfg2.0-0.0-1.0
Inception Score: 346.2634582519531
FID: 5.803230305626869
sFID: 5.440606219840561
Precision: 0.89948
Recall: 0.4363

* sit-xl-dinov2-b-enc8-repae-flux-flow-6r-1.5-z-4.0-gp-1-lm-400k_0400000_cfg1.5-0.0-1.0
FID=2.80

WAN + LDM

sit-xl-dinov2-b-enc8-ldm-wan-0.0-400k_0400000_cfg1.5-0.0-1.0
5.30

WAN + LDM + SJO
* sit-xl-dinov2-b-enc8-repae-wan-flow-6r-0.0-z-16.0-gp-1-lm-400k_0400000_cfg2.0-0.0-1.0
Inception Score: 255.6788787841797
FID: 7.077567484670226
sFID: 6.797138892391786
Precision: 0.90672
Recall: 0.3591

* sit-xl-dinov2-b-enc8-repae-wan-flow-6r-0.0-z-16.0-gp-1-lm-400k_0400000_cfg1.5-0.0-1.0
5.66

cd /workspace/cogview_dev/xutd/xu/LightningDiT/guided-diffusion/evaluations

python evaluator.py /workspace/cogview_dev/xutd/xu/LightningDiT/VIRTUAL_imagenet256_labeled.npz /workspace/cogview_dev/xutd/xu/REPA-E/samples/sit-xl-dinov2-b-enc8-repae-flux-flow-6r-1.5-z-4.0-gp-1-lm-400k_0400000_cfg2.0-0.0-1.0.npz 

python evaluator.py /workspace/cogview_dev/xutd/xu/LightningDiT/VIRTUAL_imagenet256_labeled.npz /workspace/cogview_dev/xutd/xu/REPA-E/samples/sit-xl-dinov2-b-enc8-repae-wan-flow-6r-0.0-z-16.0-gp-1-lm-400k_0400000_cfg2.0-0.0-1.0.npz 
