# Let's begin Kagglegi?
## Chapters:
[0]: #requirements "Requirements"
[1]: #train_configdocs "Train config docs"
1. [Requirements][0]
2. [Basic yaml syntax](#basic-yaml-syntax)
    1. [List](#list)
    2. [Nested lists](#nested-lists)
    3. [Dict and nesting](#dictnesting)
3. [Train config documentation][1] 

### Requirements
- GDCM requires `sudo apt-get install python-gdcm`
- `pip install -r requirements.txt`

### Basic yaml syntax
#### List
Python
```python
lst = [1, 2, 3, 4]
```
Yaml
```yaml
lst: [1, 2, 3, 4]
```
or 
```yaml
lst:
  - 1
  - 2
  - 3
  - 4
```
#### Nested lists
Python
```python
lst = [[[1, 2, 3, 4]]]
```
Yaml
```yaml
lst:
  -
    -
      - 1
      - 2
      - 3
      - 4
```
#### Dict/Nesting
Python
```python
obj = {
    "field1": 1.0,
    "field2": [1, 2, 3],
    "field3": {
        "field3_1": [
            {
                "field3_1_1": 1.0,
            },
            {
                "field3_1_2": 2.0
            }
        ]
    }
}
```
Yaml
```yaml
obj:
    field1: 1.0
    field2:
        - 1
        - 2
        - 3
    field3:
        field3_1:
            - field3_1_1: 1.0
            - field3_1_2: 2.0
```
### train_config.docs
#### Transformations
> Example
```yaml
transformations:
  image:
    - name: normalize
      params:
        threshold: 145
        left_mean: 116.46245344278002
        left_std: 13.827774125976067
        right_mean: 160.45476330904316
        right_std: 9.376197092724993
    - name: fromnumpy
    - name: tofloat
  mask:
    - name: alltosingleobject
    - name: fromnumpy
    - name: tolong
```
Possible transformations:
- alltosingleobject
- normalize
- cv2tocolor
- cv2togray
- fromnumpy
- gray2rgbtriple
- tofloat
- tolong

Docs of each transformation in [dataloaders/transformations.py](dataloaders/transformations.py)
#### Models
> Example
```yaml
model:
    type: torchhub_unet      # torch hub unet
    in_channels: 3           # in_channels, default 3
    out_channels: 1          # out_channels, default 1
    init_features: 32        # default 32, min_number_of_channels
    pretrained: True         # pretrained for 3 channels
```
Possible models:
- old-unet
- universal-unet
- torchhub_unet

Docs of each model in [models](models)
#### Losses
> Example
```yaml
stage1:
    loss: cross_entropy_weights
```
Possible losses:
- cross_entropy_weights
- dice_cc
- dice
- torch_dice  # the best one right now !

Docs of each loss in [utils/losses.py](utils/losses.py)
#### Schedulers
> Example
```yaml
stage1:
    scheduler:
        name: plateau
        mode: min
```
Possible schedulers:
- plateau
- warmrestart
- linear

Docs of each scheduler in [utils/schedulers.py](utils/schedulers.py)
