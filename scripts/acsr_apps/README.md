---
title: Arabic Calligraphy style recognition
emoji: a
colorFrom: #ff0000
colorTo: #440000
sdk: gradio
sdk_version: 2.9.1
app_file: app.py
pinned: true
---

The weights of the model aren't here, download them first and put them in the same directory as `acsr.py`

```bash
$ wget 'https://raw.githubusercontent.com/mhmoodlan/arabic-font-classification/master/codebase/code/font_classifier/weights/FontModel_RuFaDataset_cnn_weights(4).h5' -O weights.h5
```