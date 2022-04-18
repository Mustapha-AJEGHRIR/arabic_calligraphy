# arabic_calligraphy

This repository contains code for 2 projects:
- **Arabic Calligraphy Style Recognition (ACSR)** : Detect the style of the arabic calligraphy, this work is deeply inspired by the work of [ARBML](https://github.com/ARBML/ARBML), a demo is available here : [HuggingFace space](https://huggingface.co/spaces/mustapha/ACSR)
- **Arabic Calligraaphy OCR** : recognize the text of the arabic calligraphy

## Datasets

### Calliar (OCR)

Import the calliar repository into your project, by executing the bash script as follows :

```
bash get_calliar.sh
```

The script initialises the git submodules, updates it. Then extracts the dataset.

To get images for individual characters, you can use the following command :

```bash
cd scripts
python3 prepare_calliar.py --level words
```

### RuFa (Font recognition)

Fonts: Aref Ruqaa - Iran Nastaliq
<https://paperswithcode.com/dataset/rufa>
<https://mhmoodlan.github.io/blog/arabic-font-classification>
To get the data, execute the bash script as follows :

```bash
cd data
get_rufa_dataset.sh
```

### KAFD (Font recognition)

Fonts: (40)

<https://catalog.ldc.upenn.edu/docs/LDC2016T21/2014-J-KAFD%20Arabic%20Font%20Database_1-s2.0-S0031320313005463-main.pdf>

### ADAB (handwritten arabic)

Execute the bash script as follows :

```bash
cd data
bash get_adab_dataset.sh    
```

## Training

### OCR

First 
```bash
$ cd scripts/trocr
```

Then, one of the following tasks can be executed for (chars, words, sentences) :

```bash
$ python3 train.py --level chars --test_size 0.1 --num_beams 4 --limit_eval 256 --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2
```
```bash
$ python3 train.py --level words --test_size 0.1 --num_beams 4 --limit_eval 256 --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2
```
```bash
$ python3 train.py --level sentences --test_size 0.1 --num_beams 4 --limit_eval 256 --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2
```
