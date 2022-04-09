# arabic_calligraphy

## Datasets

### Calliar (OCR)

Import the calliar repository into your project, by executing the bash script as follows :

```
$ bash get_calliar.sh
```

The script initialises the git submodules, updates it. Then extracts the dataset.

### RuFa (Font recognition)

Fonts: Aref Ruqaa - Iran Nastaliq
<https://paperswithcode.com/dataset/rufa>
<https://mhmoodlan.github.io/blog/arabic-font-classification>
To get the data, execute the bash script as follows :

```bash
$ cd data
$ get_rufa_dataset.sh
```

### KAFD (Font recognition)

Fonts: (40)

<https://catalog.ldc.upenn.edu/docs/LDC2016T21/2014-J-KAFD%20Arabic%20Font%20Database_1-s2.0-S0031320313005463-main.pdf>

### ADAB (handwritten arabic)

Execute the bash script as follows :

```bash
$ cd data
$ bash get_adab_dataset.sh    
```