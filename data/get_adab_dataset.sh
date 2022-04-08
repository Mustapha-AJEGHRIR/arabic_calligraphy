#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=178SkJ_tDP6wwVtpXMW4-6EQXU60d0O8f' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=178SkJ_tDP6wwVtpXMW4-6EQXU60d0O8f" -O ADAB_DATABASE.zip && rm -rf /tmp/cookies.txt
unzip ADAB_DATABASE.zip
rm -rf ADAB_DATABASE.zip
mv 'ADAB DATABASE' ADAB_DATABASE

cd ADAB_DATABASE/
7z x set_*.7z
cd ..
