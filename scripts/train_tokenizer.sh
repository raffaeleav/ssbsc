#!/bin/bash

git clone --recurse-submodules https://github.com/PythonNut/superbpe

: '
conda deactivate 
conda env remove -n superbpe -y

conda create -n superbpe python=3.10 rust -y
conda activate superbpe

cd superbpe

pip install -r requirements.txt
'

cd superbpe

model_dir="../../data/temp/ssbsc"
output_dir="../../data/temp/ssbsc/tokenizer"

num_inherit_merges=100000
vocab_size=128000
num_bytes=$((10**10))
regex_string="\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

if [ $num_inherit_merges -ge 1000 ]; then
    num_inherit_merges_str=$(($num_inherit_merges / 1000))K
else
    num_inherit_merges_str=${num_inherit_merges}
fi

if [ $vocab_size -ge 1000 ]; then
    vocab_size_str=$(($vocab_size / 1000))K
else
    vocab_size_str=${vocab_size}
fi

if [ -n "$num_inherit_merges" ]; then
    head -n $num_inherit_merges $model_dir/merges.txt > $output_dir/merges.txt
else
    cp $model_dir/merges.txt $output_dir/merges.txt
fi

# corpus_dir has to be None
python -m train_tokenizer \
    --output_dir "$output_dir" \
    --vocab_size "$vocab_size" \
    --corpus_dir "$corpus_dir" \
    --num_bytes "$num_bytes" \
    --regex_string "$regex_string"