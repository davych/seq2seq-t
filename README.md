python -m spacy download zh_core_web_trf   
python -m spacy download en_core_web_trf   


from datasets import load_dataset
huggingface_dataset = load_dataset("swaption2009/20k-en-zh-translation-pinyin-hsk",data_dir="")
huggingface_dataset = huggingface_dataset["train"]