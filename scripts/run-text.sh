python experiment.py -m roberta-base -me outliers kurtosis rotated-kurtosis rms -d mnli mnli-mm -t 4. $@
python experiment.py -m roberta-large -me outliers kurtosis rotated-kurtosis rms -d mnli mnli-mm -t 4. $@
python experiment.py -m gpt2 -me outliers kurtosis rotated-kurtosis rms -d mnli mnli-mm -t 10. $@
python experiment.py -m gpt2-medium -me outliers kurtosis rotated-kurtosis rms -d mnli mnli-mm -t 10. $@
python experiment.py -m George-Ogden/roberta-large-cased-finetuned-mnli -me outliers kurtosis rotated-kurtosis rms accuracy -d mnli mnli-mm -t 4. $@
python experiment.py -m George-Ogden/roberta-base-cased-finetuned-mnli -me outliers kurtosis rotated-kurtosis rms accuracy -d mnli mnli-mm -t 4. $@
python experiment.py -m George-Ogden/gpt2-finetuned-mnli -me outliers kurtosis rotated-kurtosis rms accuracy -d mnli mnli-mm -t 10. $@
python experiment.py -m George-Ogden/gpt2-medium-finetuned-mnli -me outliers kurtosis rotated-kurtosis rms accuracy -d mnli mnli-mm -t 10. $@