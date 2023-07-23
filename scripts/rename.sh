cd results
ls *finetuned-mnli.json | awk '{old=$0; sub(/finetuned-mnli\.json$/, "finetuned.json"); new=$0; system("mv \"" old "\" \"" new "\"")}'
ls *-cased-*.json | awk '{old=$0; gsub(/-cased-/, "-"); new=$0; system("mv \"" old "\" \"" new "\"")}'
ls gpt2* | awk '{old=$0; if (index(old, "gpt2-medium") == 1) new="gpt2-large" substr(old, 12); else new="gpt2-base" substr(old, 5); system("mv \"" old "\" \"" new "\"")}'
cd ..