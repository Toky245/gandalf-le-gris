"""
Gandalf le Gris - traduction automatique des donnees anglaises vers le francais.

Pourquoi : la baseline a appris que les mots francais signifient "normal", car presque toutes
les attaques sont en anglais. On traduit donc les attaques ET une quantite comparable de messages
normaux, pour que la langue ne permette plus de deviner la classe.

Chaque traduction garde le "groupe" de son original : l'original anglais et sa traduction
restent toujours du meme cote (entrainement ou test), ce qui evite les fuites.

Usage :
  pip install transformers sentencepiece sacremoses langdetect
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  python scripts/traduire_en_francais.py
"""
import json
import random
import time
from pathlib import Path

from langdetect import detect, DetectorFactory
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DetectorFactory.seed = 0
random.seed(42)
RAW = Path("data/raw")
SORTIE = RAW / "traduit_fr.jsonl"
MAX_NORMAUX_OASST = 700
TAILLE_LOT = 16


def lire(nom):
    chemin = RAW / f"{nom}.jsonl"
    return [json.loads(l) for l in open(chemin, encoding="utf-8") if l.strip()]


def est_anglais(texte):
    try:
        return detect(texte) == "en"
    except Exception:
        return False


# 1. Selection des textes a traduire
a_traduire = []
a_traduire += [e for e in lire("deepset") if est_anglais(e["texte"])]           # attaques et normaux
a_traduire += lire("gandalf_ignore_instructions")                                 # attaques
a_traduire += lire("gandalf_summarization")                                       # attaques
normaux_en = [e for e in lire("oasst1") if e["langue"] == "en"]
random.shuffle(normaux_en)
a_traduire += normaux_en[:MAX_NORMAUX_OASST]                                      # normaux

nb_att = sum(e["label"] == 1 for e in a_traduire)
print(f"A traduire : {len(a_traduire)} textes ({nb_att} attaques, {len(a_traduire) - nb_att} normaux)")

# 2. Traduction (modele Helsinki-NLP, gratuit, fonctionne sur CPU)
# On utilise directement le tokenizer et le modele : la tache "translation" du pipeline
# n'existe plus dans les versions recentes de transformers.
NOM_MODELE = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(NOM_MODELE)
modele = AutoModelForSeq2SeqLM.from_pretrained(NOM_MODELE).eval()
torch.set_num_threads(max(1, torch.get_num_threads()))


def traduire(textes):
    entrees = tokenizer(textes, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        sorties = modele.generate(**entrees, max_new_tokens=512, num_beams=2)
    return [{"translation_text": t} for t in tokenizer.batch_decode(sorties, skip_special_tokens=True)]


debut = time.time()
resultats = []
for i in range(0, len(a_traduire), TAILLE_LOT):
    lot = a_traduire[i:i + TAILLE_LOT]
    sorties = traduire([e["texte"] for e in lot])
    for e, s in zip(lot, sorties):
        texte_fr = s["translation_text"].strip()
        if texte_fr:
            resultats.append({
                "id": f"fr-{e['id']}",
                "texte": texte_fr,
                "label": e["label"],
                "categorie": e["categorie"],
                "type": e["type"],
                "langue": "fr",
                "groupe": e["groupe"],
                "source": f"traduction:{e['source']}",
            })
    fait = min(i + TAILLE_LOT, len(a_traduire))
    ecoule = time.time() - debut
    reste = ecoule / fait * (len(a_traduire) - fait)
    print(f"\r{fait}/{len(a_traduire)} traduits - environ {reste / 60:.1f} min restantes", end="", flush=True)

with open(SORTIE, "w", encoding="utf-8") as f:
    for r in resultats:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n{SORTIE} : {len(resultats)} exemples")
print("Exemples :")
for r in random.sample(resultats, 4):
    print(f"  [{'attaque' if r['label'] else 'normal'}] {r['texte'][:120]}")
