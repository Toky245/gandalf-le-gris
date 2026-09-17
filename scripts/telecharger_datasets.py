"""
Gandalf le Gris - telechargement et harmonisation des datasets publics.

Convertit chaque source vers le schema commun du projet :
  id, texte, label (1 = attaque, 0 = normal), categorie, type, langue, groupe, source

Usage :
  pip install datasets pandas
  python scripts/telecharger_datasets.py
"""
import json
from pathlib import Path

from datasets import load_dataset

SORTIE = Path("data/raw")
SORTIE.mkdir(parents=True, exist_ok=True)


def toutes_les_lignes(nom):
    """Charge toutes les splits d'un dataset et ajoute le nom de la split."""
    ds = load_dataset(nom)
    for split, donnees in ds.items():
        for ligne in donnees:
            yield split, ligne


def ecrire(chemin, exemples):
    with open(chemin, "w", encoding="utf-8") as f:
        for e in exemples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{chemin} : {len(exemples)} exemples")


def deepset():
    exemples = []
    for i, (split, l) in enumerate(toutes_les_lignes("deepset/prompt-injections")):
        attaque = int(l["label"]) == 1
        exemples.append({
            "id": f"deepset-{i:05d}",
            "texte": l["text"],
            "label": 1 if attaque else 0,
            "categorie": "injection_directe" if attaque else "normal",
            "type": "attaque" if attaque else "normal",
            "langue": "inconnue",  # melange anglais / allemand, detecte plus tard
            "groupe": f"deepset-{i:05d}",
            "source": f"deepset/prompt-injections:{split}",
        })
    ecrire(SORTIE / "deepset.jsonl", exemples)


def lakera(nom_dataset, categorie):
    court = nom_dataset.split("/")[-1]
    exemples = []
    for i, (split, l) in enumerate(toutes_les_lignes(nom_dataset)):
        exemples.append({
            "id": f"{court}-{i:05d}",
            "texte": l["text"],
            "label": 1,  # ces datasets ne contiennent que des attaques
            "categorie": categorie,
            "type": "attaque",
            "langue": "en",
            "groupe": f"{court}-{i:05d}",
            "source": f"{nom_dataset}:{split}",
        })
    ecrire(SORTIE / f"{court}.jsonl", exemples)


# Mots typiques des attaques : on ecarte les prompts OpenAssistant qui les contiennent,
# pour ne pas introduire par erreur des attaques etiquetees comme normales.
MOTS_SUSPECTS = [
    "ignore", "previous instructions", "system prompt", "jailbreak", "dan ",
    "no restrictions", "developer mode", "ignorez", "ignore tes", "instructions precedentes",
    "instructions précédentes", "prompt système", "prompt systeme", "anweisungen", "ignoriere",
]

# Nombre maximum de messages normaux gardes par langue
QUOTAS_OASST = {"en": 700, "fr": 500, "de": 150, "es": 100}


def oasst():
    """Messages normaux : premiers messages ecrits par de vrais utilisateurs (OpenAssistant, Apache 2.0)."""
    import random
    random.seed(42)
    par_langue = {langue: [] for langue in QUOTAS_OASST}
    ecartes = 0
    for split, l in toutes_les_lignes("OpenAssistant/oasst1"):
        if l["role"] != "prompter" or l["parent_id"] is not None:
            continue  # on ne garde que le premier message de chaque conversation
        if l["deleted"] or l["review_result"] is False or l["lang"] not in QUOTAS_OASST:
            continue
        texte = l["text"].strip()
        if not texte:
            continue
        if any(mot in texte.lower() for mot in MOTS_SUSPECTS):
            ecartes += 1
            continue
        par_langue[l["lang"]].append((split, l, texte))

    exemples = []
    for langue, quota in QUOTAS_OASST.items():
        candidats = par_langue[langue]
        random.shuffle(candidats)
        for split, l, texte in candidats[:quota]:
            exemples.append({
                "id": f"oasst1-{l['message_id']}",
                "texte": texte,
                "label": 0,
                "categorie": "normal",
                "type": "normal",
                "langue": langue,
                "groupe": f"oasst1-{l['message_tree_id']}",
                "source": f"OpenAssistant/oasst1:{split}",
            })
        print(f"  oasst1 {langue} : {min(len(candidats), quota)} gardes sur {len(candidats)} disponibles")
    print(f"  oasst1 : {ecartes} prompts ecartes car ils contenaient des mots suspects")
    ecrire(SORTIE / "oasst1.jsonl", exemples)


if __name__ == "__main__":
    deepset()
    lakera("Lakera/gandalf_ignore_instructions", "injection_directe")
    lakera("Lakera/gandalf_summarization", "injection_indirecte")
    oasst()
