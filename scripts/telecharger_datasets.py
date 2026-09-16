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


if __name__ == "__main__":
    deepset()
    lakera("Lakera/gandalf_ignore_instructions", "injection_directe")
    lakera("Lakera/gandalf_summarization", "injection_indirecte")
