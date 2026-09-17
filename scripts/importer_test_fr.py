"""
Gandalf le Gris - importe les fichiers de test remplis a la main.

Place chaque fichier rempli dans data/test_fr/remplis/ (un fichier par personne),
puis lance :  python scripts/importer_test_fr.py

Resultat : data/test_fr/test_fr.jsonl
Ce jeu de test ne doit JAMAIS servir a l'entrainement.
"""
import json
from pathlib import Path

DOSSIER = Path("data/test_fr/remplis")
SORTIE = Path("data/test_fr/test_fr.jsonl")
SECTIONS = {
    "[ATTAQUES]": (1, "attaque"),
    "[FAUX AMIS]": (0, "faux_ami"),
    "[NORMAUX]": (0, "normal"),
}

exemples, vus = [], set()
for fichier in sorted(DOSSIER.glob("*.txt")):
    auteur, section = fichier.stem, None
    for ligne in open(fichier, encoding="utf-8"):
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#"):
            continue
        if ligne.upper().startswith("AUTEUR:"):
            auteur = ligne.split(":", 1)[1].strip() or fichier.stem
            continue
        if ligne.upper() in SECTIONS:
            section = ligne.upper()
            continue
        if section is None:
            print(f"Ignore (hors section) dans {fichier.name} : {ligne[:60]}")
            continue
        cle = " ".join(ligne.lower().split())
        if cle in vus:
            continue
        vus.add(cle)
        label, type_ = SECTIONS[section]
        exemples.append({
            "id": f"test-fr-{len(exemples) + 1:04d}",
            "texte": ligne,
            "label": label,
            "type": type_,
            "langue": "fr",
            "auteur": auteur,
            "source": "test_fr_manuel",
        })

SORTIE.parent.mkdir(parents=True, exist_ok=True)
with open(SORTIE, "w", encoding="utf-8") as f:
    for e in exemples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"{SORTIE} : {len(exemples)} exemples")
for t in ["attaque", "faux_ami", "normal"]:
    print(f"  {t:9s} : {sum(e['type'] == t for e in exemples)}")
print("  auteurs  :", sorted({e["auteur"] for e in exemples}))
