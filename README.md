# Gandalf le Gris

Pare-feu open source contre les injections de prompt, pense pour le francais.

Gandalf le Gris analyse chaque message avant qu'il n'atteigne un LLM ou un agent IA,
et decide de le laisser passer, de le signaler ou de le bloquer.

> Projet independant, sans lien avec Lakera ni avec leur jeu Gandalf.
> Il reutilise certains de leurs datasets publics sous licence MIT.

## Statut

Phase 1 : constitution des donnees (en cours).

## Schema des donnees

Chaque exemple est une ligne JSON :

| Champ     | Description |
|-----------|-------------|
| id        | Identifiant unique |
| texte     | Le message analyse |
| label     | 1 = attaque, 0 = normal |
| categorie | injection_directe, injection_indirecte, extraction_prompt_systeme, jailbreak, normal |
| type      | attaque, faux_ami, normal |
| langue    | fr, en, mg... |
| groupe    | Les variantes d'un meme exemple partagent le meme groupe et restent dans le meme ensemble (entrainement ou test) pour eviter les fuites |
| source    | Origine de l'exemple |
| domaine   | Domaine du message (ecommerce, programmation, education...), renseigne pour le dataset francais |

## Sources

| Dataset | Licence | Contenu |
|---------|---------|---------|
| deepset/prompt-injections | Apache 2.0 | Injections et messages normaux, anglais et allemand |
| Lakera/gandalf_ignore_instructions | MIT | Injections directes, anglais |
| Lakera/gandalf_summarization | MIT | Injections indirectes, anglais |
| OpenAssistant/oasst1 | Apache 2.0 | Messages normaux de vrais utilisateurs (en, fr, de, es) |
| data/fr/seed_fr.jsonl | Ce projet | Attaques, faux amis et messages normaux en francais |

## Demarrage

```bash
pip install datasets pandas
python scripts/telecharger_datasets.py
```
