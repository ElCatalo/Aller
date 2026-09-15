# Plateforme de scouting — Charleroi

Application Streamlit en lecture seule sur la base `charleroi_scouting.duckdb`.
Le calcul des scores reste hors ligne (pipeline Impect), il n'est pas dans ce depot.

## Deploiement (Streamlit Community Cloud)
1. Pousser ce dossier dans un depot GitHub (il peut etre **prive**).
2. share.streamlit.io -> New app -> ce depot -> fichier principal `app.py`.
3. Settings -> Secrets -> coller TOUT le contenu de `.streamlit/secrets.toml` :

   [utilisateurs]
   alex = "<empreinte sha256>"

   [listes]
   url = "<adresse de la base Postgres Neon>"

   Les empreintes se generent avec `gerer_comptes.py`, la section [listes]
   avec `listes_durables.py configurer` (les deux restent en local).

## Mise a jour des donnees
Relancer le scoring, puis `db_build.py`, puis `publier.py`.

## Shortlists et exclusions
Le disque de Streamlit Cloud est efface a chaque publication et redemarrage :
les listes n'y sont donc jamais stockees. Elles vivent dans une base Postgres
hebergee (Neon), independante de ce depot et de la base de scoring.
