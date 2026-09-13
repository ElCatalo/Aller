# Plateforme de scouting — Charleroi

Application Streamlit en lecture seule sur la base `charleroi_scouting.duckdb`.
Le calcul des scores reste hors ligne (pipeline Impect), il n'est pas dans ce depot.

## Deploiement (Streamlit Community Cloud)
1. Pousser ce dossier dans un depot GitHub (il peut etre **prive**).
2. share.streamlit.io -> New app -> ce depot -> fichier principal `app.py`.
3. Settings -> Secrets -> coller le contenu de `.streamlit/secrets.toml` :

   [utilisateurs]
   alex = "<empreinte sha256>"

   Les empreintes se generent avec `gerer_comptes.py` (reste en local).

## Mise a jour des donnees
Relancer le scoring, puis `db_build.py`, puis `preparer_deploiement.py`,
puis pousser le nouveau `charleroi_scouting.duckdb`.

## Limite connue
Le disque est ephemere : `shortlists.duckdb` est remis a zero a chaque
redemarrage. Pour des shortlists durables en ligne, les basculer sur une base
Postgres hebergee (Supabase ou Neon, offre gratuite).
