"""
Plateforme de scouting Charleroi -- v1 (recherche, filtres, fiche joueur).
===============================================================================
Lit UNIQUEMENT la base DuckDB construite par db_build.py. Aucun calcul de
score ici : la formule reste dans Charleroi_MultiPoste_ScoreV8.py, la base en
contient le resultat et ses composants. La plateforme ne doit jamais
recalculer un score a sa facon (cf. bug de calibration des vieux scripts ML).

LANCEMENT :
    streamlit run impect-scouting/app.py
    (depuis la racine du projet, ou charleroi_scouting.duckdb est situe)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import tomllib
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ICI = Path(__file__).resolve().parent
# A cote de app.py une fois deployee, a la racine du projet en local.
DB = next((p for p in (_ICI / "charleroi_scouting.duckdb",
                       _ICI.parent / "charleroi_scouting.duckdb") if p.exists()),
          _ICI / "charleroi_scouting.duckdb")
ARCHETYPES = {"GK": "Gardien", "CB": "Défenseur central", "FB": "Latéral", "WG": "Ailier",
              "SIX": "Milieu défensif (6)", "EIGHT": "Milieu relayeur (8)",
              "TEN": "Meneur offensif (10)", "NINE": "Avant-centre (9)"}

st.set_page_config(page_title="Scouting Impect — Charleroi", page_icon="🦓", layout="wide")


# ----------------------------------------------------------------- acces
# Comptes definis dans .streamlit/secrets.toml (mots de passe en SHA-256,
# jamais en clair). Une connexion par session de navigateur.
def _secret(section: str) -> dict:
    """Section des secrets lue depuis st.secrets, sinon depuis un secrets.toml voisin.

    st.secrets ne cherche que dans le dossier courant et le dossier personnel :
    lancer l'app depuis un autre repertoire suffit a ne plus voir les comptes.
    On retombe donc sur les emplacements relatifs au fichier app.py.
    """
    try:
        valeurs = dict(st.secrets.get(section, {}))
        if valeurs:
            return valeurs
    except Exception:
        pass
    for dossier in (_ICI, _ICI.parent):
        f = dossier / ".streamlit" / "secrets.toml"
        if f.exists():
            try:
                return dict(tomllib.loads(f.read_text(encoding="utf-8"))[section])
            except Exception:
                continue
    return {}


def _comptes() -> dict:
    return _secret("utilisateurs")


def _mot_de_passe_ok(utilisateur: str, mot_de_passe: str) -> bool:
    attendu = _comptes().get(utilisateur)
    return bool(attendu) and hmac.compare_digest(
        str(attendu), hashlib.sha256(mot_de_passe.encode()).hexdigest())


def authentifier() -> None:
    if st.session_state.get("connecte"):
        return
    st.title("🦓 Scouting Impect — Charleroi")
    if not _comptes():
        en_ligne = "/mount/src" in str(_ICI)
        if en_ligne:
            st.error("Aucun compte configuré **sur Streamlit Cloud**.\n\n"
                     "Les comptes ne sont pas dans le dépôt (volontairement). "
                     "Va dans **Manage app → ⋮ → Settings → Secrets** et colle "
                     "le contenu de ton `.streamlit/secrets.toml` local, "
                     "en commençant bien par la ligne `[utilisateurs]`.")
        else:
            st.error("Aucun compte configuré en local. Lance :\n\n"
                     "`python impect-scouting/gerer_comptes.py`\n\n"
                     f"Cherché dans : `{Path.cwd() / '.streamlit' / 'secrets.toml'}`, "
                     f"`{_ICI / '.streamlit' / 'secrets.toml'}`, "
                     f"`{_ICI.parent / '.streamlit' / 'secrets.toml'}`")
        st.stop()
    with st.form("connexion"):
        u = st.text_input("Utilisateur")
        m = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Se connecter"):
            if _mot_de_passe_ok(u, m):
                st.session_state["connecte"] = u
                st.rerun()
            st.error("Identifiants incorrects.")
    st.stop()


authentifier()


@st.cache_resource
def connexion():
    if not DB.exists():
        st.error(f"Base introuvable : {DB}\n\nLance d'abord : python impect-scouting/db_build.py")
        st.stop()
    return duckdb.connect(str(DB), read_only=True)


@st.cache_data(ttl=600)
def requete(sql: str, params: tuple = ()) -> pd.DataFrame:
    return connexion().execute(sql, params).df()


@st.cache_data(ttl=600)
def listes():
    comp = requete("""SELECT competition, pays, niveau, top5_europe
                      FROM dim_competition ORDER BY rating_moyen DESC""")
    saisons = requete("SELECT DISTINCT season FROM dim_saison ORDER BY season DESC")["season"].tolist()
    # Constantes du run (seuils, fiabilite...) : les phrases de la fiche les
    # citent, elles ne doivent pas diverger de Charleroi_MultiPoste_ScoreV8.py.
    params = requete("SELECT * FROM parametres").iloc[0]
    return comp, saisons, params


comp_df, saisons, PARAMS = listes()
genere_le = PARAMS["genere_le"]

# ----------------------------------------------------------------- filtres
st.sidebar.title("🦓 Scouting Impect")
st.sidebar.caption(f"Données calculées le {genere_le}")
st.sidebar.caption(f"Connecté : {st.session_state['connecte']}")
if st.sidebar.button("Se déconnecter"):
    st.session_state.clear(); st.rerun()

# Libelle inclus dans l'option (plutot que format_func) : identique a l'ecran,
# mais pilotable par les tests automatises de Streamlit.
archetype = st.sidebar.selectbox(
    "Poste / archétype", [f"{a} — {n}" for a, n in ARCHETYPES.items()], index=1).split(" — ")[0]


@st.cache_data(ttl=600)
def catalogue_profils() -> pd.DataFrame:
    """Profils RCSC du catalogue (vide si la base date d'avant les profils)."""
    try:
        return connexion().execute("SELECT * FROM dim_profil ORDER BY num").df()
    except duckdb.Error:
        return pd.DataFrame(columns=["profil_id", "num", "nom", "nom_en", "statut", "archetypes", "grille"])


CATALOGUE = catalogue_profils()
_profils_poste = CATALOGUE[CATALOGUE["archetypes"].str.split(", ").apply(lambda l: archetype in l)
                           & (CATALOGUE["statut"] != "non_calculable")]
_libelles_profils = {f"{r.num} · {r.nom}": r.profil_id for r in _profils_poste.itertuples()}
_choix_profil = st.sidebar.selectbox(
    "Profil RCSC", ["Tous les profils"] + list(_libelles_profils),
    help="Ne garde que les joueurs qui correspondent à ce sous-archétype, classés par correspondance.")
profil_id = _libelles_profils.get(_choix_profil)
# Pas de verdict binaire cote calcul (cf. profils_rcsc.py) : le curseur est un
# simple filtre d'affichage sur la correspondance, SEUIL_PENCHANT (50) par defaut.
seuil_correspondance = st.sidebar.slider(
    "Correspondance minimale", 0, 100, 50, step=5, disabled=profil_id is None,
    help="Ne garde que les joueurs dont la correspondance à ce profil dépasse ce seuil.")
# La correspondance mesure un style, pas un niveau : par defaut on classe les
# joueurs du profil par score V8 (qualite), la correspondance reste affichee.
tri_profil = st.sidebar.radio("Classer les joueurs du profil par", ["Score", "Correspondance"],
                              horizontal=True, disabled=profil_id is None)
saison = st.sidebar.multiselect("Saison", saisons, default=[s for s in saisons if s in ("25/26", "2026")])
recherche = st.sidebar.text_input("Recherche par nom", placeholder="ex. Beitia")

st.sidebar.markdown("**Championnats**")
exclure_top5 = st.sidebar.checkbox("Exclure les 5 grands championnats", value=False)
pays_dispo = sorted([p for p in comp_df["pays"].dropna().unique()])
pays = st.sidebar.multiselect("Pays", pays_dispo)
niveaux = st.sidebar.multiselect("Niveau de division", [1, 2, 3, 4, 5],
                                 help="1 = première division. Vide = tous, y compris les non renseignés.")

# ------------------------------------------------------- shortlists / exclusions
# Les listes ne doivent survivre ni a db_build.py (qui ecrase la base de
# scoring) ni a Streamlit Cloud, dont le disque est EFFACE a chaque publication
# et a chaque redemarrage : un fichier a cote de app.py y est perdu a coup sur.
# Elles vivent donc dans une base Postgres hebergee (Neon, gratuit), dont
# l'adresse est dans les secrets, section [listes] cle url. Sans adresse, repli
# sur shortlists.duckdb : durable sur ton PC, jamais en ligne.
# SCOUTING_LISTES_DB force le fichier local (les tests automatises ecrivent
# ainsi dans un fichier temporaire, jamais dans la vraie base) ;
# SCOUTING_LISTES_URL force une adresse Postgres.
LISTES_DB = Path(os.environ.get("SCOUTING_LISTES_DB", DB.parent / "shortlists.duckdb"))
LISTES_URL = ("" if "SCOUTING_LISTES_DB" in os.environ
              else os.environ.get("SCOUTING_LISTES_URL") or _secret("listes").get("url", ""))


@st.cache_resource
def con_listes():
    if LISTES_URL:
        import psycopg
        try:
            c = psycopg.connect(LISTES_URL, autocommit=True, connect_timeout=20)
        except psycopg.Error as e:
            # Echec bruyant, sans repli sur un fichier local : un ajout qui
            # semblerait reussir puis disparaitrait au redemarrage est pire.
            st.error(f"Base des shortlists injoignable ({type(e).__name__}). "
                     "Réessaie dans une minute ; si ça persiste, vérifie l'adresse "
                     "[listes] url dans les secrets.")
            st.stop()
    else:
        c = duckdb.connect(str(LISTES_DB))
    c.execute("""CREATE TABLE IF NOT EXISTS listes (
        liste VARCHAR, playerId BIGINT, nom VARCHAR, club VARCHAR, poste VARCHAR,
        archetype VARCHAR, score DOUBLE PRECISION, ajoute_le TIMESTAMP)""")
    # Filet de securite : une liste supprimee disparait pour tous les comptes,
    # mais ses lignes sont d'abord copiees ici (qui, quand). Pas d'ecran de
    # restauration : en cas d'erreur, on la recupere a la main depuis cette table.
    c.execute("""CREATE TABLE IF NOT EXISTS listes_supprimees (
        liste VARCHAR, playerId BIGINT, nom VARCHAR, club VARCHAR, poste VARCHAR,
        archetype VARCHAR, score DOUBLE PRECISION, ajoute_le TIMESTAMP,
        supprimee_le TIMESTAMP, supprimee_par VARCHAR)""")
    return c


def sql_listes(sql: str, params: list = ()) -> list[tuple]:
    """Execute une requete sur la base des listes et renvoie ses lignes.

    Meme SQL pour DuckDB et Postgres, seul le marqueur de parametre change.
    Neon coupe les connexions inactives apres quelques minutes : si la
    connexion en cache est morte, on la rouvre et on rejoue une fois.
    """
    if not LISTES_URL:
        cur = con_listes().execute(sql, params)
        return cur.fetchall() if cur.description else []
    import psycopg
    sql = sql.replace("?", "%s")
    for essai in (1, 2):
        c = con_listes()
        try:
            if c.closed:
                raise psycopg.OperationalError("connexion fermee")
            cur = c.execute(sql, params)
            return cur.fetchall() if cur.description else []
        except (psycopg.OperationalError, psycopg.InterfaceError):
            con_listes.clear()
            if essai == 2:
                raise


_txt = lambda v: None if pd.isna(v) else str(v)


def listes_noms() -> list[str]:
    return [r[0] for r in sql_listes(
        "SELECT DISTINCT liste FROM listes WHERE liste <> 'exclus' ORDER BY liste")]


def contenu(liste: str) -> pd.DataFrame:
    return pd.DataFrame(sql_listes(
        "SELECT nom, club, poste, archetype, score, playerId FROM listes WHERE liste=? ORDER BY score DESC",
        [liste]), columns=["nom", "club", "poste", "archetype", "score", "playerId"])


def ids_exclus() -> list[int]:
    return [r[0] for r in sql_listes("SELECT DISTINCT playerId FROM listes WHERE liste='exclus'")]


def ajouter(liste: str, j) -> None:
    sql_listes("DELETE FROM listes WHERE liste=? AND playerId=?", [liste, int(j.playerId)])
    sql_listes("INSERT INTO listes VALUES (?,?,?,?,?,?,?,now())",
               [liste, int(j.playerId), _txt(j.nom), _txt(j.club), _txt(j.poste),
                _txt(j.archetype_courant), None if pd.isna(j.score) else float(j.score)])


def retirer(liste: str, player_id: int) -> None:
    sql_listes("DELETE FROM listes WHERE liste=? AND playerId=?", [liste, int(player_id)])


def renommer_liste(ancien: str, nouveau: str) -> str | None:
    """Renomme une shortlist pour TOUS les comptes. Renvoie le motif du refus, sinon None.

    Refuse un nom deja pris (casse ignoree) plutot que de fusionner : deux
    listes melangees sans le vouloir ne se demelent plus.
    """
    nouveau = nouveau.strip()
    if not nouveau:
        return "Le nom ne peut pas être vide."
    if nouveau.casefold() == "exclus":
        return "« exclus » est réservé à la liste des joueurs exclus."
    if nouveau == ancien:
        return None
    if any(n.casefold() == nouveau.casefold() and n != ancien for n in listes_noms()):
        return f"Une liste « {nouveau} » existe déjà : choisis un autre nom."
    sql_listes("UPDATE listes SET liste=? WHERE liste=?", [nouveau, ancien])
    return None


def supprimer_liste(liste: str, par: str) -> None:
    """Supprime une shortlist pour TOUS les comptes, apres l'avoir archivee
    dans listes_supprimees (cf. con_listes)."""
    sql_listes("""INSERT INTO listes_supprimees
        SELECT liste, playerId, nom, club, poste, archetype, score, ajoute_le,
               now(), CAST(? AS VARCHAR)
        FROM listes WHERE liste=?""", [par, liste])
    sql_listes("DELETE FROM listes WHERE liste=?", [liste])


st.sidebar.markdown("**Listes**")
if not LISTES_URL and "/mount/src" in str(_ICI):
    st.sidebar.warning("⚠️ Listes **non durables** : elles seront effacées à la prochaine "
                       "publication ou au prochain redémarrage. Ajoute la section "
                       "`[listes]` dans Settings → Secrets.")
if _message := st.session_state.pop("_message_listes", None):
    st.toast(_message)
NOUVELLE_LISTE = "➕ nouvelle liste…"
_noms = listes_noms()
_options = _noms + [NOUVELLE_LISTE]
# Un widget deja affiche ne peut plus etre modifie : apres un renommage ou une
# suppression, la liste a selectionner est appliquee au rerun suivant, avant
# de recreer le selecteur.
if "_liste_suivante" in st.session_state:
    st.session_state["shortlist_active"] = st.session_state.pop("_liste_suivante")
# Liste renommee ou supprimee entre-temps, par ce compte ou un autre : on
# retombe sur la premiere liste au lieu de planter.
if st.session_state.get("shortlist_active") not in _options:
    st.session_state.pop("shortlist_active", None)
shortlist = st.sidebar.selectbox("Shortlist active", _options, key="shortlist_active")
if shortlist == NOUVELLE_LISTE:
    shortlist = st.sidebar.text_input("Nom de la nouvelle liste", value="Shortlist") or "Shortlist"
else:
    with st.sidebar.expander("⚙️ Gérer la liste"):
        st.caption("Les changements s'appliquent à **tous les comptes**.")
        nouveau_nom = st.text_input("Nouveau nom", value=shortlist, key=f"nouveau_nom_{shortlist}")
        if st.button("✏️ Renommer", key=f"renommer_{shortlist}", width="stretch"):
            erreur = renommer_liste(shortlist, nouveau_nom)
            if erreur:
                st.error(erreur)
            elif nouveau_nom.strip() != shortlist:
                st.session_state["_liste_suivante"] = nouveau_nom.strip()
                st.session_state["_message_listes"] = (f"« {shortlist} » renommée en "
                                                       f"« {nouveau_nom.strip()} »")
                st.rerun()
        _n = len(contenu(shortlist))
        confirme = st.checkbox(f"Oui, supprimer « {shortlist} » ({_n} joueur{'s' if _n > 1 else ''}) "
                               "pour tous les comptes", key=f"confirmer_{shortlist}")
        if st.button("🗑️ Supprimer définitivement", key=f"supprimer_{shortlist}",
                     disabled=not confirme, width="stretch"):
            supprimer_liste(shortlist, st.session_state["connecte"])
            st.session_state["_liste_suivante"] = ""          # -> premiere liste restante
            st.session_state["_message_listes"] = f"« {shortlist} » supprimée pour tous les comptes"
            st.rerun()
masquer_exclus = st.sidebar.checkbox("Masquer les joueurs exclus", value=True)

st.sidebar.markdown("**Profil**")
age_max = st.sidebar.slider("Âge maximum", 16, 40, 40)
minutes_min = st.sidebar.slider("Minutes minimum", 400, 3000, 900, step=100)
pieds = st.sidebar.multiselect("Pied fort", ["droit", "gauche", "les deux"])
score_min = st.sidebar.slider("Score minimum", 0, 100, 0, step=5)

where, params = ["archetype = ?", "minutes_jouees >= ?", "age_years <= ?", "score >= ?"], \
                [archetype, minutes_min, age_max, score_min]
if saison:
    where.append(f"saison IN ({','.join('?' * len(saison))})"); params += saison
if recherche:
    where.append("lower(nom) LIKE ?"); params.append(f"%{recherche.lower()}%")
if exclure_top5:
    where.append("NOT top5_europe")
if pays:
    where.append(f"pays IN ({','.join('?' * len(pays))})"); params += pays
if niveaux:
    where.append(f"niveau IN ({','.join('?' * len(niveaux))})"); params += [float(n) for n in niveaux]
if pieds:
    where.append(f"pied_fort IN ({','.join('?' * len(pieds))})"); params += pieds
_exclus = ids_exclus()
if masquer_exclus and _exclus:
    where.append(f"playerId NOT IN ({','.join('?' * len(_exclus))})"); params += _exclus

# Profil RCSC : meme joueur-saison, meme pool. Plus de verdict : filtre sur
# le score de correspondance lui-meme (cf. profils_rcsc.py).
_SOUS_PROFIL = """FROM fait_profil p WHERE p.playerId = v_joueurs.playerId AND p.squadId = v_joueurs.squadId
    AND p.iterationId = v_joueurs.iterationId AND p.position = v_joueurs.position
    AND p.archetype = v_joueurs.archetype AND p.profil_id = ?"""
_extra_select, _extra_params, _ordre = "", [], "score DESC"
if profil_id:
    where.append(f"EXISTS (SELECT 1 {_SOUS_PROFIL} AND p.correspondance >= ?)")
    params += [profil_id, seuil_correspondance]
    _extra_select = f", (SELECT round(p.correspondance, 0) {_SOUS_PROFIL}) AS corr_profil"
    _extra_params = [profil_id]
    _ordre = "corr_profil DESC, score DESC" if tri_profil == "Correspondance" else "score DESC, corr_profil DESC"

# Pagination : on ne descend jamais plus de PAR_PAGE lignes de la base, sinon
# l'affichage d'un archetype peu filtre (plusieurs milliers de joueurs) rame.
# Le classement reste global : la page 2 donne bien les 501e a 1000e meilleurs.
PAR_PAGE = 500
_filtre = f"{' AND '.join(where)}|{params}"
if st.session_state.get("_filtre_courant") != _filtre:
    st.session_state["_filtre_courant"] = _filtre
    st.session_state["page"] = 0          # tout changement de filtre ramene page 1
page = st.session_state.get("page", 0)

# Les statistiques portent sur TOUS les joueurs filtres, pas sur la page affichee.
stats = requete(f"""SELECT count(*) AS n, median(score) AS med, max(score) AS max_,
                           median(age_years) AS age_med
                    FROM v_joueurs WHERE {' AND '.join(where)}""", tuple(params)).iloc[0]
total = int(stats["n"])
n_pages = max(1, -(-total // PAR_PAGE))   # division entière arrondie au-dessus
page = min(page, n_pages - 1)
st.session_state["page"] = page

res = requete(f"""
    SELECT nom, club, competition, pays, niveau, saison, round(score,1) AS score,
           round(age_years,1) AS age, minutes_jouees AS minutes, pied_fort,
           round(score_performance,1) AS performance, round(ajust_niveau,1) AS aj_niveau,
           round(ajust_age,1) AS aj_age, round(progression_credible,1) AS progression,
           round(adv_ecart_haut_percentile,0) AS gros_matchs, round(opp_coef_avg,3) AS coef_adv,
           rang_archetype AS rang_mondial, round(score_percentile,1) AS percentile,
           round(base,1) AS base, round(excellence,1) AS excellence,
           round(fragilite,1) AS fragilite, pilier_fort, pilier_faible, taille_cm,
           n_matches_oppw AS matchs, position AS poste, side,
           round(attdef_coef_att_avg,3) AS coef_att, round(attdef_coef_def_avg,3) AS coef_def,
           round(club_rating,3) AS rating_club, round(competition_avg_rating,3) AS rating_ligue,
           round(gros_matchs_delta,1) AS gros_matchs_delta,
           profil_principal,
           playerId, squadId, iterationId, position{_extra_select}
    FROM v_joueurs WHERE {' AND '.join(where)}
    ORDER BY {_ordre} LIMIT {PAR_PAGE} OFFSET {page * PAR_PAGE}""", tuple(_extra_params + params))

# ----------------------------------------------------------------- fiche joueur
n_ = lambda v, f="{:.1f}", d="—": (f.format(v) if pd.notna(v) else d)
_date = lambda v: pd.Timestamp(v).strftime("%d/%m/%y") if pd.notna(v) else "?"
TEAL, GRIS, AUTRE_SAISON = "#1F6F6B", "#9AA5A4", "#8C9A99"
PALIERS = {"bas": "Bas de tableau", "milieu": "Milieu de tableau", "haut": "Top du championnat"}
COULEUR_PALIER = {"bas": "#A9C9C6", "milieu": "#5F9E99", "haut": TEAL}
NIVEAU_AIDE = ("**Niveau** = score de performance du poste : 50 = joueur médian, "
               "~67 = top 10 %, déjà corrigé de la difficulté de chaque match.")
MISE_EN_PAGE = dict(height=340, margin=dict(l=50, r=10, t=40, b=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)))


def lignes_joueur(table: str, cle: tuple, ordre: str) -> pd.DataFrame:
    return requete(f"""SELECT * FROM {table}
        WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype=?
        ORDER BY {ordre}""", cle)


def fin_saison(valeur) -> int:
    """'25/26' -> 2026, '2026' -> 2026 : meme regle que season_end_year de V8,
    pour classer ensemble les deux formats de saison d'Impect."""
    s = str(valeur).strip()
    if "/" in s:
        fin = s.split("/")[-1].strip()
        if fin.isdigit():
            return 2000 + int(fin) if len(fin) == 2 else int(fin)
    return int(s[:4]) if s[:4].isdigit() else 0


# --- Continuite entre saisons ------------------------------------------------
# Le graphe Adversaires ajoute les AUTRES saisons du joueur quand il y jouait
# au MEME poste, dans le MEME club et la MEME division (la courbe Evolution va
# plus loin, cf. lignes_forme_etendue plus bas).
# Pourquoi c'est comparable : db_build.py score chaque archetype contre une
# reference commune a toutes les saisons et tous les championnats, un niveau
# 55 en 24/25 vaut donc un niveau 55 en 25/26. Pourquoi ces trois conditions :
# un changement de club, de division ou de poste change le contexte (systeme,
# adversite, role) et rendrait la courbe trompeuse. Le verdict affiche reste
# celui de la saison de la fiche.
def lignes_continuite(table: str, cle: tuple, ordre: str) -> pd.DataFrame:
    """Lignes de `table` pour la saison de la fiche et les saisons continues."""
    player_id, squad_id, iteration_id, position, arch = cle
    return requete(f"""SELECT t.*, s.season AS saison, f.score_performance
        FROM {table} t
        JOIN dim_saison s USING (iterationId)
        JOIN fait_joueur_saison f USING (playerId, squadId, iterationId, position, archetype)
        WHERE t.playerId=? AND t.squadId=? AND t.position=? AND t.archetype=?
          AND s.competition = (SELECT competition FROM dim_saison WHERE iterationId=?)
        ORDER BY {ordre}""", (player_id, squad_id, position, arch, iteration_id))


def autres_saisons(lignes: pd.DataFrame, iteration: int) -> list[str]:
    s = lignes.loc[lignes["iterationId"] != iteration, "saison"].drop_duplicates()
    return sorted(s, key=fin_saison)


def historique(j, arch: str) -> pd.DataFrame:
    """Toutes les saisons du joueur a cet archetype, la plus recente en tete.
      continuite : meme poste, club et division -> dans les DEUX graphes ;
      courbe     : meme poste, autre club et/ou championnat -> courbe Evolution seule."""
    h = requete("""SELECT saison, club, competition, position AS poste, round(score,1) AS score,
            minutes_jouees AS minutes, playerId, squadId, iterationId, position
        FROM v_joueurs WHERE playerId=? AND archetype=?""", (int(j.playerId), arch))
    h = (h.assign(fin=h["saison"].map(fin_saison))
          .sort_values(["fin", "minutes"], ascending=False).reset_index(drop=True))
    h["courante"] = ((h["iterationId"] == int(j.iterationId)) & (h["squadId"] == int(j.squadId))
                     & (h["position"] == j.position))
    h["continuite"] = ((h["squadId"] == int(j.squadId)) & (h["competition"] == j.competition)
                       & (h["position"] == j.position) & ~h["courante"])
    h["courbe"] = (h["position"] == j.position) & ~h["courante"] & ~h["continuite"]
    return h


# --- Courbe Evolution etendue (demande Alex, 15/09/2026) ----------------------
# Contrairement au graphe Adversaires, la courbe Evolution accepte aussi les
# saisons dans un AUTRE club et/ou championnat (meme poste toujours), pour
# suivre la trajectoire du joueur. Limite a afficher : le niveau est corrige de
# la difficulte de chaque match, PAS de la force du championnat (celle-ci
# n'entre dans le score final que via l'ajustement "niveau") -- un 60 en
# Equateur ne vaut pas un 60 en JPL.
COULEUR_AUTRE_CONTEXTE = "#C27C2C"


def lignes_forme_etendue(cle: tuple, competition: str) -> pd.DataFrame:
    """Blocs de forme de la fiche et de toutes les saisons du joueur au meme
    poste. contexte : fiche / meme (club + division) / autre."""
    player_id, squad_id, iteration_id, position, arch = cle
    b = requete("""SELECT t.*, s.season AS saison, s.competition, c.club, f.ajust_niveau
        FROM fait_forme t
        JOIN dim_saison s USING (iterationId)
        JOIN dim_club c USING (squadId)
        JOIN fait_joueur_saison f USING (playerId, squadId, iterationId, position, archetype)
        WHERE t.playerId=? AND t.position=? AND t.archetype=?""", (player_id, position, arch))
    fiche = (b["iterationId"] == iteration_id) & (b["squadId"] == squad_id)
    meme = (b["squadId"] == squad_id) & (b["competition"] == competition) & ~fiche
    b["contexte"] = ["fiche" if f else ("meme" if m else "autre") for f, m in zip(fiche, meme)]
    return b


def ouvrir_fiche(player_id, squad_id, iteration_id, position, arch) -> None:
    """Ouvre la fiche d'une autre saison. Le lien reste partageable (?fiche=...)."""
    st.query_params["fiche"] = "~".join(
        str(v) for v in (int(player_id), int(squad_id), int(iteration_id), position, arch))
    st.rerun()


def _vrai(v) -> bool:
    return pd.notna(v) and bool(v)


def _legende_ligne(fig: go.Figure, nom: str, style: str, couleur: str) -> None:
    """Ligne de reference nommee en legende : des etiquettes posees sur les
    lignes se chevauchent des que deux valeurs sont proches."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nom,
                             line=dict(dash=style, color=couleur, width=1.5)))


def _mediane(fig: go.Figure) -> None:
    fig.add_hline(y=50, line_dash="dot", line_color=GRIS, line_width=1.5)
    _legende_ligne(fig, "Médiane du poste : 50", "dot", GRIS)


def verdict_forme(d, blocs: pd.DataFrame) -> tuple[str, str]:
    """Titre colore + phrase de lecture de l'evolution recente.
    d : ligne de fait_joueur_saison ; blocs : lignes de fait_forme."""
    if pd.isna(d.progression):
        m = int(PARAMS["progression_min_minutes"])
        return (":gray[**Évolution non mesurable**]",
                f"Il faut au moins {m} minutes sur ses derniers matchs **et** {m} minutes "
                "avant pour comparer deux périodes.")
    p, c = d.progression_percentile, d.progression_credible
    nette = _vrai(d.progression_significative)
    if nette:
        titre = ":green[**📈 En nette hausse**]" if c > 0 else ":red[**📉 En nette baisse**]"
    elif p >= 80 and c > 0:
        titre = ":green[**↗ Plutôt en hausse**]"
    elif p <= 20 and c < 0:
        titre = ":orange[**↘ Plutôt en baisse**]"
    else:
        titre = ":gray[**→ Stable**]"
    recents = blocs[blocs["bloc"] <= 1]
    periode = (f", du {_date(recents['date_debut'].min())} au {_date(recents['date_fin'].max())}"
               if not recents.empty else "")
    hasard = round(100 * (1 - PARAMS["progression_fiabilite"]))
    phrase = (f"Sur ses **{d.matchs_recent:.0f} derniers matchs** ({d.minutes_recent:.0f} min{periode}), "
              f"niveau **{d.perf_recent:.0f}**, contre **{d.perf_avant:.0f}** sur ses "
              f"{d.matchs_avant:.0f} matchs précédents ({d.minutes_avant:.0f} min). "
              f"Écart mesuré : **{d.progression:+.0f}**. Sur si peu de matchs, environ {hasard} % "
              f"d'un tel écart relève du hasard : la progression réelle est estimée à "
              f"**{c:+.1f} pt**")
    phrase += (", un écart qui dépasse nettement ce que produit le hasard." if nette else ".")
    phrase += f" Évolution plus favorable que **{p:.0f} %** des joueurs du poste."
    return titre, phrase


def graphe_forme(blocs: pd.DataFrame) -> go.Figure:
    """Niveau par blocs de ~5 matchs. `blocs` : lignes_forme_etendue (colonne contexte).
    Une courbe par saison ET par club, jamais reliees : l'intersaison ou un
    transfert n'est pas une continuite de forme. Couleur selon le contexte :
    saison de la fiche (vert), meme club + division (gris), autre club et/ou
    championnat (orange pointille, club indique)."""
    style = {"fiche": (TEAL, "solid", "circle", 3),
             "meme": (AUTRE_SAISON, "solid", "diamond", 2),
             "autre": (COULEUR_AUTRE_CONTEXTE, "dot", "circle-open", 2)}
    groupe = ["iterationId", "squadId"]
    b = (blocs.assign(fin=blocs["saison"].map(fin_saison),
                      _debut=blocs.groupby(groupe)["date_debut"].transform("min"))
              .sort_values(["fin", "_debut", "bloc"], ascending=[True, True, False])
              .reset_index(drop=True))
    b["x"] = range(len(b))
    plusieurs = b.groupby(groupe).ngroups > 1
    fig = go.Figure()
    ticks = []
    for _, g in b.groupby(groupe, sort=False):
        contexte = g["contexte"].iloc[0]
        courante = contexte == "fiche"
        couleur, trait, symbole, epaisseur = style[contexte]
        saison, club = g["saison"].iloc[0], g["club"].iloc[0]
        club_court = club if len(club) <= 18 else club[:17] + "…"
        libelle = f"{saison} · {club_court}" if contexte == "autre" else saison
        debut, fin = g["date_debut"].map(_date).tolist(), g["date_fin"].map(_date).tolist()
        ticks += [f"{a}<br>→ {z}" for a, z in zip(debut, fin)]
        fig.add_trace(go.Scatter(
            x=g["x"], y=g["performance"], mode="lines+markers+text", showlegend=False,
            text=g["performance"].round(0).astype(int), textposition="top center",
            textfont=dict(color=couleur),
            line=dict(color=couleur, width=epaisseur, dash=trait),
            marker=dict(size=9 if courante else 8, color=couleur, symbol=symbole,
                        line=dict(color=couleur, width=2)),
            customdata=[(f"{saison} · {club} ({g['competition'].iloc[0]})", a, z, m, mi)
                        for a, z, m, mi in zip(debut, fin, g["matchs"], g["minutes"])],
            hovertemplate="%{customdata[0]}<br>Du %{customdata[1]} au %{customdata[2]}<br>"
                          "%{customdata[3]:.0f} matchs · %{customdata[4]:.0f} min<br>"
                          "Niveau <b>%{y:.0f}</b><extra></extra>"))
        # Reference = moyenne (ponderee en minutes) de SES blocs, pas le score de
        # saison : un bloc de ~5 matchs est calcule avec une confiance plus
        # faible, donc tire vers la mediane. Mesure du 15/09/2026 sur toute la
        # base : a 70 de niveau de saison, les blocs sont en moyenne 11 points
        # plus bas ; a 20, 4 points plus hauts. Comparer les points au score de
        # saison ferait croire que tout bon joueur sous-performe en permanence.
        niveau = (g["performance"] * g["minutes"]).sum() / g["minutes"].sum()
        fig.add_trace(go.Scatter(
            x=[g["x"].min() - 0.4, g["x"].max() + 0.4], y=[niveau, niveau], mode="lines",
            hoverinfo="skip", line=dict(dash="dash", color=couleur, width=1.5),
            name=(f"Moyenne des blocs {libelle}"
                  + (" (cette fiche)" if courante and plusieurs else "") + f" : {niveau:.0f}")))
        if plusieurs:
            if g["x"].min() > 0:
                fig.add_vline(x=g["x"].min() - 0.5, line_color=GRIS, line_width=1)
            fig.add_annotation(x=(g["x"].min() + g["x"].max()) / 2, y=0.99, yref="paper",
                               yanchor="top", showarrow=False,
                               text=f"<b>{saison}</b>" + (f"<br>{club_court}" if contexte == "autre" else ""),
                               font=dict(color=couleur, size=12 if contexte != "autre" else 11))
        recents = g[g["bloc"] <= 1]
        if courante and not recents.empty:
            fig.add_vrect(x0=recents["x"].min() - 0.5, x1=recents["x"].max() + 0.5,
                          fillcolor=TEAL, opacity=0.08, line_width=0,
                          annotation_text="~10 derniers matchs",
                          annotation_position="bottom left" if plusieurs else "top left")
    _mediane(fig)
    valeurs = b["performance"].tolist() + [50]
    if len(b) > 8:
        # Au-dela de 8 blocs, "debut -> fin" deborde sous l'axe : date de debut
        # seule (l'annee se lit dans l'etiquette de saison, la periode au survol).
        ticks = [t.split("<br>")[0] for t in ticks]
    fig.update_layout(**MISE_EN_PAGE,
                      xaxis=dict(tickvals=b["x"].tolist(), ticktext=ticks, zeroline=False,
                                 tickfont=dict(size=10 if len(b) > 8 else 12)),
                      yaxis=dict(title="Niveau", range=[max(0, min(valeurs) - 12), max(valeurs) + 14]))
    return fig


def verdict_adversaire(d, paliers: pd.DataFrame) -> tuple[str, str]:
    """Titre colore + phrase de lecture du niveau selon l'adversaire."""
    haut = paliers[paliers["palier"] == "haut"]
    if haut.empty or pd.isna(d.adv_ecart_haut):
        m = int(PARAMS["adv_min_minutes"])
        return (":gray[**Pas assez de matchs pour comparer**]",
                f"Il faut au moins {m} minutes (~{m // 90} matchs pleins) contre le top du "
                "championnat **et** contre les autres adversaires.")
    h, q, e = haut.iloc[0], d.adv_ecart_haut_percentile, d.adv_ecart_haut
    if q >= 75 and e > 0:
        titre = ":green[**⬆ Élève son niveau contre les gros**]"
    elif q <= 25 and e < 0:
        titre = ":orange[**⬇ Moins bon contre les gros**]"
    else:
        titre = ":gray[**= Même niveau quel que soit l'adversaire**]"
    p = paliers.set_index("palier")
    detail = [(f"**{p.loc[k, 'performance']:.0f}** contre le {nom.lower()} "
               f"({p.loc[k, 'matchs']:.0f} matchs{', peu fiable' if p.loc[k, 'matchs'] < 5 else ''})")
              if k in p.index else f"trop peu de matchs contre le {nom.lower()}"
              for k, nom in PALIERS.items()]
    phrase = ("Niveau " + ", ".join(detail) + ". "
              f"Face au top, il est à **{e:+.0f}** de son niveau contre les autres adversaires "
              f"({h.performance - e:.0f}) : un écart plus favorable que **{q:.0f} %** des joueurs "
              f"du poste. Dans l'absolu, face au top, il fait mieux que **{h.percentile:.0f} %** "
              "des joueurs du poste.")
    return titre, phrase


def graphe_adversaire(paliers: pd.DataFrame, iteration: int, niveau_saison: float) -> go.Figure:
    """Niveau par palier d'adversaire. `paliers` : lignes_continuite('fait_adversaire').
    Une serie de barres par saison, la saison de la fiche en couleur, les
    saisons continues en gris hache."""
    saisons_ = (paliers[["iterationId", "saison"]].drop_duplicates()
                .assign(fin=lambda s: s["saison"].map(fin_saison)).sort_values("fin"))
    plusieurs = len(saisons_) > 1
    fig = go.Figure()
    hauteurs = [niveau_saison, 50]
    for it, saison in zip(saisons_["iterationId"], saisons_["saison"]):
        p = paliers[paliers["iterationId"] == it].set_index("palier")
        courante = it == iteration
        ok = [k in p.index for k in PALIERS]
        perf = [float(p.loc[k, "performance"]) if o else (0.0 if courante else None)
                for k, o in zip(PALIERS, ok)]
        if courante:
            couleurs = [(TEAL if plusieurs else COULEUR_PALIER[k]) if o else "#E3E7E6"
                        for k, o in zip(PALIERS, ok)]
        else:
            couleurs = AUTRE_SAISON
        hauteurs += [v for v in perf if v]
        fig.add_trace(go.Bar(
            x=list(PALIERS.values()), y=perf, marker_color=couleurs,
            marker_pattern_shape="" if courante else "/",
            name=f"{saison} (cette fiche)" if courante else f"{saison} · même club, même division",
            showlegend=plusieurs,
            text=[f"<b>{p.loc[k, 'performance']:.0f}</b><br>{p.loc[k, 'matchs']:.0f} matchs" if o
                  else ("trop peu<br>de matchs" if courante else "") for k, o in zip(PALIERS, ok)],
            textposition="outside", cliponaxis=False, textfont=dict(size=10 if plusieurs else 12),
            customdata=[(saison, p.loc[k, "minutes"], p.loc[k, "percentile"]) if o else (saison, 0, 0)
                        for k, o in zip(PALIERS, ok)],
            hovertemplate="Saison %{customdata[0]} · %{x}<br>%{customdata[1]:.0f} min<br>"
                          "Niveau <b>%{y:.0f}</b> (mieux que %{customdata[2]:.0f} % du poste)"
                          "<extra></extra>"))
    # Reference = moyenne (ponderee en minutes) des paliers de la saison de la
    # fiche, pas son score de saison : meme biais que les blocs de forme (moins
    # de matchs par palier -> tire vers la mediane ; mesure du 15/09/2026 : -10
    # points a 70 de niveau de saison, +4 a 20).
    courante = paliers[paliers["iterationId"] == iteration]
    if not courante.empty:
        moyenne = (courante["performance"] * courante["minutes"]).sum() / courante["minutes"].sum()
        fig.add_hline(y=moyenne, line_dash="dash", line_color=TEAL, line_width=1.5)
        _legende_ligne(fig, f"Moyenne de ses paliers {courante['saison'].iloc[0]} : {moyenne:.0f}",
                       "dash", TEAL)
    _mediane(fig)
    fig.update_layout(**MISE_EN_PAGE, barmode="group",
                      yaxis=dict(title="Niveau", range=[0, max(hauteurs) + 20]))
    return fig


def _texte_continuite(autres: list[str], saison_fiche: str, forme: str) -> str:
    return (f"  \n**En gris : saison{'s' if len(autres) > 1 else ''} {', '.join(autres)}**, "
            f"où il jouait aussi à ce poste, dans ce club et cette division. Même échelle de niveau "
            f"(référence commune à toutes les saisons), donc directement comparable. {forme}"
            f"Le verdict ci-dessus ne porte que sur la saison {saison_fiche}.")


def _texte_autre_contexte(tous: pd.DataFrame, d, saison_fiche: str) -> str:
    """Mise en garde pour les saisons ajoutees dans un autre club et/ou championnat."""
    autres = (tous[tous["contexte"] == "autre"].drop_duplicates(["iterationId", "squadId"])
              .assign(fin=lambda x: x["saison"].map(fin_saison)).sort_values("fin"))
    if autres.empty:
        return ""
    liste = ", ".join(f"{r.saison} à {r.club} ({r.competition})" for r in autres.itertuples())
    reperes = ", ".join(f"{r.ajust_niveau:+.1f} en {r.saison} à {r.club}" for r in autres.itertuples())
    return (f"  \n**En orange : {liste}**, au même poste mais dans un autre club et/ou un autre "
            "championnat. ⚠️ Le niveau y est corrigé de la difficulté de chaque match, **pas de la "
            "force du championnat** : un même chiffre ne vaut pas la même chose dans deux ligues de "
            "force différente. Repère, l'ajustement « niveau » de son score (force du club et du "
            f"championnat) : {d.ajust_niveau:+.1f} en {saison_fiche} ici, {reperes}. Les courbes ne "
            "sont jamais reliées d'un club ou d'une saison à l'autre.")


def section_forme(cle: tuple, d, piliers: pd.DataFrame, saison_fiche: str, competition: str) -> None:
    st.markdown("##### 📈 Évolution sur la saison")
    blocs = lignes_joueur("fait_forme", cle, "bloc")
    titre, phrase = verdict_forme(d, blocs)
    st.markdown(f"{titre}  \n{phrase}")
    tous = lignes_forme_etendue(cle, competition)
    autres = sorted(tous.loc[tous["contexte"] == "meme", "saison"].unique(), key=fin_saison)
    if len(tous) >= 2:
        st.plotly_chart(graphe_forme(tous), width="stretch", key=f"forme_{cle}")
        bloc = int(PARAMS["forme_bloc_minutes"])
        ecart = requete("""SELECT stddev(performance - moyenne) AS s FROM (
            SELECT performance, sum(performance * minutes) OVER w / sum(minutes) OVER w AS moyenne
            FROM fait_forme WHERE archetype = ?
            WINDOW w AS (PARTITION BY playerId, squadId, iterationId, position))""",
                        (cle[-1],))["s"].iloc[0]
        texte = (f"{NIVEAU_AIDE} Chaque point = un bloc d'environ {bloc} min "
                 f"(~{bloc // 90} matchs pleins), du plus ancien à gauche au plus récent à droite. "
                 f"Un bloc s'écarte en moyenne de ±{ecart:.0f} points de la moyenne de ses blocs sans "
                 "que rien ne change vraiment : c'est la **tendance** qui compte, pas un point isolé. "
                 f"La ligne en tirets est cette moyenne, pas son score de saison "
                 f"({d.score_performance:.0f}) : calculé sur ~5 matchs seulement, un bloc est "
                 "mécaniquement tiré vers 50, on le compare donc à ses pareils.")
        mouv = piliers.dropna(subset=["progression"])
        if not mouv.empty and pd.notna(d.progression):
            hausse = mouv.loc[mouv["progression"].idxmax()]
            baisse = mouv.loc[mouv["progression"].idxmin()]
            texte += (f"  \nPiliers qui ont le plus bougé sur les ~10 derniers matchs : "
                      f"**{hausse.pilier}** ({hausse.progression:+.0f} pts de percentile) et "
                      f"**{baisse.pilier}** ({baisse.progression:+.0f}). Écarts bruts, encore plus "
                      "bruités que le niveau global : une piste à vérifier en vidéo, pas une conclusion.")
        autre_contexte = _texte_autre_contexte(tous, d, saison_fiche)
        if autres:
            texte += _texte_continuite(autres, saison_fiche, "" if autre_contexte else
                                       "Les courbes ne sont pas reliées d'une saison à l'autre. ")
        texte += autre_contexte
        if autre_contexte and not autres:
            texte += f" Le verdict ci-dessus ne porte que sur la saison {saison_fiche}."
        st.caption(texte)
    elif not tous.empty:
        st.caption("Un seul bloc de matchs disponible : pas de courbe possible.")


def section_adversaire(cle: tuple, d, saison_fiche: str) -> None:
    st.markdown("##### 🆚 Selon le niveau de l'adversaire")
    paliers = lignes_joueur("fait_adversaire", cle, "ordre")
    titre, phrase = verdict_adversaire(d, paliers)
    st.markdown(f"{titre}  \n{phrase}")
    tous = lignes_continuite("fait_adversaire", cle, "ordre")
    if not tous.empty:
        st.plotly_chart(graphe_adversaire(tous, cle[2], d.score_performance), width="stretch",
                        key=f"adv_{cle}")
        texte = (f"{NIVEAU_AIDE} Paliers = tiers des adversaires selon leur rating Impect à la "
                 "date du match, dans ce championnat. La difficulté étant déjà compensée, un 50 "
                 "face au top vaut un joueur médian du poste. Peu de matchs par palier : "
                 "c'est une tendance, pas une preuve. La ligne en tirets est la moyenne de ses "
                 f"paliers, pas son score de saison ({d.score_performance:.0f}) : calculé sur moins "
                 "de matchs, un palier est mécaniquement tiré vers 50, on le compare donc à ses pareils.")
        autres = autres_saisons(tous, cle[2])
        if autres:
            texte += _texte_continuite(autres, saison_fiche, "Barres grises hachurées. ")
        st.caption(texte)


STATUT_PROFIL = {"partiel": "partiel : une partie du profil n'est pas mesurée",
                 "derive": "règle dérivée des autres profils du poste"}


def _morceaux(texte) -> list[str]:
    return [] if pd.isna(texte) or not texte else [t for t in str(texte).split(" · ") if t]


def section_profils(cle: tuple) -> None:
    """Profils RCSC (sous-archetypes) du joueur dans ce pool : les 3 plus proches
    par correspondance, puis le detail de tous les profils du poste. Pas de
    verdict ni d'exclusion (cf. profils_rcsc.py) : le classement se fait
    uniquement sur le score de correspondance, avec des points d'attention
    (vigilance) sur les non-négociables et une info de style à part."""
    arch = cle[-1]
    try:
        prof = requete("""SELECT p.correspondance, p.rang_pct, p.criteres_ok, p.vigilance,
                                 p.info_style, d.num, d.nom, d.statut, f.erreur_type
            FROM fait_profil p JOIN dim_profil d USING (profil_id)
            LEFT JOIN fiabilite_profil f USING (archetype, profil_id)
            WHERE p.playerId=? AND p.squadId=? AND p.iterationId=? AND p.position=? AND p.archetype=?""", cle)
    except duckdb.Error:
        return   # base construite avant les profils
    st.markdown("#### 🧩 Profils RCSC")
    if prof.empty:
        st.caption("Profils calculés à partir de 900 minutes jouées à ce poste.")
        return
    prof = prof.sort_values("correspondance", ascending=False).reset_index(drop=True)
    retenus = prof.head(3)
    for col, r in zip(st.columns(3), retenus.itertuples()):
        with col.container(border=True):
            statut = f"  \n:gray[{STATUT_PROFIL[r.statut]}]" if r.statut in STATUT_PROFIL else ""
            st.markdown(f"**{r.num} · {r.nom}**{statut}")
            if pd.notna(r.correspondance):
                st.progress(min(max(r.correspondance / 100, 0.0), 1.0),
                            text=f"Correspondance {r.correspondance:.0f} · P{r.rang_pct:.0f} du pool")
            lignes = ([f"✓ {t}" for t in _morceaux(r.criteres_ok)]
                      + [f"⚠ {t}" for t in _morceaux(r.vigilance)]
                      + [f"· {t}" for t in _morceaux(r.info_style)])
            st.caption("  \n".join(lignes))

    notes = []
    if len(retenus) >= 2 and pd.notna(retenus.iloc[0]["erreur_type"]):
        ecart = abs(retenus.iloc[0]["correspondance"] - retenus.iloc[1]["correspondance"])
        if ecart < retenus.iloc[0]["erreur_type"]:
            notes.append(f"**{retenus.iloc[0]['nom']}** et **{retenus.iloc[1]['nom']}** : {ecart:.0f} point(s) "
                         f"d'écart, sous l'erreur type de la correspondance (~{retenus.iloc[0]['erreur_type']:.0f}) : "
                         "profils équivalents, pas classés.")
    if arch == "CB":
        notes.append("Défenseur couvreur (06) non calculé : vitesse, anticipation et 1v1 lancé ne sont pas "
                     "mesurés par Impect.")
    if arch in ("SIX", "EIGHT", "TEN"):
        notes.append(f"Lecture dans le pool {arch} : dans un autre pool de milieu, ses profils peuvent différer.")
    notes.append("Correspondance = ressemblance de style (0-100), pas un niveau : le niveau reste le score. "
                 "Vigilance = point d'attention sur un non-négociable du profil ; l'info de style (zones de "
                 "touches, jeu en retrait) est descriptive, elle ne compte pas dans le score.")
    st.caption("  \n".join(notes))

    with st.expander(f"Détail des {len(prof)} profils du poste"):
        st.dataframe(
            prof[["num", "nom", "correspondance", "rang_pct", "criteres_ok", "vigilance", "info_style"]],
            hide_index=True, width="stretch",
            column_config={
                "num": st.column_config.TextColumn("N°"),
                "nom": st.column_config.TextColumn("Profil"),
                "correspondance": st.column_config.ProgressColumn("Correspondance", min_value=0, max_value=100,
                                                                  format="%.0f"),
                "rang_pct": st.column_config.NumberColumn("Percentile", format="P%.0f"),
                "criteres_ok": st.column_config.TextColumn("Critères validés"),
                "vigilance": st.column_config.TextColumn("Vigilance"),
                "info_style": st.column_config.TextColumn("Style (info)"),
            })


def carte_joueur(j) -> None:
    """Fiche complete d'un joueur. j doit porter une colonne archetype_courant."""
    arch = j.archetype_courant
    cle = (int(j.playerId), int(j.squadId), int(j.iterationId), j.position, arch)

    st.subheader(j.nom)
    lieu = f"{j.club} · {j.competition}"
    if pd.notna(j.pays):
        lieu += f" ({j.pays}" + (f", D{int(j.niveau)})" if pd.notna(j.niveau) else ")")
    st.caption(f"{lieu} · {j.saison} · {j.poste}" + (f" ({j.side})" if pd.notna(j.side) else "")
               + f" · archétype {arch}")

    # Acces direct aux fiches des autres saisons (seules existent celles ou il
    # a assez joue a ce poste pour etre evalue par la pipeline).
    hist = historique(j, arch)
    autres = hist[~hist["courante"]].head(5)
    if not autres.empty:
        c_ = st.columns([1.1] + [1] * len(autres) + [max(0.1, 5 - len(autres))])
        c_[0].markdown("**📅 Ses autres saisons :**")
        for col, h in zip(c_[1:], autres.itertuples()):
            aide = f"{h.competition} · {h.poste} · {h.minutes:.0f} min"
            if h.continuite:
                aide += " — même club, division et poste : intégrée aux deux graphes de cette fiche"
            elif h.courbe:
                aide += " — même poste, autre club et/ou championnat : intégrée à la courbe Évolution"
            marque = " ✓" if h.continuite else (" 📈" if h.courbe else "")
            if col.button(f"{h.saison} · {h.club} · {h.score:.0f}{marque}",
                          key=f"saison_{cle}_{h.squadId}_{h.iterationId}_{h.position}",
                          width="stretch", help=aide):
                ouvrir_fiche(h.playerId, h.squadId, h.iterationId, h.position, arch)
        aides = []
        if autres["continuite"].any():
            aides.append("✓ = même club, même division, même poste : ajoutée en gris aux graphes "
                         "Évolution et Adversaires")
        if autres["courbe"].any():
            aides.append("📈 = même poste mais autre club et/ou championnat : ajoutée en orange à la "
                         "courbe Évolution seulement")
        if aides:
            st.caption(" · ".join(aides) + ".")

    k_ = st.columns(6)
    k_[0].metric("Score", n_(j.score), f"rang mondial {int(j.rang_mondial)}")
    k_[1].metric("Performance", n_(j.performance), f"percentile {n_(j.percentile, '{:.0f}')}")
    k_[2].metric("Âge", n_(j.age))
    k_[3].metric("Minutes", n_(j.minutes, "{:.0f}"), f"{n_(j.matchs, '{:.0f}')} matchs")
    k_[4].metric("Pied", j.pied_fort or "—")
    k_[5].metric("Taille", n_(j.taille_cm, "{:.0f} cm"))

    st.markdown(
        f"**Score {n_(j.score)}** = performance {n_(j.performance)} "
        f"({n_(j.base)} de base {j.excellence:+.1f} excellence {-j.fragilite:+.1f} fragilité) "
        f"{j.aj_niveau:+.1f} niveau {j.aj_age:+.1f} âge")

    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    if b1.button(f"➕ Ajouter à « {shortlist} »", width="stretch", key=f"add_{cle}"):
        ajouter(shortlist, j); st.toast(f"{j.nom} ajouté à « {shortlist} »"); st.rerun()
    if b2.button("🚫 Exclure ce joueur", width="stretch", key=f"exc_{cle}",
                 help="Il ne sera plus proposé dans les recherches par poste."):
        ajouter("exclus", j); st.toast(f"{j.nom} exclu"); st.rerun()
    if b3.button(f"🏟️ Effectif de {j.club}", width="stretch", key=f"clu_{cle}"):
        if "fiche" in st.query_params:
            del st.query_params["fiche"]
        st.query_params["club"] = f"{int(j.squadId)}-{int(j.iterationId)}"
        st.rerun()

    section_profils(cle)

    g1, g2 = st.columns([3, 2])
    with g1:
        piliers = requete("""SELECT pilier, percentile, poids, progression FROM fait_pilier
            WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype=?
            ORDER BY poids DESC""", cle)
        piliers["pilier"] = piliers["pilier"].str.replace("_", " ")
        fig = px.bar(piliers, x="percentile", y="pilier", orientation="h",
                     range_x=[0, 100], text="percentile",
                     labels={"percentile": "Percentile du poste", "pilier": ""},
                     hover_data={"poids": ":.1%"})
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside",
                          marker_color="#1F6F6B", cliponaxis=False)
        fig.update_layout(height=360, margin=dict(l=0, r=20, t=30, b=0),
                          yaxis=dict(autorange="reversed"),
                          title="Piliers — survol : poids dans le score")
        fig.add_vline(x=50, line_dash="dot", line_color="#999")
        st.plotly_chart(fig, width="stretch", key=f"fig_{cle}")

    with g2:
        st.markdown("**Contexte de match**")
        st.dataframe(pd.DataFrame({
            "indicateur": ["Coefficient adversaire", "Coef att/def (attaque)",
                           "Coef att/def (défense)", "Rating du club",
                           "Rating moyen du championnat"],
            "valeur": [n_(j.coef_adv, "{:.3f}"), n_(j.coef_att, "{:.3f}"),
                       n_(j.coef_def, "{:.3f}"), n_(j.rating_club, "{:.3f}"),
                       n_(j.rating_ligue, "{:.3f}")]}),
            hide_index=True, width="stretch")
        st.markdown(f"**Points marquants**\n- Pilier le plus fort : **{str(j.pilier_fort).replace('_', ' ')}**\n"
                    f"- Pilier le plus faible : **{str(j.pilier_faible).replace('_', ' ')}**")

    st.markdown("#### Forme et adversaires")
    d = lignes_joueur("fait_joueur_saison", cle, "archetype").iloc[0]
    h1, h2 = st.columns(2)
    with h1:
        section_forme(cle, d, piliers, j.saison, j.competition)
    with h2:
        section_adversaire(cle, d, j.saison)

    met = requete("""SELECT metrique, pilier, valeur_brute, z FROM fait_metrique
        WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype=?
        ORDER BY z DESC""", cle)
    if not met.empty:
        f1, f2 = st.columns(2)
        f1.markdown("**Points forts**")
        # Au plus la moitie des metriques de chaque cote : avec les 10 metriques
        # du gardien, 7 + 7 ferait apparaitre les memes lignes dans les deux listes.
        n_pf = min(7, len(met) // 2)
        f1.dataframe(met.head(n_pf)[["metrique", "pilier", "z", "valeur_brute"]].round(2),
                     hide_index=True, width="stretch")
        f2.markdown("**Points faibles**")
        f2.dataframe(met.tail(n_pf)[["metrique", "pilier", "z", "valeur_brute"]].round(2).iloc[::-1],
                     hide_index=True, width="stretch")

    a1, a2 = st.columns(2)
    autres_arch = requete("""SELECT archetype, round(score,1) AS score, rang_archetype AS rang
        FROM fait_joueur_saison
        WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype<>?
        ORDER BY score DESC""", cle)
    with a1:
        st.markdown("**Autres archétypes de ce poste**")
        if not autres_arch.empty:
            st.dataframe(autres_arch, hide_index=True, width="stretch")
        else:
            st.caption("Ce poste ne correspond qu'à un seul archétype.")
    with a2:
        st.markdown("**Historique du joueur**")
        if len(hist) > 1:
            vue = hist.assign(fiche=["▶ affichée" if c else ("✓ dans les 2 graphes" if k else
                                                              ("📈 courbe Évolution" if b else ""))
                                     for c, k, b in zip(hist["courante"], hist["continuite"],
                                                        hist["courbe"])])
            st.dataframe(vue, hide_index=True, width="stretch",
                         column_order=["saison", "club", "competition", "poste", "score",
                                       "minutes", "fiche"])
            st.caption("Saisons où il a assez joué pour être évalué à cet archétype. Boutons "
                       "« 📅 Ses autres saisons » en haut de la fiche pour les ouvrir. "
                       "« ✓ dans les 2 graphes » = même poste, club et division ; "
                       "« 📈 courbe Évolution » = même poste, autre club et/ou championnat.")
        else:
            st.caption("Une seule saison évaluée à ce poste pour ce joueur.")


CHAMPS = """nom, club, competition, pays, niveau, saison, round(score,1) AS score,
    round(age_years,1) AS age, minutes_jouees AS minutes, pied_fort,
    round(score_performance,1) AS performance, round(ajust_niveau,1) AS aj_niveau,
    round(ajust_age,1) AS aj_age, round(progression_credible,1) AS progression,
    round(adv_ecart_haut_percentile,0) AS gros_matchs, round(opp_coef_avg,3) AS coef_adv,
    rang_archetype AS rang_mondial, round(score_percentile,1) AS percentile,
    round(base,1) AS base, round(excellence,1) AS excellence, round(fragilite,1) AS fragilite,
    pilier_fort, pilier_faible, taille_cm, n_matches_oppw AS matchs,
    position AS poste, side, round(attdef_coef_att_avg,3) AS coef_att,
    round(attdef_coef_def_avg,3) AS coef_def, round(club_rating,3) AS rating_club,
    round(competition_avg_rating,3) AS rating_ligue,
    round(gros_matchs_delta,1) AS gros_matchs_delta, archetype AS archetype_courant,
    profil_principal,
    playerId, squadId, iterationId, position"""

club_param = st.query_params.get("club")
fiche_param = st.query_params.get("fiche")

# ----------------------------------------------------------------- vue fiche
# Ouverte depuis "Ses autres saisons" : prime sur la vue club et la recherche,
# le bouton retour ramene la ou l'on etait (effectif du club ou recherche).
if fiche_param:
    try:
        _p, _s, _i, _pos, _a = fiche_param.split("~")
        cle_fiche = (int(_p), int(_s), int(_i), _pos, _a)
    except ValueError:
        cle_fiche = None
    ligne = (requete(f"""SELECT {CHAMPS} FROM v_joueurs
        WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype=?""", cle_fiche)
             if cle_fiche else pd.DataFrame())
    if st.button("← Retour à l'effectif" if club_param else "← Retour à la recherche"):
        del st.query_params["fiche"]
        st.rerun()
    if ligne.empty:
        st.error("Fiche introuvable : le lien vient peut-être d'avant une mise à jour des données.")
    else:
        carte_joueur(ligne.iloc[0])

# ----------------------------------------------------------------- vue club
elif club_param:
    squad_id, iter_id = (int(x) for x in club_param.split("-"))
    entete = requete("""SELECT club, competition, saison, pays, niveau,
            round(club_rating,3) AS rating, count(*) AS lignes
        FROM v_joueurs WHERE squadId=? AND iterationId=? GROUP BY ALL""", (squad_id, iter_id))
    if entete.empty:
        st.error("Club introuvable."); st.stop()
    e = entete.iloc[0]
    if st.button("← Retour à la recherche"):
        st.query_params.clear(); st.rerun()
    st.title(f"🏟️ {e.club}")
    st.caption(f"{e.competition} · {e.saison}"
               + (f" · {e.pays}" if pd.notna(e.pays) else "")
               + f" · rating club {n_(e.rating, '{:.3f}')}")

    effectif = requete(f"""SELECT {CHAMPS} FROM v_joueurs
        WHERE squadId=? AND iterationId=?
        QUALIFY row_number() OVER (PARTITION BY playerId ORDER BY score DESC) = 1
        ORDER BY position, score DESC""", (squad_id, iter_id))
    m1, m2, m3 = st.columns(3)
    m1.metric("Joueurs", len(effectif))
    m2.metric("Score médian", n_(effectif["score"].median()))
    m3.metric("Âge médian", n_(effectif["age"].median()))
    st.caption("👉 Clique sur une ligne pour ouvrir la fiche du joueur.")
    ev = st.dataframe(
        effectif, hide_index=True, width="stretch", height=430, key="effectif",
        on_select="rerun", selection_mode="single-row",
        column_order=["nom", "poste", "archetype_courant", "profil_principal", "score", "age", "minutes",
                      "pied_fort", "performance", "progression", "gros_matchs"],
        column_config={"score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=110, format="%.1f"),
            "archetype_courant": st.column_config.TextColumn("Archétype"),
            "profil_principal": st.column_config.TextColumn("Profil RCSC")})
    st.divider()
    sel = ev.selection["rows"] if ev and "rows" in ev.selection else []
    carte_joueur(effectif.iloc[sel[0] if sel else 0])

else:
    # ------------------------------------------------------------- vue joueurs
    st.title(f"{archetype} — {ARCHETYPES[archetype]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Joueurs trouvés", f"{total}")
    c2.metric("Score médian", n_(stats["med"]))
    c3.metric("Meilleur score", n_(stats["max_"]))
    c4.metric("Âge médian", n_(stats["age_med"]))

    if res.empty:
        st.warning("Aucun joueur ne correspond à ces filtres.")
        st.stop()

    res = res.copy()
    res["archetype_courant"] = archetype
    premier, dernier = page * PAR_PAGE + 1, page * PAR_PAGE + len(res)
    if profil_id:
        _tri = "correspondance au profil" if tri_profil == "Correspondance" else "score"
        st.caption(f"Joueurs **{premier} à {dernier}** sur {total} qui correspondent au profil "
                   f"**{_choix_profil}**, classés par {_tri}. 👉 Clique sur une ligne pour ouvrir la fiche.")
        _colonnes_profil = ["corr_profil"]
    else:
        st.caption(f"Joueurs **{premier} à {dernier}** sur {total}, classés par score. "
                   "👉 Clique sur une ligne pour ouvrir la fiche du joueur.")
        _colonnes_profil = ["profil_principal"]
    event = st.dataframe(
        res, hide_index=True, width="stretch", height=430, key=f"tableau_{page}",
        on_select="rerun", selection_mode="single-row",
        column_order=["nom", "club", "competition", "pays", "niveau", "saison",
                      "score"] + _colonnes_profil + ["age", "minutes", "pied_fort", "performance", "aj_niveau",
                      "aj_age", "progression", "gros_matchs", "coef_adv", "rang_mondial"],
        column_config={
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=110, format="%.1f"),
            "profil_principal": st.column_config.TextColumn(
                "Profil RCSC", help="Profil du poste dont la correspondance est la plus haute. "
                                    "Détail dans la fiche."),
            "corr_profil": st.column_config.ProgressColumn(
                "Correspondance", min_value=0, max_value=100, format="%.0f",
                help="Ressemblance au profil (0-100) : un style, pas un niveau."),
            "gros_matchs": st.column_config.NumberColumn("Gros matchs", help="Écart de niveau entre ses matchs contre le top du championnat et ses autres matchs, en percentile du poste : 75+ = élève son niveau contre les gros, 25- = baisse. Détail dans la fiche."),
            "progression": st.column_config.NumberColumn("Progression", help="Évolution réelle estimée (points de niveau) entre ses ~10 derniers matchs et le reste de la saison, hasard retiré : au-delà de ±3 = changement notable. Courbe dans la fiche."),
            "coef_adv": st.column_config.NumberColumn("Coef adv.", help="Coefficient adversaire moyen : >1 = calendrier plus dur que son club"),
        })
    if n_pages > 1:
        p1, p2, p3 = st.columns([1, 2, 1])
        if p1.button("◀ 500 précédents", disabled=page == 0, width="stretch"):
            st.session_state["page"] = page - 1
            st.rerun()
        p2.markdown(f"<div style='text-align:center;padding-top:6px'>Page {page + 1} / {n_pages}</div>",
                    unsafe_allow_html=True)
        if p3.button("500 suivants ▶", disabled=page >= n_pages - 1, width="stretch"):
            st.session_state["page"] = page + 1
            st.rerun()

    d1, d2, d3 = st.columns([2, 2, 1])
    d1.download_button("Télécharger la page affichée (CSV)", res.to_csv(index=False).encode("utf-8"),
                       f"scouting_{archetype}.csv", "text/csv", width="stretch")
    clubs = res[["club", "squadId", "iterationId"]].drop_duplicates().sort_values("club")
    club_choisi = d2.selectbox("Voir l'effectif d'un club", ["—"] + clubs["club"].tolist(),
                               label_visibility="collapsed")
    if club_choisi != "—" and d3.button("🏟️ Ouvrir", width="stretch"):
        c_ = clubs[clubs["club"] == club_choisi].iloc[0]
        st.query_params["club"] = f"{int(c_.squadId)}-{int(c_.iterationId)}"
        st.rerun()
    st.divider()
    lignes = event.selection["rows"] if event and "rows" in event.selection else []
    carte_joueur(res.iloc[lignes[0] if lignes else 0])

# ----------------------------------------------------------------- mes listes
st.divider()
st.subheader("Mes listes")
o1, o2 = st.tabs([f"Shortlist « {shortlist} »", "Joueurs exclus"])
for onglet, nom_liste in ((o1, shortlist), (o2, "exclus")):
    with onglet:
        contenu_df = contenu(nom_liste)
        if contenu_df.empty:
            st.caption("Liste vide.")
            continue
        st.dataframe(contenu_df.drop(columns=["playerId"]), hide_index=True, width="stretch")
        a_retirer = st.selectbox("Retirer un joueur", ["—"] + contenu_df["nom"].tolist(),
                                 key=f"retrait_{nom_liste}")
        if a_retirer != "—" and st.button("Retirer", key=f"btn_retrait_{nom_liste}"):
            retirer(nom_liste, contenu_df.loc[contenu_df["nom"] == a_retirer, "playerId"].iloc[0])
            st.rerun()
        st.download_button("Exporter (CSV)", contenu_df.to_csv(index=False).encode("utf-8"),
                           f"{nom_liste}.csv", "text/csv", key=f"dl_{nom_liste}")
