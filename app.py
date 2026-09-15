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
ARCHETYPES = {"CB": "Défenseur central", "FB": "Latéral", "WG": "Ailier",
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
    "Poste / archétype", [f"{a} — {n}" for a, n in ARCHETYPES.items()]).split(" — ")[0]
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


st.sidebar.markdown("**Listes**")
if not LISTES_URL and "/mount/src" in str(_ICI):
    st.sidebar.warning("⚠️ Listes **non durables** : elles seront effacées à la prochaine "
                       "publication ou au prochain redémarrage. Ajoute la section "
                       "`[listes]` dans Settings → Secrets.")
_noms = listes_noms()
shortlist = st.sidebar.selectbox("Shortlist active", _noms + ["➕ nouvelle liste…"],
                                 index=0 if _noms else len(_noms))
if shortlist == "➕ nouvelle liste…":
    shortlist = st.sidebar.text_input("Nom de la nouvelle liste", value="Shortlist") or "Shortlist"
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
           playerId, squadId, iterationId, position
    FROM v_joueurs WHERE {' AND '.join(where)}
    ORDER BY score DESC LIMIT {PAR_PAGE} OFFSET {page * PAR_PAGE}""", tuple(params))

# ----------------------------------------------------------------- fiche joueur
n_ = lambda v, f="{:.1f}", d="—": (f.format(v) if pd.notna(v) else d)
_date = lambda v: pd.Timestamp(v).strftime("%d/%m/%y") if pd.notna(v) else "?"
TEAL, GRIS = "#1F6F6B", "#9AA5A4"
PALIERS = {"bas": "Bas de tableau", "milieu": "Milieu de tableau", "haut": "Top du championnat"}
COULEUR_PALIER = {"bas": "#A9C9C6", "milieu": "#5F9E99", "haut": TEAL}
NIVEAU_AIDE = ("**Niveau** = score de performance du poste : 50 = joueur médian, "
               "~67 = top 10 %, déjà corrigé de la difficulté de chaque match.")


def lignes_joueur(table: str, cle: tuple, ordre: str) -> pd.DataFrame:
    return requete(f"""SELECT * FROM {table}
        WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype=?
        ORDER BY {ordre}""", cle)


def _vrai(v) -> bool:
    return pd.notna(v) and bool(v)


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


def graphe_forme(blocs: pd.DataFrame, niveau_saison: float) -> go.Figure:
    b = blocs.sort_values("bloc", ascending=False).reset_index(drop=True)   # ancien -> recent
    x = list(range(len(b)))
    debut, fin = b["date_debut"].map(_date).tolist(), b["date_fin"].map(_date).tolist()
    fig = go.Figure(go.Scatter(
        x=x, y=b["performance"], mode="lines+markers+text",
        text=b["performance"].round(0).astype(int), textposition="top center",
        line=dict(color=TEAL, width=3), marker=dict(size=9, color=TEAL),
        customdata=list(zip(debut, fin, b["matchs"], b["minutes"])),
        hovertemplate="Du %{customdata[0]} au %{customdata[1]}<br>"
                      "%{customdata[2]:.0f} matchs · %{customdata[3]:.0f} min<br>"
                      "Niveau <b>%{y:.0f}</b><extra></extra>"))
    n_recents = int((b["bloc"] <= 1).sum())
    if n_recents:
        fig.add_vrect(x0=len(b) - n_recents - 0.5, x1=len(b) - 0.5, fillcolor=TEAL, opacity=0.08,
                      line_width=0, annotation_text="~10 derniers matchs",
                      annotation_position="top left")
    _lignes_reference(fig, niveau_saison)
    valeurs = b["performance"].tolist() + [niveau_saison, 50]
    fig.update_layout(**MISE_EN_PAGE,
                      xaxis=dict(tickvals=x, ticktext=[f"{a}<br>→ {z}" for a, z in zip(debut, fin)],
                                 zeroline=False),
                      yaxis=dict(title="Niveau", range=[max(0, min(valeurs) - 12), max(valeurs) + 12]))
    return fig


MISE_EN_PAGE = dict(height=340, margin=dict(l=50, r=10, t=40, b=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)))


def _lignes_reference(fig: go.Figure, niveau_saison: float) -> None:
    """Niveau de saison et mediane du poste, nommes en legende : des etiquettes
    posees sur les lignes se chevauchent des que les deux valeurs sont proches."""
    fig.update_traces(showlegend=False)
    for y, nom, style, couleur in ((niveau_saison, f"Sa saison : {niveau_saison:.0f}", "dash", TEAL),
                                   (50, "Médiane du poste : 50", "dot", GRIS)):
        fig.add_hline(y=y, line_dash=style, line_color=couleur, line_width=1.5)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nom,
                                 line=dict(dash=style, color=couleur, width=1.5)))


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


def graphe_adversaire(paliers: pd.DataFrame, niveau_saison: float) -> go.Figure:
    p = paliers.set_index("palier")
    ok = [k in p.index for k in PALIERS]
    perf = [p.loc[k, "performance"] if o else 0 for k, o in zip(PALIERS, ok)]
    fig = go.Figure(go.Bar(
        x=list(PALIERS.values()), y=perf,
        marker_color=[COULEUR_PALIER[k] if o else "#E3E7E6" for k, o in zip(PALIERS, ok)],
        text=[f"<b>{p.loc[k, 'performance']:.0f}</b><br>{p.loc[k, 'matchs']:.0f} matchs" if o
              else "trop peu<br>de matchs" for k, o in zip(PALIERS, ok)],
        textposition="outside", cliponaxis=False,
        customdata=[(p.loc[k, "minutes"], p.loc[k, "percentile"]) if o else (0, 0)
                    for k, o in zip(PALIERS, ok)],
        hovertemplate="%{x}<br>%{customdata[0]:.0f} min<br>Niveau <b>%{y:.0f}</b> "
                      "(mieux que %{customdata[1]:.0f} % du poste)<extra></extra>"))
    _lignes_reference(fig, niveau_saison)
    fig.update_layout(**MISE_EN_PAGE,
                      yaxis=dict(title="Niveau", range=[0, max(perf + [niveau_saison, 50]) + 20]))
    return fig


def section_forme(cle: tuple, d, piliers: pd.DataFrame) -> None:
    st.markdown("##### 📈 Évolution sur la saison")
    blocs = lignes_joueur("fait_forme", cle, "bloc")
    titre, phrase = verdict_forme(d, blocs)
    st.markdown(f"{titre}  \n{phrase}")
    if len(blocs) >= 2:
        st.plotly_chart(graphe_forme(blocs, d.score_performance), width="stretch", key=f"forme_{cle}")
        bloc = int(PARAMS["forme_bloc_minutes"])
        ecart = requete("""SELECT stddev(b.performance - f.score_performance) AS s
            FROM fait_forme b JOIN fait_joueur_saison f
            USING (playerId, squadId, iterationId, position, archetype)
            WHERE b.archetype = ?""", (cle[-1],))["s"].iloc[0]
        texte = (f"{NIVEAU_AIDE} Chaque point = un bloc d'environ {bloc} min "
                 f"(~{bloc // 90} matchs pleins), du plus ancien à gauche au plus récent à droite. "
                 f"Un bloc s'écarte en moyenne de ±{ecart:.0f} points du niveau de saison sans que "
                 "rien ne change vraiment : c'est la **tendance** qui compte, pas un point isolé.")
        mouv = piliers.dropna(subset=["progression"])
        if not mouv.empty and pd.notna(d.progression):
            hausse = mouv.loc[mouv["progression"].idxmax()]
            baisse = mouv.loc[mouv["progression"].idxmin()]
            texte += (f"  \nPiliers qui ont le plus bougé sur les ~10 derniers matchs : "
                      f"**{hausse.pilier}** ({hausse.progression:+.0f} pts de percentile) et "
                      f"**{baisse.pilier}** ({baisse.progression:+.0f}). Écarts bruts, encore plus "
                      "bruités que le niveau global : une piste à vérifier en vidéo, pas une conclusion.")
        st.caption(texte)
    elif not blocs.empty:
        st.caption("Un seul bloc de matchs disponible : pas de courbe possible.")


def section_adversaire(cle: tuple, d) -> None:
    st.markdown("##### 🆚 Selon le niveau de l'adversaire")
    paliers = lignes_joueur("fait_adversaire", cle, "ordre")
    titre, phrase = verdict_adversaire(d, paliers)
    st.markdown(f"{titre}  \n{phrase}")
    if not paliers.empty:
        st.plotly_chart(graphe_adversaire(paliers, d.score_performance), width="stretch",
                        key=f"adv_{cle}")
        st.caption(f"{NIVEAU_AIDE} Paliers = tiers des adversaires selon leur rating Impect à la "
                   "date du match, dans ce championnat. La difficulté étant déjà compensée, un 50 "
                   "face au top vaut un joueur médian du poste. Peu de matchs par palier : "
                   "c'est une tendance, pas une preuve.")


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
        st.query_params["club"] = f"{int(j.squadId)}-{int(j.iterationId)}"
        st.rerun()

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
        section_forme(cle, d, piliers)
    with h2:
        section_adversaire(cle, d)

    met = requete("""SELECT metrique, pilier, valeur_brute, z FROM fait_metrique
        WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype=?
        ORDER BY z DESC""", cle)
    if not met.empty:
        f1, f2 = st.columns(2)
        f1.markdown("**Points forts**")
        f1.dataframe(met.head(7)[["metrique", "pilier", "z", "valeur_brute"]].round(2),
                     hide_index=True, width="stretch")
        f2.markdown("**Points faibles**")
        f2.dataframe(met.tail(7)[["metrique", "pilier", "z", "valeur_brute"]].round(2).iloc[::-1],
                     hide_index=True, width="stretch")

    a1, a2 = st.columns(2)
    autres = requete("""SELECT archetype, round(score,1) AS score, rang_archetype AS rang
        FROM fait_joueur_saison
        WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype<>?
        ORDER BY score DESC""", cle)
    with a1:
        st.markdown("**Autres archétypes de ce poste**")
        if not autres.empty:
            st.dataframe(autres, hide_index=True, width="stretch")
        else:
            st.caption("Ce poste ne correspond qu'à un seul archétype.")
    hist = requete("""SELECT saison, competition, round(score,1) AS score, minutes_jouees AS minutes
        FROM v_joueurs WHERE playerId=? AND archetype=? ORDER BY saison""",
                   (int(j.playerId), arch))
    with a2:
        st.markdown("**Historique du joueur**")
        if len(hist) > 1:
            st.dataframe(hist, hide_index=True, width="stretch")
        else:
            st.caption("Une seule saison disponible pour ce joueur.")


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
    playerId, squadId, iterationId, position"""

# ----------------------------------------------------------------- vue club
club_param = st.query_params.get("club")
if club_param:
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
        column_order=["nom", "poste", "archetype_courant", "score", "age", "minutes",
                      "pied_fort", "performance", "progression", "gros_matchs"],
        column_config={"score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=110, format="%.1f"),
            "archetype_courant": st.column_config.TextColumn("Archétype")})
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
    st.caption(f"Joueurs **{premier} à {dernier}** sur {total}, classés par score. "
               "👉 Clique sur une ligne pour ouvrir la fiche du joueur.")
    event = st.dataframe(
        res, hide_index=True, width="stretch", height=430, key=f"tableau_{page}",
        on_select="rerun", selection_mode="single-row",
        column_order=["nom", "club", "competition", "pays", "niveau", "saison",
                      "score", "age", "minutes", "pied_fort", "performance", "aj_niveau",
                      "aj_age", "progression", "gros_matchs", "coef_adv", "rang_mondial"],
        column_config={
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=110, format="%.1f"),
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
