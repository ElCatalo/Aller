"""
Plateforme de scouting Charleroi -- v1 (recherche, filtres, fiche joueur),
onglet Monitoring (forme des N derniers jours, base charleroi_monitoring.duckdb).
===============================================================================
Lit UNIQUEMENT la base DuckDB construite par db_build.py. Aucun calcul de
score ici : la formule reste dans Charleroi_MultiPoste_ScoreV10.py, la base en
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
import re
import tomllib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

_ICI = Path(__file__).resolve().parent
# A cote de app.py une fois deployee, a la racine du projet en local.
DB = next((p for p in (_ICI / "charleroi_scouting.duckdb",
                       _ICI.parent / "charleroi_scouting.duckdb") if p.exists()),
          _ICI / "charleroi_scouting.duckdb")
# Base du monitoring (db_build.py -> Charleroi_MultiPoste_MonitoringV10.py) :
# separee de la base de scoring, qui frole la limite de 100 Mo de GitHub.
MON_DB = next((p for p in (_ICI / "charleroi_monitoring.duckdb",
                           _ICI.parent / "charleroi_monitoring.duckdb") if p.exists()), None)
ARCHETYPES = {"GK": "Gardien", "CB": "Défenseur central", "FB": "Latéral", "WG": "Ailier",
              "SIX": "Milieu défensif (6)", "EIGHT": "Milieu relayeur (8)",
              "TEN": "Meneur offensif (10)", "NINE": "Avant-centre (9)"}

st.set_page_config(page_title="Scouting Impect — Charleroi", page_icon="🦓", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================
# IDENTITE VISUELLE -- RCSC, noir et blanc (02/10/2026)
# =============================================================================
# Tout tient dans app.py a dessein : le deploiement ne copie que ce fichier et
# les deux bases (cf. preparer_deploiement.py), un .streamlit/config.toml ne
# suivrait pas sur Streamlit Cloud. Le theme est donc en CSS injecte, et il
# s'adapte au mode clair ET sombre du navigateur.
#
# Charte : encre quasi noire sur papier blanc cassé, une seule couleur d'accent
# (le noir lui-meme), zero couleur decorative. Les rayures noir/blanc du
# Sporting -- les Zebres -- servent de motif de separation et d'en-tete.
# Le vert/rouge reste reserve aux ecarts chiffres (progression, verdicts).
# Ecusson officiel depose par Alex a la racine du projet (03/10/2026). On
# accepte plusieurs noms/formats : le fichier reel s'appelle
# "Sporting_de_Charleroi_(logo).svg". A defaut, _blason() dessine un
# monogramme raye de repli (cf. plus bas).
_NOMS_LOGO = ("Sporting_de_Charleroi_(logo).svg", "Sporting_de_Charleroi_(logo).png",
              "logo_rcsc.svg", "logo_rcsc.png")
LOGO_RCSC = next((d / n for d in (_ICI, _ICI.parent) for n in _NOMS_LOGO
                  if (d / n).exists()), None)

# Injecte en UNE SEULE chaine sans ligne vide : markdown ferme la balise
# de style des qu il rencontre une ligne vide, et tout le CSS suivant
# s affichait alors en texte sur la page (bug constate le 03/10/2026).
_CSS_RCSC = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{color-scheme:light;--encre:#0C0C0D;--encre-2:#3A3C40;--encre-3:#71757B;--papier:#FBFBF9;--carte:#FFFFFF;--trait:#E2E2DD;--trait-fort:#C9C9C2;--surbrillance:#F2F2EE;--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;--sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stSidebar"],[data-testid="stSidebarContent"]{background:#FBFBF9!important;color:var(--encre);font-family:var(--sans);font-feature-settings:"ss01","cv01","tnum";}
[data-testid="stHeader"]{background:transparent;border-bottom:1px solid var(--trait);}
[data-testid="stAppViewContainer"] .block-container{padding-top:1.4rem;max-width:1480px;}
h1,h2,h3{font-family:var(--sans);letter-spacing:-.022em;color:var(--encre);font-weight:600;}
h1{font-size:1.72rem;font-weight:700;}
h2{font-size:1.24rem;}
h3{font-size:1.04rem;}
h4,h5{font-family:var(--mono);font-size:.72rem!important;letter-spacing:.13em;text-transform:uppercase;color:var(--encre-3);font-weight:600;margin:1.5rem 0 .5rem;}
p,li,label{color:var(--encre);}
small,[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:var(--encre-3)!important;font-size:.795rem;line-height:1.5;}
a{color:var(--encre);text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:2px;text-decoration-color:var(--trait-fort);}
a:hover{text-decoration-color:var(--encre);}
code,kbd{font-family:var(--mono);font-size:.82em;background:var(--surbrillance);padding:.08em .34em;border-radius:3px;border:1px solid var(--trait);}
.rcsc-rayures{height:5px;margin:0 0 1.15rem;background:repeating-linear-gradient(90deg,var(--encre) 0 11px,transparent 11px 22px);opacity:.9;}
.rcsc-rayures.fine{height:3px;background:repeating-linear-gradient(90deg,var(--encre) 0 7px,transparent 7px 14px);opacity:.5;margin:.35rem 0 1rem;}
.rcsc-marque{display:flex;align-items:center;gap:.7rem;margin:.1rem 0;}
.rcsc-marque svg{flex:none;display:block;}
.rcsc-marque .nom{font-family:var(--mono);font-weight:600;font-size:.86rem;letter-spacing:.1em;text-transform:uppercase;line-height:1.25;}
.rcsc-marque .sous{font-family:var(--mono);font-size:.64rem;letter-spacing:.11em;text-transform:uppercase;color:var(--encre-3);margin-top:.18rem;}
.rcsc-marque.grand{gap:1rem;}
.rcsc-marque.grand .nom{font-size:1.16rem;letter-spacing:.13em;}
.rcsc-marque.grand .sous{font-size:.72rem;}
[data-testid="stSidebar"]{border-right:1px solid var(--trait);}
[data-testid="stSidebar"] .block-container{padding-top:1.1rem;}
[data-testid="stSidebar"] label p{font-size:.775rem!important;font-weight:500;color:var(--encre-2)!important;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong{font-family:var(--mono);font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;color:var(--encre-3);font-weight:600;}
[data-testid="stSidebar"] hr{border-color:var(--trait);}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="textarea"]{border-radius:4px!important;border-color:var(--trait-fort)!important;background:var(--carte)!important;}
[data-baseweb="input"]:focus-within,[data-baseweb="select"]>div:focus-within{border-color:var(--encre)!important;box-shadow:none!important;}
[data-baseweb="tag"]{background:var(--encre)!important;color:var(--papier)!important;border-radius:3px!important;font-family:var(--mono);font-size:.72rem!important;}
[data-baseweb="tag"] svg{fill:var(--papier)!important;}
.stButton>button,.stDownloadButton>button{border-radius:4px;border:1px solid var(--trait-fort);background:var(--carte);color:var(--encre);font-weight:500;font-size:.82rem;padding:.36rem .8rem;box-shadow:none;transition:border-color .12s,background .12s;}
.stButton>button p,.stDownloadButton>button p{color:inherit!important;margin:0;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--encre);background:var(--surbrillance);color:var(--encre);}
.stButton>button[kind="primary"]{background:var(--encre);color:var(--papier);border-color:var(--encre);}
.stButton>button[kind="primary"] p,.stButton>button[kind="primary"] div,.stButton>button[kind="primary"] span{color:var(--papier)!important;}
[data-testid="stFormSubmitButton"]>button{background:var(--encre);color:var(--papier);border-color:var(--encre);font-weight:600;}
[data-testid="stFormSubmitButton"]>button p,[data-testid="stFormSubmitButton"]>button div,[data-testid="stFormSubmitButton"]>button span{color:var(--papier)!important;}
[data-testid="stFormSubmitButton"]>button:hover{background:var(--encre-2);border-color:var(--encre-2);}
.stButton>button[kind="primary"]:hover{background:var(--encre-2);border-color:var(--encre-2);color:var(--papier);}
[data-testid="stSidebar"] .stButton>button{text-align:left;justify-content:flex-start;border:none;border-bottom:1px solid var(--trait);border-radius:0;padding:.4rem .15rem;font-size:.8rem;background:transparent;}
[data-testid="stSidebar"] .stButton>button:hover{background:var(--surbrillance);border-bottom-color:var(--encre);}
/* onglets : lisibles d abord -- ils portent la navigation principale */
.stTabs [data-baseweb="tab-list"]{gap:1.9rem;border-bottom:1px solid var(--trait);margin-bottom:.4rem;}
.stTabs [data-baseweb="tab"]{background:transparent!important;padding:.6rem 0 .65rem;font-family:var(--sans);font-size:.95rem;letter-spacing:-.01em;font-weight:500;color:var(--encre-2)!important;}
.stTabs [data-baseweb="tab"]:hover{color:var(--encre)!important;}
.stTabs [data-baseweb="tab"] p{font-size:.95rem!important;font-weight:500;color:inherit!important;}
.stTabs [aria-selected="true"]{color:var(--encre)!important;font-weight:700;}
.stTabs [aria-selected="true"] p{font-weight:700!important;color:var(--encre)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--encre);height:2.5px;}
.stTabs [data-baseweb="tab-border"]{display:none;}
[data-testid="stMetric"]{background:#FFFFFF;border:1px solid var(--trait);border-radius:5px;padding:.68rem .85rem;}
[data-testid="stMetricLabel"] p{font-family:var(--mono)!important;font-size:.66rem!important;letter-spacing:.11em;text-transform:uppercase;color:var(--encre-3)!important;font-weight:600;}
[data-testid="stMetricValue"]{font-family:var(--mono);font-weight:600;font-size:1.5rem;letter-spacing:-.018em;color:var(--encre);}
[data-testid="stMetricDelta"]{font-family:var(--mono);font-size:.76rem;}
[data-testid="stDataFrame"]{border:1px solid var(--trait);border-radius:5px;}
[data-testid="stDataFrame"] [role="columnheader"]{background:var(--surbrillance)!important;font-family:var(--mono)!important;font-size:.68rem!important;letter-spacing:.075em;text-transform:uppercase;color:var(--encre-3)!important;font-weight:600;}
[data-testid="stDataFrame"] [role="gridcell"]{font-size:.815rem;font-variant-numeric:tabular-nums;}
[data-testid="stExpander"]{border:1px solid var(--trait)!important;border-radius:5px;background:var(--carte);}
[data-testid="stExpander"] summary{font-size:.84rem;font-weight:500;}
div[data-testid="stAlert"]{border-radius:5px;border:1px solid var(--trait);background:var(--carte);color:var(--encre);font-size:.83rem;}
hr{border-color:var(--trait);margin:1.3rem 0;}
[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"]{font-family:var(--mono);font-size:.66rem;color:var(--encre-3);}
[data-baseweb="slider"] [role="slider"]{background:var(--encre)!important;border-color:var(--encre)!important;}
/* equipe type : cases de poste, lecture rapide */
.rcsc-case{font-family:var(--mono);font-size:.62rem;letter-spacing:.11em;text-transform:uppercase;color:var(--encre-3);font-weight:600;margin-bottom:.3rem;}
.rcsc-meta{font-size:.73rem;line-height:1.45;color:var(--encre-2);margin:.15rem 0 .45rem;}
.rcsc-meta b{color:var(--encre);}
.rcsc-vide{font-family:var(--mono);color:var(--trait-fort);font-size:1.1rem;text-align:center;padding:.5rem 0 .7rem;line-height:1.3;}
.rcsc-vide span{font-size:.66rem;letter-spacing:.06em;color:var(--encre-3);}
/* ecran de connexion : respiration verticale */
.rcsc-accueil{padding-top:4.5rem;}
</style>"""
st.markdown(_CSS_RCSC, unsafe_allow_html=True)


def rayures(fine: bool = False) -> None:
    """Separateur raye noir/blanc -- le maillot du Sporting."""
    st.markdown(f"<div class='rcsc-rayures{' fine' if fine else ''}'></div>",
                unsafe_allow_html=True)


# --- Graphiques : meme charte que le reste -----------------------------------
# Plotly ne lit pas le CSS : sans template, les courbes repartent sur la
# palette par defaut (bleu/orange) et cassent l'ensemble. Un seul template
# gris/noir, enregistre comme defaut, suffit -- les traces qui imposent deja
# leur couleur (vert/rouge des ecarts) ne sont pas touchees.
_ENCRE, _PAPIER_G, _TRAIT_G, _GRIS = "#0C0C0D", "rgba(0,0,0,0)", "#E2E2DD", "#71757B"
# Serie categorielle en degrade de gris : lisible en noir et blanc, et
# differenciable a l'impression (une comparaison de 5 joueurs reste lisible).
SEQUENCE_RCSC = ["#0C0C0D", "#6B6E73", "#A8ABAF", "#3A3C40", "#8E9195"]
pio.templates["rcsc"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12, color=_ENCRE),
    paper_bgcolor=_PAPIER_G, plot_bgcolor=_PAPIER_G,
    colorway=SEQUENCE_RCSC,
    xaxis=dict(gridcolor=_TRAIT_G, zerolinecolor=_TRAIT_G, linecolor=_TRAIT_G,
               tickfont=dict(size=11, color=_GRIS), title_font=dict(size=11, color=_GRIS)),
    yaxis=dict(gridcolor=_TRAIT_G, zerolinecolor=_TRAIT_G, linecolor=_TRAIT_G,
               tickfont=dict(size=11, color=_GRIS), title_font=dict(size=11, color=_GRIS)),
    legend=dict(font=dict(size=11), bgcolor=_PAPIER_G, borderwidth=0),
    hoverlabel=dict(font=dict(family="Inter, sans-serif", size=12),
                    bgcolor="#FFFFFF", bordercolor=_TRAIT_G),
    margin=dict(t=18, b=8, l=8, r=8),
))
pio.templates.default = "plotly_white+rcsc"
px.defaults.color_discrete_sequence = SEQUENCE_RCSC


@st.cache_data(show_spinner=False)
def _logo_inline(chemin: str, hauteur: int) -> str:
    """SVG de l'ecusson, injecte dans la page a la hauteur voulue.

    On retire la taille d'origine (305x575 pt) pour ne garder que le viewBox :
    c'est lui qui permet la mise a l'echelle. La largeur suit le ratio.
    """
    brut = Path(chemin).read_text(encoding="utf-8", errors="replace")
    brut = re.sub(r"<\?xml[^>]*\?>", "", brut)
    brut = re.sub(r"<!DOCTYPE[^>]*>", "", brut)
    brut = re.sub(r"<!--.*?-->", "", brut, flags=re.S)
    ouv = re.search(r"<svg[^>]*>", brut)
    if not ouv:
        return ""
    balise = ouv.group(0)
    vb = re.search(r'viewBox="([^"]+)"', balise)
    ratio = 0.55
    if vb:
        try:
            _, _, w, h = (float(x) for x in vb.group(1).replace(",", " ").split())
            ratio = w / h if h else ratio
        except ValueError:
            pass
    nouvelle = re.sub(r'\s(width|height)="[^"]*"', "", balise)
    largeur = round(hauteur * ratio)
    nouvelle = nouvelle.replace(
        "<svg", f'<svg height="{hauteur}" width="{largeur}" '
                f'style="height:{hauteur}px;width:{largeur}px;flex:none;display:block" '
                'role="img" aria-label="Sporting de Charleroi"', 1)
    return brut.replace(balise, nouvelle, 1).strip()


# Blason : monogramme raye dessine en SVG, pas le vrai ecusson (marque deposee,
# et un fichier binaire ne doit pas finir dans un depot public). Si un
# logo_rcsc.png/svg est pose a cote d'app.py ou a la racine, il prend le relais.
def _blason(taille: int = 34) -> str:
    """Ecusson raye du Sporting, dessine en SVG.

    Ce n'est PAS l'ecusson officiel : c'est une marque deposee, et le dossier
    deploiement/ part dans un depot public. Pose un logo_rcsc.png a cote
    d'app.py ou a la racine et marque() l'utilisera a la place.

    La taille est inscrite dans les attributs du SVG, pas seulement en CSS :
    sans feuille de style appliquee l'ecusson prenait toute la largeur.
    """
    h = round(taille * 44 / 40)
    return (
        f'<svg width="{taille}" height="{h}" viewBox="0 0 40 44" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Sporting de Charleroi" '
        f'style="width:{taille}px;height:{h}px;flex:none;display:block">'
        '<defs><clipPath id="ecu-rcsc">'
        '<path d="M3 3h34v24.5c0 7.6-7.4 11.4-17 13.5C10.4 38.9 3 35.1 3 27.5V3z"/>'
        '</clipPath></defs>'
        '<g clip-path="url(#ecu-rcsc)">'
        '<rect x="0" y="0" width="40" height="44" fill="currentColor"/>'
        '<rect x="9.5" y="0" width="4.2" height="44" fill="var(--papier,#FBFBF9)"/>'
        '<rect x="17.9" y="0" width="4.2" height="44" fill="var(--papier,#FBFBF9)"/>'
        '<rect x="26.3" y="0" width="4.2" height="44" fill="var(--papier,#FBFBF9)"/>'
        '</g>'
        '<path d="M3 3h34v24.5c0 7.6-7.4 11.4-17 13.5C10.4 38.9 3 35.1 3 27.5V3z" '
        'fill="none" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/>'
        '</svg>'
    )


def marque(sous_titre: str = "Scouting · Impect", hauteur: int = 38,
           grand: bool = False) -> None:
    """En-tete de marque : ecusson du club + nom.

    L'ecusson officiel est portrait (plus haut que large) : on raisonne en
    HAUTEUR, la largeur suit le ratio du fichier.
    """
    classe = "rcsc-marque grand" if grand else "rcsc-marque"
    ecusson = _logo_inline(str(LOGO_RCSC), hauteur) if LOGO_RCSC else ""
    if not ecusson:
        ecusson = _blason(round(hauteur * 0.78))      # repli dessine
    st.markdown(f"<div class='{classe}'>{ecusson}"
                f"<div><div class='nom'>Sporting<br>de Charleroi</div>"
                f"<div class='sous'>{sous_titre}</div></div></div>",
                unsafe_allow_html=True)


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
    # Ecran d'accueil centre : blason, rayures, puis le formulaire. Rien d'autre.
    _, _mid, _ = st.columns([1, 1.25, 1])
    with _mid:
        st.markdown("<div class='rcsc-accueil'></div>", unsafe_allow_html=True)
        marque("Plateforme de scouting", hauteur=86, grand=True)
        rayures()
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
    _, _mid, _ = st.columns([1, 1.25, 1])
    with _mid:
        with st.form("connexion", border=False):
            u = st.text_input("Utilisateur")
            m = st.text_input("Mot de passe", type="password")
            if st.form_submit_button("Se connecter", width="stretch", type="primary"):
                if _mot_de_passe_ok(u, m):
                    st.session_state["connecte"] = u
                    st.rerun()
                st.error("Identifiants incorrects.")
        st.caption("Accès réservé à la cellule recrutement.")
    st.stop()


authentifier()


@st.cache_resource
def connexion():
    if not DB.exists():
        st.error(f"Base introuvable : {DB}\n\nLance d'abord : python impect-scouting/db_build.py")
        st.stop()
    con = duckdb.connect(str(DB), read_only=True)
    # Tables du monitoring sous le prefixe "mon." : elles se joignent ainsi
    # directement aux tables de saison (fiche, score de saison).
    if MON_DB is not None:
        try:
            con.execute(f"ATTACH '{str(MON_DB).replace(chr(39), chr(39) * 2)}' AS mon (READ_ONLY)")
        except duckdb.Error:
            pass
    return con


@st.cache_data(ttl=600)
def _requete(sql: str, params: tuple = ()) -> pd.DataFrame:
    return connexion().execute(sql, params).df()


# ------------------------------------------------ poids de l'age et du niveau
# (06/10/2026, Alex) Le Score est ADDITIF dans la V10 :
#     score = score_performance + ajust_niveau + ajust_age
# et les deux ajustements sont stockes tels quels dans la base (verifie : 0 ecart
# sur 76 042 lignes). Changer leur poids est donc une multiplication exacte, sans
# relancer la pipeline ni rien telecharger : la plateforme recalcule a la volee
#     score = score_performance + k_niveau x ajust_niveau + k_age x ajust_age
# (le plafond de l'ajustement d'age est multiplie avec lui), puis les rangs et percentiles par archetype. A 100 % / 100 % rien n'est
# recalcule : les requetes lisent les colonnes de la base, comme avant.
#
# La ponderation est faite ICI, dans requete(), et pas requete par requete : toute
# lecture de v_joueurs, de fait_joueur_saison ou de mon.mon_joueur passe par une
# sous-requete qui remplace les colonnes derivees du Score. Classement, fiche,
# effectif, shortlists, recherche et monitoring restent ainsi coherents entre eux,
# y compris une requete ajoutee plus tard.
#
# Non concerne : la Qualite actuelle. Elle n'a pas d'ajustement d'age, et sa
# traduction au niveau JPL est une pente MESUREE sur les transferts, pas un poids
# choisi -- la multiplier n'aurait pas de sens.
POIDS_DEFAUT = (1.0, 1.0)      # (age, niveau), multiplicateurs des ajustements de la V10


def poids_courants() -> tuple[float, float]:
    """(k_age, k_niveau) regles dans la barre laterale, pour cette session."""
    return (st.session_state.get("poids_age", 100) / 100,
            st.session_state.get("poids_niveau", 100) / 100)


@st.cache_data(show_spinner=False)
def _colonnes(table: str) -> frozenset:
    return frozenset(r[0] for r in connexion().execute(f"DESCRIBE {table}").fetchall())


def _rangs(col: str, par: str, rang: str, pct: str | None) -> str:
    """Rang (methode min, comme la pipeline) et percentile (rang moyen / effectif,
    comme rank(pct=True) de pandas) de `col` a l'interieur de `par`."""
    sql = (f"CASE WHEN {col} IS NOT NULL THEN rank() OVER (PARTITION BY {par} "
           f"ORDER BY {col} DESC NULLS LAST) END AS {rang}")
    if pct:
        sql += (f", CASE WHEN {col} IS NOT NULL THEN round(100.0 * (rank() OVER (PARTITION BY {par} "
                f"ORDER BY {col} NULLS LAST) + (count(*) OVER (PARTITION BY {par}, {col}) - 1) / 2.0) "
                f"/ count({col}) OVER (PARTITION BY {par}), 1) END AS {pct}")
    return sql


def _table_ponderee(table: str, ka: float, kn: float) -> str:
    """Sous-requete equivalente a `table`, Score et derives recalcules."""
    cols = _colonnes(table)
    # Ecrit a partir du Score STOCKE (score + (k - 1) x ajustement) et non
    # recompose depuis la performance : la pipeline arrondit le Score apres
    # l'addition, ses trois termes separement, et la somme recomposee s'en
    # ecarte de 0,01 sur un tiers des lignes. Ainsi 100 % redonne la base au
    # centieme pres, et le rang ne saute pas en quittant le reglage d'origine.
    d_niv, d_age = f"ajust_niveau * {kn - 1!r}", f"ajust_age * {ka - 1!r}"
    score = f"round(score + {d_niv} + {d_age}, 2)"
    # "+ 0.0" : un ajustement negatif multiplie par 0 donne -0.0, que la fiche
    # afficherait « -0.0 âge ».
    valeurs = [f"round(ajust_age * {ka!r}, 2) + 0.0 AS ajust_age",
               f"round(ajust_niveau * {kn!r}, 2) + 0.0 AS ajust_niveau", f"{score} AS score"]
    if table == "mon.mon_joueur":
        rangs = [_rangs("score", "periode_jours, archetype", "rang", None)]
    else:
        rangs = [_rangs("score", "archetype", "rang_archetype", "score_percentile")]
        if "score_sans_age" in cols:
            # Pas d'arrondi ici : la base stocke score - ajust_age tel quel, et
            # ses rangs sont calcules sur cette valeur non arrondie.
            valeurs.append(f"score_sans_age + {d_niv} AS score_sans_age")
            rangs.append(_rangs("score_sans_age", "archetype", "rang_sans_age", "score_sans_age_percentile"))
    return (f"(SELECT * REPLACE ({', '.join(rangs)}) "
            f"FROM (SELECT * REPLACE ({', '.join(valeurs)}) FROM {table}))")


_TABLES_PONDEREES = re.compile(
    r"\b(FROM|JOIN)\s+(v_joueurs|fait_joueur_saison|mon\.mon_joueur)\b(\s+\w+)?", re.IGNORECASE)
# Mot qui suit le nom de la table : si c'est un de ceux-ci, la table n'avait pas
# d'alias et la sous-requete prend son nom (les requetes ecrivent v_joueurs.playerId).
_MOTS_SQL = {"where", "left", "right", "inner", "cross", "join", "order", "group", "limit",
             "using", "on", "qualify", "union"}


def sql_pondere(sql: str, ka: float, kn: float) -> str:
    def remplace(m: re.Match) -> str:
        mot, table, suite = m.group(1), m.group(2), m.group(3) or ""
        sous = _table_ponderee(table.lower(), ka, kn)
        if suite.strip() and suite.strip().lower() not in _MOTS_SQL:
            return f"{mot} {sous}{suite}"                      # alias deja ecrit dans la requete
        return f"{mot} {sous} AS {table.split('.')[-1]}{suite}"
    return _TABLES_PONDEREES.sub(remplace, sql)


def requete(sql: str, params: tuple = ()) -> pd.DataFrame:
    poids = poids_courants()
    if poids != POIDS_DEFAUT:
        sql = sql_pondere(sql, *poids)
    return _requete(sql, params)


# ----------------------------------------------------------------- exports
def en_csv(df: pd.DataFrame) -> bytes:
    """CSV en UTF-8 AVEC marque d'ordre (BOM). Sans elle, Excel ouvre le fichier
    en Windows-1252 et les accents sortent en « JeremÃ­as LÃ¡zaro », les smileys
    de priorite en « ðŸ”´ ». Les autres logiciels ignorent cette marque."""
    return df.to_csv(index=False).encode("utf-8-sig")


try:
    import openpyxl  # noqa: F401
    EXCEL_DISPO = True
except ImportError:          # environnement sans openpyxl : le bouton Excel n'est pas propose
    EXCEL_DISPO = False


def en_excel(df: pd.DataFrame, feuille: str) -> bytes:
    """Classeur .xlsx d'une seule feuille : en-tete figee, filtre automatique,
    colonnes a la largeur de leur contenu. Le format est Unicode de bout en bout :
    accents et smileys passent sans reglage d'encodage."""
    import io
    from openpyxl.utils import get_column_letter
    d = df.copy()
    for c in d.columns:      # Excel refuse les dates avec fuseau horaire
        if isinstance(d[c].dtype, pd.DatetimeTZDtype):
            d[c] = d[c].dt.tz_localize(None)
    # Nom de feuille : 31 caracteres au plus, sans  [ ] : * ? / \
    feuille = re.sub(r"[\[\]:*?/\\]", " ", feuille)
    feuille = re.sub(r" +", " ", feuille).strip()[:31].strip() or "Export"
    tampon = io.BytesIO()
    with pd.ExcelWriter(tampon, engine="openpyxl") as xw:
        d.to_excel(xw, sheet_name=feuille, index=False)
        ws = xw.sheets[feuille]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for i, c in enumerate(d.columns, start=1):
            # Longueur calculee valeur par valeur : sur une colonne entierement
            # vide, .str.len().max() rend NaN, et int(NaN) fait planter l'export
            # (cas d'une liste dont aucun joueur n'est plus dans la base).
            contenu_max = max((len(str(v)) for v in d[c] if pd.notna(v)), default=0)
            largeur = max(len(str(c)), contenu_max) + 2
            ws.column_dimensions[get_column_letter(i)].width = min(max(largeur, 8), 60)
    return tampon.getvalue()


@st.cache_data(ttl=600)
def listes():
    comp = requete("""SELECT competition, pays, niveau, top5_europe
                      FROM dim_competition ORDER BY rating_moyen DESC""")
    saisons = requete("SELECT DISTINCT season FROM dim_saison ORDER BY season DESC")["season"].tolist()
    # Constantes du run (seuils, fiabilite...) : les phrases de la fiche les
    # citent, elles ne doivent pas diverger de Charleroi_MultiPoste_ScoreV10.py.
    params = requete("SELECT * FROM parametres").iloc[0]
    return comp, saisons, params


comp_df, saisons, PARAMS = listes()
genere_le = PARAMS["genere_le"]
# Profils RCSC et jeu sous pression : calcules des MINUTES_PROFILS, a lire avec
# precaution sous MINUTES_FIABLES (cf. profils_rcsc.py : sous 900 minutes le
# joueur est situe dans le pool de reference sans en faire partie). Une base
# d'avant le 06/10/2026 n'a pas ces parametres : les profils y commencent a 900
# minutes, les deux seuils se confondent et aucune alerte ne s'affiche.
MINUTES_PROFILS = int(PARAMS.get("profils_minutes_min", 900))
MINUTES_FIABLES = int(PARAMS.get("minutes_fiables", 900))


def echantillon_reduit(minutes) -> bool:
    """Le joueur a-t-il un profil et un jeu sous pression calcules sur moins de
    MINUTES_FIABLES minutes ? C'est ce qui declenche l'alerte de la fiche."""
    return bool(pd.notna(minutes) and MINUTES_PROFILS <= minutes < MINUTES_FIABLES)


@st.cache_data(ttl=600)
def params_monitoring():
    """Parametres du monitoring (date des donnees, periodes precalculees), None
    si la base du monitoring est absente."""
    try:
        return requete("SELECT * FROM mon.mon_parametres").iloc[0]
    except (duckdb.Error, IndexError):
        return None


MON = params_monitoring()

# ----------------------------------------------------------------- onglets
# Onglets a execution paresseuse : seul l'onglet ouvert calcule sa page, et la
# barre laterale s'adapte (periode et minutes sur la periode en Monitoring).
onglet_saison, onglet_mon, onglet_pression, onglet_rcsc = st.tabs(
    ["Scouting saison", "Monitoring", "Jeu sous pression", "Sporting de Charleroi"],
    key="onglet", on_change="rerun")
MONITORING = bool(onglet_mon.open)
# Onglet « Jeu sous pression » : classements des metriques de l'etude de pression.
# Comme l'onglet maison, il a ses propres filtres et n'utilise pas le classement.
VUE_PRESSION = bool(onglet_pression.open)
# Onglet maison : effectif du RSC Charleroi et equipe type. Il n'utilise aucun
# des filtres de la barre laterale (poste, championnat, minutes...), on evite
# donc la grosse requete de classement quand il est ouvert.
VUE_RCSC = bool(onglet_rcsc.open)

# ----------------------------------------------------------------- filtres
with st.sidebar:
    marque("Scouting · Impect", hauteur=46)
    rayures(fine=True)
    st.caption(f"Données du {genere_le} · {st.session_state['connecte']}")
    if st.button("Se déconnecter", width="stretch"):
        st.session_state.clear(); st.rerun()
    st.markdown("**Recherche & poste**")

# Libelle inclus dans l'option (plutot que format_func) : identique a l'ecran,
# mais pilotable par les tests automatises de Streamlit.
_OPT_ARCH = [f"{a} — {n}" for a, n in ARCHETYPES.items()]
# Saut de poste demande ailleurs dans la page (recherche globale, shortlist) :
# un widget deja affiche ne peut plus etre modifie, la valeur est donc posee
# ici, AVANT la creation du selecteur, et consommee au rerun.
if _a := st.session_state.pop("_archetype_a_appliquer", None):
    for _o in _OPT_ARCH:
        if _o.startswith(f"{_a} —"):
            st.session_state["archetype_sel"] = _o
archetype = st.sidebar.selectbox(
    "Poste / archétype", _OPT_ARCH,
    index=1 if "archetype_sel" not in st.session_state else None,
    key="archetype_sel").split(" — ")[0]

# Monitoring : fenetre de N jours avant la date des donnees (derniere
# synchronisation du run V10), precalculee pour chaque N de PERIODES.
if MONITORING and MON is not None:
    _fin_mon = pd.Timestamp(MON["date_reference"])
    _periodes = [int(p) for p in str(MON["periodes"]).split(",")]
    periode = st.sidebar.select_slider(
        "Période monitorée (jours)", _periodes, value=int(MON["periode_defaut"]),
        help=f"Matchs joués dans les N jours précédant le {_fin_mon:%d/%m/%Y}, date des dernières "
             "données de matchs publiées.")
    st.sidebar.caption(f"Du {_fin_mon - pd.Timedelta(days=periode):%d/%m/%Y} au {_fin_mon:%d/%m/%Y}")

# V10 : Score (formule V8) = lecture par defaut, Qualite actuelle en second
# (cf. bloc VERSION 10 du pipeline).
LECTURES = {
    "Score (performance + niveau + âge)": dict(
        col="score", rang="rang_archetype", pct="score_percentile", court="Score",
        aide="Performance terrain + ajustement linéaire du niveau du club + ajustement d'âge "
             "(formule V8). Avantage les joueurs des gros clubs."),
    "Qualité actuelle (niveau JPL)": dict(
        col="qualite_actuelle", rang="rang_qualite", pct="qualite_percentile", court="Qualité",
        aide="Performance projetée au niveau d'un club moyen de JPL, sans l'âge : "
             "ce que le joueur vaut aujourd'hui."),
    # 06/10/2026 (Alex) : le Score V8 prive de son seul ajustement d'age. Celui-ci
    # etant ADDITIF dans la formule (score = base + excellence - fragilite
    # + ajust_niveau + ajust_age), le retirer est une soustraction exacte, pas
    # une approximation -- aucune formule nouvelle n'est introduite.
    "Score sans l'âge (performance + niveau)": dict(
        col="score_sans_age", rang="rang_sans_age", pct="score_sans_age_percentile",
        court="Score hors âge",
        aide="Le Score privé de son seul ajustement d'âge : même performance, même ajustement "
             "du niveau du club, mais un joueur de 31 ans n'est plus pénalisé pour son âge. "
             "La lecture à prendre pour chercher un profil expérimenté."),
}
@st.cache_data(show_spinner=False)
def buts_dispo() -> bool:
    """La base contient-elle les buts / passes decisives ?

    Ajoutes le 06/10/2026 (db_build.buts_passes). Une base construite avant ne
    les a pas : la fiche doit continuer a s'afficher sans eux plutot que de
    planter sur une colonne absente.
    """
    try:
        requete("SELECT buts FROM v_joueurs LIMIT 1")
        return True
    except duckdb.Error:
        return False


# Une base d'avant le 06/10/2026 n'a pas ces colonnes : sans ce retrait, choisir
# cette lecture ferait echouer toutes les requetes de classement.
try:
    requete("SELECT score_sans_age FROM v_joueurs LIMIT 1")
    SANS_AGE = True
except duckdb.Error:
    SANS_AGE = False
    LECTURES.pop("Score sans l'âge (performance + niveau)", None)
# Monitoring : une seule lecture, le Score (la Qualite actuelle repose sur une
# calibration de saison) -- les fiches de saison ouvertes depuis l'onglet
# restent donc sur le Score.
if MONITORING:
    lecture = list(LECTURES)[0]
else:
    lecture = st.sidebar.radio("Lecture", list(LECTURES),
                               help="  \n".join(f"**{k}** : {v['aide']}" for k, v in LECTURES.items()))
SC, RG, PCT, COURT = (LECTURES[lecture][k] for k in ("col", "rang", "pct", "court"))

# Poids de l'age et du niveau du club dans le Score (cf. requete()). Reglage de
# session : il vaut pour toute la plateforme tant que l'on reste connecte, et
# revient a 100 % a la connexion suivante.
K_AGE, K_NIVEAU = poids_courants()
POIDS_MODIFIES = (K_AGE, K_NIVEAU) != POIDS_DEFAUT


def _poids_origine() -> None:
    st.session_state["poids_age"] = st.session_state["poids_niveau"] = 100


with st.sidebar.expander("⚖️ Poids de l'âge et du niveau" + (" · modifiés" if POIDS_MODIFIES else ""),
                         expanded=POIDS_MODIFIES):
    # Valeurs posees dans la session avant la creation des curseurs (et non en
    # valeur par defaut du widget) : le bouton de retour les modifie par la session.
    st.session_state.setdefault("poids_age", 100)
    st.session_state.setdefault("poids_niveau", 100)
    st.slider("Poids de l'âge", 0, 200, step=10, format="%d %%", key="poids_age",
              help="100 % = réglage d'origine. 0 % = l'âge ne compte plus dans le Score. "
                   "200 % = un joueur jeune est deux fois plus avantagé, un joueur âgé deux fois "
                   "plus pénalisé.")
    st.slider("Poids du niveau du club", 0, 200, step=10, format="%d %%", key="poids_niveau",
              help="100 % = réglage d'origine. 0 % = le niveau du club et de son championnat ne compte "
                   "plus : le Score se rapproche de la performance pure. 200 % = l'écart entre gros et "
                   "petits clubs est doublé.")
    _pente_age = float(PARAMS.get("age_pente", 1.6)) * K_AGE
    _plafond_age = float(PARAMS.get("age_plafond", 12)) * K_AGE
    _pente_niv = float(PARAMS.get("niveau_pente", 40)) * K_NIVEAU
    st.caption(
        f"**Âge** : {_pente_age:.1f} point par année d'écart à {float(PARAMS.get('age_pivot', 24)):.0f} ans "
        f"(27 pour un gardien), plafonné à ±{_plafond_age:.0f}.  \n"
        f"**Niveau** : {_pente_niv:.0f} points par point de rating du club, autour de "
        f"{float(PARAMS.get('niveau_reference', 0.49)):.2f}.  \n"
        "Score, Score hors âge, rangs et percentiles sont recalculés partout, monitoring compris. "
        "La Qualité actuelle ne change pas : sa traduction au niveau JPL est mesurée, pas réglée.")
    st.button("Revenir aux poids d'origine", on_click=_poids_origine, disabled=not POIDS_MODIFIES,
              width="stretch")


CHAMPS = f"""nom, club, competition, pays, niveau, saison, round({SC},1) AS score,
    round(qualite_actuelle,1) AS qualite, round(score,1) AS score_v8,
    round(ajust_traduction,1) AS aj_traduction, round(ajust_niveau,1) AS aj_niveau,
    role_milieu, round(age_years,1) AS age, minutes_jouees AS minutes, pied_fort,
    round(score_performance,1) AS performance,
    round(ajust_age,1) AS aj_age, round(progression_credible,1) AS progression,
    round(adv_ecart_haut_percentile,0) AS gros_matchs, round(opp_coef_avg,3) AS coef_adv,
    {RG} AS rang_mondial, round({PCT},1) AS percentile,
    round(base,1) AS base, round(excellence,1) AS excellence, round(fragilite,1) AS fragilite,
    pilier_fort, pilier_faible, taille_cm, n_matches_oppw AS matchs,
    position AS poste, side, round(attdef_coef_att_avg,3) AS coef_att,
    round(attdef_coef_def_avg,3) AS coef_def, round(club_rating,3) AS rating_club,
    round(competition_avg_rating,3) AS rating_ligue,
    round(gros_matchs_delta,1) AS gros_matchs_delta, archetype AS archetype_courant,
    profil_principal,
    playerId, squadId, iterationId, position"""
# Buts et passes decisives : ajoutes a la liste SEULEMENT si la base les
# contient (cf. buts_dispo). Une base d'avant le 06/10/2026 n'a pas ces
# colonnes, et toutes les requetes de la plateforme passent par CHAMPS : les
# demander sans condition casserait l'application entiere.
if buts_dispo():
    CHAMPS += ", buts, passes_d, buts_penalty"
if SANS_AGE:
    CHAMPS += ", round(score_sans_age,1) AS score_sans_age"


@st.cache_data(ttl=600)
@st.cache_data(show_spinner=False)
def pression_dispo() -> bool:
    """La base contient-elle le jeu sous pression (cf. Etude_Pression_Passes) ?"""
    try:
        requete("SELECT pct_part_prog_fp FROM v_joueurs LIMIT 1")
        return True
    except duckdb.Error:
        return False


@st.cache_data(show_spinner=False)
def pression_fiabilite() -> dict:
    """Fidelite mesuree de chaque metrique de pression (matchs pairs vs impairs,
    corrigee Spearman-Brown). Affichee dans la fiche : au-dessus de 0,6 la
    mesure decrit le joueur, en dessous elle decrit surtout son equipe."""
    try:
        f = requete("SELECT metrique, avg(fiabilite_saison) AS f FROM fiabilite_pression "
                    "GROUP BY metrique")
        return dict(zip(f["metrique"], f["f"]))
    except duckdb.Error:
        return {}


@st.cache_data(show_spinner=False)
def pression_fiabilite_reduite() -> dict:
    """Meme fidelite, mesuree sur les seuls joueurs entre 400 et 900 minutes
    (colonne ajoutee le 06/10/2026 ; vide sur une base plus ancienne)."""
    try:
        f = requete("SELECT metrique, avg(fiabilite_sous_900) AS f FROM fiabilite_pression "
                    "GROUP BY metrique HAVING avg(fiabilite_sous_900) IS NOT NULL")
        return dict(zip(f["metrique"], f["f"]))
    except duckdb.Error:
        return {}


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
# Profils RCSC et saison : propres au scouting de saison (les profils sont
# calcules sur une saison complete).
profil_id, seuil_correspondance, tri_profil, saison = None, 50, COURT, []
_choix_profil = "Tous les profils"
if MONITORING:
    tri_monitoring = st.sidebar.radio(
        "Classer par", ["Score", "Performance"], horizontal=True,
        help="**Score** : performance sur la période + niveau du club + âge (formule du score de "
             "saison). **Performance** : la performance terrain seule.")
else:
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
    # joueurs du profil par la lecture choisie, la correspondance reste affichee.
    tri_profil = st.sidebar.radio("Classer les joueurs du profil par", [COURT, "Correspondance"],
                                  horizontal=True, disabled=profil_id is None)
    saison = st.sidebar.multiselect("Saison", saisons, default=[s for s in saisons if s in ("25/26", "2026")])
recherche = st.sidebar.text_input(
    "Recherche par nom", placeholder="ex. Beitia",
    help="Cherche dans TOUS les postes. Les résultats s'affichent juste en dessous : "
         "un clic ouvre la fiche et bascule le classement sur le bon poste.")


@st.cache_data(show_spinner=False)
def chercher_partout(texte: str, saisons_filtre: tuple, poids: tuple = POIDS_DEFAUT) -> pd.DataFrame:
    """Tous les joueurs-saison dont le nom correspond, TOUS POSTES confondus.

    L'ancien comportement ne cherchait que dans l'archetype ouvert : un joueur
    classe ailleurs restait introuvable sans avoir devine son poste. Ici on
    interroge la base entiere, et un joueur qui existe sous plusieurs
    archetypes sort autant de fois -- a lui de choisir lequel il veut voir,
    plutot que de deviner a sa place (un CB qui apparait aussi en FB n'est pas
    le meme dossier).

    poids : sert seulement de cle de cache (le Score depend des poids regles
    dans la barre laterale, appliques par requete()).
    """
    ou, pa = ["lower(nom) LIKE ?"], [f"%{texte.lower()}%"]
    if saisons_filtre:
        ou.append(f"saison IN ({','.join('?' * len(saisons_filtre))})"); pa += list(saisons_filtre)
    return requete(f"""
        SELECT nom, club, competition, saison, archetype, position,
               round(score,1) AS score, round(score_percentile,1) AS pct,
               round(age_years,1) AS age, minutes_jouees AS minutes, rang_archetype AS rang,
               playerId, squadId, iterationId
        FROM v_joueurs WHERE {' AND '.join(ou)}
        ORDER BY score DESC LIMIT 60""", tuple(pa))


if recherche and len(recherche.strip()) >= 2:
    _trouves = chercher_partout(recherche.strip(), tuple(saison), poids_courants())
    with st.sidebar.container(border=True):
        if _trouves.empty:
            st.caption(f"Aucun joueur pour « {recherche} »"
                       + (" sur les saisons sélectionnées." if saison else "."))
        else:
            _joueurs = _trouves["nom"].nunique()
            _postes = sorted(_trouves["archetype"].unique())
            st.caption(f"**{len(_trouves)} résultat(s)** · {_joueurs} joueur(s) · "
                       f"postes : {', '.join(_postes)}")
            if len(_postes) > 1:
                st.caption("⚠️ Ce nom sort sur plusieurs postes : choisis la ligne qui t'intéresse.")
            for _r in _trouves.head(20).itertuples():
                _ici = _r.archetype == archetype
                _sc = f"{_r.score:.0f}" if pd.notna(_r.score) else "—"
                if st.button(
                        f"{'📍 ' if _ici else ''}{_r.nom} · **{_r.archetype}** — {_r.club} "
                        f"({_r.saison}) · {_sc}",
                        key=f"go_{_r.playerId}_{_r.squadId}_{_r.iterationId}_{_r.archetype}",
                        width="stretch",
                        help=f"{_r.competition} · {_r.position} · rang {_r.rang} de son poste · "
                             f"{_r.minutes:.0f} min · {_r.age:.0f} ans"):
                    # Bascule le classement sur le bon poste ET ouvre la fiche :
                    # le bouton "Retour" ramene donc sur la bonne liste.
                    # (ouvrir_fiche() est definie plus bas dans le fichier, on
                    # pose directement le parametre d'URL qu'elle utilise.)
                    st.session_state["_archetype_a_appliquer"] = _r.archetype
                    st.query_params["fiche"] = "~".join(str(v) for v in (
                        int(_r.playerId), int(_r.squadId), int(_r.iterationId),
                        _r.position, _r.archetype))
                    st.rerun()
            if len(_trouves) > 20:
                st.caption(f"…et {len(_trouves) - 20} autres : précise le nom.")

st.sidebar.markdown("**Championnats**")
exclure_top5 = st.sidebar.checkbox("Exclure les 5 grands championnats", value=False)
# La MLS seule : MLS Next Pro (reserves) et les USL sont d'autres championnats,
# ils restent dans la liste.
exclure_mls = st.sidebar.checkbox("Exclure la MLS", value=False)
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
    # Annotations de scouting (ajoutees 02/10/2026) : une note libre et une
    # priorite par joueur dans une liste. En ADD COLUMN IF NOT EXISTS pour ne
    # rien casser sur les bases deja remplies (syntaxe commune DuckDB/Postgres).
    for _col, _type in (("note", "VARCHAR"), ("priorite", "VARCHAR")):
        c.execute(f"ALTER TABLE listes ADD COLUMN IF NOT EXISTS {_col} {_type}")
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
        "SELECT nom, club, poste, archetype, score, playerId, note, priorite, ajoute_le "
        "FROM listes WHERE liste=? ORDER BY score DESC", [liste]),
        columns=["nom", "club", "poste", "archetype", "score", "playerId",
                 "note", "priorite", "ajoute_le"])


PRIORITES = ["", "🔴 Priorité", "🟡 À suivre", "🔵 Vivier"]
# Export Excel d'une liste : colonne de la liste -> intitule, dans l'ordre du fichier.
EXPORT_LISTE = {
    "priorite": "Priorité", "nom": "Nom", "archetype": "Poste", "club": "Club",
    "competition": "Championnat", "pays": "Pays", "saison": "Saison", "score_actuel": "Score",
    "performance": "Performance", "qualite": "Qualité actuelle", "progression": "Progression",
    "age": "Âge", "minutes": "Minutes", "pied_fort": "Pied fort", "taille_cm": "Taille (cm)",
    "profil_principal": "Profil RCSC", "pilier_fort": "Point fort", "pilier_faible": "Point faible",
    "rang": "Rang mondial", "note": "Note", "ajoute_le": "Ajouté le",
}


def annoter(liste: str, player_id: int, note: str | None, priorite: str | None) -> None:
    sql_listes("UPDATE listes SET note=?, priorite=? WHERE liste=? AND playerId=?",
               [note or None, priorite or None, liste, int(player_id)])


def deplacer(src: str, dest: str, player_ids: list[int], copier: bool = False) -> int:
    """Copie (ou deplace) des joueurs d'une liste vers une autre.

    Ecrase la cible pour ces joueurs plutot que de creer un doublon : la meme
    ligne deux fois dans une liste n'a aucun sens et casse les retraits.
    """
    n = 0
    for pid in player_ids:
        ligne = sql_listes("SELECT nom, club, poste, archetype, score, note, priorite "
                           "FROM listes WHERE liste=? AND playerId=?", [src, int(pid)])
        if not ligne:
            continue
        nom, club, poste, arch, score, note, prio = ligne[0]
        sql_listes("DELETE FROM listes WHERE liste=? AND playerId=?", [dest, int(pid)])
        sql_listes("INSERT INTO listes (liste, playerId, nom, club, poste, archetype, score, "
                   "ajoute_le, note, priorite) VALUES (?,?,?,?,?,?,?,now(),?,?)",
                   [dest, int(pid), nom, club, poste, arch, score, note, prio])
        if not copier:
            sql_listes("DELETE FROM listes WHERE liste=? AND playerId=?", [src, int(pid)])
        n += 1
    return n


def ids_exclus() -> list[int]:
    return [r[0] for r in sql_listes("SELECT DISTINCT playerId FROM listes WHERE liste='exclus'")]


def ajouter(liste: str, j) -> None:
    sql_listes("DELETE FROM listes WHERE liste=? AND playerId=?", [liste, int(j.playerId)])
    # Colonnes nommees : la table gagne des colonnes au fil des versions
    # (note, priorite), un INSERT positionnel casserait a la prochaine.
    sql_listes("INSERT INTO listes (liste, playerId, nom, club, poste, archetype, score, ajoute_le) "
               "VALUES (?,?,?,?,?,?,?,now())",
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
NOUVELLE_LISTE = "+ nouvelle liste…"
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
    with st.sidebar.expander("Gérer la liste"):
        st.caption("Les changements s'appliquent à **tous les comptes**.")
        nouveau_nom = st.text_input("Nouveau nom", value=shortlist, key=f"nouveau_nom_{shortlist}")
        if st.button("Renommer", key=f"renommer_{shortlist}", width="stretch"):
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
        if st.button("Supprimer définitivement", key=f"supprimer_{shortlist}",
                     disabled=not confirme, width="stretch"):
            supprimer_liste(shortlist, st.session_state["connecte"])
            st.session_state["_liste_suivante"] = ""          # -> premiere liste restante
            st.session_state["_message_listes"] = f"« {shortlist} » supprimée pour tous les comptes"
            st.rerun()
masquer_exclus = st.sidebar.checkbox("Masquer les joueurs exclus", value=True)

st.sidebar.markdown("**Profil**")
age_max = st.sidebar.slider("Âge maximum", 16, 40, 40)
if MONITORING:
    # Seuil d'entree du monitoring : 30 min au poste sur la periode (cf.
    # Charleroi_MultiPoste_MonitoringV10.MINUTES_MIN).
    _min_mon = int(MON["minutes_min"]) if MON is not None else 30
    minutes_min = st.sidebar.slider("Minutes minimum sur la période", _min_mon, 1200, _min_mon, step=30)
else:
    minutes_min = st.sidebar.slider(
        "Minutes minimum", 400, 3000, 900, step=100,
        help=f"Sous {MINUTES_FIABLES} minutes, le profil RCSC et le jeu sous pression d'un joueur "
             "reposent sur un échantillon réduit : ils sont affichés, avec une alerte sur la fiche "
             "et un ⚠ dans le classement." if MINUTES_PROFILS < MINUTES_FIABLES else None)
pieds = st.sidebar.multiselect("Pied fort", ["droit", "gauche", "les deux"])
score_min = st.sidebar.slider(f"{COURT} minimum", 0, 100, 0, step=5)

# Jeu sous pression (Etude_Pression_Passes) : filtre de STYLE, il ne change
# aucun score ni aucun classement, il restreint seulement la liste.
PRESSION_FILTRES = {
    "—": None,
    "Progresse sous pression (P≥70)": "pct_part_prog_fp >= 70",
    "Garde son ambition sous pression (P≥70)": "pct_part_prog_fp >= 70 AND maintien_ambition >= 0.8",
    "Résiste sous pression (P≥70)": "pct_reussite_fp_vs_attendu_100 >= 70",
    "Élimine des adversaires (P≥70)": "pct_bypassed_vs_attendu_p90 >= 70",
    "Prudent sous pression (P≤30)": "pct_part_prog_fp BETWEEN 0 AND 30",
}
filtre_pression = "—"
if not MONITORING and pression_dispo():
    filtre_pression = st.sidebar.selectbox(
        "Jeu sous pression", list(PRESSION_FILTRES),
        help="Filtre de style, descriptif : il ne modifie ni le score ni le classement. "
             "Percentiles calculés par poste sur une centaine de championnats.")

# Memes filtres pour les deux onglets ; seuls le volume (saison / periode) et
# l'age different : le monitoring garde les joueurs sans date de naissance
# tant que l'age maximum n'est pas regle.
if MONITORING:
    where, params = ["archetype = ?", "periode_jours = ?", "minutes >= ?", "score >= ?"], \
                    [archetype, periode if MON is not None else 0, minutes_min, score_min]
    if age_max < 40:
        where.append("age_years <= ?"); params.append(age_max)
else:
    where, params = ["archetype = ?", "minutes_jouees >= ?", "age_years <= ?", f"{SC} >= ?"], \
                    [archetype, minutes_min, age_max, score_min]
if saison:
    where.append(f"saison IN ({','.join('?' * len(saison))})"); params += saison
if recherche:
    where.append("lower(nom) LIKE ?"); params.append(f"%{recherche.lower()}%")
if exclure_top5:
    where.append("NOT top5_europe")
if exclure_mls:
    where.append("competition <> 'Major League Soccer'")
if pays:
    where.append(f"pays IN ({','.join('?' * len(pays))})"); params += pays
if niveaux:
    where.append(f"niveau IN ({','.join('?' * len(niveaux))})"); params += [float(n) for n in niveaux]
if pieds:
    where.append(f"pied_fort IN ({','.join('?' * len(pieds))})"); params += pieds
if PRESSION_FILTRES.get(filtre_pression):
    where.append(PRESSION_FILTRES[filtre_pression])
_exclus = ids_exclus()
if masquer_exclus and _exclus:
    where.append(f"playerId NOT IN ({','.join('?' * len(_exclus))})"); params += _exclus

# Pagination : on ne descend jamais plus de PAR_PAGE lignes de la base, sinon
# l'affichage d'un archetype peu filtre (plusieurs milliers de joueurs) rame.
# Le classement reste global : la page 2 donne bien les 501e a 1000e meilleurs.
PAR_PAGE = 500


def page_courante(filtre: str) -> int:
    """Page affichee ; tout changement de filtre ramene a la page 1."""
    if st.session_state.get("_filtre_courant") != filtre:
        st.session_state["_filtre_courant"] = filtre
        st.session_state["page"] = 0
    return st.session_state.get("page", 0)


if not MONITORING and not VUE_RCSC and not VUE_PRESSION:
    # Profil RCSC : meme joueur-saison, meme pool. Plus de verdict : filtre sur
    # le score de correspondance lui-meme (cf. profils_rcsc.py).
    _SOUS_PROFIL = """FROM fait_profil p WHERE p.playerId = v_joueurs.playerId AND p.squadId = v_joueurs.squadId
        AND p.iterationId = v_joueurs.iterationId AND p.position = v_joueurs.position
        AND p.archetype = v_joueurs.archetype AND p.profil_id = ?"""
    _extra_select, _extra_params, _ordre = "", [], f"{SC} DESC"
    if profil_id:
        where.append(f"EXISTS (SELECT 1 {_SOUS_PROFIL} AND p.correspondance >= ?)")
        params += [profil_id, seuil_correspondance]
        _extra_select = f", (SELECT round(p.correspondance, 0) {_SOUS_PROFIL}) AS corr_profil"
        _extra_params = [profil_id]
        _ordre = f"corr_profil DESC, {SC} DESC" if tri_profil == "Correspondance" else f"{SC} DESC, corr_profil DESC"

    page = page_courante(f"{' AND '.join(where)}|{params}")

    # Les statistiques portent sur TOUS les joueurs filtres, pas sur la page affichee.
    stats = requete(f"""SELECT count(*) AS n, median({SC}) AS med, max({SC}) AS max_,
                               median(age_years) AS age_med
                        FROM v_joueurs WHERE {' AND '.join(where)}""", tuple(params)).iloc[0]
    total = int(stats["n"])
    n_pages = max(1, -(-total // PAR_PAGE))   # division entière arrondie au-dessus
    page = min(page, n_pages - 1)
    st.session_state["page"] = page

    # Le classement lit EXACTEMENT les memes colonnes que la fiche : c'est de
    # cette requete que sort la ligne passee a carte_joueur quand on clique un
    # joueur. Elle dupliquait la liste a la main, et chaque colonne ajoutee a
    # CHAMPS (buts, passes decisives, score hors age...) manquait donc dans la
    # fiche ouverte depuis le classement -- le chemin le plus frequent.
    res = requete(f"""
        SELECT {CHAMPS}{_extra_select}
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
    h = requete(f"""SELECT saison, club, competition, position AS poste, round({SC},1) AS score,
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
# n'entre dans la qualite actuelle que via la traduction au niveau JPL) -- un
# 60 en Equateur ne vaut pas un 60 en JPL.
COULEUR_AUTRE_CONTEXTE = "#C27C2C"


def lignes_forme_etendue(cle: tuple, competition: str) -> pd.DataFrame:
    """Blocs de forme de la fiche et de toutes les saisons du joueur au meme
    poste. contexte : fiche / meme (club + division) / autre."""
    player_id, squad_id, iteration_id, position, arch = cle
    b = requete("""SELECT t.*, s.season AS saison, s.competition, c.club, f.ajust_traduction
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
    reperes = ", ".join(f"{r.ajust_traduction:+.1f} en {r.saison} à {r.club}" for r in autres.itertuples())
    return (f"  \n**En orange : {liste}**, au même poste mais dans un autre club et/ou un autre "
            "championnat. ⚠️ Le niveau y est corrigé de la difficulté de chaque match, **pas de la "
            "force du championnat** : un même chiffre ne vaut pas la même chose dans deux ligues de "
            "force différente. Repère, la traduction au niveau JPL de sa qualité actuelle (force du "
            f"club et du championnat) : {d.ajust_traduction:+.1f} en {saison_fiche} ici, {reperes}. Les courbes ne "
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


def section_profils(cle: tuple, minutes=None) -> None:
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
        st.caption(f"Profils calculés à partir de {MINUTES_PROFILS} minutes jouées à ce poste.")
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
    if echantillon_reduit(minutes):
        notes.append(f"⚠️ **Moins de {MINUTES_FIABLES} minutes** : ses correspondances sont situées par "
                     f"rapport aux joueurs à {MINUTES_FIABLES} minutes et plus, dont il ne fait pas partie. "
                     "Elles lisent des percentiles de piliers calculés sur moins de dix matchs pleins : "
                     "un écart de quelques points entre deux profils ne les départage pas.")
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
                 "touches, jeu en retrait, jeu sous pression) est descriptive, elle ne compte pas dans le "
                 "score.")
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


PRESSION_LIGNES = [
    ("pct_pression_passe", "pression_passe", "Pression subie à la passe", "{:.1f}",
     "Pression adverse moyenne (0-100) au moment où il donne le ballon. Décrit surtout son rôle "
     "et sa zone de jeu."),
    ("pct_part_prog_fp", "part_prog_fp", "Passes progressives sous pression", "{:.0f} %",
     "Part de ses passes sous forte pression (≥ 50) qui gagnent du terrain vers le but. C'est "
     "l'audace : sous pression, la moyenne des championnats tombe à 23 %."),
    (None, "maintien_ambition", "Maintien de l'ambition", "{:.2f}",
     "Cette part rapportée à la même part sans pression. 1,00 = il joue pareil pressé ou libre ; "
     "0,69 est la médiane des championnats."),
    ("pct_reussite_fp_vs_attendu_100", "reussite_fp_vs_attendu_100", "Réussite sous pression vs attendu",
     "{:+.1f}", "Passes réussies au-dessus de l'attendu, pour 100 passes sous forte pression, à poste, "
     "zone et distance comparables."),
    ("pct_bypassed_vs_attendu_p90", "bypassed_vs_attendu_p90", "Adversaires éliminés vs attendu / 90",
     "{:+.1f}", "Packing : adversaires entre le ballon et le but qu'il supprime par ses passes, "
     "au-dessus de l'attendu."),
    ("pct_prog_fp_p90", "prog_fp_p90", "Passes progressives sous pression / 90", "{:.1f}",
     "Volume brut, très dépendant du volume de jeu de l'équipe."),
    (None, "entrees_surface_p90", "Entrées dans la surface / 90", "{:.2f}",
     "Passes réussies qui font entrer le ballon dans la surface adverse."),
]


# Sous cette fidelite (matchs pairs vs impairs, joueurs de 400 a 900 minutes), une
# mesure de pression n'est pas affichee sur une fiche a echantillon reduit.
FIDELITE_MIN_REDUIT = 0.20


def section_pression(j) -> None:
    """Bloc « Sous pression » : descriptif, hors score.

    Diagnostic du 30/09/2026 (5 056 joueurs suivis d'une saison a l'autre) : ces
    metriques ne sont pas redondantes avec le score (|rho| <= 0,27 a championnat
    et archetype fixes) mais n'ont presque aucune valeur predictive une fois le
    score connu (+0,4 point de R2 la ou le score en explique 15,5). Elles sont
    donc affichees comme contexte et comme style, et n'entrent dans aucun calcul.
    """
    if not pression_dispo():
        return
    try:
        pz = requete("""SELECT p.* FROM fait_pression p
            JOIN dim_poste_pression m USING (pos)
            WHERE p.playerId = ? AND p.iterationId = ? AND m.position = ?""",
                     (int(j.playerId), int(j.iterationId), j.position))
    except duckdb.Error:
        return
    if pz.empty:
        st.markdown("#### 🫸 Sous pression")
        st.caption("Pas de mesure à ce poste : le jeu sous pression est affiché à partir de 80 passes "
                   "jouées à ce poste, et sa partie progression à partir de 400 minutes et de 20 passes "
                   "sous forte pression. Certains championnats ne sont pas couverts par l'étude.")
        return
    d = pz.iloc[0]

    st.markdown("#### 🫸 Sous pression")
    # Echantillon reduit : une mesure dont la fidelite, mesuree entre 400 et 900
    # minutes, est quasi nulle n'est pas affichee. C'est le cas du volume de
    # passes progressives par 90 (-0,09) : aucun signal, et un percentile en
    # plus tire vers le bas par le retrecissement des volumes (prior de 3 x 90
    # minutes, qui pese d'autant plus que le joueur a peu joue).
    reduit = echantillon_reduit(j.minutes)
    court = pression_fiabilite_reduite() if reduit else {}
    masquees = []
    lignes = []
    for cle_pct, cle_val, libelle, fmt, aide in PRESSION_LIGNES:
        val = d.get(cle_val)
        if val is None or pd.isna(val):
            continue
        if court.get(cle_val, 1.0) < FIDELITE_MIN_REDUIT:
            masquees.append(libelle.lower())
            continue
        pct = d.get(cle_pct) if cle_pct else None
        lignes.append({"indicateur": libelle, "valeur": fmt.format(val),
                       "percentile": float(pct) if pct is not None and pd.notna(pct) and pct >= 0 else None,
                       "lecture": aide})
    if not lignes:
        return
    st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch",
                 column_config={
                     "indicateur": st.column_config.TextColumn("Indicateur"),
                     "valeur": st.column_config.TextColumn("Valeur", width="small"),
                     "percentile": st.column_config.ProgressColumn(
                         "Percentile du poste", min_value=0, max_value=100, format="P%.0f"),
                     "lecture": st.column_config.TextColumn("Comment le lire", width="large")})

    # Contexte de traduction : pression subie a ce poste ici vs en JPL, meme jeu de saisons.
    try:
        ctx = requete("""SELECT c.pression_passe AS ici, c.pression_reception AS ici_r,
                   r.pression_passe AS jpl, r.pression_reception AS jpl_r
            FROM dim_pression_championnat c
            JOIN dim_pression_championnat r
              ON r.pos = c.pos AND r.jeu_saisons = c.jeu_saisons
            JOIN dim_saison s ON s.iterationId = r.iterationId
            WHERE c.iterationId = ? AND c.pos = ? AND s.competition = 'Jupiler Pro League'""",
                      (int(j.iterationId), d["pos"]))
    except duckdb.Error:
        ctx = pd.DataFrame()

    notes = []
    if reduit:
        notes.append(f"⚠️ **Moins de {MINUTES_FIABLES} minutes** : ces mesures reposent sur peu de passes, "
                     "à lire comme une tendance et non comme un trait établi du joueur. Ses percentiles "
                     "le situent parmi les joueurs de l'échantillon de référence, dont il ne fait pas partie."
                     + (f" Non affiché sur cet échantillon : {', '.join(masquees)} (fidélité mesurée "
                        "quasi nulle à ce volume de jeu)." if masquees else ""))
    if not ctx.empty and pd.notna(ctx.iloc[0]["jpl"]) and j.competition != "Jupiler Pro League":
        r = ctx.iloc[0]
        dp, dr = r["jpl"] - r["ici"], r["jpl_r"] - r["ici_r"]
        sens = "plus" if dp > 0 else "moins"
        sens_r = "plus" if dr > 0 else "moins"
        notes.append(f"**Traduction vers la JPL** : à ce poste, la JPL presse **{abs(dp):.1f} point"
                     f"{'s' if abs(dp) >= 2 else ''} {sens}** à la passe et **{abs(dr):.1f} "
                     f"{sens_r}** à la réception qu'en {j.competition}. Mesuré sur 5 308 transferts, "
                     "cet écart n'explique pas la perte de performance (rho 0,007) : c'est un "
                     "contexte de lecture, pas une prédiction.")
    notes.append("Ces indicateurs sont **descriptifs : ils n'entrent dans aucun score**. Mesurés sur "
                 "5 056 joueurs suivis d'une saison à l'autre, ils n'ajoutent que 0,4 point de R² "
                 "pour prédire la saison suivante, quand le score en explique 15,5.")
    fia = pression_fiabilite()
    if fia:
        # Echantillon reduit : la fidelite mesuree entre 400 et 900 minutes est
        # donnee a cote de celle de la reference -- c'est elle qui vaut pour lui.
        fiab_txt = " · ".join(f"{k.replace('_', ' ')} {v:.2f}"
                              + (f" → {court[k]:.2f}" if k in court else "")
                              for k, v in sorted(fia.items(), key=lambda x: -x[1]))
        notes.append(f"Fidélité mesurée (matchs pairs vs impairs) : {fiab_txt}. "
                     + (f"La seconde valeur est celle des joueurs entre {MINUTES_PROFILS} et "
                        f"{MINUTES_FIABLES} minutes, donc la sienne. " if court else "")
                     + "Au-dessus de 0,60 la mesure décrit le joueur ; en dessous elle décrit surtout "
                       "son équipe et son volume de jeu.")
    st.caption("  \n".join(notes))


# ================================================================= heatmap
# Carte des touches de balle (Etude_Pression_Passes/09_heatmap.py). Les actions
# sont placees a leurs COORDONNEES reelles (evenements Impect), sur une grille
# de 2,5 m : rien a voir avec les zones des KPI de saison. Deux fichiers parquet
# a cote de l'app (saisons terminees / saison en cours), lus directement -- ils
# ne passent pas par la base, qui est deja pres de la limite de taille.
_CARTES = sorted({*_ICI.glob("heatmap_*.parquet"), *(_ICI / "data").glob("heatmap_*.parquet")})
# Cartes de saison (une ligne par fiche) et cartes par match, pour le monitoring
# (une ligne par joueur x match des ~3 derniers mois, un fichier par semaine).
HEATMAP_FICHIERS = [f for f in _CARTES if not f.name.startswith("heatmap_matchs_")]
HEATMAP_MATCHS = [f for f in _CARTES if f.name.startswith("heatmap_matchs_")]
# Sous ce temps de jeu, une carte reflete surtout le deroulement de quelques matchs.
HEATMAP_MINUTES_PRUDENCE = 400
TERRAIN_L, TERRAIN_l = 105.0, 68.0
# mode -> couches additionnees (cf. 09_heatmap.py). off = chaque contact du joueur
# avec le ballon quand son equipe l'a : passes, receptions, tirs, conduites de
# balle ; def = interceptions, degagements, contres, duels, ballons recuperes,
# arrets ; cpa = coups de pied arretes qu'il tire, ajoutes sur demande seulement.
HEATMAP_MODES = {"Toutes les actions": ("off", "def"), "Offensives": ("off",), "Défensives": ("def",)}
HEATMAP_DEFAUT = "Toutes les actions"
# Une seule teinte, du papier de la page au sarcelle fonce : c'est une intensite.
HEATMAP_ECHELLE = [[0.0, "#FBFBF9"], [0.12, "#E3EEEC"], [0.35, "#A3C9C4"], [0.6, "#5A9E98"],
                   [0.8, "#1F6F6B"], [1.0, "#0D3F3C"]]
# Limites des zones Impect, en metres depuis le centre du terrain.
TIERS = [("Tiers défensif", -52.5, -17.5), ("Tiers médian", -17.5, 17.5), ("Tiers offensif", 17.5, 52.5)]
COULOIRS = [("Aile gauche", 20.2, 34.0), ("Demi-espace gauche", 9.2, 20.2), ("Axe", -9.2, 9.2),
            ("Demi-espace droit", -20.2, -9.2), ("Aile droite", -34.0, -20.2)]


@st.cache_data(ttl=600, show_spinner=False)
def heatmap_joueur(player_id: int, squad_id: int, iteration_id: int, position: str) -> dict | None:
    """Les couches d'une fiche (offensive, defensive, coups de pied arretes), ou None."""
    if not HEATMAP_FICHIERS:
        return None
    liste = ", ".join("'" + str(f).replace("'", "''") + "'" for f in HEATMAP_FICHIERS)
    try:
        d = connexion().execute(
            f"""SELECT * FROM read_parquet([{liste}], union_by_name = true)
                WHERE playerId = ? AND iterationId = ? AND squadId = ? AND position = ?""",
            (player_id, iteration_id, squad_id, position)).df()
    except duckdb.Error:
        return None
    if d.empty:
        return None
    r = d.sort_values("n_off", ascending=False).iloc[0]
    nx, ny = int(r["nx"]), int(r["ny"])

    def grille(nom: str) -> np.ndarray:
        # Couche absente d'un fichier ecrit avant son ajout (cpa, 07/10/2026) : vide.
        b = r.get(nom)
        if b is None or (not isinstance(b, (bytes, bytearray, memoryview)) and pd.isna(b)):
            return np.zeros((nx, ny))
        return np.frombuffer(bytes(b), dtype="<u2").reshape(nx, ny).astype(float)

    entier = lambda nom: int(r[nom]) if nom in r.index and pd.notna(r[nom]) else 0      # noqa: E731
    return {"off": grille("off"), "def": grille("def"), "cpa": grille("cpa"),
            "complet": bool(r["complet"]), "n_tirs": entier("n_tirs"),
            "n_conduites": entier("n_conduites"), "n_matchs": entier("n_matchs")}


@st.cache_data(ttl=600, show_spinner=False)
def heatmap_periode(player_id: int, squad_id: int, iteration_id: int, position: str,
                    debut, fin) -> dict | None:
    """Carte d'un joueur sur une periode du monitoring : somme de ses matchs joues
    a ce poste entre `debut` (exclu) et `fin` (inclus). None si aucun match."""
    if not HEATMAP_MATCHS:
        return None
    liste = ", ".join("'" + str(f).replace("'", "''") + "'" for f in HEATMAP_MATCHS)
    try:
        d = connexion().execute(
            f"""SELECT * FROM read_parquet([{liste}], union_by_name = true)
                WHERE playerId = ? AND squadId = ? AND iterationId = ? AND position = ?
                  AND date > ? AND date <= ?""",
            (player_id, squad_id, iteration_id, position, debut, fin)).df()
    except duckdb.Error:
        return None
    if d.empty:
        return None
    nx, ny = int(d["nx"].iloc[0]), int(d["ny"].iloc[0])

    def grille(nom: str) -> np.ndarray:
        # Chaque match porte la liste des cases touchees, une par action.
        cases = np.concatenate([np.frombuffer(bytes(b), dtype="<u2") for b in d[nom]] or [np.array([], dtype="<u2")])
        return np.bincount(cases, minlength=nx * ny).reshape(nx, ny).astype(float)

    return {"off": grille("off"), "def": grille("def"), "cpa": grille("cpa"), "complet": True,
            "n_tirs": int(d["n_tirs"].sum()), "n_conduites": int(d["n_conduites"].sum()),
            "n_matchs": int(d["matchId"].nunique())}


def _lisser(g: np.ndarray, sigma: float) -> np.ndarray:
    """Flou gaussien separable, bords en miroir : une touche le long de la ligne
    ne « fuit » pas hors du terrain, ce qui effacerait les joueurs de couloir."""
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    g = np.pad(g, r, mode="reflect")
    g = np.apply_along_axis(np.convolve, 0, g, k, "valid")
    return np.apply_along_axis(np.convolve, 1, g, k, "valid")


def _lissage_cases(n: int) -> float:
    """Ecart-type du flou, en cases : plus le joueur a de touches, plus la carte
    peut etre fine sans devenir un nuage de points."""
    return 1.2 if n >= 1500 else 1.6 if n >= 500 else 2.2


def graphe_heatmap(g: np.ndarray) -> go.Figure:
    """Carte d'une grille (nx, ny) de touches : attaque vers la droite, cote
    gauche du joueur en haut."""
    nx, ny = g.shape
    z = _lisser(g, _lissage_cases(int(g.sum())))
    z = z / z.max() if z.max() > 0 else z
    xs = (np.arange(nx) + 0.5) * TERRAIN_L / nx - TERRAIN_L / 2
    ys = (np.arange(ny) + 0.5) * TERRAIN_l / ny - TERRAIN_l / 2
    fig = go.Figure(go.Heatmap(
        z=z.T, x=xs, y=ys, customdata=g.T, zsmooth="best", zmin=0, zmax=1,
        colorscale=HEATMAP_ECHELLE, showscale=False,
        hovertemplate="%{customdata:.0f} action(s) dans cette case de 2,5 m<extra></extra>"))
    L, l = TERRAIN_L / 2, TERRAIN_l / 2
    trait = dict(color="#3A3C40", width=1)
    lignes = [dict(type="rect", x0=-L, x1=L, y0=-l, y1=l),
              dict(type="line", x0=0, x1=0, y0=-l, y1=l),
              dict(type="circle", x0=-9.15, x1=9.15, y0=-9.15, y1=9.15)]
    for sens in (-1, 1):       # surfaces, 6 metres, but, de chaque cote
        lignes += [dict(type="rect", x0=sens * L, x1=sens * (L - 16.5), y0=-20.16, y1=20.16),
                   dict(type="rect", x0=sens * L, x1=sens * (L - 5.5), y0=-9.16, y1=9.16),
                   dict(type="rect", x0=sens * L, x1=sens * (L + 1.8), y0=-3.66, y1=3.66),
                   dict(type="circle", x0=sens * (L - 11) - 0.3, x1=sens * (L - 11) + 0.3, y0=-0.3, y1=0.3)]
    fig.update_layout(
        shapes=[dict(line=trait, opacity=0.55, layer="above", **f) for f in lignes],
        height=390, margin=dict(l=4, r=4, t=6, b=26),
        xaxis=dict(visible=False, range=[-L - 2.5, L + 2.5], fixedrange=True),
        yaxis=dict(visible=False, range=[-l - 1.5, l + 1.5], scaleanchor="x", fixedrange=True),
        # Pas de barre de couleur : l'echelle est relative au joueur (sa case la
        # plus frequentee = le plus fonce), une graduation chiffree n'aurait pas de sens.
        annotations=[dict(x=0.5, y=-0.07, xref="paper", yref="paper", showarrow=False,
                          text="sens de l'attaque  ⟶   ·   plus foncé = plus d'actions",
                          font=dict(size=11, color=_GRIS))])
    return fig


def repartition_zones(g: np.ndarray) -> pd.DataFrame:
    """Part des touches par couloir et par tiers Impect (cases rattachees a la
    zone de leur centre : a 1 m pres sur les limites de couloir)."""
    nx, ny = g.shape
    xs = (np.arange(nx) + 0.5) * TERRAIN_L / nx - TERRAIN_L / 2
    ys = (np.arange(ny) + 0.5) * TERRAIN_l / ny - TERRAIN_l / 2
    total = g.sum() or 1.0
    lignes = []
    for nom, y0, y1 in COULOIRS:
        dans_y = (ys >= y0) & (ys < y1)
        ligne = {"Couloir": nom}
        for tiers, x0, x1 in TIERS:
            ligne[tiers] = round(100 * g[np.ix_((xs >= x0) & (xs < x1), dans_y)].sum() / total, 1)
        ligne["Total"] = round(sum(ligne[t] for t, _, _ in TIERS), 1)
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def section_heatmap(j) -> None:
    """Bloc « Zones d'action » de la fiche : descriptif, hors score."""
    if not HEATMAP_FICHIERS:
        return                       # base publiee sans les cartes : le bloc n'existe pas
    # La saison est dans le titre : dans une fiche ouverte depuis le Monitoring,
    # cette carte suit celle de la periode, et la fiche de saison affichee peut
    # etre celle d'une autre saison ou d'un autre club que la periode monitoree.
    st.markdown(f"#### 🗺️ Zones d'action · saison {j.saison}")
    h = heatmap_joueur(int(j.playerId), int(j.squadId), int(j.iterationId), j.position)
    if h is None:
        st.caption("Pas de carte pour cette fiche : championnat hors de la collecte d'événements "
                   "(championnats de jeunes, saisons d'avant 2025) ou moins de 80 actions à ce poste.")
        return
    _bloc_heatmap(h, f"{int(j.playerId)}_{int(j.squadId)}_{int(j.iterationId)}_{j.position}",
                  libelle=f"Saison {j.saison} entière · {j.club} · {j.competition}")


def section_heatmap_periode(m) -> None:
    """Carte de la periode monitoree : memes boutons que la carte de saison, sur
    les seuls matchs de la periode. Forcement plus approximative -- d'ou l'alerte
    sous HEATMAP_MINUTES_PRUDENCE minutes."""
    if not HEATMAP_MATCHS:
        return
    debut, fin = bornes_periode()
    st.markdown(f"##### 🗺️ Zones d'action · {periode} derniers jours, du {debut:%d/%m} au {fin:%d/%m/%Y}")
    h = heatmap_periode(int(m.playerId), int(m.squadId), int(m.iterationId), m.position,
                        debut.to_pydatetime(), fin.to_pydatetime())
    if h is None:
        st.caption("Pas de carte sur cette période : les événements de ses matchs ne sont pas encore "
                   "collectés, ou son championnat est hors de la collecte.")
        return
    alertes = []
    if pd.notna(m.minutes) and m.minutes < HEATMAP_MINUTES_PRUDENCE:
        alertes.append(f"⚠️ **{m.minutes:.0f} minutes sur la période, moins de {HEATMAP_MINUTES_PRUDENCE}** : "
                       "carte à prendre avec précaution. Sur si peu de temps de jeu elle reflète surtout le "
                       "déroulement de quelques matchs (adversaire, score, consigne du jour), pas la zone "
                       "d'activité habituelle du joueur.")
    if pd.notna(m.matchs) and h["n_matchs"] < int(m.matchs):
        alertes.append(f"Carte sur **{h['n_matchs']} des {int(m.matchs)} matchs** de la période : les "
                       "événements des autres ne sont pas encore collectés.")
    _bloc_heatmap(h, f"mon_{periode}_{int(m.playerId)}_{int(m.squadId)}_{int(m.iterationId)}_{m.position}",
                  alertes,
                  libelle=f"Période monitorée : {periode} derniers jours, du {debut:%d/%m/%Y} au "
                          f"{fin:%d/%m/%Y} · {m.club} · {m.competition} ({m.saison})")


def _bloc_heatmap(h: dict, cle: str, alertes: list[str] | None = None, libelle: str = "") -> None:
    """Boutons, carte, repartition par zone et notes : commun a la carte de saison
    et a celle d'une periode du monitoring. alertes : notes placees en tete.
    libelle : ce que la carte couvre (saison ou periode), ecrit juste au-dessus
    d'elle -- deux cartes se suivent dans une fiche du Monitoring."""
    # Les trois boutons sont TOUJOURS la, meme quand le championnat n'a encore que
    # ses passes et receptions : un mode absent se lirait comme une fonction qui
    # n'existe pas. Le message dit alors ce qui manque et pourquoi.
    b1, b2 = st.columns([3, 2])
    mode = b1.segmented_control("Actions affichées", list(HEATMAP_MODES), default=HEATMAP_DEFAUT,
                                key=f"hm_mode_{cle}", label_visibility="collapsed") or HEATMAP_DEFAUT
    arretes = b2.toggle("Avec les coups de pied arrêtés", key=f"hm_cpa_{cle}",
                        disabled=not h["complet"] or mode == "Défensives",
                        help="Ajoute les touches, corners, coups francs et six mètres qu'il tire. "
                             "Décoché par défaut : ils dessinent le poteau de corner et la ligne de "
                             "touche du tireur plus que son jeu.")
    partiel = not h["complet"]
    if partiel:
        st.info("**Championnat en cours de téléchargement.** Seules ses passes et ses réceptions sont "
                "disponibles pour l'instant : la carte ci-dessous ne montre qu'elles, quel que soit le "
                "bouton. Tirs, conduites de balle, coups de pied arrêtés et interventions défensives "
                "arriveront d'un bloc quand tous les matchs du championnat seront récupérés.", icon="⏳")
        couches = ("off",)
    else:
        couches = HEATMAP_MODES[mode] + (("cpa",) if arretes and mode != "Défensives" else ())
    g = sum(h[c] for c in couches)
    n = int(g.sum())
    if n == 0:
        st.caption("Aucune action de ce type enregistrée à ce poste.")
        return
    c1, c2 = st.columns([3, 2])
    with c1:
        if libelle:
            st.markdown(f"**{libelle}**")
        st.plotly_chart(graphe_heatmap(g), width="stretch", key=f"hm_fig_{cle}_{'_'.join(couches)}",
                        config={"displayModeBar": False})
    with c2:
        m1, m2 = st.columns(2)
        m1.metric("Actions sur la carte", f"{n:,}".replace(",", " "))
        m2.metric("Matchs", f"{h['n_matchs']}")
        zones = repartition_zones(g)
        pct = st.column_config.NumberColumn(format="%.0f %%")
        st.dataframe(zones, hide_index=True, width="stretch",
                     column_config={t: pct for t in zones.columns if t != "Couloir"})
    notes = list(alertes or [])
    if n < 300:
        notes.append(f"⚠️ **{n} actions seulement** : la carte donne une tendance, pas une zone d'activité "
                     "établie. Elle est d'autant plus lissée que le joueur a peu d'actions.")
    if partiel:
        notes.append("Affiché : **passes** (à leur point de départ) et **réceptions**.")
    else:
        notes.append("**Offensives** = chaque contact avec le ballon quand son équipe l'a : passes (à leur "
                     "point de départ), réceptions, tirs, et conduites de balle — un point tous les 5 m "
                     "de course balle au pied, plus l'endroit où il perd le ballon quand la conduite "
                     f"n'aboutit ni à une passe ni à un tir ({h['n_conduites']} conduites, "
                     f"{h['n_tirs']} tirs). **Défensives** = interceptions, dégagements, contres, duels "
                     "au sol, ballons récupérés, arrêts du gardien.")
    notes.append("Chaque action est placée à ses coordonnées réelles (événements Impect, cases de 2,5 m), "
                 "pour ce poste seulement. Le haut de la carte est le côté gauche du joueur. Le tableau "
                 "donne la part des actions par couloir et par tiers Impect. Descriptif : hors score.")
    st.caption("  \n".join(notes))


# ======================================================== profils similaires
# « Trouver des profils similaires » (09/10/2026, Alex) : les joueurs du meme
# poste qui ressemblent le plus a celui de la fiche. Six blocs de comparaison,
# chacun mesure par une distance, puis ponderes :
#   piliers         percentiles des piliers du poste, ponderes par leur poids
#                   dans le score -- ce que le joueur fait sur le terrain
#   profils         correspondances aux profils RCSC du poste -- son style
#   niveau          Score sans l'age : meme niveau, quel que soit l'age
#   championnat     rating moyen du championnat ou il joue
#   pression        son jeu sous pression (etude de pression), en percentiles
#   pression_ligue  pression subie a ce poste dans son championnat
# Les deux blocs de pression pesent peu, a dessein : l'etude a montre qu'ils
# decrivent un style sans presque rien predire.
#
# Chaque distance est rapportee a la distance mediane entre deux joueurs du
# poste pris AU HASARD (mesuree sur la base, par poste). D'ou une echelle
# lisible : similarite = 100 x 0,5^distance -- 100 = identique, 50 = aussi
# proches que deux joueurs quelconques du poste, sous 50 = plus eloignes.
SIM_POIDS = {"piliers": 0.45, "profils": 0.18, "niveau": 0.15, "championnat": 0.10,
             "pression": 0.07, "pression_ligue": 0.05}
SIM_LIBELLES = {"piliers": "Piliers", "profils": "Profils RCSC", "niveau": "Niveau",
                "championnat": "Championnat", "pression": "Sous pression",
                "pression_ligue": "Pression du championnat"}
# Fiabilite d'une comparaison : pleine a partir de ce volume de jeu pour les DEUX
# joueurs. En dessous, la similarite est ramenee vers 50 (le niveau du hasard) :
# sur peu de matchs, deux joueurs se ressemblent ou different surtout par bruit.
SIM_MINUTES_PLEINES, SIM_MATCHS_PLEINS = 900, 10
SIM_CLE = ["playerId", "squadId", "iterationId", "position"]
SIM_PRESSION = ["pct_part_prog_fp", "pct_reussite_fp_vs_attendu_100", "pct_bypassed_vs_attendu_p90",
                "pct_pression_passe", "ambition"]


def _sim_distances(d: dict, i: int) -> dict[str, np.ndarray]:
    """Distance de chaque joueur du poste au joueur d'indice i, bloc par bloc
    (NaN quand le bloc n'est pas mesure pour l'un des deux)."""
    def rms(m: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        e = (m - m[i]) ** 2
        w = np.ones(m.shape[1]) if w is None else w
        ok = ~np.isnan(e)
        den = (ok * w).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(np.where(ok, e, 0.0).dot(w) / np.where(den > 0, den, np.nan))
    return {"piliers": rms(d["piliers"], d["poids_piliers"]), "profils": rms(d["profils"]),
            "niveau": np.abs(d["niveau"] - d["niveau"][i]), "championnat": np.abs(d["ligue"] - d["ligue"][i]),
            "pression": rms(d["pression"]), "pression_ligue": rms(d["pression_ligue"])}


@st.cache_data(ttl=600, show_spinner=False)
def donnees_similarite(arch: str, poids: tuple = POIDS_DEFAUT) -> dict:
    """Tout ce qu'il faut pour comparer les joueurs d'un poste, en tableaux
    alignes sur la meme liste de joueurs. poids : cle de cache seulement (le
    Score sans l'age depend des poids regles dans la barre laterale)."""
    avec_pression = pression_dispo()
    champs_p = (""", v.pct_part_prog_fp, v.pct_reussite_fp_vs_attendu_100, v.pct_bypassed_vs_attendu_p90,
               v.pct_pression_passe, v.maintien_ambition""" if avec_pression else "")
    j = requete(f"""SELECT v.playerId, v.squadId, v.iterationId, v.position, v.nom, v.club, v.competition,
               v.saison, v.pays, v.top5_europe, round(v.age_years, 1) AS age, v.minutes_jouees AS minutes,
               v.n_matches_oppw AS matchs, v.score - v.ajust_age AS niveau,
               v.competition_avg_rating AS ligue, v.profil_principal{champs_p}
        FROM v_joueurs v WHERE v.archetype = ?""", (arch,)).drop_duplicates(SIM_CLE).reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(j[SIM_CLE])
    pil = requete("""SELECT playerId, squadId, iterationId, position, pilier, percentile, poids
        FROM fait_pilier WHERE archetype = ?""", (arch,))
    P = pil.pivot_table(index=SIM_CLE, columns="pilier", values="percentile").reindex(idx)
    w = pil.groupby("pilier")["poids"].first().reindex(P.columns).fillna(0).to_numpy(dtype=float)
    try:
        pro = requete("""SELECT playerId, squadId, iterationId, position, profil_id, correspondance
            FROM fait_profil WHERE archetype = ?""", (arch,))
        R = pro.pivot_table(index=SIM_CLE, columns="profil_id", values="correspondance").reindex(idx)
    except duckdb.Error:
        R = pd.DataFrame(index=idx)
    X = np.full((len(j), len(SIM_PRESSION)), np.nan)
    L = np.full((len(j), 2), np.nan)
    if avec_pression:
        x = j[SIM_PRESSION[:-1]].apply(pd.to_numeric, errors="coerce")
        x = x.where(x >= 0)                                    # -1 = percentile non mesure
        # Maintien de l'ambition : un rapport (mediane ~0,7), mis sur 0-100 comme les percentiles.
        x["ambition"] = (pd.to_numeric(j["maintien_ambition"], errors="coerce") * 70).clip(0, 100)
        X = x.to_numpy(dtype=float)
        try:
            lig = requete("""SELECT c.iterationId, m.position, avg(c.pression_passe) AS lp,
                       avg(c.pression_reception) AS lr
                FROM dim_pression_championnat c JOIN dim_poste_pression m USING (pos) GROUP BY ALL""")
            L = j[["iterationId", "position"]].merge(lig, on=["iterationId", "position"], how="left")[["lp", "lr"]] \
                .to_numpy(dtype=float)
        except duckdb.Error:
            pass
    d = {"joueurs": j, "piliers": P.to_numpy(dtype=float), "poids_piliers": w,
         "profils": R.to_numpy(dtype=float) if R.shape[1] else np.full((len(j), 1), np.nan),
         "niveau": j["niveau"].to_numpy(dtype=float), "ligue": j["ligue"].to_numpy(dtype=float),
         "pression": X, "pression_ligue": L}
    # Echelle de chaque bloc : distance mediane a 40 joueurs de reference tires au
    # hasard (graine fixe : memes echelles d'un affichage a l'autre).
    tirage = np.random.default_rng(7).choice(len(j), size=min(40, len(j)), replace=False)
    morceaux = {b: [] for b in SIM_POIDS}
    for i in tirage:
        for b, v in _sim_distances(d, int(i)).items():
            morceaux[b].append(v)
    d["echelles"] = {}
    for b, v in morceaux.items():
        v = np.concatenate(v)
        v = v[~np.isnan(v) & (v > 0)]
        d["echelles"][b] = float(np.median(v)) if len(v) else np.nan
    return d


def profils_similaires(cle: tuple, poids: tuple = POIDS_DEFAUT) -> pd.DataFrame:
    """Joueurs du meme poste, du plus au moins semblable a celui de la fiche.
    cle : (playerId, squadId, iterationId, position, archetype)."""
    d = donnees_similarite(cle[4], poids)
    j = d["joueurs"]
    moi = np.flatnonzero((j["playerId"] == cle[0]) & (j["squadId"] == cle[1])
                         & (j["iterationId"] == cle[2]) & (j["position"] == cle[3]))
    if not len(moi):
        return pd.DataFrame()
    i = int(moi[0])
    out = j.copy()
    somme, poids_dispo = np.zeros(len(j)), np.zeros(len(j))
    distances = _sim_distances(d, i)
    for bloc, dist in distances.items():
        ech = d["echelles"].get(bloc)
        if not ech or np.isnan(ech):
            continue
        rel = dist / ech                                   # 1 = ecart de deux joueurs au hasard
        ok = ~np.isnan(rel)
        somme += np.where(ok, rel, 0.0) * SIM_POIDS[bloc]
        poids_dispo += ok * SIM_POIDS[bloc]
        out[f"sim_{bloc}"] = np.round(100 * 0.5 ** rel, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        brute = 100 * 0.5 ** (somme / np.where(poids_dispo > 0, poids_dispo, np.nan))
    # Les piliers sont le coeur de la comparaison : sans eux, pas de similarite.
    brute = np.where(np.isnan(distances["piliers"]), np.nan, brute)
    volume = np.minimum(out["minutes"].fillna(0) / SIM_MINUTES_PLEINES,
                        out["matchs"].fillna(0) / SIM_MATCHS_PLEINS).clip(0, 1).to_numpy(dtype=float)
    fiab = np.sqrt(volume * volume[i])                      # les deux joueurs comptent
    out["similarite_brute"] = np.round(brute, 1)
    out["fiabilite"] = np.round(100 * fiab, 0)
    out["similarite"] = np.round(50 + (brute - 50) * (0.5 + 0.5 * fiab), 1)
    out["blocs_mesures"] = np.round(100 * poids_dispo / sum(SIM_POIDS.values()), 0)
    out = out[out["playerId"] != cle[0]].dropna(subset=["similarite"])
    # Un joueur, une ligne : sa saison la plus ressemblante.
    return (out.sort_values("similarite", ascending=False).drop_duplicates("playerId")
            .reset_index(drop=True))


def section_similaires(j, cle: tuple) -> None:
    """Resultat du bouton « Trouver des profils similaires »."""
    arch = cle[4]
    st.markdown(f"#### 🔎 Profils similaires à {j.nom}")
    sim = profils_similaires(cle, poids_courants())
    if sim.empty:
        st.caption("Comparaison impossible : les piliers de ce joueur ne sont pas mesurés.")
        return
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    toutes = sorted(sim["saison"].dropna().unique(), key=fin_saison, reverse=True)
    saisons_sim = f1.multiselect("Saisons", toutes, default=[x for x in toutes if x in (saison or [])],
                                 key=f"sim_saisons_{cle}", placeholder="Toutes les saisons")
    age_sim = f2.slider("Âge maximum", 16, 40, 40, key=f"sim_age_{cle}")
    min_sim = f3.slider("Minutes minimum", 400, 3000, 400, step=100, key=f"sim_min_{cle}")
    sans_top5 = f4.checkbox("Sans les 5 grands championnats", key=f"sim_top5_{cle}")
    vue = sim[(sim["minutes"] >= min_sim) & ((sim["age"] <= age_sim) | sim["age"].isna() if age_sim == 40
                                             else sim["age"] <= age_sim)]
    if saisons_sim:
        vue = vue[vue["saison"].isin(saisons_sim)]
    if sans_top5:
        vue = vue[~vue["top5_europe"].fillna(False).astype(bool)]
    vue = vue.head(50).reset_index(drop=True)
    if vue.empty:
        st.caption("Aucun joueur ne passe ces filtres.")
        return
    blocs = [b for b in SIM_POIDS if f"sim_{b}" in vue]
    pct = lambda nom, aide: st.column_config.NumberColumn(nom, format="%.0f", help=aide)    # noqa: E731
    ev = st.dataframe(
        vue, hide_index=True, width="stretch", height=min(430, 80 + 36 * len(vue)),
        key=f"sim_tbl_{cle}", on_select="rerun", selection_mode="single-row",
        column_order=["nom", "club", "competition", "saison", "age", "minutes", "similarite", "fiabilite"]
                     + [f"sim_{b}" for b in blocs] + ["profil_principal"],
        column_config={
            "similarite": st.column_config.ProgressColumn(
                "Similarité", min_value=0, max_value=100, format="%.0f",
                help="100 = identique. 50 = aussi proches que deux joueurs du poste pris au hasard. "
                     "Déjà ramenée vers 50 quand la comparaison est peu fiable."),
            "fiabilite": st.column_config.NumberColumn(
                "Fiabilité", format="%.0f %%",
                help=f"Volume de jeu des DEUX joueurs : 100 % à partir de {SIM_MINUTES_PLEINES} minutes "
                     f"et {SIM_MATCHS_PLEINS} matchs chacun."),
            **{f"sim_{b}": pct(SIM_LIBELLES[b], f"Similarité sur ce seul bloc (poids "
                                                 f"{SIM_POIDS[b] * 100:.0f} % dans le total).") for b in blocs},
            "minutes": st.column_config.NumberColumn("Min", format="%d"),
            "profil_principal": st.column_config.TextColumn("Profil RCSC"),
        })
    choix = ev.selection["rows"] if ev and "rows" in ev.selection else []
    if choix:
        c = vue.iloc[choix[0]]
        if st.button(f"Ouvrir la fiche de {c.nom} ({c.saison}, {c.club})", key=f"sim_open_{cle}", type="primary"):
            ouvrir_fiche(int(c.playerId), int(c.squadId), int(c.iterationId), c.position, arch)
    poids_txt = " · ".join(f"{SIM_LIBELLES[b].lower()} {SIM_POIDS[b] * 100:.0f} %" for b in SIM_POIDS)
    _nb = lambda v: float(v) if pd.notna(v) else 0.0                                      # noqa: E731
    moi_fiab = min(1.0, _nb(j.minutes) / SIM_MINUTES_PLEINES, _nb(j.matchs) / SIM_MATCHS_PLEINS)
    notes = [f"Comparaison dans le pool **{arch}**, toutes saisons ; un joueur n'apparaît qu'une fois, avec sa "
             f"saison la plus ressemblante. Poids : {poids_txt}.",
             "**Similarité** : 100 = identique, 50 = aussi proches que deux joueurs du poste pris au hasard. "
             "Elle dit que deux joueurs se ressemblent, pas que l'un vaut l'autre — le niveau n'en est "
             "qu'un bloc. 👉 Coche une ligne pour ouvrir sa fiche."]
    if moi_fiab < 1:
        notes.insert(0, f"⚠️ **{j.nom} n'a que {j.minutes:.0f} minutes** sur cette fiche : toutes les "
                        "comparaisons sont moins fiables, et les similarités sont resserrées vers 50.")
    st.caption("  \n".join(notes))


def carte_joueur(j, extra=None) -> None:
    """Fiche complete d'un joueur. j doit porter une colonne archetype_courant.
    extra : fonction affichant un bloc supplementaire sous les boutons (la
    fiche ouverte depuis le Monitoring y ajoute la periode monitoree)."""
    arch = j.archetype_courant
    cle = (int(j.playerId), int(j.squadId), int(j.iterationId), j.position, arch)

    # En-tete de fiche : identifiant du poste en micro-titre, nom en grand,
    # rayures, puis la ligne de contexte -- meme grammaire que les classements.
    st.markdown(f"#### Fiche joueur · {arch}")
    st.markdown(f"## {j.nom}")
    lieu = f"{j.club} · {j.competition}"
    if pd.notna(j.pays):
        lieu += f" ({j.pays}" + (f", D{int(j.niveau)})" if pd.notna(j.niveau) else ")")
    role = (f" · rôle Impect {j.role_milieu}" if arch in ("SIX", "EIGHT") and pd.notna(j.role_milieu) else "")
    st.caption(f"{lieu} · {j.saison} · {j.poste}" + (f" ({j.side})" if pd.notna(j.side) else "")
               + f" · archétype {arch}{role}")
    rayures(fine=True)

    # Acces direct aux fiches des autres saisons (seules existent celles ou il
    # a assez joue a ce poste pour etre evalue par la pipeline).
    hist = historique(j, arch)
    autres = hist[~hist["courante"]].head(5)
    if not autres.empty:
        c_ = st.columns([1.1] + [1] * len(autres) + [max(0.1, 5 - len(autres))])
        c_[0].markdown("**Ses autres saisons**")
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

    # Buts et passes decisives : A TITRE INDICATIF, ils n'entrent dans aucun
    # score. Un ailier a 4 buts dans une ligue faible n'est pas meilleur qu'un
    # ailier a 2 buts en Premier League -- c'est ce que dit deja le score. Les
    # penalties sont detailles en petit : 15 buts dont 8 penalties, ce n'est pas
    # 15 buts, et un total brut induirait en erreur.
    _avec_buts = buts_dispo() and "buts" in j.index
    # Les trois lectures cote a cote : meme performance terrain, lue de trois
    # manieres. Les voir ensemble evite de changer de lecture dans la barre
    # laterale juste pour comparer.
    _avec_sans_age = SANS_AGE and "score_sans_age" in j.index and pd.notna(j.score_sans_age)
    # Le rang mondial s'affiche sous la lecture active, quelle qu'elle soit.
    _rang = f"rang mondial {int(j.rang_mondial)}" if pd.notna(j.rang_mondial) else None
    _pen = j.buts_penalty if _avec_buts and pd.notna(j.buts_penalty) else 0
    _mets = [
        ("Score", n_(j.score_v8), _rang if SC == "score" else None,
         LECTURES["Score (performance + niveau + âge)"]["aide"]),
        ("Qualité actuelle", n_(j.qualite), _rang if SC == "qualite_actuelle" else None,
         LECTURES["Qualité actuelle (niveau JPL)"]["aide"]),
    ]
    if _avec_sans_age:
        _mets.append(("Score hors âge", n_(j.score_sans_age),
                      _rang if SC == "score_sans_age" else None,
                      LECTURES["Score sans l'âge (performance + niveau)"]["aide"]))
    _mets += [("Âge", n_(j.age), None, None),
              ("Minutes", n_(j.minutes, "{:.0f}"), f"{n_(j.matchs, '{:.0f}')} matchs", None)]
    if _avec_buts:
        _mets += [
            ("Buts", f"{int(j.buts)}" if pd.notna(j.buts) else "—",
             # « pén. » et non « pénaltys » : le libelle d'un delta est tronque
             # des que la colonne descend sous ~200 px, et c'est justement le
             # chiffre qu'Alex veut voir.
             (f"dont {int(_pen)} pén." if _pen else None),
             "Buts marqués sur la saison, toutes compétitions de ce championnat. "
             "Donnée indicative : elle n'entre dans aucun score."),
            ("Passes déc.", f"{int(j.passes_d)}" if pd.notna(j.passes_d) else "—", None,
             "Passes décisives sur la saison. Indicatif : hors score. "
             "La création est mesurée par le pilier Création (xA, actions créées)."),
        ]
    _mets += [("Pied", j.pied_fort or "—", None, None),
              ("Taille", n_(j.taille_cm, "{:.0f} cm"), None, None)]
    # AU PLUS 5 PAR LIGNE. Avec les trois lectures, les buts et les passes on
    # monte a 9 metriques : sur une seule ligne, Streamlit tronque les libelles
    # ET les valeurs (« S… », « 72… », « MINU… ») des 1 400 px -- illisible.
    # st.columns(5) a chaque ligne garde aussi les colonnes alignees entre elles.
    for _debut in range(0, len(_mets), 5):
        _cols = st.columns(5)
        for _col, (_lab, _val, _delta, _aide) in zip(_cols, _mets[_debut:_debut + 5]):
            _col.metric(_lab, _val, _delta, delta_color="off", help=_aide)

    st.markdown(
        f"**Score {n_(j.score_v8)}** = performance {n_(j.performance)} "
        f"({n_(j.base)} de base {j.excellence:+.1f} excellence {-j.fragilite:+.1f} fragilité) "
        f"{j.aj_niveau:+.1f} niveau du club {j.aj_age:+.1f} âge  \n"
        f"**Qualité actuelle {n_(j.qualite)}** = performance {n_(j.performance)} "
        f"{j.aj_traduction:+.1f} traduction au niveau JPL"
        + (f"  \n**Score sans l'âge {n_(j.score_sans_age)}** = le Score ci-dessus "
           f"{-j.aj_age:+.1f}, c'est-à-dire sans son ajustement d'âge"
           if SANS_AGE and pd.notna(j.get("score_sans_age")) else ""))

    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    if b1.button(f"Ajouter à « {shortlist} »", width="stretch", key=f"add_{cle}"):
        ajouter(shortlist, j); st.toast(f"{j.nom} ajouté à « {shortlist} »"); st.rerun()
    if b2.button("🚫 Exclure ce joueur", width="stretch", key=f"exc_{cle}",
                 help="Il ne sera plus proposé dans les recherches par poste."):
        ajouter("exclus", j); st.toast(f"{j.nom} exclu"); st.rerun()
    if b3.button(f"🏟️ Effectif de {j.club}", width="stretch", key=f"clu_{cle}"):
        if "fiche" in st.query_params:
            del st.query_params["fiche"]
        st.query_params["club"] = f"{int(j.squadId)}-{int(j.iterationId)}"
        st.rerun()
    # Bascule : un premier clic affiche la liste, un second la referme. L'etat est
    # garde par fiche, pour qu'elle reste ouverte quand on regle ses filtres.
    _sim_ouvert = f"sim_ouvert_{cle}"
    if b4.button("🔎 Trouver des profils similaires", width="stretch", key=f"sim_btn_{cle}",
                 help="Les joueurs du même poste qui lui ressemblent le plus : piliers, profils RCSC, "
                      "niveau, championnat, jeu sous pression."):
        st.session_state[_sim_ouvert] = not st.session_state.get(_sim_ouvert, False)
    if st.session_state.get(_sim_ouvert):
        section_similaires(j, cle)

    if extra is not None:
        extra()

    # Alerte d'echantillon : profils et jeu sous pression sont calcules des
    # MINUTES_PROFILS, mais sous MINUTES_FIABLES ils ne valent pas ceux d'une
    # saison pleine. Placee juste au-dessus des deux blocs qu'elle concerne.
    if echantillon_reduit(j.minutes):
        st.warning(
            f"**Échantillon réduit : {j.minutes:.0f} minutes à ce poste**, moins de {MINUTES_FIABLES}. "
            "Les profils RCSC et le jeu sous pression ci-dessous sont à prendre avec précaution : "
            "quelques matchs suffisent encore à les déplacer. Le score, lui, est calculé comme pour "
            f"tous les joueurs à partir de {MINUTES_PROFILS} minutes.", icon="⚠️")
    section_profils(cle, j.minutes)
    section_pression(j)
    section_heatmap(j)

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
    autres_arch = requete(f"""SELECT archetype, round(score,1) AS score,
            round(qualite_actuelle,1) AS qualite, {RG} AS rang
        FROM fait_joueur_saison
        WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype<>?
        ORDER BY {SC} DESC""", cle)
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




# ================================================================= monitoring
# Forme des N derniers jours, precalculee par Charleroi_MultiPoste_MonitoringV10
# (appele par db_build.py) : la plateforme ne fait que lire. Meme formule que le
# Score de saison, meme reference figee (un niveau 60 sur la periode vaut un 60
# de saison), plus deux garde-fous plus stricts a petit volume : confiance
# reduite du z metrique et metriques aberrantes ramenees dans une plage de
# confiance. "Perf. brute" = la meme periode sans ces deux garde-fous.
TRI_MONITORING = {"Score": "score", "Performance": "score_performance"}
MON_CHAMPS = """m.nom, m.club, m.competition, m.pays, m.niveau, m.saison,
    round(m.score,1) AS score, round(m.score_performance,1) AS performance,
    round(m.perf_brute,1) AS perf_brute, round(m.ajust_niveau,1) AS aj_niveau,
    round(m.ajust_age,1) AS aj_age, round(m.age_years,1) AS age, m.minutes,
    m.n_matchs AS matchs, round(m.matchs_eq,1) AS matchs_eq,
    round(m.confiance * 100, 0) AS confiance, m.nb_metriques_corrigees AS corrigees,
    m.metriques_corrigees, m.pied_fort, m.taille_cm, m.rang, m.position AS poste,
    m.premier_match, m.dernier_match, round(m.base,1) AS base,
    round(m.excellence,1) AS excellence, round(m.fragilite,1) AS fragilite,
    m.pilier_fort, m.pilier_faible, round(m.opp_coef_avg,3) AS coef_adv,
    round(m.club_rating,3) AS rating_club,
    round(f.score,1) AS score_saison, round(f.score_performance,1) AS perf_saison,
    m.playerId, m.squadId, m.iterationId, m.position, m.archetype AS archetype_courant,
    m.fiche_squadId, m.fiche_iterationId, m.fiche_position, m.fiche_archetype,
    m.fiche_meme_saison"""
# Fiche de saison correspondante (meme joueur, club, saison et archetype ; a
# defaut sa saison evaluee la plus recente, cf. db_build._lien_fiche).
MON_JOINTURE = """LEFT JOIN fait_joueur_saison f
    ON f.playerId = m.playerId AND f.squadId = m.fiche_squadId
   AND f.iterationId = m.fiche_iterationId AND f.position = m.fiche_position
   AND f.archetype = m.archetype"""
LIBELLE_PALIER = {"bas": "Bas de tableau", "milieu": "Milieu de tableau", "haut": "Top du championnat"}


def bornes_periode() -> tuple[pd.Timestamp, pd.Timestamp]:
    fin = pd.Timestamp(MON["date_reference"])
    return fin - pd.Timedelta(days=periode), fin


def graphe_matchs(matchs: pd.DataFrame, m) -> go.Figure:
    """Niveau de chaque match de la periode (barres colorees par palier
    d'adversaire), avec la performance de la periode brute et retenue."""
    fig = go.Figure()
    x = [f"{pd.Timestamp(d):%d/%m}<br>{(a if isinstance(a, str) else '?')[:16]}"
         for d, a in zip(matchs["date"], matchs["adversaire"])]
    fig.add_trace(go.Bar(
        x=x, y=matchs["niveau"], showlegend=False,
        marker_color=[COULEUR_PALIER.get(p, GRIS) for p in matchs["palier"]],
        text=matchs["niveau"].round(0).astype("Int64").astype(str), textposition="outside",
        cliponaxis=False,
        customdata=[(pd.Timestamp(d).strftime("%d/%m/%Y"), a, LIBELLE_PALIER.get(p, "?"), r, mi, po)
                    for d, a, p, r, mi, po in zip(matchs["date"], matchs["adversaire"], matchs["palier"],
                                                  matchs["opp_rating"], matchs["minutes"],
                                                  matchs["position"])],
        hovertemplate="%{customdata[0]} contre %{customdata[1]}<br>%{customdata[2]} "
                      "(rating %{customdata[3]:.2f})<br>%{customdata[4]:.0f} min · %{customdata[5]}"
                      "<br>Niveau du match <b>%{y:.0f}</b><extra></extra>"))
    for palier, libelle in LIBELLE_PALIER.items():
        if (matchs["palier"] == palier).any():
            fig.add_trace(go.Bar(x=[None], y=[None], name=f"Adversaire : {libelle.lower()}",
                                 marker_color=COULEUR_PALIER[palier]))
    for valeur, nom, style, couleur in ((m.perf_brute, "Performance brute de la période", "dash", TEAL),
                                         (m.performance, "Performance retenue (classement)", "solid",
                                          COULEUR_AUTRE_CONTEXTE)):
        if pd.notna(valeur):
            fig.add_hline(y=valeur, line_dash=style, line_color=couleur, line_width=1.5)
            _legende_ligne(fig, f"{nom} : {valeur:.0f}", style, couleur)
    _mediane(fig)
    haut = max([50] + matchs["niveau"].dropna().tolist())
    fig.update_layout(**MISE_EN_PAGE, bargap=0.35,
                      yaxis=dict(title="Niveau", range=[0, haut + 18]),
                      xaxis=dict(tickfont=dict(size=10 if len(matchs) > 8 else 12)))
    return fig


def graphe_piliers_periode(pm: pd.DataFrame, ps: pd.DataFrame) -> go.Figure:
    """Percentile de chaque pilier sur la periode, a cote de celui de la saison."""
    ordre = (ps if not ps.empty else pm).sort_values("poids", ascending=False)["pilier"].tolist()
    ordre += [p for p in pm["pilier"] if p not in ordre]
    fig = go.Figure()
    for df, nom, couleur in ((ps, "Saison (fiche)", GRIS), (pm, "Période monitorée", TEAL)):
        if df.empty:
            continue
        d = df.set_index("pilier").reindex(ordre)
        fig.add_trace(go.Bar(y=[p.replace("_", " ") for p in ordre], x=d["percentile"], name=nom,
                             orientation="h", marker_color=couleur,
                             text=d["percentile"].round(0).astype("Int64").astype(str),
                             textposition="outside", cliponaxis=False))
    fig.add_vline(x=50, line_dash="dot", line_color="#999")
    fig.update_layout(height=60 + 34 * len(ordre), margin=dict(l=0, r=20, t=30, b=0),
                      barmode="group", xaxis=dict(range=[0, 105], title="Percentile du poste"),
                      yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    return fig


def section_monitoring(m, fiche) -> None:
    """Bloc 'periode monitoree' de la fiche : chiffres de la periode, niveau
    de chaque match (et niveau des adversaires), piliers periode vs saison."""
    debut, fin = bornes_periode()
    arch = m.archetype_courant
    st.markdown(f"#### 📡 Sur les {periode} derniers jours — du {debut:%d/%m} au {fin:%d/%m/%Y}")
    autre_poste = fiche is not None and fiche.archetype_courant != arch
    if autre_poste:
        st.caption(f"ℹ️ Fiche de saison affichée : **{fiche.saison} à {fiche.club}, archétype "
                   f"{fiche.archetype_courant}** — il n'a jamais été évalué en {arch} sur une saison "
                   "(400 minutes à ce poste). Le Score de saison n'est donc pas comparable à celui de la période.")
    elif fiche is not None and not _vrai(m.fiche_meme_saison):
        st.caption(f"ℹ️ Fiche de saison affichée : **{fiche.saison} à {fiche.club}** — sa saison en cours "
                   f"({m.competition}, {m.club}) ne compte pas encore 400 minutes évaluées à ce poste.")
    k = st.columns(6)
    k[0].metric("Score période", n_(m.score), f"rang {int(m.rang)} du poste" if pd.notna(m.rang) else None,
                delta_color="off", help="Même formule que le Score de saison : performance sur la période "
                                        "+ niveau du club + âge. Rang parmi tous les joueurs du poste actifs "
                                        "sur la période.")
    ecart = (m.performance - m.perf_saison) if pd.notna(m.perf_saison) else None
    k[1].metric("Performance", n_(m.performance),
                f"{ecart:+.0f} vs saison" if ecart is not None else None,
                help="Performance retenue pour le classement, après les garde-fous petits échantillons. "
                     "Comparée à sa performance de saison (fiche).")
    k[2].metric("Perf. brute", n_(m.perf_brute),
                help="La même période sans les garde-fous du monitoring (confiance de saison, aucune "
                     "métrique corrigée) : ce que le joueur a montré, bruit compris.")
    k[3].metric("Matchs", f"{int(m.matchs)}", f"{m.minutes:.0f} min · {m.matchs_eq:.1f} pleins",
                delta_color="off")
    k[4].metric("Fiabilité", f"{m.confiance:.0f} %",
                help="Part du signal conservée pour chaque métrique : 30 % sous un match plein, 60 % à "
                     "400 minutes (le minimum d'une saison), 100 % à 15 matchs. Plus il a joué, moins sa "
                     "performance est ramenée vers 50.")
    k[5].metric("Métriques corrigées", f"{int(m.corrigees)}",
                help="Métriques hors de la plage de confiance, ramenées à sa borne (jamais exclues). "
                     "Plage d'autant plus étroite qu'il a peu joué.")
    st.markdown(f"**Score période {n_(m.score)}** = performance {n_(m.performance)} "
                f"({n_(m.base)} de base {m.excellence:+.1f} excellence {-m.fragilite:+.1f} fragilité) "
                f"{m.aj_niveau:+.1f} niveau du club {m.aj_age:+.1f} âge"
                + (f" · Score de saison **{n_(m.score_saison)}**" if pd.notna(m.score_saison) else ""))

    matchs = requete("""SELECT * FROM mon.mon_match
        WHERE playerId=? AND squadId=? AND archetype=? AND date > ? AND date <= ?
        ORDER BY date""", (int(m.playerId), int(m.squadId), arch,
                           debut.to_pydatetime(), fin.to_pydatetime()))
    tous = requete("""SELECT count(DISTINCT matchId) AS n FROM mon.mon_match
        WHERE playerId=? AND squadId=? AND date > ? AND date <= ?""",
                   (int(m.playerId), int(m.squadId), debut.to_pydatetime(), fin.to_pydatetime()))["n"].iloc[0]
    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown("##### 🗓️ Niveau de chaque match")
        if matchs.empty:
            st.caption("Aucun match détaillé à ce poste sur la période.")
        else:
            st.plotly_chart(graphe_matchs(matchs, m), width="stretch", key=f"mon_matchs_{m.playerId}_{arch}")
            n_pal = matchs["palier"].value_counts()
            adv = ", ".join(f"**{n_pal[p]}** contre le {LIBELLE_PALIER[p].lower()}"
                            for p in ("haut", "milieu", "bas") if p in n_pal)
            autres = int(tous) - matchs["matchId"].nunique()
            texte = (f"{NIVEAU_AIDE} Chaque barre = un match, scoré seul contre la référence de saison "
                     f"(couleur = niveau de l'adversaire : tiers de son championnat selon le rating Impect à "
                     f"la date du match). Adversaires sur la période : {adv}. Un match isolé est très bruité "
                     "(±15 à 20 points sans que rien ne change) : c'est la série qui compte. La ligne en "
                     "tirets est la performance de toute la période, la ligne pleine celle retenue pour le "
                     "classement (ramenée vers 50 à proportion du peu de temps joué).")
            if autres > 0:
                texte += f" {autres} autre(s) match(s) joué(s) à un autre poste, hors de ce classement."
            st.caption(texte)
            with st.expander(f"Détail des {len(matchs)} matchs"):
                vue = matchs.assign(
                    date=pd.to_datetime(matchs["date"]).dt.strftime("%d/%m/%Y"),
                    palier=matchs["palier"].map(LIBELLE_PALIER),
                    niveau=matchs["niveau"].round(0), opp_rating=matchs["opp_rating"].round(3))
                st.dataframe(vue, hide_index=True, width="stretch",
                             column_order=["date", "adversaire", "palier", "opp_rating", "minutes",
                                           "position", "niveau"],
                             column_config={"opp_rating": st.column_config.NumberColumn("Rating adverse"),
                                            "palier": st.column_config.TextColumn("Niveau de l'adversaire"),
                                            "niveau": st.column_config.NumberColumn("Niveau du match")})
    with g2:
        st.markdown("##### 🧱 Piliers : période vs saison")
        pm = requete("""SELECT p.pilier, p.percentile, w.poids FROM mon.mon_pilier p
            JOIN mon.mon_pilier_poids w USING (archetype, pilier)
            WHERE p.periode_jours=? AND p.playerId=? AND p.squadId=? AND p.archetype=?""",
                     (periode, int(m.playerId), int(m.squadId), arch))
        ps = (requete("""SELECT pilier, percentile, poids FROM fait_pilier
            WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype=?""",
                      (int(fiche.playerId), int(fiche.squadId), int(fiche.iterationId), fiche.position, arch))
              if fiche is not None and not autre_poste
              else pd.DataFrame(columns=["pilier", "percentile", "poids"]))
        if not pm.empty:
            st.plotly_chart(graphe_piliers_periode(pm, ps), width="stretch", key=f"mon_piliers_{m.playerId}_{arch}")
            absents = [p.replace("_", " ") for p in ps["pilier"] if p not in set(pm["pilier"])]
            if absents:
                st.caption(f"Non mesurés sur une période ({', '.join(absents)}) : ils reposent sur des "
                           "statistiques de saison, sans équivalent match par match.")
        if m.corrigees and isinstance(m.metriques_corrigees, str) and m.metriques_corrigees:
            st.caption("Métriques ramenées dans la plage de confiance : "
                       + ", ".join(m.metriques_corrigees.split(";")) + ".")
    section_heatmap_periode(m)


def carte_monitoring(m) -> None:
    """Fiche d'un joueur du monitoring : sa fiche de saison habituelle, avec
    la periode monitoree en plus. Sans fiche de saison (moins de 400 minutes
    evaluees a ce poste, toutes saisons), en-tete minimal + la periode."""
    fiche = pd.DataFrame()
    if pd.notna(m.fiche_iterationId):
        fiche = requete(f"""SELECT {CHAMPS} FROM v_joueurs
            WHERE playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype=?""",
                        (int(m.playerId), int(m.fiche_squadId), int(m.fiche_iterationId),
                         m.fiche_position, m.fiche_archetype))
    if not fiche.empty:
        f = fiche.iloc[0]
        carte_joueur(f, extra=lambda: section_monitoring(m, f))
        return
    st.subheader(m.nom)
    lieu = f"{m.club} · {m.competition}"
    if pd.notna(m.pays):
        lieu += f" ({m.pays}" + (f", D{int(m.niveau)})" if pd.notna(m.niveau) else ")")
    st.caption(f"{lieu} · {m.saison} · {m.poste} · archétype {m.archetype_courant}")
    j = m.copy()
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button(f"Ajouter à « {shortlist} »", width="stretch", key=f"mon_add_{m.playerId}"):
        ajouter(shortlist, j); st.toast(f"{m.nom} ajouté à « {shortlist} »"); st.rerun()
    if b2.button("🚫 Exclure ce joueur", width="stretch", key=f"mon_exc_{m.playerId}"):
        ajouter("exclus", j); st.toast(f"{m.nom} exclu"); st.rerun()
    section_monitoring(m, None)
    st.info("Pas encore de fiche de saison : il faut 400 minutes jouées à ce poste sur une saison "
            "pour être évalué par le scoring de saison.")


def vue_monitoring() -> None:
    if MON is None:
        st.warning("Base du monitoring absente : relance `python impect-scouting/db_build.py`, "
                   "puis `publier.py` pour la mettre en ligne.")
        return
    debut, fin = bornes_periode()
    st.markdown(f"#### Monitoring {archetype}")
    st.title(ARCHETYPES[archetype])
    rayures(fine=True)
    filtre = " AND ".join(where)
    stats_m = requete(f"""SELECT count(*) AS n, median(score) AS med, max(score) AS max_,
                                 median(n_matchs) AS matchs_med
                          FROM mon.mon_joueur WHERE {filtre}""", tuple(params)).iloc[0]
    total_m = int(stats_m["n"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Joueurs actifs", f"{total_m}")
    c2.metric("Score médian", n_(stats_m["med"]))
    c3.metric("Meilleur score", n_(stats_m["max_"]))
    c4.metric("Matchs médians", n_(stats_m["matchs_med"], "{:.0f}"))
    if total_m == 0:
        st.warning("Aucun joueur ne correspond à ces filtres sur la période.")
        return
    page_m = page_courante(f"mon|{filtre}|{params}|{tri_monitoring}")
    n_pages_m = max(1, -(-total_m // PAR_PAGE))
    page_m = min(page_m, n_pages_m - 1)
    tri = TRI_MONITORING[tri_monitoring]
    res_m = requete(f"""SELECT {MON_CHAMPS}
        FROM (SELECT * FROM mon.mon_joueur WHERE {filtre}) m {MON_JOINTURE}
        ORDER BY m.{tri} DESC, m.score DESC
        LIMIT {PAR_PAGE} OFFSET {page_m * PAR_PAGE}""", tuple(params))
    premier, dernier = page_m * PAR_PAGE + 1, page_m * PAR_PAGE + len(res_m)
    st.caption(f"Joueurs ayant joué au moins {minutes_min} min à ce poste entre le {debut:%d/%m/%Y} et le "
               f"{fin:%d/%m/%Y} (données de matchs publiées le {fin:%d/%m/%Y}) : **{premier} à {dernier}** "
               f"sur {total_m}, classés par {tri_monitoring.lower()}. 👉 Clique sur une ligne pour ouvrir la "
               "fiche : sa fiche de saison, plus le niveau de chacun de ses matchs de la période.")
    event = st.dataframe(
        res_m, hide_index=True, width="stretch", height=430, key=f"tableau_mon_{page_m}",
        on_select="rerun", selection_mode="single-row",
        column_order=["rang", "nom", "club", "competition", "pays", "niveau", "matchs", "minutes", "score",
                      "performance", "perf_brute", "score_saison", "aj_niveau", "aj_age", "confiance",
                      "corrigees", "age", "pied_fort"],
        column_config={
            "rang": st.column_config.NumberColumn("Rang", format="%d",
                                                  help="Rang au Score parmi tous les joueurs du poste actifs "
                                                       "sur la période, avant tes filtres."),
            "matchs": st.column_config.NumberColumn("Matchs", help="Matchs joués à ce poste sur la période."),
            "minutes": st.column_config.NumberColumn("Min", format="%d"),
            "score": st.column_config.ProgressColumn(
                "Score période", min_value=0, max_value=120, format="%.1f",
                help="Performance sur la période + niveau du club + âge : même formule et même échelle que "
                     "le Score de saison."),
            "performance": st.column_config.NumberColumn(
                "Performance", format="%.1f",
                help="Performance terrain sur la période, après les garde-fous petits échantillons (c'est "
                     "elle qui entre dans le Score). 50 = joueur médian du poste."),
            "perf_brute": st.column_config.NumberColumn(
                "Perf. brute", format="%.1f",
                help="La même période sans les garde-fous du monitoring : ce qu'il a montré, bruit compris. "
                     "Très au-dessus de la performance retenue = gros match(s) sur peu de temps de jeu."),
            "score_saison": st.column_config.NumberColumn("Score saison", format="%.1f",
                                                          help="Score de sa fiche de saison."),
            "aj_niveau": st.column_config.NumberColumn("Aj. niveau", format="%+.1f"),
            "aj_age": st.column_config.NumberColumn("Aj. âge", format="%+.1f"),
            "confiance": st.column_config.ProgressColumn(
                "Fiabilité", min_value=0, max_value=100, format="%.0f %%",
                help="Part du signal conservée : 30 % sous un match plein, 60 % à 400 min, 100 % à 15 "
                     "matchs."),
            "corrigees": st.column_config.NumberColumn(
                "Métr. corrigées", help="Métriques ramenées dans la plage de confiance (plus étroite à "
                                        "petit volume)."),
        })
    if n_pages_m > 1:
        p1, p2, p3 = st.columns([1, 2, 1])
        if p1.button("◀ 500 précédents", disabled=page_m == 0, width="stretch", key="mon_prec"):
            st.session_state["page"] = page_m - 1
            st.rerun()
        p2.markdown(f"<div style='text-align:center;padding-top:6px'>Page {page_m + 1} / {n_pages_m}</div>",
                    unsafe_allow_html=True)
        if p3.button("500 suivants ▶", disabled=page_m >= n_pages_m - 1, width="stretch", key="mon_suiv"):
            st.session_state["page"] = page_m + 1
            st.rerun()
    st.download_button("Télécharger la page affichée (CSV)", en_csv(res_m),
                       f"monitoring_{archetype}_{periode}j.csv", "text/csv")
    st.divider()
    lignes = event.selection["rows"] if event and "rows" in event.selection else []
    carte_monitoring(res_m.iloc[lignes[0] if lignes else 0])


club_param = st.query_params.get("club")
fiche_param = st.query_params.get("fiche")

# Chaque vue s'affiche dans l'onglet ouvert. Une fiche ou un effectif ouvert
# depuis le Monitoring y reste : le bouton retour ramene au classement.
with (onglet_mon if MONITORING else
      onglet_rcsc if VUE_RCSC else
      onglet_pression if VUE_PRESSION else onglet_saison):
    # Rappel visible sur chaque vue : un Score lu avec des poids modifies ne doit
    # jamais pouvoir etre pris pour le Score d'origine.
    if POIDS_MODIFIES:
        st.info(f"**Poids modifiés** — âge {K_AGE * 100:.0f} %, niveau du club {K_NIVEAU * 100:.0f} %. "
                "Score, Score hors âge, rangs et percentiles sont recalculés avec ces poids ; la Qualité "
                "actuelle est inchangée. Réglage dans la barre latérale, « ⚖️ Poids de l'âge et du niveau ».",
                icon="⚖️")
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
        st.markdown("#### Effectif")
        st.title(str(e.club))
        st.caption(f"{e.competition} · {e.saison}"
                   + (f" · {e.pays}" if pd.notna(e.pays) else "")
                   + f" · rating club {n_(e.rating, '{:.3f}')}")

        effectif = requete(f"""SELECT {CHAMPS} FROM v_joueurs
            WHERE squadId=? AND iterationId=?
            QUALIFY row_number() OVER (PARTITION BY playerId ORDER BY {SC} DESC) = 1
            ORDER BY position, {SC} DESC""", (squad_id, iter_id))
        m1, m2, m3 = st.columns(3)
        m1.metric("Joueurs", len(effectif))
        m2.metric(f"{COURT} médiane", n_(effectif["score"].median()))
        m3.metric("Âge médian", n_(effectif["age"].median()))
        st.caption("👉 Clique sur une ligne pour ouvrir la fiche du joueur.")
        ev = st.dataframe(
            effectif, hide_index=True, width="stretch", height=430, key="effectif",
            on_select="rerun", selection_mode="single-row",
            column_order=["nom", "poste", "archetype_courant", "profil_principal", "score", "age", "minutes",
                          "pied_fort", "performance", "progression", "gros_matchs"],
            column_config={"score": st.column_config.ProgressColumn(
                lecture, min_value=0, max_value=120, format="%.1f"),
                "archetype_courant": st.column_config.TextColumn("Archétype"),
                "profil_principal": st.column_config.TextColumn("Profil RCSC")})
        st.divider()
        sel = ev.selection["rows"] if ev and "rows" in ev.selection else []
        carte_joueur(effectif.iloc[sel[0] if sel else 0])

    elif MONITORING:
        vue_monitoring()

    elif VUE_RCSC or VUE_PRESSION:
        # L'onglet maison se dessine dans son propre bloc plus bas : ici on ne
        # fait rien, surtout pas le classement (ses donnees ne sont meme pas
        # calculees quand cet onglet est ouvert).
        pass

    else:
        # ------------------------------------------------------------- vue joueurs
        st.markdown(f"#### Classement {archetype}")
        st.title(ARCHETYPES[archetype])
        rayures(fine=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Joueurs trouvés", f"{total}")
        c2.metric(f"{COURT} médiane", n_(stats["med"]))
        c3.metric(f"Meilleure {COURT.lower()}", n_(stats["max_"]))
        c4.metric("Âge médian", n_(stats["age_med"]))

        if res.empty:
            st.warning("Aucun joueur ne correspond à ces filtres.")
            st.stop()

        res = res.copy()
        res["archetype_courant"] = archetype
        # Repere d'echantillon reduit : seulement si le filtre laisse passer des
        # joueurs sous MINUTES_FIABLES (au reglage par defaut, la colonne n'existe pas).
        _reduits = res["minutes"].map(echantillon_reduit)
        _col_prudence = []
        if minutes_min < MINUTES_FIABLES and _reduits.any():
            res["prudence"] = _reduits.map({True: "⚠", False: ""})
            _col_prudence = ["prudence"]
        premier, dernier = page * PAR_PAGE + 1, page * PAR_PAGE + len(res)
        if profil_id:
            _tri = "correspondance au profil" if tri_profil == "Correspondance" else lecture.lower()
            st.caption(f"Joueurs **{premier} à {dernier}** sur {total} qui correspondent au profil "
                       f"**{_choix_profil}**, classés par {_tri}. 👉 Clique sur une ligne pour ouvrir la fiche.")
            _colonnes_profil = ["corr_profil"]
        else:
            st.caption(f"Joueurs **{premier} à {dernier}** sur {total}, classés par {lecture.lower()}. "
                       "👉 Clique sur une ligne pour ouvrir la fiche du joueur.")
            _colonnes_profil = ["profil_principal"]
        event = st.dataframe(
            res, hide_index=True, width="stretch", height=430, key=f"tableau_{page}",
            on_select="rerun", selection_mode="single-row",
            column_order=["nom", "club", "competition", "pays", "niveau", "saison",
                          "score"] + (["role_milieu"] if archetype in ("SIX", "EIGHT") else [])
                         + _colonnes_profil + [("score_v8" if SC == "qualite_actuelle" else "qualite"),
                          "age", "minutes"] + _col_prudence + ["pied_fort", "performance", "aj_niveau",
                          "aj_age", "aj_traduction", "progression", "gros_matchs", "coef_adv",
                          "rang_mondial"],
            column_config={
                "prudence": st.column_config.TextColumn(
                    "⚠", width="small",
                    help=f"Moins de {MINUTES_FIABLES} minutes à ce poste : profil RCSC et jeu sous "
                         "pression calculés sur un échantillon réduit, à prendre avec précaution."),
                "score": st.column_config.ProgressColumn(lecture, min_value=0, max_value=120, format="%.1f",
                                                         help=LECTURES[lecture]["aide"]),
                "qualite": st.column_config.NumberColumn("Qualité", format="%.1f",
                                                         help=LECTURES["Qualité actuelle (niveau JPL)"]["aide"]),
                "score_v8": st.column_config.NumberColumn(
                    "Score", format="%.1f", help=LECTURES["Score (performance + niveau + âge)"]["aide"]),
                "aj_niveau": st.column_config.NumberColumn(
                    "Aj. niveau", format="%+.1f",
                    help=f"Ajustement du Score pour le niveau du club : {_pente_niv:.0f} points par point "
                         "de rating d'écart avec la référence (0,49), linéaire."),
                "aj_traduction": st.column_config.NumberColumn(
                    "Trad. JPL", format="%+.1f",
                    help="Traduction au niveau JPL (lecture Qualité actuelle) : de combien sa performance "
                         "baisserait (ou monterait) dans un club moyen de JPL, pente mesurée sur les "
                         "transferts réels."),
                "role_milieu": st.column_config.TextColumn(
                    "Rôle", help="Rôle qu'Impect lui attribue le plus souvent : 6 (milieu défensif) ou 8 "
                                 "(milieu central), au moins 2/3 du temps de jeu ; 6/8 sinon."),
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
        if _col_prudence:
            st.caption(f"⚠ = moins de {MINUTES_FIABLES} minutes à ce poste ({int(_reduits.sum())} joueur"
                       f"{'s' if _reduits.sum() > 1 else ''} sur cette page) : profil RCSC et jeu sous "
                       "pression calculés sur un échantillon réduit, à prendre avec précaution.")
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
        d1.download_button("Télécharger la page affichée (CSV)", en_csv(res),
                           f"scouting_{archetype}.csv", "text/csv", width="stretch")
        clubs = res[["club", "squadId", "iterationId"]].drop_duplicates().sort_values("club")
        club_choisi = d2.selectbox("Voir l'effectif d'un club", ["—"] + clubs["club"].tolist(),
                                   label_visibility="collapsed")
        if club_choisi != "—" and d3.button("Ouvrir", width="stretch"):
            c_ = clubs[clubs["club"] == club_choisi].iloc[0]
            st.query_params["club"] = f"{int(c_.squadId)}-{int(c_.iterationId)}"
            st.rerun()
        st.divider()
        lignes = event.selection["rows"] if event and "rows" in event.selection else []
        carte_joueur(res.iloc[lignes[0] if lignes else 0])

# ============================================================================
# ONGLET SPORTING DE CHARLEROI -- effectif maison et equipe type 4-2-3-1
# ============================================================================
# Repond a trois questions : qui du RSC Charleroi est evalue par la plateforme,
# qui joue le plus a chaque poste, et quel archetype s'applique a ce poste.
# Le squadId est fige (374) : chercher "charleroi" par le nom ramenerait aussi
# l'Olympic Charleroi, qui est un autre club (Challenger Pro League).
RCSC_SQUAD_ID = 374

# Un poste Impect -> une case du 4-2-3-1. Les deux axes centraux (charniere,
# double pivot) prennent deux joueurs, d'ou la colonne "places".
FORMATION_4231 = [
    # (ligne, [(case, poste Impect, places)])
    ("Attaque",   [("Buteur", "CENTER_FORWARD", 1)]),
    ("Soutien",   [("Ailier gauche", "LEFT_WINGER", 1),
                   ("Meneur", "ATTACKING_MIDFIELD", 1),
                   ("Ailier droit", "RIGHT_WINGER", 1)]),
    ("Double pivot", [("Milieu", "DEFENSE_CENTRAL_MIDFIELD", 2)]),
    ("Defense",   [("Latéral gauche", "LEFT_WINGBACK_DEFENDER", 1),
                   ("Charnière", "CENTRAL_DEFENDER", 2),
                   ("Latéral droit", "RIGHT_WINGBACK_DEFENDER", 1)]),
    ("But",       [("Gardien", "GOALKEEPER", 1)]),
]


@st.cache_data(show_spinner=False)
def effectif_rcsc(saison_choisie: str, poids: tuple = POIDS_DEFAUT) -> pd.DataFrame:
    """Joueurs du RSC Charleroi evalues par la plateforme sur cette saison.

    Un joueur de DEFENSE_CENTRAL_MIDFIELD sort DEUX fois (archetypes SIX et
    EIGHT, memes minutes) : on regroupe par joueur+poste et on garde le
    meilleur archetype comme principal, en conservant la liste complete pour
    l'affichage -- c'est precisement ce qu'Alex veut voir poste par poste.
    """
    d = requete("""
        SELECT playerId, squadId, iterationId, position, archetype, nom,
               minutes_jouees AS minutes, round(score,1) AS score,
               round(score_performance,1) AS performance,
               round(qualite_actuelle,1) AS qualite, round(age_years,1) AS age,
               pied_fort, taille_cm, profil_principal, pilier_fort, pilier_faible,
               round(progression_credible,1) AS progression, rang_archetype AS rang,
               side, saison
        FROM v_joueurs WHERE squadId = ? AND saison = ?
        ORDER BY minutes_jouees DESC, score DESC""",
        (RCSC_SQUAD_ID, saison_choisie))
    if d.empty:
        return d
    # Archetype principal = celui au meilleur score pour ce joueur a ce poste.
    d = d.sort_values(["playerId", "position", "score"], ascending=[True, True, False])
    grp = d.groupby(["playerId", "position"], as_index=False)
    princ = grp.head(1).copy()
    tous = (d.groupby(["playerId", "position"])["archetype"]
              .apply(lambda x: " / ".join(dict.fromkeys(x))).rename("archetypes").reset_index())
    princ = princ.merge(tous, on=["playerId", "position"], how="left")
    # Colonnes objet contenant None : st.dataframe ecrit litteralement "None".
    # On force le numerique (vide = vide) et on remplace les textes manquants.
    for c in ("score", "performance", "qualite", "progression", "age",
              "taille_cm", "rang", "minutes"):
        princ[c] = pd.to_numeric(princ[c], errors="coerce")
    for c in ("profil_principal", "pilier_fort", "pilier_faible", "pied_fort"):
        princ[c] = princ[c].astype("object").where(princ[c].notna(), "—")
    return princ.sort_values("minutes", ascending=False).reset_index(drop=True)


def _carte_poste(col, titre: str, joueur, saison_choisie: str) -> None:
    """Une case de l'equipe type. Sans joueur, la case reste visible et dit
    pourquoi -- une equipe type trouee est une information, pas un bug."""
    with col.container(border=True):
        st.markdown(f"<div class='rcsc-case'>{titre}</div>", unsafe_allow_html=True)
        if joueur is None:
            st.markdown("<div class='rcsc-vide'>—<br><span>aucun joueur "
                        "évalué</span></div>", unsafe_allow_html=True)
            return
        st.markdown(f"**{joueur.nom}**")
        st.markdown(
            f"<div class='rcsc-meta'><b>{joueur.archetypes}</b>"
            f"{' · ' + str(joueur.profil_principal) if str(joueur.profil_principal) not in ('nan', 'None', '—') else ''}"
            f"<br>{joueur.minutes:,.0f} min · {n_(joueur.age, '{:.0f}')} ans"
            f"<br>{COURT} <b>{n_(joueur.score)}</b> · perf {n_(joueur.performance)}</div>"
            .replace(",", " "), unsafe_allow_html=True)
        if st.button("Fiche", key=f"rcsc_{saison_choisie}_{joueur.playerId}_{joueur.position}",
                     width="stretch"):
            st.session_state["_archetype_a_appliquer"] = joueur.archetype
            st.query_params["fiche"] = "~".join(str(v) for v in (
                int(joueur.playerId), int(joueur.squadId), int(joueur.iterationId),
                joueur.position, joueur.archetype))
            st.rerun()


# ============================================================ jeu sous pression
# Onglet « Jeu sous pression » (09/10/2026, Alex) : qui sont les meilleurs sous
# pression, metrique par metrique, en tenant compte du NIVEAU et de la PRESSION
# du championnat -- puis un score d'ensemble.
#
# AJUSTEMENT AU CONTEXTE. Une reussite de +3 sous pression en D2 ne vaut pas +3
# en Premier League. L'ecart est MESURE, pas suppose : sur les joueurs presents
# dans les deux jeux de saisons de l'etude (saison terminee -> saison en cours,
# meme poste), on regresse la variation de chaque metrique sur la variation du
# rating du championnat et de la pression qu'on y subit a ce poste. Mesure du
# 09/10/2026, paires a 800 minutes et plus dans la saison en cours (~4 200) :
#     reussite sous pression vs attendu   -5,4 par point de rating (t -5,9)
#     reussite vs attendu (toutes passes) -2,9 par point de rating (t -6,1),
#                                         -0,23 par point de pression (t -5,1)
#     entrees dans la surface / 90        -0,54 par point de rating (t -10,3)
#     passes progressives sous pression/90  -0,39 (t -4,4) ; +0,05 par point de
#                                         pression (t +5,6)
#     maintien de l'ambition              +0,26 par point de rating (t +3,3)
#     part de passes progressives sous pression, adversaires elimines vs
#     attendu : pas d'effet net -- non ajustes
# Les coefficients sont recalcules a chaque chargement de la base, et un
# coefficient n'est applique que s'il est net (|t| >= PRESSION_T_MIN). Chaque
# valeur est ensuite ramenee a ce qu'elle vaudrait en JPL (niveau et pression de
# la JPL au meme poste) : c'est la meme logique que la lecture Qualite actuelle.
#
# SCORE DE PRESSION. Trois axes, en percentiles du poste (apres ajustement) :
#   Resister  ne pas perdre le ballon quand on est presse          40 %
#   Oser      continuer a jouer vers l'avant quand on est presse    30 %
#   Peser     ce que ses passes produisent                          30 %
# Les poids suivent la demande (la reussite sous pression d'abord) et la fidelite
# mesuree de chaque metrique : le volume par 90, fidele a 0,26 seulement, ne
# pese que 5 %. Le score est ramene vers 50 sous PRESSION_MINUTES_PLEINES.
PRESSION_POSTES = {"CB": "Défenseurs centraux", "FB": "Latéraux", "DMCM": "Milieux défensifs et centraux",
                   "AM": "Milieux offensifs", "WG": "Ailiers", "CF": "Avant-centres", "GK": "Gardiens"}
# metrique -> (intitule, axe, poids dans le score, format, lecture)
PRESSION_METRIQUES = {
    "reussite_fp_vs_attendu_100": (
        "Réussite sous pression vs attendu", "Résister", 0.30, "%+.1f",
        "Passes réussies au-dessus de l'attendu, pour 100 passes sous forte pression, à poste, zone "
        "et distance comparables."),
    "reussite_vs_attendu_100": (
        "Réussite vs attendu (toutes passes)", "Résister", 0.10, "%+.1f",
        "La même mesure sur toutes ses passes, pressé ou non."),
    "part_prog_fp": (
        "Passes progressives sous pression (%)", "Oser", 0.20, "%.0f",
        "Part de ses passes sous forte pression qui gagnent du terrain vers le but."),
    "maintien_ambition": (
        "Maintien de l'ambition", "Oser", 0.10, "%.2f",
        "Cette part rapportée à la même part sans pression : 1,00 = il joue pareil pressé ou libre."),
    "bypassed_vs_attendu_p90": (
        "Adversaires éliminés vs attendu / 90", "Peser", 0.20, "%+.1f",
        "Adversaires supprimés par ses passes, au-dessus de l'attendu."),
    "entrees_surface_p90": (
        "Entrées dans la surface / 90", "Peser", 0.05, "%.2f",
        "Passes réussies qui font entrer le ballon dans la surface adverse."),
    "prog_fp_p90": (
        "Passes progressives sous pression / 90", "Peser", 0.05, "%.1f",
        "Volume brut, très dépendant du volume de jeu de l'équipe."),
}
PRESSION_AXES = ["Résister", "Oser", "Peser"]
PRESSION_T_MIN = 2.5                 # un coefficient d'ajustement n'est applique que s'il est net
PRESSION_CALIB_MINUTES = 800         # minutes de la saison en cours pour entrer dans la calibration
PRESSION_MINUTES_PLEINES = 900
PRESSION_REFERENCE = "Jupiler Pro League"


def _centile_ref(valeurs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Percentile de chaque valeur dans la population `ref` (masque), comme
    profils_rcsc._rang : les joueurs de reference sont classes entre eux, les
    autres sont situes parmi eux sans y entrer."""
    out = np.full(len(valeurs), np.nan)
    base = np.sort(valeurs[ref & ~np.isnan(valeurs)])
    if not len(base):
        return out
    ok = ~np.isnan(valeurs)
    g, d = np.searchsorted(base, valeurs[ok], "left"), np.searchsorted(base, valeurs[ok], "right")
    out[ok] = np.where(d > g, (g + d + 1) / 2, g) / len(base) * 100
    return np.clip(out, 0, 100)


@st.cache_data(ttl=600, show_spinner=False)
def calibration_pression() -> pd.DataFrame:
    """Effet mesure d'un changement de championnat sur chaque metrique : une
    ligne par metrique, coefficients par point de rating et par point de pression
    du championnat, avec leur t. Paires = meme joueur, meme poste, jeu de saisons
    'passe' puis 'cours'."""
    p = requete("""SELECT p.playerId, p.pos, p.jeu_saisons, s.rating_moyen_championnat AS rating,
               l.pression_passe AS ligue_pp, p.* EXCLUDE (playerId, pos, jeu_saisons)
        FROM fait_pression p JOIN dim_saison s USING (iterationId)
        LEFT JOIN dim_pression_championnat l ON l.iterationId = p.iterationId AND l.pos = p.pos""")
    a = p[p["jeu_saisons"] == "passe"].drop_duplicates(["playerId", "pos"])
    b = p[p["jeu_saisons"] == "cours"].drop_duplicates(["playerId", "pos"])
    x = a.merge(b, on=["playerId", "pos"], suffixes=("_a", "_b"))
    # Seulement les paires ou la saison en cours a assez de minutes. Les mesures
    # sont retrecies vers zero quand le joueur a peu joue : avec toutes les
    # paires, la variation est ecrasee et l'effet du niveau sous-estime (reussite
    # sous pression : -3,0 par point de rating sur toutes les paires, -5,4 sur
    # celles a 800 minutes et plus).
    x = x[pd.to_numeric(x["minutes_b"], errors="coerce") >= PRESSION_CALIB_MINUTES]
    lignes = []
    for m in PRESSION_METRIQUES:
        y = pd.to_numeric(x[f"{m}_b"], errors="coerce") - pd.to_numeric(x[f"{m}_a"], errors="coerce")
        dr, dp = x["rating_b"] - x["rating_a"], x["ligue_pp_b"] - x["ligue_pp_a"]
        ok = (y.notna() & dr.notna() & dp.notna()).to_numpy()
        ligne = {"metrique": m, "paires": int(ok.sum()), "coef_rating": 0.0, "t_rating": np.nan,
                 "coef_pression": 0.0, "t_pression": np.nan}
        if ok.sum() >= 500:
            X = np.column_stack([np.ones(ok.sum()), dr[ok].to_numpy(float), dp[ok].to_numpy(float)])
            beta, *_ = np.linalg.lstsq(X, y[ok].to_numpy(float), rcond=None)
            residus = y[ok].to_numpy(float) - X @ beta
            se = np.sqrt(np.diag(residus.var(ddof=3) * np.linalg.inv(X.T @ X)))
            ligne.update(coef_rating=float(beta[1]), t_rating=float(beta[1] / se[1]),
                         coef_pression=float(beta[2]), t_pression=float(beta[2] / se[2]))
        # Un coefficient flou vaut zero : on n'ajuste pas sur du bruit.
        ligne["applique_rating"] = ligne["coef_rating"] if abs(ligne["t_rating"]) >= PRESSION_T_MIN else 0.0
        ligne["applique_pression"] = ligne["coef_pression"] if abs(ligne["t_pression"]) >= PRESSION_T_MIN else 0.0
        lignes.append(ligne)
    return pd.DataFrame(lignes)


@st.cache_data(ttl=600, show_spinner=False)
def donnees_pression() -> pd.DataFrame:
    """Une ligne par joueur x club x saison x poste ayant des mesures de pression :
    valeurs brutes, valeurs ramenees au contexte de la JPL (aj_*), percentiles du
    poste (pc_*), les trois axes et le score."""
    metriques = ", ".join(f"any_value(p.{m}) AS {m}" for m in PRESSION_METRIQUES)
    d = requete(f"""SELECT v.playerId, v.squadId, v.iterationId, v.position, min(v.archetype) AS archetype,
               any_value(v.nom) AS nom, any_value(v.club) AS club, any_value(v.competition) AS competition,
               any_value(v.pays) AS pays, any_value(v.top5_europe) AS top5_europe, any_value(v.saison) AS saison,
               any_value(round(v.age_years, 1)) AS age, any_value(v.minutes_jouees) AS minutes,
               any_value(s.rating_moyen_championnat) AS rating, any_value(p.pos) AS pos,
               any_value(p.jeu_saisons) AS jeu, any_value(p.pression_passe) AS pression_subie,
               any_value(p.pct_passes_forte_pression) AS part_forte_pression,
               any_value(l.pression_passe) AS ligue_pp, {metriques}
        FROM v_joueurs v
        JOIN dim_poste_pression m USING (position)
        JOIN fait_pression p ON p.playerId = v.playerId AND p.iterationId = v.iterationId AND p.pos = m.pos
        JOIN dim_saison s ON s.iterationId = v.iterationId
        LEFT JOIN dim_pression_championnat l ON l.iterationId = p.iterationId AND l.pos = p.pos
        GROUP BY v.playerId, v.squadId, v.iterationId, v.position""")
    # Deux clubs dans la meme saison : la mesure de pression est par joueur x saison
    # x poste, on la rattache au club ou il a le plus joue.
    d = (d.sort_values("minutes", ascending=False).drop_duplicates(["playerId", "iterationId", "pos"])
          .reset_index(drop=True))
    cal = calibration_pression().set_index("metrique")
    # Contexte de reference : la JPL du meme jeu de saisons, au meme poste.
    jpl = d[d["competition"] == PRESSION_REFERENCE]
    rating_ref = jpl.groupby("jeu")["rating"].mean()
    pp_ref = jpl.groupby(["jeu", "pos"])["ligue_pp"].mean()
    d_rating = (d["rating"] - d["jeu"].map(rating_ref).fillna(d["rating"].mean())).fillna(0.0)
    ref_pp = pd.Series(list(zip(d["jeu"], d["pos"])), index=d.index).map(pp_ref)
    d_pp = (d["ligue_pp"] - ref_pp).fillna(0.0)
    fiable = (d["minutes"] >= PRESSION_MINUTES_PLEINES).to_numpy()
    groupes = d.groupby(["jeu", "pos"]).indices
    for m in PRESSION_METRIQUES:
        v = pd.to_numeric(d[m], errors="coerce")
        # valeur en JPL = valeur ici - effet du contexte d'ici par rapport a la JPL
        d[f"aj_{m}"] = v - cal.loc[m, "applique_rating"] * d_rating - cal.loc[m, "applique_pression"] * d_pp
        pc = np.full(len(d), np.nan)
        for _, lignes in groupes.items():
            pc[lignes] = _centile_ref(d[f"aj_{m}"].to_numpy(dtype=float)[lignes], fiable[lignes])
        d[f"pc_{m}"] = pc
    poids_tot, somme = np.zeros(len(d)), np.zeros(len(d))
    for axe in PRESSION_AXES:
        ms = [m for m, x in PRESSION_METRIQUES.items() if x[1] == axe]
        w = np.array([PRESSION_METRIQUES[m][2] for m in ms])
        vals = d[[f"pc_{m}" for m in ms]].to_numpy(dtype=float)
        ok = ~np.isnan(vals)
        den = (ok * w).sum(axis=1)
        d[f"axe_{axe}"] = np.where(den > 0, np.where(ok, vals, 0.0).dot(w) / np.where(den > 0, den, 1), np.nan)
        somme += np.where(ok, vals, 0.0).dot(w)
        poids_tot += den
    brut = np.where(poids_tot > 0, somme / np.where(poids_tot > 0, poids_tot, 1), np.nan)
    # Sans la reussite sous pression, pas de score : c'est la mesure centrale. Et il
    # faut au moins 60 % du poids mesure (gardiens : la partie progression manque).
    brut = np.where(d["pc_reussite_fp_vs_attendu_100"].isna() | (poids_tot < 0.6), np.nan, brut)
    volume = (d["minutes"].fillna(0) / PRESSION_MINUTES_PLEINES).clip(0, 1).to_numpy(dtype=float)
    d["score_brut"] = brut
    d["fiabilite"] = np.round(100 * volume, 0)
    d["score_pression"] = np.round(50 + (brut - 50) * (0.5 + 0.5 * volume), 1)
    d["couverture"] = np.round(100 * poids_tot, 0)
    return d


def vue_pression() -> None:
    st.markdown("#### Étude de pression")
    st.title("Jeu sous pression")
    rayures(fine=True)
    if not pression_dispo():
        st.warning("La base ne contient pas l'étude de pression.")
        return
    d = donnees_pression()
    f1, f2, f3 = st.columns([2, 2, 3])
    pos = f1.selectbox("Poste", list(PRESSION_POSTES), format_func=PRESSION_POSTES.get, key="pr_poste")
    jeu = f2.radio("Saisons", ["cours", "passe"], horizontal=True, key="pr_jeu",
                   format_func={"cours": "Saison en cours", "passe": "Saisons terminées"}.get)
    CLASSER = {"score_pression": "Score de pression",
               **{m: x[0] for m, x in PRESSION_METRIQUES.items()}}
    tri = f3.selectbox("Classer par", list(CLASSER), format_func=CLASSER.get, key="pr_tri")
    g1, g2, g3, g4 = st.columns([2, 2, 2, 2])
    age_max_p = g1.slider("Âge maximum", 16, 40, 40, key="pr_age")
    min_p = g2.slider("Minutes minimum", 400, 3000, 900, step=100, key="pr_min")
    sans_top5 = g3.checkbox("Sans les 5 grands championnats", key="pr_top5")
    # La MLS seule, comme dans la barre laterale : MLS Next Pro et les USL restent.
    sans_mls = g3.checkbox("Exclure la MLS", key="pr_mls")
    ajuste = g4.toggle("Ajusté au contexte de la JPL", value=True, key="pr_ajuste",
                       help="Chaque valeur est ramenée à ce qu'elle vaudrait en JPL, d'après l'effet mesuré "
                            "du niveau et de la pression du championnat. Décoché : valeurs brutes.")
    pays_p = st.multiselect("Pays", sorted(d["pays"].dropna().unique()), key="pr_pays",
                            placeholder="Tous les pays")
    v = d[(d["pos"] == pos) & (d["jeu"] == jeu) & (d["minutes"] >= min_p)]
    if age_max_p < 40:
        v = v[v["age"] <= age_max_p]
    if sans_top5:
        v = v[~v["top5_europe"].fillna(False).astype(bool)]
    if sans_mls:
        v = v[v["competition"] != "Major League Soccer"]
    if pays_p:
        v = v[v["pays"].isin(pays_p)]
    pref = "aj_" if ajuste else ""
    col_tri = tri if tri == "score_pression" else f"{pref}{tri}"
    v = v.dropna(subset=[col_tri]).sort_values(col_tri, ascending=False).reset_index(drop=True)
    if v.empty:
        st.warning("Aucun joueur ne correspond à ces filtres.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Joueurs", f"{len(v):,}".replace(",", " "))
    c2.metric("Score de pression médian", n_(v["score_pression"].median()))
    c3.metric("Championnats", v["competition"].nunique())
    c4.metric("Âge médian", n_(v["age"].median()))
    vue = v.head(PAR_PAGE).copy()
    vue.insert(0, "rang", np.arange(1, len(vue) + 1))
    vue["prudence"] = np.where(vue["minutes"] < PRESSION_MINUTES_PLEINES, "⚠", "")
    st.caption(f"Joueurs **1 à {len(vue)}** sur {len(v)}, classés par **{CLASSER[tri].lower()}**"
               + (" (valeur ramenée au contexte de la JPL)" if ajuste and tri != "score_pression" else "")
               + ". 👉 Clique sur une ligne pour ouvrir la fiche du joueur.")
    cols_m = [f"{pref}{m}" for m in PRESSION_METRIQUES]
    ev = st.dataframe(
        vue, hide_index=True, width="stretch", height=430, key=f"pr_tbl_{pos}_{jeu}_{tri}_{ajuste}",
        on_select="rerun", selection_mode="single-row",
        column_order=["rang", "nom", "club", "competition", "saison", "age", "minutes", "prudence",
                      "score_pression"] + [f"axe_{a}" for a in PRESSION_AXES] + cols_m
                     + ["pression_subie", "part_forte_pression"],
        column_config={
            "rang": st.column_config.NumberColumn("Rang", format="%d"),
            "minutes": st.column_config.NumberColumn("Min", format="%d"),
            "prudence": st.column_config.TextColumn(
                "⚠", width="small", help=f"Moins de {PRESSION_MINUTES_PLEINES} minutes : mesures moins fiables, "
                                         "score ramené vers 50."),
            "score_pression": st.column_config.ProgressColumn(
                "Score de pression", min_value=0, max_value=100, format="%.1f",
                help="Résister 40 % · Oser 30 % · Peser 30 %, en percentiles du poste après ajustement "
                     "au contexte de la JPL. 50 = joueur médian du poste."),
            **{f"axe_{a}": st.column_config.NumberColumn(a, format="%.0f",
                                                         help=f"Percentile du poste sur l'axe « {a} ».")
               for a in PRESSION_AXES},
            **{f"{pref}{m}": st.column_config.NumberColumn(x[0], format=x[3], help=x[4])
               for m, x in PRESSION_METRIQUES.items()},
            "pression_subie": st.column_config.NumberColumn(
                "Pression subie", format="%.1f",
                help="Pression adverse moyenne (0-100) quand il donne le ballon. Contexte, hors score : "
                     "elle décrit son rôle et sa zone."),
            "part_forte_pression": st.column_config.NumberColumn(
                "% passes pressées", format="%.0f", help="Part de ses passes faites sous forte pression."),
        })
    choix = ev.selection["rows"] if ev and "rows" in ev.selection else []
    if choix:
        c = vue.iloc[choix[0]]
        if st.button(f"Ouvrir la fiche de {c.nom} ({c.saison}, {c.club})", key="pr_ouvrir", type="primary"):
            st.session_state["_archetype_a_appliquer"] = c.archetype
            ouvrir_fiche(int(c.playerId), int(c.squadId), int(c.iterationId), c.position, c.archetype)
    st.download_button("Télécharger le classement affiché (CSV)",
                       en_csv(vue[[c for c in vue.columns if not c.startswith("pc_")]]),
                       f"pression_{pos}_{jeu}.csv", "text/csv")

    # ---- Les meilleurs, metrique par metrique
    st.markdown("##### Les meilleurs sur chaque métrique")
    st.caption("Mêmes filtres que le classement. Cinq premiers par métrique"
               + (", valeurs ramenées au contexte de la JPL." if ajuste else ", valeurs brutes."))
    liste = list(PRESSION_METRIQUES.items())
    for debut in range(0, len(liste), 3):
        for col, (m, x) in zip(st.columns(3), liste[debut:debut + 3]):
            top = v.dropna(subset=[f"{pref}{m}"]).nlargest(5, f"{pref}{m}")
            with col.container(border=True):
                st.markdown(f"**{x[0]}**  \n:gray[axe {x[1]} · {x[2] * 100:.0f} % du score]")
                st.dataframe(top[["nom", "club", f"{pref}{m}"]], hide_index=True, width="stretch",
                             column_config={f"{pref}{m}": st.column_config.NumberColumn("Valeur", format=x[3]),
                                            "nom": st.column_config.TextColumn("Joueur"),
                                            "club": st.column_config.TextColumn("Club")})

    # ---- Methode
    with st.expander("Comment ces chiffres sont construits"):
        cal = calibration_pression()
        fia = pression_fiabilite()
        st.markdown(
            "**1. Les mesures** viennent de l'étude de pression : chaque passe est comparée à l'attendu de "
            "sa situation (poste, niveau de pression, zone, distance). Elles sont calculées par poste.\n\n"
            "**2. L'ajustement au contexte.** Sur les joueurs présents dans deux saisons de l'étude au même "
            "poste, on mesure de combien chaque métrique bouge quand le championnat change de niveau "
            "(rating Impect) et de pression. Un coefficient n'est appliqué que s'il est net "
            f"(|t| ≥ {PRESSION_T_MIN}) ; sinon la métrique n'est pas ajustée. Chaque valeur est ensuite "
            f"ramenée à ce qu'elle vaudrait en {PRESSION_REFERENCE}, au même poste.")
        _effet = float(cal.set_index("metrique").loc["reussite_fp_vs_attendu_100", "coef_rating"])
        t = cal.assign(metrique=cal["metrique"].map(lambda m: PRESSION_METRIQUES[m][0]),
                       fidelite=cal["metrique"].map(fia))
        st.dataframe(
            t[["metrique", "paires", "coef_rating", "t_rating", "applique_rating", "coef_pression",
               "t_pression", "applique_pression", "fidelite"]], hide_index=True, width="stretch",
            column_config={
                "metrique": st.column_config.TextColumn("Métrique"),
                "paires": st.column_config.NumberColumn("Paires", format="%d"),
                "coef_rating": st.column_config.NumberColumn("Effet par point de rating", format="%+.2f"),
                "t_rating": st.column_config.NumberColumn("t", format="%+.1f"),
                "applique_rating": st.column_config.NumberColumn("Appliqué", format="%+.2f"),
                "coef_pression": st.column_config.NumberColumn("Effet par point de pression", format="%+.3f"),
                "t_pression": st.column_config.NumberColumn("t", format="%+.1f"),
                "applique_pression": st.column_config.NumberColumn("Appliqué", format="%+.3f"),
                "fidelite": st.column_config.NumberColumn(
                    "Fidélité", format="%.2f", help="Matchs pairs vs impairs. Au-dessus de 0,60 la mesure "
                                                    "décrit le joueur.")})
        st.markdown(
            f"Lecture : un effet de {_effet:+.1f} par point de rating veut dire que la réussite sous "
            f"pression d'un joueur bouge de {_effet:+.1f} quand il monte d'un point de rating (de la D2 "
            f"belge à la JPL : +0,26 de rating, soit {_effet * 0.26:+.1f}). "
            "Un joueur d'un championnat plus fort que la JPL est donc relevé, un joueur d'un championnat "
            "plus faible abaissé.\n\n"
            "**3. Le score de pression** : percentile du poste sur chaque métrique ajustée, puis "
            "**Résister 40 %** (réussite sous pression 30, réussite toutes passes 10), **Oser 30 %** "
            "(part de passes progressives sous pression 20, maintien de l'ambition 10), **Peser 30 %** "
            "(adversaires éliminés vs attendu 20, entrées dans la surface 5, volume par 90 : 5). Les "
            f"percentiles sont pris parmi les joueurs à {PRESSION_MINUTES_PLEINES} minutes et plus ; sous ce "
            "volume le score est ramené vers 50 et la ligne porte un ⚠.\n\n"
            "**À garder en tête** : ces mesures décrivent un style et une solidité sous pression. L'étude a "
            "montré qu'elles prédisent très peu la performance de la saison suivante une fois le score "
            "connu : ce classement complète le Score, il ne le remplace pas.")


with onglet_pression:
    if VUE_PRESSION and not fiche_param and not club_param:
        vue_pression()


with onglet_rcsc:
    if fiche_param:
        pass                      # la fiche ouverte depuis cet onglet s'affiche plus haut
    else:
        st.markdown("#### Effectif maison")
        st.title("Sporting de Charleroi")
        rayures(fine=True)

        _saisons_rcsc = requete(
            "SELECT DISTINCT saison FROM v_joueurs WHERE squadId = ? ORDER BY saison DESC",
            (RCSC_SQUAD_ID,))["saison"].tolist()
        if not _saisons_rcsc:
            st.warning("Aucun joueur du RSC Charleroi dans cette base.")
        else:
            saison_rcsc = st.radio("Saison", _saisons_rcsc, horizontal=True,
                                   key="saison_rcsc")
            eff = effectif_rcsc(saison_rcsc, poids_courants())

            if eff.empty:
                st.warning(f"Aucun joueur évalué sur {saison_rcsc}.")
            else:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Joueurs évalués", eff["playerId"].nunique())
                k2.metric(f"{COURT} médian", n_(eff["score"].median()))
                k3.metric("Âge médian", n_(eff["age"].median()))
                k4.metric("Minutes cumulées", f"{eff['minutes'].sum():,.0f}".replace(",", " "))
                st.caption(
                    "Seuls les joueurs ayant assez joué à un poste sont évalués par la pipeline "
                    "(400 minutes au minimum) : un joueur peu utilisé n'apparaît pas, "
                    "et un joueur ayant tenu deux postes apparaît une fois par poste.")

                # ---------------- equipe type ----------------
                st.markdown("#### Équipe type 4-2-3-1")
                st.caption("À chaque poste, le joueur qui a le plus joué **à ce poste** sur la "
                           "saison. 👉 Le bouton ouvre sa fiche complète.")
                pris: set = set()
                for _ligne, cases in FORMATION_4231:
                    largeurs = []
                    for _t, _p, places in cases:
                        largeurs += [1] * places
                    # centrage : marges laterales sur les lignes courtes
                    marge = max(0, (4 - sum(1 for _ in largeurs)) / 2)
                    cols = st.columns(([marge] if marge else []) + largeurs
                                      + ([marge] if marge else []))
                    i = 1 if marge else 0
                    for titre, poste, places in cases:
                        candidats = eff[(eff["position"] == poste)
                                        & (~eff["playerId"].isin(pris))]
                        for k in range(places):
                            j = None
                            if len(candidats) > k:
                                j = candidats.iloc[k]
                                pris.add(int(j.playerId))
                            lbl = titre if places == 1 else f"{titre} {k + 1}"
                            _carte_poste(cols[i], lbl, j, saison_rcsc)
                            i += 1

                # ---------------- archetypes par poste ----------------
                st.markdown("#### Nos archétypes, poste par poste")
                st.caption("Quel archétype la plateforme applique à chaque poste du club, "
                           "et combien de joueurs y sont évalués.")
                lignes_arch = []
                for _ligne, cases in FORMATION_4231:
                    for titre, poste, _pl in cases:
                        sub = eff[eff["position"] == poste]
                        lignes_arch.append({
                            "Case": titre,
                            "Poste Impect": poste,
                            "Archétype(s)": " / ".join(dict.fromkeys(sub["archetypes"]))
                                            if not sub.empty else "—",
                            "Joueurs": len(sub),
                            "Minutes": int(sub["minutes"].sum()) if not sub.empty else 0,
                            # texte : une colonne numerique vide s'affiche "None"
                            f"Meilleur {COURT.lower()}":
                                f"{sub['score'].max():.1f}" if not sub.empty else "—",
                        })
                st.dataframe(pd.DataFrame(lignes_arch), hide_index=True, width="stretch")

                # ---------------- effectif complet ----------------
                st.markdown("#### Tous les joueurs évalués")
                st.caption("Un joueur ayant tenu deux postes apparaît une ligne par poste. "
                           "👉 Coche une ligne pour ouvrir sa fiche.")
                # Une colonne entierement vide est affichee "None" par
                # st.dataframe des qu'elle a un format : on la retire plutot.
                # (la progression demande deux fenetres de match, elle n'existe
                # donc pas en debut de saison)
                _cols_rcsc = [c for c in
                              ["nom", "position", "archetypes", "profil_principal", "minutes",
                               "score", "performance", "qualite", "progression", "age",
                               "pied_fort", "taille_cm", "pilier_fort", "pilier_faible", "rang"]
                              if eff[c].notna().any()]
                ev_rcsc = st.dataframe(
                    eff, hide_index=True, width="stretch", key="tbl_rcsc",
                    on_select="rerun", selection_mode="single-row",
                    column_order=_cols_rcsc,
                    column_config={
                        "nom": st.column_config.TextColumn("Joueur"),
                        "position": st.column_config.TextColumn("Poste"),
                        "minutes": st.column_config.NumberColumn("Minutes", format="%d"),
                        "age": st.column_config.NumberColumn("Âge", format="%.1f"),
                        "pied_fort": st.column_config.TextColumn("Pied"),
                        "archetypes": st.column_config.TextColumn("Archétype(s)"),
                        "profil_principal": st.column_config.TextColumn("Profil RCSC"),
                        "score": st.column_config.ProgressColumn(
                            COURT, min_value=0, max_value=120, format="%.1f"),
                        "performance": st.column_config.NumberColumn("Perf.", format="%.1f"),
                        "qualite": st.column_config.NumberColumn("Qualité", format="%.1f"),
                        "progression": st.column_config.NumberColumn("Progr.", format="%+.1f"),
                        "taille_cm": st.column_config.NumberColumn("Taille", format="%d"),
                        "pilier_fort": st.column_config.TextColumn("Point fort"),
                        "pilier_faible": st.column_config.TextColumn("Point faible"),
                        "rang": st.column_config.NumberColumn("Rang mondial", format="%d"),
                    })
                _sel_r = ev_rcsc.selection["rows"] if ev_rcsc and "rows" in ev_rcsc.selection else []
                if _sel_r:
                    _j = eff.iloc[_sel_r[0]]
                    if st.button(f"Ouvrir la fiche de {_j.nom}", type="primary",
                                 key="rcsc_ouvrir", width="stretch"):
                        st.session_state["_archetype_a_appliquer"] = _j.archetype
                        st.query_params["fiche"] = "~".join(str(v) for v in (
                            int(_j.playerId), int(_j.squadId), int(_j.iterationId),
                            _j.position, _j.archetype))
                        st.rerun()
                st.download_button(
                    "Exporter l'effectif (CSV)", en_csv(eff),
                    f"rcsc_{saison_rcsc.replace('/', '-')}.csv", "text/csv", key="dl_rcsc")

# ----------------------------------------------------------------- mes listes
# Refonte 02/10/2026 : la liste n'est plus un tableau mort. Les lignes stockees
# ne gardent que le nom/club/score du jour de l'ajout -- on les RE-JOINT a
# v_joueurs pour afficher des valeurs a jour, ouvrir la fiche d'un clic,
# comparer plusieurs joueurs et deplacer des joueurs entre listes.
st.divider()


@st.cache_data(show_spinner=False)
def enrichir_liste(player_ids: tuple, lecture_col: str, poids: tuple = POIDS_DEFAUT) -> pd.DataFrame:
    """Valeurs actuelles des joueurs d'une liste, meilleure ligne par joueur.

    Un joueur peut avoir plusieurs lignes (saisons, postes) : on garde celle de
    score le plus haut, sinon la liste afficherait des doublons sans que l'on
    sache lequel est le bon.
    """
    if not player_ids:
        return pd.DataFrame()
    marques = ",".join("?" * len(player_ids))
    return requete(f"""
        SELECT playerId, nom AS nom_actuel, club AS club_actuel, competition, pays, saison,
               archetype AS archetype_actuel, position AS position_actuelle,
               round({lecture_col},1) AS score_actuel,
               round(score_performance,1) AS performance, round(qualite_actuelle,1) AS qualite,
               round(progression_credible,1) AS progression, round(age_years,1) AS age,
               minutes_jouees AS minutes, pied_fort, taille_cm, profil_principal,
               pilier_fort, pilier_faible, rang_archetype AS rang, squadId, iterationId
        FROM v_joueurs WHERE playerId IN ({marques})
        QUALIFY row_number() OVER (PARTITION BY playerId ORDER BY {lecture_col} DESC) = 1
        """, tuple(int(p) for p in player_ids))


@st.cache_data(show_spinner=False)
def piliers_de(cles: tuple) -> pd.DataFrame:
    """Percentiles par pilier pour un lot de joueurs (comparaison)."""
    if not cles:
        return pd.DataFrame()
    ou = " OR ".join(["(playerId=? AND squadId=? AND iterationId=? AND position=? AND archetype=?)"] * len(cles))
    pa = [v for c in cles for v in c]
    return requete(f"SELECT playerId, pilier, poids, percentile FROM fait_pilier WHERE {ou}", tuple(pa))


_toutes = listes_noms()
_onglets_noms = ([shortlist] if shortlist not in _toutes else []) + _toutes
st.markdown("#### Suivi")
st.subheader(f"Mes listes · {len(_toutes)}")
st.caption("Navigue d'une liste à l'autre par les onglets. Les valeurs affichées sont **recalculées "
           "à chaque build** : un joueur ajouté l'an dernier montre son niveau actuel.")
_onglets = st.tabs([f"{n} ({len(contenu(n))})" for n in _onglets_noms] + ["Joueurs exclus"])

for _ong, nom_liste in zip(_onglets, _onglets_noms + ["exclus"]):
    with _ong:
        contenu_df = contenu(nom_liste)
        if contenu_df.empty:
            st.info("Liste vide. Ajoute des joueurs depuis leur fiche "
                    f"(bouton « ➕ Ajouter à « {nom_liste} » »).")
            continue

        vivant = enrichir_liste(tuple(contenu_df["playerId"].tolist()), SC, poids_courants())
        df = contenu_df.merge(vivant, on="playerId", how="left")
        # Valeurs a jour quand le joueur est encore dans le build, valeurs
        # figees a l'ajout sinon (joueur sorti du perimetre).
        df["nom"] = df["nom_actuel"].fillna(df["nom"])
        df["club"] = df["club_actuel"].fillna(df["club"])
        df["score_actuel"] = df["score_actuel"].fillna(df["score"])
        df["archetype"] = df["archetype_actuel"].fillna(df["archetype"])
        df["position"] = df["position_actuelle"].fillna(df["poste"])
        df["priorite"] = df["priorite"].fillna("")
        df["note"] = df["note"].fillna("")
        _perdus = int(df["squadId"].isna().sum())

        # ---- Synthese de la liste
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Joueurs", len(df))
        k2.metric(f"{COURT} médian", n_(df["score_actuel"].median()))
        k3.metric("Âge médian", n_(df["age"].median()))
        k4.metric("Postes", df["archetype"].nunique())
        _rep = df["archetype"].value_counts()
        st.caption("Répartition : " + " · ".join(f"**{a}** {n}" for a, n in _rep.items())
                   + (f"  \n⚠️ {_perdus} joueur(s) absent(s) du dernier build "
                      "(plus assez de minutes, ou saison non reprise) : valeurs figées à l'ajout."
                      if _perdus else ""))

        # ---- Tri / filtre interne
        f1, f2, f3 = st.columns([2, 2, 2])
        TRIS = {f"{COURT} ▼": ("score_actuel", False), "Progression ▼": ("progression", False),
                "Âge ▲": ("age", True), "Nom A→Z": ("nom", True),
                "Ajout récent ▼": ("ajoute_le", False), "Priorité": ("priorite", True)}
        _tri = f1.selectbox("Trier par", list(TRIS), key=f"tri_{nom_liste}")
        _col_tri, _asc = TRIS[_tri]
        _postes_liste = sorted(df["archetype"].dropna().unique())
        _f_poste = f2.multiselect("Filtrer par poste", _postes_liste, key=f"fp_{nom_liste}")
        _f_prio = f3.multiselect("Filtrer par priorité",
                                 [p for p in PRIORITES if p and p in set(df["priorite"])],
                                 key=f"fpr_{nom_liste}")
        vue = df.copy()
        if _f_poste:
            vue = vue[vue["archetype"].isin(_f_poste)]
        if _f_prio:
            vue = vue[vue["priorite"].isin(_f_prio)]
        vue = vue.sort_values(_col_tri, ascending=_asc, na_position="last").reset_index(drop=True)

        st.caption("👉 Coche une ligne pour ouvrir sa fiche, plusieurs pour comparer ou déplacer.")
        ev_liste = st.dataframe(
            vue, hide_index=True, width="stretch", height=min(430, 80 + 36 * len(vue)),
            key=f"tbl_{nom_liste}", on_select="rerun", selection_mode="multi-row",
            column_order=["priorite", "nom", "archetype", "club", "competition", "saison",
                          "score_actuel", "performance", "progression", "age", "minutes",
                          "pied_fort", "profil_principal", "pilier_fort", "pilier_faible",
                          "rang", "note"],
            column_config={
                "priorite": st.column_config.TextColumn("", width="small"),
                "score_actuel": st.column_config.ProgressColumn(
                    COURT, min_value=0, max_value=120, format="%.1f",
                    help=LECTURES[lecture]["aide"]),
                "performance": st.column_config.NumberColumn("Perf.", format="%.1f"),
                "progression": st.column_config.NumberColumn(
                    "Progr.", format="%+.1f",
                    help="Évolution réelle estimée entre ses ~10 derniers matchs et le reste "
                         "de la saison. Au-delà de ±3 = changement notable."),
                "archetype": st.column_config.TextColumn("Poste"),
                "profil_principal": st.column_config.TextColumn("Profil RCSC"),
                "pilier_fort": st.column_config.TextColumn("Point fort"),
                "pilier_faible": st.column_config.TextColumn("Point faible"),
                "rang": st.column_config.NumberColumn("Rang", format="%d"),
                "note": st.column_config.TextColumn("Note", width="medium"),
            })
        _sel = ev_liste.selection["rows"] if ev_liste and "rows" in ev_liste.selection else []
        choisis = vue.iloc[_sel] if _sel else vue.iloc[0:0]

        # ---- Actions groupees
        with st.container(border=True):
            if choisis.empty:
                st.caption("Sélectionne au moins un joueur pour agir dessus.")
            else:
                st.markdown(f"**{len(choisis)} sélectionné(s)** : "
                            + ", ".join(choisis["nom"].astype(str).head(6))
                            + ("…" if len(choisis) > 6 else ""))
                a1, a2, a3 = st.columns([2, 2, 2])
                _ids = [int(x) for x in choisis["playerId"]]
                if a1.button("Retirer de la liste", key=f"rm_{nom_liste}", width="stretch"):
                    for _p in _ids:
                        retirer(nom_liste, _p)
                    st.toast(f"{len(_ids)} joueur(s) retiré(s) de « {nom_liste} »")
                    st.rerun()
                _cibles = [n for n in _toutes if n != nom_liste] or []
                _dest = a2.selectbox("Vers la liste", ["—"] + _cibles,
                                     key=f"dest_{nom_liste}", label_visibility="collapsed")
                c1, c2 = a3.columns(2)
                if c1.button("Déplacer →", key=f"mv_{nom_liste}", width="stretch",
                             disabled=_dest == "—"):
                    n = deplacer(nom_liste, _dest, _ids, copier=False)
                    st.toast(f"{n} joueur(s) déplacé(s) vers « {_dest} »")
                    st.rerun()
                if c2.button("Copier", key=f"cp_{nom_liste}", width="stretch",
                             disabled=_dest == "—"):
                    n = deplacer(nom_liste, _dest, _ids, copier=True)
                    st.toast(f"{n} joueur(s) copié(s) vers « {_dest} »")
                    st.rerun()

        # ---- Annotation du joueur selectionne
        if len(choisis) == 1:
            j_ = choisis.iloc[0]
            with st.expander(f"Annoter {j_.nom}", expanded=False):
                n1, n2 = st.columns([1, 3])
                _p0 = j_["priorite"] if j_["priorite"] in PRIORITES else ""
                _prio = n1.selectbox("Priorité", PRIORITES, index=PRIORITES.index(_p0),
                                     key=f"prio_{nom_liste}_{j_.playerId}")
                _note = n2.text_input("Note", value=str(j_["note"] or ""),
                                      placeholder="ex. à revoir contre un bloc bas",
                                      key=f"note_{nom_liste}_{j_.playerId}")
                if st.button("Enregistrer", key=f"save_{nom_liste}_{j_.playerId}"):
                    annoter(nom_liste, int(j_.playerId), _note, _prio)
                    st.toast("Annotation enregistrée")
                    st.rerun()

        # ---- Comparaison par piliers
        if 2 <= len(choisis) <= 5:
            cles = tuple((int(r.playerId), int(r.squadId), int(r.iterationId), r.position, r.archetype)
                         for r in choisis.itertuples() if pd.notna(r.squadId))
            pil = piliers_de(cles)
            if not pil.empty:
                st.markdown("##### Comparaison par piliers")
                _memes = choisis["archetype"].nunique() == 1
                if not _memes:
                    st.warning("⚠️ Postes différents : les piliers ne portent pas sur les mêmes "
                               "compétences, la comparaison n'a de sens que pilier par pilier "
                               "quand le nom est identique.")
                noms = dict(zip(choisis["playerId"].astype(int), choisis["nom"].astype(str)))
                pil["joueur"] = pil["playerId"].map(noms)
                fig = go.Figure()
                for jn, g in pil.groupby("joueur"):
                    g = g.sort_values("poids", ascending=False)
                    fig.add_trace(go.Bar(x=g["pilier"], y=g["percentile"], name=str(jn)))
                fig.add_hline(y=50, line_dash="dot", line_color="grey")
                fig.update_layout(barmode="group", height=360, yaxis_title="Percentile du poste",
                                  yaxis_range=[0, 100], margin=dict(t=10, b=0, l=0, r=0),
                                  legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, width="stretch")
                st.caption("50 = médiane du poste. Piliers classés par poids dans le score.")
        elif len(choisis) > 5:
            st.caption("Comparaison limitée à 5 joueurs : réduis la sélection.")

        # ---- Ouvrir la fiche
        if len(choisis) == 1 and pd.notna(choisis.iloc[0].get("squadId")):
            j_ = choisis.iloc[0]
            if st.button(f"Ouvrir la fiche de {j_.nom}", key=f"open_{nom_liste}",
                         width="stretch", type="primary"):
                st.session_state["_archetype_a_appliquer"] = j_.archetype
                st.query_params["fiche"] = "~".join(str(v) for v in (
                    int(j_.playerId), int(j_.squadId), int(j_.iterationId),
                    j_.position, j_.archetype))
                st.rerun()

        # ---- Export : la liste telle qu'elle est triee et filtree a l'ecran
        x1, x2, _ = st.columns([1, 1, 2])
        x1.download_button("Exporter la liste (CSV)",
                           en_csv(vue.drop(columns=[c for c in ("squadId", "iterationId") if c in vue])),
                           f"{nom_liste}.csv", "text/csv", key=f"dl_{nom_liste}", width="stretch")
        if EXCEL_DISPO:
            # Colonnes du tableau a l'ecran, dans le meme ordre, avec des intitules
            # lisibles : c'est un document a transmettre, pas un fichier technique.
            _export = vue[[c for c in EXPORT_LISTE if c in vue]].rename(columns=EXPORT_LISTE) \
                .rename(columns={"Score": COURT})
            for _c in ("Point fort", "Point faible"):
                if _c in _export:
                    _export[_c] = _export[_c].astype("string").str.replace("_", " ")
            # Un export qui echoue ne doit jamais emporter la page : sans ce garde-fou
            # l'erreur d'une liste empechait d'afficher toutes les listes suivantes.
            try:
                _xlsx = en_excel(_export, nom_liste)
            except Exception as _e:  # noqa: BLE001
                _xlsx = None
                x2.caption(f"Export Excel indisponible pour cette liste ({type(_e).__name__}).")
            if _xlsx is not None:
                x2.download_button(
                    "Exporter la liste (Excel)", _xlsx, f"{nom_liste}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dlx_{nom_liste}", width="stretch")
