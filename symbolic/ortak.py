"""Symbolic regression katmani (pyoperon, gplearn) icin ortak altyapi.

Sinyal/arka plan tanimlari, olay agirliklari, SR kesimi, bolunme, anlamlilik
(asimov, pick_cut, apply_cut), BDT (fit_bdt), permutasyon onemi ve Bell ve ark.
kesimleri burada YENIDEN YAZILMAZ: notebooks/monoz_dark_matter_analizi.ipynb
icindeki tanimlar oldugu gibi okunup calistirilir. Veri, defterle ayni
_extract.py ile yuklenir. Defterde bir tanim degisirse bu modul de degisir.
"""
import argparse
import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

KOK = Path(__file__).resolve().parents[1]
DEFTER = KOK / 'notebooks' / 'monoz_dark_matter_analizi.ipynb'

# Defterden alinan tanimlar (degistirilmeden calistirilir)
_ISIMLER = ('LUMI', 'MZ', 'OUT', 'signal_region', 'SINYAL', 'RESAMPLE', 'FRAC',
            'split3', 'LIT8', 'NEFF_MIN', 'n_eff', 'asimov', 'pick_cut',
            'apply_cut', 'BELL', 'fit_bdt', 'perm_importance')


def _defter_tanimlari():
    """Defterin kod hucrelerinden yukaridaki isimlerin tanimlarini calistirir."""
    with open(DEFTER, encoding='utf-8') as f:
        hucreler = [c for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    ns = {'np': np, 'pd': pd, 'train_test_split': train_test_split,
          'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
          'log_loss': log_loss, 'roc_auc_score': roc_auc_score}
    bulunan, extract_yolu, stil = set(), None, {}
    for c in hucreler:
        kaynak = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        for dugum in ast.parse(kaynak).body:
            ad = None
            if isinstance(dugum, ast.FunctionDef):
                ad = dugum.name
            elif (isinstance(dugum, ast.Assign) and len(dugum.targets) == 1
                  and isinstance(dugum.targets[0], ast.Name)):
                ad = dugum.targets[0].id
            elif (isinstance(dugum, ast.Expr) and isinstance(dugum.value, ast.Call)
                  and ast.unparse(dugum.value.func) == 'plt.rcParams.update'):
                stil = ast.literal_eval(dugum.value.args[0])
            if ad in _ISIMLER:
                exec(compile(ast.Module([dugum], []), str(DEFTER), 'exec'), ns)
                bulunan.add(ad)
        m = re.search(r"exec\(open\('([^']+_extract\.py)'\)", kaynak)
        if m:
            extract_yolu = m.group(1)
    eksik = set(_ISIMLER) - bulunan
    if eksik:
        raise RuntimeError(f'defterde bulunamayan tanimlar: {sorted(eksik)}')
    return ns, extract_yolu, stil


D, _EXTRACT_DEFTER, STIL = _defter_tanimlari()
LUMI, MZ, FRAC, NEFF_MIN = D['LUMI'], D['MZ'], D['FRAC'], D['NEFF_MIN']
SINYAL, RESAMPLE, BELL, LIT8 = D['SINYAL'], D['RESAMPLE'], D['BELL'], D['LIT8']
signal_region, split3, n_eff, asimov = (D['signal_region'], D['split3'],
                                        D['n_eff'], D['asimov'])
pick_cut, apply_cut, fit_bdt, perm_importance = (D['pick_cut'], D['apply_cut'],
                                                 D['fit_bdt'], D['perm_importance'])
OUT = Path(os.environ.get('MONOZ_OUT', D['OUT']))
EXTRACT = os.environ.get('MONOZ_EXTRACT', _EXTRACT_DEFTER)
OLCEK = 1 / FRAC        # dogrulama/test verimini tum veriye tasir (defterle ayni)

GAMMA_Z = 2.4952        # GeV, PDG
SR_DEGISKENLERI = ['r', 'x', 'dphi', 'dR', 'dm', 'dphij']
SURECLER = None         # load sirasiyla arka plan surecleri (veri_yukle doldurur)


def stil_uygula():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(STIL)
    return plt


def argumanlar(aciklama, ekle=None):
    """--oran: verinin kesri (hizli deneme icin 0.1). Ciktilar ayri klasore gider."""
    p = argparse.ArgumentParser(description=aciklama)
    p.add_argument('--oran', type=float, default=1.0,
                   help='kullanilacak olay kesri (once 0.1 ile dene)')
    if ekle:
        ekle(p)
    a = p.parse_args()
    return a, dizinler(a.oran)


def dizinler(oran):
    """(ara sonuclar, son ciktilar). Tam veride son ciktilar OUT'a yazilir."""
    if oran >= 1:
        ara, son = OUT / 'symbolic', OUT
    else:
        ara = son = OUT / f'symbolic_oran{oran:g}'
    ara.mkdir(parents=True, exist_ok=True)
    son.mkdir(parents=True, exist_ok=True)
    return ara, son


def kisa_ad(ad):
    ad = ad.replace('+', 'arti').replace('-', 'eksi')   # 'MET +5%' ile 'MET -5%' ayrissin
    return re.sub(r'[^0-9A-Za-z]+', '_', ad).strip('_')


def extract_yukle():
    """Defterle ayni _extract.py: load, BKG, SIG, cell_resample, ..."""
    ns = {'__name__': '_extract', 'np': np, 'pd': pd, 'LUMI': LUMI, 'MZ': MZ}
    with open(EXTRACT, encoding='utf-8') as f:
        exec(compile(f.read(), EXTRACT, 'exec'), ns)
    return ns


def sr_degiskenleri(d):
    """Boyutsuz symbolic regression girdileri; mevcut sutunlar yeniden kullanilir."""
    return pd.DataFrame({
        'r': d.met_over_ptZ,                 # MET / pT(ll)
        'x': d.met / MZ,                     # MET / m_Z
        'dphi': d.dphi_Z_met,                # dphi(ll, MET)
        'dR': d.dR_ll,                       # dR(l, l)
        'dm': (d.m_ll - MZ) / GAMMA_Z,       # (m_ll - m_Z) / Gamma_Z
        'dphij': d.min_dphi_met_jets,        # min dphi(jet, MET), jet yoksa pi
    }, index=d.index)


# gplearn'un korumali fonksiyonlari (gplearn/functions.py ile ayni tanim). sympy'de
# adlariyla (pdiv, plog, psqrt) tutulur ki sadelestirme davranislarini degistirmesin.
def _pdiv(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(b) > 0.001, np.divide(a, b), 1.)


def _plog(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(a) > 0.001, np.log(np.abs(a)), 0.)


def _psqrt(a):
    return np.sqrt(np.abs(a))


KORUMALI = {'pdiv': _pdiv, 'plog': _plog, 'psqrt': _psqrt}


def formul_derle(ifade):
    """sympy ifadesi metnini guvenli bicimde numpy fonksiyonuna cevirir (eval yok)."""
    import sympy
    semboller = sympy.symbols(SR_DEGISKENLERI)
    yerel = {**dict(zip(SR_DEGISKENLERI, semboller)),
             **{k: sympy.Function(k) for k in KORUMALI}}
    e = sympy.sympify(ifade, locals=yerel)
    f = sympy.lambdify(semboller, e, modules=[KORUMALI, 'numpy'])

    def hesap(d):
        with np.errstate(all='ignore'):
            v = np.asarray(f(*[np.asarray(d[c], float) for c in SR_DEGISKENLERI]), float)
        v = np.broadcast_to(v, (len(d[SR_DEGISKENLERI[0]]),)).copy()
        # tanim disi (NaN) olaylar hicbir esikte secilmez
        return np.nan_to_num(v, nan=-1e30, posinf=1e30, neginf=-1e30)
    return e, hesap


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sinif_dengeli(d, F):
    """Egitim girdisi; defterdeki gibi sinif dengeleyici agirlik 0.5 / N_sinif."""
    y = d.y.to_numpy()
    w = np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1))
    return d[F].to_numpy(), y, w


def calisma_noktasi(sb_va, ss_va, wb_va, ws_va, sb_te, ss_te, wb_te, ws_te,
                    src_te=None):
    """Esik dogrulamada (pick_cut), rapor testte (apply_cut) -- defterin protokolu."""
    cut = pick_cut(sb_va, ss_va, wb_va, ws_va, OLCEK)
    _, z_va = apply_cut(cut, sb_va, ss_va, wb_va, ws_va, OLCEK)
    nokta, z_te = apply_cut(cut, sb_te, ss_te, wb_te, ws_te, OLCEK)
    sonuc = {'esik': cut, 'Z_dogrulama': z_va, 'Z_test': z_te, **nokta}
    if src_te is not None:
        for p in SURECLER:
            m = (src_te == p) & (sb_te > cut) if np.isfinite(cut) else np.zeros(len(sb_te), bool)
            sonuc[f'B_{p}'] = wb_te[m].sum() * LUMI * OLCEK
    return sonuc


def kesim_tabanli(d):
    """Bell ve ark. kesimleri (defterdeki BELL) bir olay kumesine; Z tum veriye olceklenir."""
    b, s = d[d.y == 0], d[d.y == 1]
    mb, ms = np.ones(len(b), bool), np.ones(len(s), bool)
    satir = []
    for ad, kesim in BELL:
        mb &= np.asarray(kesim(b))
        ms &= np.asarray(kesim(s))
        wb = b.weight.to_numpy()[mb]
        S = s.weight.to_numpy()[ms].sum() * LUMI * OLCEK
        B = wb.sum() * LUMI * OLCEK
        ne = n_eff(wb)
        satir.append({'kesim': ad, 'S': S, 'B': B, 'Neff': ne,
                      'epsS': ms.mean() if len(ms) else np.nan,
                      'Z_test': asimov(S, B) if ne >= NEFF_MIN else np.nan})
    return pd.DataFrame(satir).set_index('kesim')


# --- Sistematik kaydirmalar --------------------------------------------------
# MET ve jet enerjisi olceklenince degisen turetilmis degiskenler. Her aday
# (dogrulama, mutlak_mi, kaydir) uclusudur: dogrulama(d) nominal veride sutunla
# karsilastirilir; ilk tutan aday kaydirmada kullanilir. Tutmayan degisken
# nominal birakilir ve raporda "dogrulanamadi" olarak isaretlenir.
def _rec(d, km):
    return np.sqrt((km * d.met) ** 2 + d.Z_pt ** 2
                   + 2 * km * d.met * d.Z_pt * np.cos(d.dphi_Z_met))


def _upar(d, km):
    return -(d.Z_pt + km * d.met * np.cos(d.dphi_Z_met))


def _mt_kutleli(d, km):
    et = np.sqrt(d.Z_pt ** 2 + d.m_ll ** 2)
    return np.sqrt(np.maximum((et + km * d.met) ** 2 - _rec(d, km) ** 2, 0))


def _ht_orani(d, km, kj):
    jet = d.jet1_pt.where(d.jet1_pt > 0, 1.0)
    return np.where(d.n_jets > 0, km * d.met / np.sqrt(kj * jet), 0.0)


KURALLAR = {
    'met_over_ptZ': [
        (lambda d: np.minimum(d.met / d.Z_pt, 10.0), False,
         lambda d, km, kj: np.minimum(km * d.met / d.Z_pt, 10.0)),
        (lambda d: d.met / d.Z_pt, False, lambda d, km, kj: km * d.met / d.Z_pt)],
    'abs_met_minus_ptZ': [
        (lambda d: np.abs(d.met - d.Z_pt), False,
         lambda d, km, kj: np.abs(km * d.met - d.Z_pt))],
    'hadronic_recoil_pt': [
        (lambda d: _rec(d, 1), False, lambda d, km, kj: _rec(d, km))],
    'u_parallel': [
        (lambda d: _upar(d, 1), False, lambda d, km, kj: _upar(d, km)),
        (lambda d: -_upar(d, 1), False, lambda d, km, kj: -_upar(d, km))],
    'u_parallel_over_ptZ': [
        (lambda d: _upar(d, 1) / d.Z_pt, False, lambda d, km, kj: _upar(d, km) / d.Z_pt),
        (lambda d: -_upar(d, 1) / d.Z_pt, False, lambda d, km, kj: -_upar(d, km) / d.Z_pt)],
    'u_perp': [   # isaretli aciyla tanimli; |u_perp| = MET sin(dphi), MET ile dogrusal
        (lambda d: d.met * np.sin(d.dphi_Z_met), True, lambda d, km, kj: km * d.u_perp)],
    'u_perp_over_ptZ': [
        (lambda d: d.met * np.sin(d.dphi_Z_met) / d.Z_pt, True,
         lambda d, km, kj: km * d.u_perp_over_ptZ)],
    'mT': [
        (lambda d: _mt_kutleli(d, 1), False, lambda d, km, kj: _mt_kutleli(d, km)),
        (lambda d: np.sqrt(2 * d.met * d.Z_pt * (1 - np.cos(d.dphi_Z_met))), False,
         lambda d, km, kj: np.sqrt(2 * km * d.met * d.Z_pt * (1 - np.cos(d.dphi_Z_met))))],
    'met_over_sqrtHT': [   # SR'de en fazla bir jet: HT = jet1_pt
        (lambda d: _ht_orani(d, 1, 1), False, _ht_orani)],
}


def kurallari_dogrula(d, tol=1e-2):
    """Her turetilmis degisken icin tutan adayin sirasi (yoksa None) ve sapma."""
    sonuc = {}
    for ad, adaylar in KURALLAR.items():
        if ad not in d:
            continue
        en_iyi = np.inf
        for i, (dog, mutlak, _) in enumerate(adaylar):
            with np.errstate(all='ignore'):
                beklenen = np.asarray(dog(d), float)
            gercek = d[ad].to_numpy(float)
            if mutlak:
                beklenen, gercek = np.abs(beklenen), np.abs(gercek)
            sapma = np.abs(beklenen - gercek) / np.maximum(np.abs(gercek), 1e-3)
            q99 = float(np.nanquantile(sapma, 0.99))
            en_iyi = min(en_iyi, q99)
            if q99 < tol:
                sonuc[ad] = {'aday': i, 'q99_goreli_sapma': q99}
                break
        else:
            sonuc[ad] = {'aday': None, 'q99_goreli_sapma': en_iyi}
    return sonuc


def kaydir(d, km, kj, dogrulama):
    """MET'i km, jet enerjisini kj ile olcekler; dogrulanan turetilmis degiskenleri yeniden hesaplar."""
    yeni = {}
    for ad, bilgi in dogrulama.items():
        if bilgi['aday'] is not None:
            with np.errstate(all='ignore'):
                yeni[ad] = KURALLAR[ad][bilgi['aday']][2](d, km, kj)
    d = d.copy()
    for ad, v in yeni.items():
        d[ad] = v
    d['met'] = km * d.met
    d['jet1_pt'] = kj * d.jet1_pt
    return d


# --- Kayit yardimcilari -------------------------------------------------------
def json_yaz(yol, nesne):
    def donustur(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    with open(yol, 'w', encoding='utf-8') as f:
        json.dump(nesne, f, indent=2, ensure_ascii=False, default=donustur)


def json_oku(yol):
    with open(yol, encoding='utf-8') as f:
        return json.load(f)


def oznitelikler():
    """Defterin §1g'de sectigi oznitelikler (defter OUT/features.json'a yazar)."""
    yol = OUT / 'features.json'
    if not yol.exists():
        raise FileNotFoundError(
            f'{yol} yok: once defteri §1g (oznitelik secimi) dahil calistirin.')
    return json_oku(yol)


def surecleri_ayarla(surecler):
    global SURECLER
    SURECLER = list(surecler)
