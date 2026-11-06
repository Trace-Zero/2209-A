"""Adim 3: sembolik regresyon (Julia gerektirmez).

    A) Damitma -- pyoperon SymbolicRegressor; hedef ogretmenin OOF logiti,
       amaclar ['r2', 'length'] (NSGA-II, Pareto cephesi).
    B) Dogrudan siniflandirma -- gplearn SymbolicClassifier (log-kayip);
       karmasiklik parsimony_coefficient ile cezalandirilir, her parsimony
       degerinin en iyi programi Pareto listesine girer.

Formuller yalnizca 'egitim' kumesinden alinan alt orneklemde aranir; esik ve
formul secimi 'dogrulama', rapor 'test' kumesinde yapilir (adim 4). Alt orneklem
katmanlidir: sinyal n/2, arka plan n/2 ve arka planda her surecten esit sayida
olay (kucuk surecler tamamen alinir, kalan kota digerlerine dagitilir). Her
surecin agirliklari o surecin egitim kumesindeki toplam verimini koruyacak
sekilde yeniden normalize edilir; ardindan iki sinif esit toplam agirliga getirilir.

pyoperon 0.6 sample_weight kabul etmez: A modunda agirlik, olaylarin agirlikla
orantili olasilikla (yerine koyarak) yeniden orneklenmesiyle uygulanir. Damitma
hedefi deterministik bir fonksiyon oldugu icin bu yalnizca girdi bolgelerinin
onemini degistirir. gplearn agirligi sample_weight olarak alir.

Her formul sympy'ye cevrilir ve kutuphanenin kendi hesapladigi degerlerle
sayisal olarak karsilastirilir (adim 4 formulleri bu metinden derler).
Yazilanlar (son dizin): sr_pareto_A.csv, sr_pareto_B.csv.
"""
import numpy as np
import pandas as pd
import sympy

from symbolic import ortak

OPERON = dict(
    allowed_symbols='add,sub,mul,div,square,sqrt,log,exp,cos,constant,variable',
    max_length=25,
    objectives=['r2', 'length'],
    random_state=0,
    n_threads=1,                 # tek is parcacigi: tekrarlanabilir
)
GPLEARN = dict(
    function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'cos'),
    random_state=0,
    n_jobs=1,
)
PARSIMONY = (0.0005, 0.001, 0.005, 0.01)


def su_doldur(mevcut, toplam):
    """Kotayi surecler arasinda esit dagitir; tukenen surecin payi digerlerine gecer."""
    kota, kalan, hedef = {k: 0 for k in mevcut}, dict(mevcut), toplam
    while hedef > 0 and any(kalan.values()):
        acik = [k for k, v in kalan.items() if v > 0]
        pay = max(hedef // len(acik), 1)
        for k in acik:
            ek = min(pay, kalan[k], hedef)
            kota[k] += ek
            kalan[k] -= ek
            hedef -= ek
            if hedef == 0:
                break
    return kota


def alt_orneklem(egit, n, agirlik, tohum=0):
    sinyal, arka = egit[egit.y == 1], egit[egit.y == 0]
    kota = su_doldur(arka.groupby('src').size().to_dict(), n - n // 2)
    parca = [sinyal.sample(n=min(n // 2, len(sinyal)), random_state=tohum)]
    parca += [g.sample(n=kota[k], random_state=tohum) for k, g in arka.groupby('src')]
    alt = pd.concat(parca)
    # her surec (sinyal dahil) egitim kumesindeki toplam verimini korur
    oran = egit.groupby('src').weight.sum() / alt.groupby('src').weight.sum()
    alt['w_fiz'] = alt.weight * alt.src.map(oran)
    w = np.ones(len(alt)) if agirlik == 'esit' else alt.w_fiz.to_numpy().copy()
    y = alt.y.to_numpy()
    w[y == 1] /= w[y == 1].sum()
    w[y == 0] /= w[y == 0].sum()
    alt['w_sr'] = w * len(alt) / w.sum()
    return alt, kota


# --- metinden sympy'ye ----------------------------------------------------------
_SEMBOL = {v: sympy.Symbol(v) for v in ortak.SR_DEGISKENLERI}
# kesin bicim: gplearn'un korumali fonksiyonlari adlariyla (numpy karsiliklari
# ortak.KORUMALI'da, gplearn/functions.py ile ayni); okunabilir bicim: esiksiz
_GPLEARN_KESIN = {
    'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b,
    'div': sympy.Function('pdiv'), 'log': sympy.Function('plog'),
    'sqrt': sympy.Function('psqrt'), 'cos': sympy.cos}
_GPLEARN_OKUNUR = {**_GPLEARN_KESIN, 'div': lambda a, b: a / b,
                   'log': lambda a: sympy.log(sympy.Abs(a)),
                   'sqrt': lambda a: sympy.sqrt(sympy.Abs(a))}


def operon_sympy(metin):
    # pyoperon kareyi 'a ^ 2' yazar; sympify '^' isaretini ussa cevirir
    return sympy.sympify(metin, locals=_SEMBOL, convert_xor=True)


def gplearn_sympy(prog, kesin=True):
    """gplearn programini dugum listesinden kurar. str(program) sabitleri 3 ondaliga
    yuvarladigi icin metin kullanilmaz; sabitler tam hassasiyetle alinir."""
    tablo = _GPLEARN_KESIN if kesin else _GPLEARN_OKUNUR
    dugum = list(prog.program)

    def kur(i):
        d = dugum[i]
        if isinstance(d, (int, np.integer)):          # degisken sirasi
            return _SEMBOL[ortak.SR_DEGISKENLERI[d]], i + 1
        if isinstance(d, (float, np.floating)):       # sabit
            return sympy.Float(float(d), 15), i + 1
        arg, j = [], i + 1
        for _ in range(d.arity):
            a, j = kur(j)
            arg.append(a)
        return tablo[d.name](*arg), j
    return kur(0)[0]


def sapma(beklenen, ifade, X):
    """Kutuphanenin degeri ile sympy cevirisi arasindaki en buyuk goreli fark."""
    _, f = ortak.formul_derle(str(ifade))
    deger = f(X)
    olcek = np.std(beklenen) + 1e-12
    return float(np.max(np.abs(deger - beklenen)) / olcek)


def operon_patch():
    """pyoperon 0.6.2: Pareto cephesi hesaplanirken model float32, hedef float64
    tutuldugu icin FitLeastSquares cokuyor; iki diziyi de float64'e cevir."""
    import pyoperon.sklearn as sk
    if getattr(sk.op.FitLeastSquares, '_monoz', False):
        return sk
    asil = sk.op.FitLeastSquares

    def sar(a, b):
        return asil(np.array(a, np.float64), np.array(b, np.float64))
    sar._monoz = True
    sk.op.FitLeastSquares = sar
    return sk


def mod_a(alt, ayar, tohum=0):
    sk = operon_patch()
    rng = np.random.default_rng(tohum)
    secim = rng.choice(len(alt), size=len(alt), replace=True,
                       p=alt.w_sr.to_numpy() / alt.w_sr.sum())
    egit = alt.iloc[secim]
    # pandas 3 to_numpy() salt okunur dizi dondurur; pyoperon yazilabilir dizi ister
    X = np.array(egit[ortak.SR_DEGISKENLERI].to_numpy(float), order='F')
    reg = sk.SymbolicRegressor(**ayar)
    reg.fit(X, egit.ogretmen_logit.to_numpy(float))
    satir = []
    for m in reg.pareto_front_:
        metin = reg.get_model_string(m['tree'], 10, ortak.SR_DEGISKENLERI)
        e = operon_sympy(metin)
        beklenen = reg.evaluate_model(m['tree'], X)
        satir.append({'formul': metin, 'uzunluk': m['length'],
                      'r2': -float(m['objective_values'][0]),
                      'sympy': str(e), 'okunabilir': str(e), 'latex': sympy.latex(e),
                      'ceviri_sapmasi': sapma(beklenen, e, egit)})
    bilgi = {'olay': len(alt), 'tekil_olay_orani': len(np.unique(secim)) / len(alt)}
    return pd.DataFrame(satir).sort_values('uzunluk'), bilgi


def mod_b(alt, ayar, parsimony):
    from gplearn.genetic import SymbolicClassifier
    X = alt[ortak.SR_DEGISKENLERI].to_numpy(float)
    satir = []
    for p in parsimony:
        est = SymbolicClassifier(**ayar, parsimony_coefficient=p,
                                 feature_names=ortak.SR_DEGISKENLERI)
        est.fit(X, alt.y.to_numpy(), sample_weight=alt.w_sr.to_numpy())
        prog = est._program
        e, e_ok = gplearn_sympy(prog), gplearn_sympy(prog, kesin=False)
        satir.append({'parsimony': p, 'formul': str(prog), 'uzunluk': prog.length_,
                      'kayip': prog.raw_fitness_, 'sympy': str(e),
                      'okunabilir': str(e_ok), 'latex': sympy.latex(e_ok),
                      'ceviri_sapmasi': sapma(prog.execute(X), e, alt)})
        print(f'  parsimony {p}: uzunluk {prog.length_}, log-kayip {prog.raw_fitness_:.4f}')
    return pd.DataFrame(satir).sort_values('uzunluk')


def main():
    def ekle(p):
        p.add_argument('--mod', default='AB', help="'A', 'B' ya da 'AB'")
        p.add_argument('--n_a', type=int, default=15000, help='A alt orneklemi')
        p.add_argument('--n_b', type=int, default=5000, help='B alt orneklemi (gplearn yavas)')
        p.add_argument('--agirlik', choices=['fiziksel', 'esit'], default='fiziksel',
                       help='sinif ici agirlik: fiziksel (yeniden normalize) ya da esit')
        p.add_argument('--nesil', type=int, default=None,
                       help='hizli deneme icin nesil sayisi (A ve B); verilmezse kutuphane varsayilani')
        p.add_argument('--nufus', type=int, default=None, help='nufus buyuklugu (A ve B)')

    a, (ara, son) = ortak.argumanlar('Adim 3: sembolik regresyon', ekle)
    olaylar = pd.read_pickle(ara / 'olaylar.pkl').join(pd.read_pickle(ara / 'ogretmen.pkl'))
    egit = olaylar[olaylar.kume == 'egitim']
    ek = {k: v for k, v in (('generations', a.nesil), ('population_size', a.nufus))
          if v is not None}
    bilgi = {'agirlik': a.agirlik, 'ek_ayarlar': ek}

    if 'A' in a.mod:
        alt, kota = alt_orneklem(egit, a.n_a, a.agirlik)
        tablo, bilgi['A'] = mod_a(alt, {**OPERON, **ek})
        bilgi['A']['arka_plan_kotasi'] = kota
        tablo.to_csv(son / 'sr_pareto_A.csv', index=False)
        print(f'\nmod A (pyoperon): {len(tablo)} formul')
        print(tablo[['uzunluk', 'r2', 'ceviri_sapmasi', 'okunabilir']].to_string(index=False))
    if 'B' in a.mod:
        alt, kota = alt_orneklem(egit, a.n_b, a.agirlik)
        print(f'\nmod B (gplearn): {len(alt)} olay')
        tablo = mod_b(alt, {**GPLEARN, **ek}, PARSIMONY)
        bilgi['B'] = {'olay': len(alt), 'arka_plan_kotasi': kota, 'parsimony': PARSIMONY}
        tablo.to_csv(son / 'sr_pareto_B.csv', index=False)
        print(tablo[['parsimony', 'uzunluk', 'kayip', 'ceviri_sapmasi', 'okunabilir']]
              .to_string(index=False))
    ortak.json_yaz(ara / 'sr_bilgi.json', bilgi)
    for mod in a.mod:
        t = pd.read_csv(son / f'sr_pareto_{mod}.csv')
        if (t.ceviri_sapmasi > 1e-3).any():
            print(f'UYARI: mod {mod} formullerinin bir kisminda sympy cevirisi kutuphane '
                  f'degerinden sapiyor (en buyuk {t.ceviri_sapmasi.max():.2g})')


if __name__ == '__main__':
    main()
