"""Adim 4: Pareto formullerinin degerlendirilmesi, secim ve ozet.

Her formul icin esik 'dogrulama' kumesinde (pick_cut), anlamlilik 'test'
kumesinde (apply_cut) -- defterin protokolu ve defterin asimov fonksiyonu
(arka plan belirsizligi yok). Secim kurali, dogrulama anlamliliginin
maksimumunun %95'ine ulasan en kisa (en az dugumlu) formuldur. A (pyoperon) ve
B (gplearn) listeleri sr_pareto_{A,B}.csv'den okunur; formuller sympy metninden
sympy.lambdify ile derlenir (eval kullanilmaz).
"""
import numpy as np
import pandas as pd
import sympy

from symbolic import ortak

YORUM = {
    'r': 'MET/pT(Z) dengesi: sinyalde Z gorunmez sisteme karsi geri teper (r ~ 1); '
         'DY\'deki sahte MET dengesizdir',
    'x': 'MET olcegi (m_Z biriminde): karanlik madde cifti buyuk MET uretir',
    'dphi': 'Z ile MET arasindaki aci: sinyalde sirt sirta (pi)',
    'dR': 'iki lepton arasindaki aci: hizli (boost edilmis) Z dar acili cift verir',
    'dm': 'Z kutle cekimi (Gamma_Z biriminde): gercek Z icermeyen arka plani ayirir',
    'dphij': 'jet ile MET arasindaki en kucuk aci: kucukse MET yanlis olculmus bir '
             'jetten gelir (sahte MET); jet yoksa pi',
}
SECIM_ORANI = 0.95


def nokta(df, skor):
    va, te = (df.kume == 'dogrulama').to_numpy(), (df.kume == 'test').to_numpy()
    b, s = (df.y == 0).to_numpy(), (df.y == 1).to_numpy()
    w = df.weight.to_numpy()
    return ortak.calisma_noktasi(skor[va & b], skor[va & s], w[va & b], w[va & s],
                                 skor[te & b], skor[te & s], w[te & b], w[te & s],
                                 df.src.to_numpy()[te & b])


def sabit_esikte(t, skor, esik):
    """Kaydirilmis test olaylarinda, nominal esikle anlamlilik."""
    b, s = (t.y == 0).to_numpy(), (t.y == 1).to_numpy()
    w = t.weight.to_numpy()
    n, z = ortak.apply_cut(esik, skor[b], skor[s], w[b], w[s], ortak.OLCEK)
    return z, n['S'], n['B']


def pareto_degerlendir(olaylar, tablo):
    satir = []
    for _, r in tablo.iterrows():
        _, f = ortak.formul_derle(r.sympy)
        ek = {k: r[k] for k in ('r2', 'parsimony', 'kayip') if k in r}
        satir.append({'uzunluk': r.uzunluk, **ek, 'sympy': r.sympy,
                      'okunabilir': r.okunabilir, 'latex': r.latex,
                      **nokta(olaylar, f(olaylar))})
    sonuc = pd.DataFrame(satir)
    if not np.isfinite(sonuc.Z_dogrulama).any():
        raise ValueError('hicbir formulde N_eff >= NEFF_MIN olan calisma noktasi yok')
    zmax = np.nanmax(sonuc.Z_dogrulama)
    aday = sonuc[sonuc.Z_dogrulama >= SECIM_ORANI * zmax]
    return sonuc, aday.sort_values('uzunluk').iloc[0]


def yorumla(e, olaylar):
    """Formuldeki her degisken: fiziksel anlam ve sinyal/arka plan medyaninda turev isareti."""
    satir = []
    medyan = {y: olaylar[olaylar.y == y][ortak.SR_DEGISKENLERI].median() for y in (0, 1)}
    for v in sorted(e.free_symbols, key=str):
        turev = sympy.diff(e, v)
        isaret = {}
        for y, ad in ((1, 'sinyal'), (0, 'arka plan')):
            deger = complex(turev.subs({sympy.Symbol(k): medyan[y][k]
                                        for k in ortak.SR_DEGISKENLERI}).evalf())
            if not np.isfinite(deger) or deger.imag:
                isaret[ad] = 'tanimsiz'
            else:
                isaret[ad] = ('artar' if deger.real > 0 else
                              'azalir' if deger.real < 0 else 'sabit')
        satir.append(f"- `{v}`: {YORUM[str(v)]}. Formul, `{v}` artinca sinyal medyaninda "
                     f"{isaret['sinyal']}, arka plan medyaninda {isaret['arka plan']}.")
    return satir


def md_tablo(df, ondalik=3):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: f'{x:.{ondalik}g}' if np.isfinite(x) else '—')
    basliklar = [str(df.index.name or '')] + [str(c) for c in df.columns]
    satirlar = ['| ' + ' | '.join(basliklar) + ' |', '|' + '---|' * len(basliklar)]
    for i, r in df.iterrows():
        satirlar.append('| ' + ' | '.join([str(i)] + [str(x) for x in r]) + ' |')
    return '\n'.join(satirlar)


def main():
    a, (ara, son) = ortak.argumanlar('Adim 4: degerlendirme')
    plt = ortak.stil_uygula()
    girdi = ortak.json_oku(ara / 'girdiler_ozet.json')
    ortak.surecleri_ayarla(girdi['surecler'])
    olaylar = pd.read_pickle(ara / 'olaylar.pkl').join(pd.read_pickle(ara / 'ogretmen.pkl'))
    og = ortak.json_oku(ara / 'ogretmen_ozet.json')
    og_nokta = og['calisma_noktasi']
    test = olaylar[olaylar.kume == 'test']

    # Mevcut kesim tabanli secim (Bell ve ark. kesimleri, defterdeki BELL), test kumesinde
    bell = ortak.kesim_tabanli(test)
    bell.to_csv(son / 'sr_kesim_tabanli.csv')
    gecerli = bell.Z_test.dropna()
    bell_satir = gecerli.index[-1] if len(gecerli) else bell.index[-1]
    bell_ref = bell.loc[bell_satir]

    # --- Pareto formulleri -------------------------------------------------
    sonuc, secilen, ifade = {}, {}, {}
    for mod in 'AB':
        yol = son / f'sr_pareto_{mod}.csv'
        if not yol.exists():
            print(f'{yol} yok, mod {mod} atlandi')
            continue
        sonuc[mod], secilen[mod] = pareto_degerlendir(olaylar, pd.read_csv(yol))
        sonuc[mod].to_csv(son / f'sr_degerlendirme_{mod}.csv', index=False)
        ifade[mod] = ortak.formul_derle(secilen[mod].sympy)
        print(f"mod {mod}: secilen uzunluk {secilen[mod].uzunluk}, "
              f"Z_test {secilen[mod].Z_test:.3f}: {secilen[mod].sympy}")

    # Grafik: formul uzunlugu - anlamlilik
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mod, renk in (('A', 'tab:blue'), ('B', 'tab:green')):
        if mod in sonuc:
            ax.plot(sonuc[mod].uzunluk, sonuc[mod].Z_test, 'o-', color=renk,
                    label=f'formul {mod} (test)')
            ax.plot(sonuc[mod].uzunluk, sonuc[mod].Z_dogrulama, 's--', color=renk,
                    alpha=.4, ms=4, label=f'formul {mod} (dogrulama)')
            ax.plot(secilen[mod].uzunluk, secilen[mod].Z_test, '*', color=renk, ms=16)
    ax.axhline(og_nokta['Z_test'], color='k', ls='-', label='ogretmen (BDT)')
    ax.axhline(bell_ref.Z_test, color='red', ls='--',
               label=f'kesim tabanli (Bell ve ark.: {bell_satir})')
    ax.set_xlabel('formul uzunlugu (dugum sayisi)')
    ax.set_ylabel('$Z_A$')
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(son / 'sr_uzunluk_anlamlilik.png', dpi=120)

    # Grafik: secilen formullerin surec bazinda yiginli dagilimi (test)
    for mod in secilen:
        f = ifade[mod][1]
        v = f(test)
        b, s = test[test.y == 0], test[test.y == 1]
        vb, vs = v[(test.y == 0).to_numpy()], v[(test.y == 1).to_numpy()]
        kenar = np.linspace(*np.percentile(v[np.abs(v) < 1e29], [0.5, 99.5]), 31)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.hist([vb[(b.src == p).to_numpy()] for p in ortak.SURECLER], bins=kenar,
                stacked=True, label=ortak.SURECLER,
                weights=[b.weight[b.src == p] * ortak.LUMI * ortak.OLCEK
                         for p in ortak.SURECLER])
        ax.hist(vs, bins=kenar, weights=s.weight * ortak.LUMI * ortak.OLCEK,
                histtype='step', lw=2.4, color='black', label='sinyal')
        ax.axvline(secilen[mod].esik, color='red', ls='--', label='dogrulamada secilen esik')
        ax.set_yscale('log')
        ax.set_xlabel(f'formul {mod} degeri')
        ax.set_ylabel('olay / bin  (test, tum veriye olcekli)')
        ax.legend(fontsize=8, frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(son / f'sr_secilen_{mod}_yigin.png', dpi=120)

    # --- Tasinabilirlik: formul sabit, esik her noktada dogrulamada yeniden secilir
    og_noktalar = ortak.json_oku(ara / 'nokta_ogretmen.json')
    bkg = olaylar[olaylar.y == 0]
    tas = []
    for ad in girdi['noktalar']:
        s = pd.read_pickle(ara / f'nokta_{ortak.kisa_ad(ad)}.pkl')
        df = pd.concat([bkg, s[bkg.columns.intersection(s.columns)]], ignore_index=True)
        z_og = og_noktalar[ad]['Z_test']
        r = {'sinyal noktasi': ad, 'Z ogretmen': z_og}
        for mod in secilen:
            z = nokta(df, ifade[mod][1](df))['Z_test']
            r[f'Z formul {mod}'] = z
            r[f'oran {mod}'] = z / z_og if z_og and np.isfinite(z_og) and z_og > 0 else np.nan
        tas.append(r)
    tas = pd.DataFrame(tas).set_index('sinyal noktasi') if tas else pd.DataFrame()
    tas.to_csv(son / 'sr_tasinabilirlik.csv')

    # --- Sistematik: MET ve jet +-%5, esik nominal ---------------------------
    sist = []
    z0 = {'ogretmen': og_nokta['Z_test'],
          **{f'formul {m}': secilen[m].Z_test for m in secilen}}
    for ad in girdi['sistematik']:
        t = pd.read_pickle(ara / f'sist_{ortak.kisa_ad(ad)}.pkl')
        og_t = pd.read_pickle(ara / f'sist_{ortak.kisa_ad(ad)}_ogretmen.pkl').to_numpy()
        r = {'kaydirma': ad}
        z, _, _ = sabit_esikte(t, og_t, og_nokta['esik'])
        r['Z ogretmen'], r['dZ/Z ogretmen'] = z, z / z0['ogretmen'] - 1
        for mod in secilen:
            z, _, _ = sabit_esikte(t, ifade[mod][1](t), secilen[mod].esik)
            r[f'Z formul {mod}'], r[f'dZ/Z formul {mod}'] = z, z / z0[f'formul {mod}'] - 1
        sist.append(r)
    sist = pd.DataFrame(sist).set_index('kaydirma')
    sist.to_csv(son / 'sr_sistematik.csv')

    # --- Ozet tablosu ---------------------------------------------------------
    ana = [{'secim': f'kesim tabanli (Bell ve ark.: {bell_satir})', 'Z_A': bell_ref.Z_test,
            'S': bell_ref.S, 'B': bell_ref.B, 'N_eff': bell_ref.Neff, 'eps_S': bell_ref.epsS}]
    for mod in secilen:
        r = secilen[mod]
        ana.append({'secim': f'formul {mod} (uzunluk {r.uzunluk})', 'Z_A': r.Z_test,
                    'S': r.S, 'B': r.B, 'N_eff': r.Neff, 'eps_S': r.epsS})
    ana.append({'secim': 'ogretmen (BDT, defterdeki fit_bdt)', 'Z_A': og_nokta['Z_test'],
                'S': og_nokta['S'], 'B': og_nokta['B'], 'N_eff': og_nokta['Neff'],
                'eps_S': og_nokta['epsS']})
    ana = pd.DataFrame(ana).set_index('secim')

    dog = pd.DataFrame(girdi['turetilmis_dogrulama']).T
    dog['durum'] = np.where(dog.aday.isna(), 'dogrulanamadi (nominal birakildi)', 'tamam')
    eksik = [f for f in og['oznitelikler'] if f in dog.index and pd.isna(dog.loc[f, 'aday'])]

    md = [f"# Sembolik regresyon ozeti ({ortak.SINYAL}, oran {a.oran:g})", '',
          'Anlamlilik: defterdeki `asimov` (arka plan belirsizligi yok), '
          f'L = {ortak.LUMI / 1000:.2f} fb^-1. Esik dogrulama kumesinde, rapor test '
          'kumesinde; test verimleri tum veriye olceklidir (x 1/FRAC).', '',
          '## 1. Karsilastirma (test kumesi)', '', md_tablo(ana), '']
    md += ['## 2. Secilen formuller', '']
    baslik = {'A': 'damitma, pyoperon', 'B': 'dogrudan siniflandirma, gplearn'}
    for mod in secilen:
        r = secilen[mod]
        md += [f'### Formul {mod} ({baslik[mod]}), uzunluk {r.uzunluk}', '',
               f'Okunabilir: `{r.okunabilir}`', '', f'$$ f_{mod} = {r.latex} $$', '',
               f'Esik (dogrulamada secildi): $f_{mod} > {r.esik:.4g}$', '']
        if mod == 'B':
            md += ['gplearn\'in `div`, `log`, `sqrt` fonksiyonlari korumalidir (|payda| ya da '
                   '|arguman| < 0.001 iken sabit deger, `log` ve `sqrt` mutlak deger uzerinden); '
                   'degerlendirme bu kesin bicimle (`sympy` sutunundaki `pdiv`, `plog`, `psqrt`) '
                   'yapilir, yukaridaki okunabilir bicim esikleri gostermez.', '']
        md += yorumla(ortak.formul_derle(r.okunabilir)[0], olaylar) + ['',
               '*Not: terim yorumlari degisken anlamina ve turev isaretine dayanir; '
               'formul belli olunca elle gozden gecirilmelidir.*', '']
    md += ['## 3. Tasinabilirlik (formul sabit, esik her noktada dogrulamada)', '',
           md_tablo(tas) if len(tas) else '_diger sinyal noktasi yok_', '',
           '## 4. Sistematik (MET / jet enerjisi +-%5, esik nominal)', '', md_tablo(sist), '',
           'Jet kaydirmasi jet sayisi gocunu (esik etrafinda) ve MET yayilimini icermez. '
           'Turetilmis degiskenlerin yeniden hesabi nominal veride dogrulandi:', '',
           md_tablo(dog[['durum', 'q99_goreli_sapma']]), '']
    if eksik:
        md += [f'**Uyari:** ogretmen ozniteliklerinden {eksik} dogrulanamadi; bu degiskenler '
               'kaydirmada nominal kaldi, ogretmenin sistematigi eksik tahmin edilebilir.', '']
    md += ['## 5. Grafikler', '',
           '- `sr_girdileri.png`: sembolik regresyon girdileri, sinyal ve arka plan',
           '- `ogretmen_asiri_ogrenme.png`: ogretmen egitim / OOF skorlari, KS testi',
           '- `ogretmen_degisken_onemi.png`: ogretmen permutasyon onemi',
           '- `sr_uzunluk_anlamlilik.png`: formul uzunlugu - anlamlilik',
           *[f'- `sr_secilen_{m}_yigin.png`: formul {m}, surec bazinda yiginli dagilim'
             for m in secilen], '']
    (son / 'summary.md').write_text('\n'.join(md), encoding='utf-8')
    print(ana.round(3).to_string())
    print(f"\nyazildi: {son / 'summary.md'}")


if __name__ == '__main__':
    main()
