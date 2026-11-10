"""Adim 1: symbolic regression girdileri ve olay tablolari.

Defterle ayni yukleme (_extract.load), SR kesimi (signal_region), hucre yeniden
orneklemesi (cell_resample, RESAMPLE) ve bolunme (split3, tohum 0) kullanilir.
Yazilanlar (ara dizin):
    olaylar.pkl           ana sinyal noktasi + arka plan, SR, kume etiketli
    nokta_<ad>.pkl        diger sinyal noktalari (SR, kume etiketli)
    sist_<ad>.pkl         MET / jet +-%5 kaydirilmis TEST olaylari (SR yeniden uygulanmis)
    girdiler_ozet.json    sayilar, turetilmis degisken dogrulamasi, kontroller
"""
import numpy as np
import pandas as pd

from symbolic import ortak

SISTEMATIK = {'MET +5%': (1.05, 1.0), 'MET -5%': (0.95, 1.0),
              'jet +5%': (1.0, 1.05), 'jet -5%': (1.0, 0.95)}
KUMELER = ('egitim', 'dogrulama', 'test')


def on_secim(ex, oran, tohum=0):
    """load ile on secilmis (SR oncesi) olaylar; her olaya kalici olay_no verilir."""
    rng = np.random.default_rng(tohum)

    def al(etiket, kaynak):
        d = ex['load']({etiket: kaynak}).reset_index(drop=True)
        d['olay_no'] = np.arange(len(d))
        if oran < 1:                       # hizli deneme: verim korunur
            d = d[rng.uniform(size=len(d)) < oran].reset_index(drop=True)
            d['weight'] = d.weight / oran
        return d

    bkg = {k: al(k, v) for k, v in ex['BKG'].items()}
    sig = {ad: al('A', v) for ad, v in ex['SIG'].items()}
    return bkg, sig


def kume_etiketi(df, tohum=0):
    tr, va, te = ortak.split3(df, tohum)
    k = pd.Series('egitim', index=df.index)
    k[va.index], k[te.index] = 'dogrulama', 'test'
    return k


def sr_olaylari(on_bkg, on_sig, ex):
    """Defterin §1c'si: SR, birlestirme, hucre yeniden orneklemesi kume_ici_ornekle'de."""
    bkg = pd.concat([ortak.signal_region(d).assign(src=k) for k, d in on_bkg.items()],
                    ignore_index=True)
    sig = ortak.signal_region(on_sig).assign(src='sinyal')
    return bkg.assign(y=0), sig.assign(y=1)


def kume_ici_ornekle(bkg, ex):
    """Defterin split3_bkg'si: her kume (egitim/dogrulama/test) kendi icinde,
    ham agirliktan baslayarak yeniden orneklenir; kumeler arasi sizinti olmaz."""
    w = bkg.weight.to_numpy().copy()
    for k in KUMELER:
        m = (bkg.kume == k).to_numpy()
        w[m] = ex['cell_resample'](bkg[m], ortak.RESAMPLE)
    return w


def on_secim_etiketleri(on, sr, tohum):
    """Her on secilmis olaya kume etiketi: SR olaylari nominal etiketini korur,
    SR disindakiler (modeller hic gormedi) ayni oranlarla rastgele dagitilir."""
    rng = np.random.default_rng(tohum)
    etiket = pd.Series(rng.choice(KUMELER, size=len(on),
                                  p=[1 - 2 * ortak.FRAC, ortak.FRAC, ortak.FRAC]),
                       index=on.olay_no.to_numpy())
    etiket.loc[sr.olay_no.to_numpy()] = sr.kume.to_numpy()
    return etiket.to_numpy()


def main():
    a, (ara, son) = ortak.argumanlar('Adim 1: symbolic regression girdileri')
    ex = ortak.extract_yukle()
    on_bkg, on_sig = on_secim(ex, a.oran)
    ortak.surecleri_ayarla(on_bkg)

    # --- nominal: ana sinyal noktasi -------------------------------------
    bkg, sig = sr_olaylari(on_bkg, on_sig[ortak.SINYAL], ex)
    bkg['kume'], sig['kume'] = kume_etiketi(bkg), kume_etiketi(sig)
    bkg['weight'] = kume_ici_ornekle(bkg, ex)
    olaylar = pd.concat([bkg, sig], ignore_index=True)
    olaylar = olaylar.join(ortak.sr_degiskenleri(olaylar))
    olaylar.to_pickle(ara / 'olaylar.pkl')

    jetsiz = olaylar.n_jets == 0
    kontrol = {
        'dphij_jetsiz_pi_mi': bool(np.allclose(olaylar.dphij[jetsiz], np.pi, atol=1e-3)),
        'r_ust_siniri': float(olaylar.r.max()),
    }

    # --- diger sinyal noktalari (arka plan ortak) -------------------------
    noktalar = {}
    for ad, d in on_sig.items():
        if ad == ortak.SINYAL:
            continue
        s = ortak.signal_region(d).assign(src='sinyal', y=1)
        if len(s) < 10:
            print(f'{ad}: SR\'de {len(s)} olay, atlandi')
            continue
        s['kume'] = kume_etiketi(s)
        s = s.join(ortak.sr_degiskenleri(s))
        s.to_pickle(ara / f'nokta_{ortak.kisa_ad(ad)}.pkl')
        noktalar[ad] = {'olay': len(s), 'verim': s.weight.sum() * ortak.LUMI}

    # --- sistematik kaydirmalar: SR kaydirmadan SONRA yeniden uygulanir ----
    dogrulama = ortak.kurallari_dogrula(olaylar)
    on_bkg_e = {k: d.assign(kume=on_secim_etiketleri(d, bkg[bkg.src == k], 1 + i))
                for i, (k, d) in enumerate(on_bkg.items())}
    on_sig_e = on_sig[ortak.SINYAL].assign(
        kume=on_secim_etiketleri(on_sig[ortak.SINYAL], sig, 99))
    sist = {}
    for ad, (km, kj) in SISTEMATIK.items():
        kb = {k: ortak.kaydir(d, km, kj, dogrulama) for k, d in on_bkg_e.items()}
        ks = ortak.kaydir(on_sig_e, km, kj, dogrulama)
        b, s = sr_olaylari(kb, ks, ex)
        b['weight'] = kume_ici_ornekle(b, ex)   # kaydirilmis SR'de, kume icinde
        t = pd.concat([b, s], ignore_index=True).query("kume == 'test'")
        t = t.join(ortak.sr_degiskenleri(t))
        t.to_pickle(ara / f'sist_{ortak.kisa_ad(ad)}.pkl')
        sist[ad] = {'test_olay': len(t), 'km': km, 'kj': kj}

    sayim = olaylar.groupby(['src', 'kume']).size().unstack(fill_value=0)
    verim = (olaylar.groupby('src').weight.sum() * ortak.LUMI).rename('verim')
    ozet = {'oran': a.oran, 'sinyal': ortak.SINYAL, 'surecler': ortak.SURECLER,
            'sayim': sayim.to_dict(), 'verim': verim.to_dict(), 'kontrol': kontrol,
            'turetilmis_dogrulama': dogrulama, 'noktalar': noktalar, 'sistematik': sist}
    ortak.json_yaz(ara / 'girdiler_ozet.json', ozet)

    plt = ortak.stil_uygula()
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a_, v in zip(ax.flat, ortak.SR_DEGISKENLERI):
        kenar = np.linspace(*np.nanpercentile(olaylar[v], [0.5, 99.5]), 41)
        for y, renk, etiket in ((0, 'blue', 'arka plan (verim agirlikli)'),
                                (1, 'black', f'sinyal {ortak.SINYAL}')):
            d = olaylar[olaylar.y == y]
            a_.hist(d[v], bins=kenar, weights=d.weight, density=True,
                    histtype='step', lw=2, color=renk, label=etiket)
        a_.set_xlabel(v)
        a_.set_ylabel('a.u.')
    ax.flat[0].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(son / 'sr_girdileri.png', dpi=120)

    print(sayim.to_string())
    print('\nturetilmis degisken dogrulamasi (sistematik icin):')
    for k, v in dogrulama.items():
        print(f"  {k:22s} {'tamam' if v['aday'] is not None else 'DOGRULANAMADI'}"
              f"  (q99 goreli sapma {v['q99_goreli_sapma']:.2g})")
    print('kontroller:', kontrol)
    print(f'yazildi: {ara}')


if __name__ == '__main__':
    main()
