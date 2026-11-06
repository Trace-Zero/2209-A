"""Adim 2: ogretmen model (defterdeki BDT, fit_bdt) ve out-of-fold logitler.

Ogretmen, defterin §1g'de sectigi oznitelikleri (features.json) kullanir.
Test kumesi disindaki olaylar 5 kata bolunur; her olayin skoru onu gormemis
kattan gelir (OOF). Test olaylarinin skoru 5 kat modelinin ortalamasidir; hicbiri
test olaylarini gormemistir. Hedef: s = log(p / (1 - p)), p in [1e-6, 1 - 1e-6].
Erken durdurma her katin egitim kisminin %20'sinde yapilir (fit_bdt).

Yazilanlar (ara dizin): ogretmen.pkl, ogretmen_ozet.json,
sist_<ad>_ogretmen.pkl, nokta_<ad>_ogretmen.json; grafikler (son dizin).
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from symbolic import ortak

KAT = 5


def ogretmen_egit(df, F, tohum=0):
    """OOF olasiliklari (test disi), test ortalamasi, kat modelleri ve tani."""
    gor, test = df[df.kume != 'test'], df[df.kume == 'test']
    oof = pd.Series(np.nan, index=gor.index)
    modeller, tani = [], []
    skf = StratifiedKFold(n_splits=KAT, shuffle=True, random_state=tohum)
    for i, (a, b) in enumerate(skf.split(gor, gor.y)):
        katlar, disari = gor.iloc[a], gor.iloc[b]
        tr, es = train_test_split(katlar, test_size=0.2, stratify=katlar.y,
                                  random_state=tohum + i)
        skor, n_agac = ortak.fit_bdt(*ortak.sinif_dengeli(tr, F),
                                     *ortak.sinif_dengeli(es, F), tohum + i)
        p = (lambda d, s=skor: (s(d[F].to_numpy()) + 1) / 2)
        oof[disari.index] = p(disari)
        modeller.append(p)
        tani.append({'kat': i, 'agac': n_agac, 'skor': skor, 'tr': tr, 'disari': disari})
    p_test = pd.Series(np.mean([m(test) for m in modeller], axis=0), index=test.index)
    return pd.concat([oof, p_test]).loc[df.index], modeller, tani


def ortalama_p(modeller, d):
    return np.mean([m(d) for m in modeller], axis=0)


def calisma_noktasi(df, skor):
    va, te = (df.kume == 'dogrulama').to_numpy(), (df.kume == 'test').to_numpy()
    b, s = (df.y == 0).to_numpy(), (df.y == 1).to_numpy()
    skor, w = np.asarray(skor), df.weight.to_numpy()
    return ortak.calisma_noktasi(skor[va & b], skor[va & s], w[va & b], w[va & s],
                                 skor[te & b], skor[te & s], w[te & b], w[te & s],
                                 df.src.to_numpy()[te & b])


def tani_grafikleri(tani, F, son):
    """Asiri ogrenme (egitim vs OOF, KS) ve permutasyon onemi -- kat 0."""
    plt = ortak.stil_uygula()
    t0 = tani[0]
    sk = lambda d: ortak.logit((t0['skor'](d[F].to_numpy()) + 1) / 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = {}
    kenar = np.linspace(*np.percentile(sk(t0['disari']), [0.5, 99.5]), 40)
    for y, renk, ad in ((0, 'tab:blue', 'arka plan'), (1, 'crimson', 'sinyal')):
        tr, ds = sk(t0['tr'][t0['tr'].y == y]), sk(t0['disari'][t0['disari'].y == y])
        ks[ad] = ks_2samp(tr, ds).pvalue
        ax.hist(tr, bins=kenar, density=True, histtype='step', color=renk, lw=2,
                label=f'{ad} egitim')
        ax.hist(ds, bins=kenar, density=True, histtype='stepfilled', color=renk,
                alpha=.25, label=f'{ad} OOF (KS p={ks[ad]:.3f})')
    ax.set_xlabel('ogretmen logit  $s=\\log[p/(1-p)]$')
    ax.set_ylabel('a.u.')
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(son / 'ogretmen_asiri_ogrenme.png', dpi=120)

    ds = t0['disari']
    imp = ortak.perm_importance(t0['skor'], lambda d: d[F].to_numpy(),
                                ds[ds.y == 0], ds[ds.y == 1], F)
    sira = sorted(F, key=lambda f: -imp[f][0])
    fig, ax = plt.subplots(figsize=(0.7 * len(F) + 2.5, 4))
    ax.bar(sira, [imp[f][0] for f in sira], yerr=[imp[f][1] for f in sira],
           color='#4878a8', capsize=3)
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(range(len(sira)), sira, rotation=35, ha='right')
    ax.set_ylabel('AUC dususu (karistirilinca)')
    plt.tight_layout()
    plt.savefig(son / 'ogretmen_degisken_onemi.png', dpi=120)
    return ks, {f: imp[f][0] for f in sira}


def main():
    a, (ara, son) = ortak.argumanlar('Adim 2: ogretmen model')
    F = ortak.oznitelikler()
    olaylar = pd.read_pickle(ara / 'olaylar.pkl')
    ozet_girdi = ortak.json_oku(ara / 'girdiler_ozet.json')
    ortak.surecleri_ayarla(ozet_girdi['surecler'])

    p, modeller, tani = ogretmen_egit(olaylar, F)
    skor = ortak.logit(p.to_numpy())
    pd.DataFrame({'ogretmen_p': p, 'ogretmen_logit': skor},
                 index=olaylar.index).to_pickle(ara / 'ogretmen.pkl')

    ks, onem = tani_grafikleri(tani, F, son)
    kat_auc = [{'kat': t['kat'], 'agac': t['agac'],
                'AUC_egitim': roc_auc_score(t['tr'].y, t['skor'](t['tr'][F].to_numpy())),
                'AUC_OOF': roc_auc_score(t['disari'].y,
                                         t['skor'](t['disari'][F].to_numpy()))}
               for t in tani]
    nokta = calisma_noktasi(olaylar, skor)

    # Sistematik: kaydirilmis test olaylari, ayni kat modelleriyle
    for ad in ozet_girdi['sistematik']:
        t = pd.read_pickle(ara / f'sist_{ortak.kisa_ad(ad)}.pkl')
        pd.Series(ortak.logit(ortalama_p(modeller, t)), index=t.index,
                  name='ogretmen_logit').to_pickle(
            ara / f'sist_{ortak.kisa_ad(ad)}_ogretmen.pkl')

    # Diger sinyal noktalari: ogretmen her nokta icin ayni yontemle yeniden egitilir
    bkg = olaylar[olaylar.y == 0]
    noktalar = {}
    for ad in ozet_girdi['noktalar']:
        s = pd.read_pickle(ara / f'nokta_{ortak.kisa_ad(ad)}.pkl')
        df = pd.concat([bkg, s[bkg.columns.intersection(s.columns)]], ignore_index=True)
        pp, _, _ = ogretmen_egit(df, F)
        noktalar[ad] = calisma_noktasi(df, ortak.logit(pp.to_numpy()))
        print(f"  {ad:22s} ogretmen Z_test = {noktalar[ad]['Z_test']:.3f}")
    ortak.json_yaz(ara / 'nokta_ogretmen.json', noktalar)

    ortak.json_yaz(ara / 'ogretmen_ozet.json', {
        'oznitelikler': F, 'calisma_noktasi': nokta, 'KS_p': ks,
        'kat_auc': kat_auc, 'permutasyon_onemi': onem})
    print(pd.DataFrame(kat_auc).round(4).to_string(index=False))
    print(f"KS p (egitim vs OOF): {ks}")
    print(f"ogretmen: esik {nokta['esik']:.3f}  Z_dogrulama {nokta['Z_dogrulama']:.3f}  "
          f"Z_test {nokta['Z_test']:.3f}  S {nokta['S']:.2f}  B {nokta['B']:.2f}")


if __name__ == '__main__':
    main()
