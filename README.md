# 2209-A
TÜBİTAK 2209-A Undergraduate Research Project

## İçerik

| Yol | Açıklama |
| --- | --- |
| `notebooks/monoz_dark_matter_analizi.ipynb` | Ana analiz: mono-$Z$ ($Z\to\ell\ell$ + MET), CMS Open Data `RunIIFall15MiniAODv2`; BDT / DNN / NSF karşılaştırması, Bell ve ark. (2012) kesim tabanlı referansı |
| `notebooks/adim_adim_monoz.ipynb` | Aynı analizin adım adım, öğretici sürümü |
| `symbolic/` | Sembolik regresyon katmanı (aşağıda) |
| `Theotical Background/` | Haftalık teorik notlar |

Defter, veriyi `/work/_extract.py` ile yükler ve sonuçları `/work/results/` klasörüne
(`OUT`) yazar.

## Sembolik regresyon katmanı (`symbolic/`)

Öğretmen modelin (defterdeki BDT) kararını kısa, okunabilir bir formüle indirger ve bu
formülü mevcut kesim tabanlı seçimle ve öğretmenle karşılaştırır. Julia gerektirmez.

| Mod | Kütüphane | Hedef |
| --- | --- | --- |
| A, damıtma | `pyoperon` `SymbolicRegressor` | öğretmenin out-of-fold logiti, amaçlar `['r2', 'length']` (Pareto cephesi) |
| B, doğrudan sınıflandırma | `gplearn` `SymbolicClassifier` | sinyal / arka plan etiketi; karmaşıklık `parsimony_coefficient` ile cezalandırılır |

Girdiler boyutsuzdur: $r=E_T^{\rm miss}/p_T^{\ell\ell}$, $x=E_T^{\rm miss}/m_Z$,
$\Delta\phi(\ell\ell,E_T^{\rm miss})$, $\Delta R(\ell\ell)$, $(m_{\ell\ell}-m_Z)/\Gamma_Z$,
$\min\Delta\phi({\rm jet},E_T^{\rm miss})$ (jet yoksa $\pi$).

**Mevcut koddan yeniden kullanılanlar.** Sinyal/arka plan tanımları, ağırlıklar, SR kesimi
(`signal_region`), hücre yeniden örneklemesi (`cell_resample`), bölünme (`split3`), öğretmen
(`fit_bdt`), anlamlılık (`asimov`, `pick_cut`, `apply_cut`), permütasyon önemi ve Bell
kesimleri **defterden okunur** (`symbolic/ortak.py`); kopyalanmaz. Öğretmen, defterin §1g'de
seçip `OUT/features.json`'a yazdığı öznitelikleri kullanır.

### Kurulum

```bash
pip install pyoperon gplearn sympy scikit-learn pandas matplotlib scipy
```

Not: pyoperon 0.6.2'de Pareto cephesi hesaplanırken bir tür uyuşmazlığı (float32/float64)
vardır; `symbolic/step3_sr.py` bunu küçük bir sarmalayıcıyla giderir.

### Çalıştırma

Önce defteri en az §1g'ye kadar çalıştırın (`features.json` oluşur). Ardından depo kökünden,
**önce verinin onda biriyle** deneyin:

```bash
python -m symbolic.step1_inputs   --oran 0.1
python -m symbolic.step2_teacher  --oran 0.1
python -m symbolic.step3_sr       --oran 0.1 --nesil 20 --nufus 300
python -m symbolic.step4_evaluate --oran 0.1
```

Sorun yoksa aynı komutları `--oran` olmadan (ve adım 3'te `--nesil`/`--nufus` olmadan,
kütüphanelerin varsayılanlarıyla) tam veride çalıştırın.

| Adım | Ne yapar | Yazdıkları |
| --- | --- | --- |
| 1 | Olaylar, SR, bölünme, girdiler; diğer sinyal noktaları; MET ve jet ±%5 kaydırılmış test olayları (SR kaydırmadan sonra yeniden uygulanır) | `OUT/symbolic/*.pkl`, `girdiler_ozet.json`, `sr_girdileri.png` |
| 2 | Öğretmen: 5 kat out-of-fold logit, aşırı öğrenme ve değişken önemi; diğer sinyal noktaları için yeniden eğitim | `ogretmen*.pkl/json`, `ogretmen_*.png` |
| 3 | A: pyoperon (alt örnek 15k), B: gplearn (alt örnek 5k; parsimony 0.0005, 0.001, 0.005, 0.01) | `OUT/sr_pareto_A.csv`, `OUT/sr_pareto_B.csv` |
| 4 | Her formül: eşik doğrulamada, anlamlılık testte; seçim, taşınabilirlik, sistematik | `OUT/summary.md`, `sr_*.csv`, `sr_*.png` |

`--oran` verildiğinde bütün çıktılar `OUT/symbolic_oran<oran>/` altına gider. `MONOZ_OUT`
ve `MONOZ_EXTRACT` ortam değişkenleri `OUT` ve `_extract.py` yolunu değiştirir.

### Yöntem notları

- **Bölünme:** defterle aynı (`split3`, tohum 0): eğitim %60, doğrulama %20, test %20.
  Öğretmenin OOF skorları test dışındaki olaylarda 5 katla üretilir; test olaylarının skoru
  5 kat modelinin ortalamasıdır, dolayısıyla formül araması test etiketlerini hiç görmez.
- **Alt örnek:** katmanlıdır (sinyal yarı, arka planda her süreçten eşit sayıda olay; küçük
  süreçler tamamen alınır). Ağırlıklar her sürecin toplam verimini koruyacak şekilde yeniden
  normalize edilir ve iki sınıf eşit toplam ağırlığa getirilir. pyoperon `sample_weight`
  desteklemediği için A modunda ağırlık, ağırlıkla orantılı yeniden örneklemeyle uygulanır;
  gplearn ağırlığı doğrudan alır.
- **Formüller** metinden `sympy` ile ifadeye, oradan `sympy.lambdify` ile numpy fonksiyonuna
  çevrilir (`eval` yok). Her çeviri, kütüphanenin kendi hesapladığı değerle karşılaştırılır
  (`ceviri_sapmasi` sütunu). gplearn'ün korumalı `div`/`log`/`sqrt` fonksiyonları birebir
  korunur (`pdiv`, `plog`, `psqrt`).
- **Seçim:** doğrulama anlamlılığının en büyük değerinin %95'ine ulaşan en kısa formül.
- **Anlamlılık:** defterdeki `asimov` (arka plan belirsizliği içermez), $\mathcal{L}$ defterden.
- **Sistematik:** MET ve jet enerjisi ±%5 ölçeklenir; türetilmiş değişkenler (ör. `mT`,
  `met_over_ptZ`) yalnızca tanımları nominal veride doğrulanırsa yeniden hesaplanır.
  Jet sayısı göçü ve jet kaydırmasının MET'e yayılımı modellenmez.
