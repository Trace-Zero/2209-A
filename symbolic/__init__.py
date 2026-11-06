"""Mono-Z analizine sembolik regresyon katmani (Julia gerektirmez).

    A) damitma: pyoperon SymbolicRegressor, hedef ogretmenin out-of-fold logiti
    B) dogrudan siniflandirma: gplearn SymbolicClassifier

Adimlar ayri calisir ve ara sonuclari diske yazar:

    python -m symbolic.step1_inputs   [--oran 0.1]
    python -m symbolic.step2_teacher  [--oran 0.1]
    python -m symbolic.step3_sr       [--oran 0.1]
    python -m symbolic.step4_evaluate [--oran 0.1]

Sinyal/arka plan tanimlari, agirliklar, SR kesimi, bolunme, BDT ve anlamlilik
fonksiyonlari notebooks/monoz_dark_matter_analizi.ipynb icinden okunur (ortak.py).
"""
