# Exploration - highest uncertainties
def exploration(predictions, uncertainties, **kwargs):
    unc = uncertainties.copy()
    target = unc.columns.values
    unc['score'] = unc[target].rank(pct=True)
    return unc
