
import numpy as np
import pandas as pd

# Parametres modifiables pour utiliser le futur dataset du professeur.
COLONNES = ['Annual Income (k$)', 'Spending Score (1-100)']
Q = 3              # Nombre d'etalons par classe.
R = 1              # Dimension du sous-espace affine.
ALPHA = 0.2        # Penalisation des largeurs d'intervalles.
PLANCHER = 0.02    # Plancher des valeurs propres des covariances.
N_INIT = 5
MAX_ITER = 100
GRAINE = 42

MODES = ('point', 'points', 'axes', 'distribution', 'structure')


def distances2(X, Y):
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)


class NueesDynamiques:
    """Optimisation alternee; une representation distincte par classe.

    points : q etalons observes, recherche locale a un remplacement.
    axes : sous-espace affine de dimension r (ACP locale).
    distribution : gaussienne, covariance bornee inferieurement.
    structure : boite de quantiles avec penalisation de sa largeur.
    """
    def __init__(self, k=5, mode='points', q=3, r=1, alpha=0.2,
                 floor=0.02, n_init=5, max_iter=100, seed=42):
        if mode not in MODES:
            raise ValueError('Mode inconnu.')
        for name, value in [('k', k), ('q', q), ('r', r),
                            ('n_init', n_init), ('max_iter', max_iter)]:
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(name + ' doit etre un entier positif.')
        if not 0 < alpha < 0.5 or not np.isfinite(floor) or floor <= 0:
            raise ValueError('Exiger 0 < alpha < 0.5 et floor > 0.')
        self.k, self.mode, self.q, self.r = k, mode, q, r
        self.alpha, self.floor = alpha, floor
        self.n_init, self.max_iter, self.seed = n_init, max_iter, seed

    def cost(self, X, rep):
        if self.mode == 'point':
            return ((X - rep['mean']) ** 2).sum(axis=1)
        if self.mode == 'points':
            return distances2(X, rep['prototypes']).min(axis=1)
        if self.mode == 'axes':
            Z = X - rep['mean']
            residual = Z - (Z @ rep['basis']) @ rep['basis'].T
            return (residual ** 2).sum(axis=1)
        if self.mode == 'distribution':
            Z = (X - rep['mean']) @ rep['vectors']
            return 0.5 * ((Z ** 2 / rep['values']).sum(axis=1)
                          + np.log(rep['values']).sum()
                          + X.shape[1] * np.log(2 * np.pi))
        lo, hi = rep['low'], rep['high']
        return (np.maximum(lo-X, 0) + np.maximum(X-hi, 0)).sum(axis=1) + self.alpha * (hi-lo).sum()

    def update(self, group, old=None):
        if len(group) == 0:
            return old  # La classe vide conserve sa representation.
        mean = group.mean(axis=0)
        if self.mode == 'point':
            return {'mean': mean}
        if self.mode == 'points':
            # Candidats fixes : toutes les observations d'apprentissage.
            D = distances2(group, self.X_)
            if old is None:
                chosen = [int(np.argmin(D.sum(axis=0)))]
                while len(chosen) < self.q:
                    scores = np.minimum(D[:, chosen].min(axis=1)[:, None], D).sum(axis=0)
                    scores[chosen] = np.inf
                    chosen.append(int(np.argmin(scores)))
            else:
                chosen = old['indices'].tolist()
            # Des echanges strictement ameliorants garantissent la descente.
            for _ in range(20):
                changed = False
                for h in range(self.q):
                    others = chosen[:h] + chosen[h+1:]
                    scores = (np.minimum(D[:, others].min(axis=1)[:, None], D).sum(axis=0)
                              if others else D.sum(axis=0))
                    scores[others] = np.inf
                    candidate = int(np.argmin(scores))
                    if scores[candidate] < scores[chosen[h]] - 1e-12:
                        chosen[h] = candidate
                        changed = True
                if not changed:
                    break
            ids = np.array(chosen, dtype=int)
            return {'indices': ids, 'prototypes': self.X_[ids].copy()}
        if self.mode in ('axes', 'distribution'):
            Z = group-mean
            values, vectors = np.linalg.eigh(Z.T @ Z / len(group))
            if self.mode == 'axes':
                return {'mean': mean, 'basis': vectors[:, -self.r:]}
            return {'mean': mean, 'values': np.maximum(values, self.floor), 'vectors': vectors}
        # Quantiles empiriques par inverse de la fonction de repartition.
        ordered = np.sort(group, axis=0)
        low = ordered[max(0, int(np.ceil(self.alpha*len(group)))-1)]
        high = ordered[max(0, int(np.ceil((1-self.alpha)*len(group)))-1)]
        return {'low': low, 'high': high}

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
            raise ValueError('X doit etre une matrice finie non vide.')
        if self.k > len(X) or self.q > len(X):
            raise ValueError('k et q doivent etre inferieurs ou egaux a n.')
        if self.mode == 'axes' and self.r >= X.shape[1]:
            raise ValueError('Axes : exiger 1 <= r < nombre de variables.')
        self.X_ = X.copy()
        best = None
        self.runs_ = []
        for restart in range(self.n_init):
            rng = np.random.default_rng(self.seed + restart)
            seeds = X[rng.choice(len(X), self.k, replace=False)]
            labels = distances2(X, seeds).argmin(axis=1)
            reps = [self.update(X[labels == j] if np.any(labels == j) else seeds[j:j+1])
                    for j in range(self.k)]
            costs = np.column_stack([self.cost(X, rep) for rep in reps])
            labels = costs.argmin(axis=1)
            history = [float(costs[np.arange(len(X)), labels].sum())]
            converged = False
            for iteration in range(1, self.max_iter+1):
                new = [self.update(X[labels == j], reps[j]) for j in range(self.k)]
                costs = np.column_stack([self.cost(X, rep) for rep in new])
                updated = costs.argmin(axis=1)
                value = float(costs[np.arange(len(X)), updated].sum())
                if value > history[-1] + 1e-8 * max(1, abs(history[-1])):
                    raise ArithmeticError('Le critere a augmente.')
                history.append(value)
                # Exiger aussi la stabilite des prototypes dans le mode discret.
                same_rep = self.mode != 'points' or all(np.array_equal(a['indices'], b['indices']) for a,b in zip(reps,new))
                converged = np.array_equal(labels, updated) and same_rep
                labels, reps = updated, new
                if converged:
                    break
            run = {'objective': value, 'iterations': iteration, 'converged': converged,
                   'occupied': int(len(np.unique(labels))), 'history': history,
                   'labels': labels.copy()}
            self.runs_.append(run)
            if best is None or value < best[0]:
                best = (value, labels.copy(), reps, run.copy())
        self.objective_, self.labels_, self.representations_, self.info_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1] or not np.isfinite(X).all():
            raise ValueError('Dimensions ou valeurs invalides.')
        return np.column_stack([self.cost(X, rep) for rep in self.representations_]).argmin(axis=1)


def silhouette(X, labels):
    """Silhouette euclidienne commune aux cinq representations."""
    groups = np.unique(labels)
    if len(groups) < 2 or len(groups) >= len(X):
        return None
    D = np.sqrt(distances2(X, X))
    values = []
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() == 1:
            values.append(0.0)
            continue
        a = D[i, own].sum()/(own.sum()-1)
        b = min(D[i, labels == j].mean() for j in groups if j != labels[i])
        values.append((b-a)/max(a,b) if max(a,b) else 0.)
    return float(np.mean(values))


def ari(a, b):
    """Indice de Rand ajuste, invariant a la numerotation des classes."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    N = np.zeros((a.max()+1, b.max()+1), dtype=int)
    np.add.at(N, (a,b), 1)
    choose = lambda z: np.sum(z*(z-1)/2)
    total = len(a)*(len(a)-1)/2
    if total == 0:
        return 1.0
    observed, A, B = choose(N), choose(N.sum(1)), choose(N.sum(0))
    expected = A*B/total
    den = (A+B)/2-expected
    return float((observed-expected)/den) if den else 1.0


def echapper(texte):
    """Protection du texte HTML avec les fonctions natives de Python."""
    return str(texte).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def svg_plot(X, labels, names):
    """SVG sans Matplotlib; projection sur les deux premieres variables."""
    colors = ['#1768ac','#de541e','#138a36','#8d3b96','#b8860b','#008b8b','#dc143c']
    low, high = X[:, :2].min(0), X[:, :2].max(0)
    coords = (X[:, :2]-low)/np.maximum(high-low, 1e-12)
    out = ['<svg viewBox="0 0 620 420" role="img">', '<rect x="55" y="20" width="535" height="340" fill="#fafafa" stroke="#888"/>']
    for v in (0, .25, .5, .75, 1):
        out.append(f'<text x="{55+535*v}" y="380" text-anchor="middle" font-size="12">{low[0]+v*(high[0]-low[0]):.1f}</text>')
        out.append(f'<text x="50" y="{365-340*v}" text-anchor="end" font-size="12">{low[1]+v*(high[1]-low[1]):.1f}</text>')
    for i, (x,y) in enumerate(coords):
        out.append(f'<circle cx="{55+535*x:.2f}" cy="{360-340*y:.2f}" r="4" fill="{colors[labels[i]%len(colors)]}" opacity=".8"><title>Ligne {i+1}, classe {labels[i]+1}</title></circle>')
    out.append(f'<text x="320" y="405" text-anchor="middle">{echapper(names[0])}</text>')
    out.append(f'<text transform="translate(14,190) rotate(-90)" text-anchor="middle">{echapper(names[1])}</text></svg>')
    return ''.join(out)


def main():
    print('NUEES DYNAMIQUES')
    chemin = input('Chemin du CSV [Mall_Customers.csv] : ').strip() or 'Mall_Customers.csv'
    print('1 Point (K-means)\n2 Ensemble de points\n3 Axes factoriels')
    print('4 Distribution gaussienne\n5 Structure par intervalles\n6 Comparer les cinq')
    choix = input('Votre choix [2] : ').strip() or '2'
    if choix not in ('1', '2', '3', '4', '5', '6'):
        raise ValueError('Choisir un numero de 1 a 6.')
    k = int(input('Nombre de classes [5] : ').strip() or '5')



    # Lecture des colonnes du fichier CSV.
    X = pd.read_csv(chemin, usecols=COLONNES)[COLONNES].to_numpy(dtype=float)
    X = np.ascontiguousarray(X)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] < 2 or not np.isfinite(X).all():
        raise ValueError('Au moins deux colonnes numeriques et aucune valeur manquante sont requises.')



    # Tous les traitements numeriques utilisent uniquement NumPy.
    moyenne, echelle = X.mean(axis=0), X.std(axis=0)
    echelle[echelle == 0] = 1.0
    Z = (X-moyenne)/echelle
    modes = MODES if choix == '6' else [MODES[int(choix)-1]]
    page = ['<!doctype html><meta charset="utf-8"><title>Segmentation</title>',
            '<style>body{font:16px Arial;max-width:950px;margin:30px auto}svg{width:100%;max-width:760px}</style>',
            '<h1>Nuees dynamiques</h1><p>Les couleurs indiquent les classes. Les numeros ne correspondent pas entre methodes. Les objectifs de modes differents ne sont pas comparables.</p>']
    for mode in modes:
        modele = NueesDynamiques(k=k, mode=mode, q=Q, r=R, alpha=ALPHA,
                                 floor=PLANCHER, n_init=N_INIT,
                                 max_iter=MAX_ITER, seed=GRAINE).fit(Z)
        etiquettes = modele.labels_
        score = silhouette(Z, etiquettes)
        stabilites = [ari(a['labels'], b['labels'])
                      for i, a in enumerate(modele.runs_)
                      for b in modele.runs_[i+1:]]
        stabilite = float(np.mean(stabilites)) if stabilites else None
        print('\nRepresentation :', mode)
        print('Critere :', modele.objective_, '| Silhouette :', score)
        print('ARI moyen :', stabilite, '| Stabilise :', modele.info_['converged'])
        page.append('<h2>'+mode+'</h2>'+svg_plot(X, etiquettes, COLONNES))
        page.append('<p>'+echapper(f'Silhouette : {score}; ARI moyen : {stabilite}')+'</p>')
        lignes = [f'Representation : {mode}', f'Colonnes : {COLONNES}',
                  f'Parametres : k={k}, q={Q}, r={R}, alpha={ALPHA}, plancher={PLANCHER}',
                  f'Demarrages : {N_INIT}; graine initiale : {GRAINE}; max_iter : {MAX_ITER}',
                  f'Objectif : {modele.objective_}; silhouette : {score}; ARI : {stabilite}',
                  f'Iterations : {modele.info_["iterations"]}; stabilise : {modele.info_["converged"]}']
        sauvegarde = {'moyenne': moyenne, 'echelle': echelle,
                      'etiquettes': etiquettes, 'colonnes': np.array(COLONNES),
                      'historique': np.array(modele.info_['history'])}


        for j in range(k):
            groupe = X[etiquettes == j]
            profil = groupe.mean(axis=0) if len(groupe) else None
            description = f'Classe {j+1} : {len(groupe)} individus; moyennes : {profil}'
            print(description)
            lignes.append(description)
            page.append('<p>'+echapper(description)+'</p>')
            for nom, valeur in modele.representations_[j].items():
                sauvegarde[f'classe_{j+1}_{nom}'] = np.asarray(valeur)
        for i, run in enumerate(modele.runs_):
            sauvegarde[f'demarrage_{i}_historique'] = np.array(run['history'])
            sauvegarde[f'demarrage_{i}_etiquettes'] = run['labels']
            lignes.append(f'Demarrage {i+1} : objectif={run["objective"]}; stabilise={run["converged"]}; iterations={run["iterations"]}')


        # Export des variables analysees et de leur segment par NumPy.
        entete = ','.join('"'+c.replace('"', '""')+'"' for c in COLONNES)+',Segment'
        np.savetxt(mode+'_segments.csv', np.column_stack((X, etiquettes+1)),
                   delimiter=',', header=entete, comments='',
                   fmt=['%.10g']*X.shape[1]+['%d'], encoding='utf-8')
        np.savez(mode+'_modele.npz', **sauvegarde)
        with open(mode+'_resume.txt', 'w', encoding='utf-8') as fichier:
            fichier.write('\n'.join(lignes))


    # open et write sont des fonctions natives : aucun module HTML importe.
    with open('segmentation.html', 'w', encoding='utf-8') as fichier:
        fichier.write(''.join(page))
    print('\nTermine. Ouvrir segmentation.html pour voir les groupes.')
    print('CSV, resumes TXT et representations NPZ enregistres dans le dossier courant.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, ArithmeticError) as erreur:
        print('Erreur :', erreur)
