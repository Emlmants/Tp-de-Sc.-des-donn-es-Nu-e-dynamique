import numpy as np


def nuees_dynamiques(X, k, max_iter=100, tol=1e-6, seed=42):
    """
    X : matrice des données (n individus, p variables).
    k : nombre de classes.
    Retour : classes, centres, inertie et nombre d'itérations.
    """
    X = np.asarray(X, dtype=float)

    # Vérification des paramètres
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X doit être une matrice non vide.")
    if not np.isfinite(X).all():
        raise ValueError("X doit contenir uniquement des valeurs finies.")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(X):
        raise ValueError("k doit être un entier entre 1 et n.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter doit être un entier positif.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol doit être un réel positif ou nul.")

    # 1. Initialisation des représentants
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=k, replace=False)
    centres = X[indices].copy()

    for iteration in range(1, max_iter + 1):

        # 2. Distances au carré entre individus et centres
        distances = np.sum(
            (X[:, np.newaxis, :] - centres[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # 3. Affectation au représentant le plus proche
        classes = np.argmin(distances, axis=1)

        # 4. Mise à jour des représentants par les barycentres
        nouveaux_centres = centres.copy()

        for j in range(k):
            individus = X[classes == j]

            if len(individus) > 0:
                nouveaux_centres[j] = np.mean(individus, axis=0)
            # Une classe vide conserve son ancien représentant.

        # 5. Mesure du déplacement maximal des centres
        deplacement = np.max(
            np.sqrt(np.sum((nouveaux_centres - centres) ** 2, axis=1))
        )
        centres = nouveaux_centres

        if deplacement <= tol:
            break

    # Affectation finale aux centres obtenus
    distances = np.sum(
        (X[:, np.newaxis, :] - centres[np.newaxis, :, :]) ** 2,
        axis=2
    )
    classes = np.argmin(distances, axis=1)

    # Inertie intraclasse : somme des distances au carré
    inertie = np.sum((X - centres[classes]) ** 2)

    return classes, centres, inertie, iteration


# Exemple d'application
X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [8, 8],
    [9, 8],
    [8, 9]
], dtype=float)

classes, centres, inertie, iterations = nuees_dynamiques(X, k=2)

print("Classes des individus :", classes + 1)
print("Centres des classes :\n", centres)
print("Inertie intraclasse :", round(float(inertie), 4))
print("Nombre d'itérations :", iterations)