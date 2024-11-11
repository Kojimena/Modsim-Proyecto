import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

DATASET = None
TARGET = ""
MODEL = None


def preprocess(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            if len(data[column].unique()) == 2:
                data[column] = pd.Categorical(data[column]).codes
            else:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
    return data

def fitness_function(particle, X, y, model, min_features, max_features):
    num_features = np.count_nonzero(particle)
    if num_features == 0:
        return 0
    if num_features < min_features or num_features > max_features:
        penalty = 0.5  
    else:
        penalty = 1
    X_selected = X.iloc[:, particle == 1]
    model_clone = clone(model)
    try:
        score = np.mean(cross_val_score(model_clone, X_selected, y, cv=5, scoring='accuracy'))
    except ValueError:
        score = 0
    return score * penalty

def feature_selection_pso(data, target_variable, model, n_particles=30, n_iterations=8, min_features=1, max_features=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    data = preprocess(data)
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    n_features = X.shape[1]

    if max_features is None:
        max_features = n_features

    # Inicializar las partículas y velocidades
    particles = np.random.choice([0, 1], size=(n_particles, n_features))
    velocities = np.random.uniform(-1, 1, (n_particles, n_features))

    # Asegurar que cada partícula tenga al menos el número mínimo de características y no más del máximo
    for i in range(n_particles):
        if np.count_nonzero(particles[i]) < min_features:
            selected_indices = np.random.choice(n_features, min_features, replace=False)
            particles[i, selected_indices] = 1
        elif np.count_nonzero(particles[i]) > max_features:
            deselected_indices = np.random.choice(np.where(particles[i] == 1)[0], np.count_nonzero(particles[i]) - max_features, replace=False)
            particles[i, deselected_indices] = 0

    # Inicializar el mejor global y los mejores personales
    pbest = particles.copy()
    pbest_scores = np.array([fitness_function(p, X, y, model, min_features, max_features) for p in particles])
    gbest_idx = np.argmax(pbest_scores)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]

    # Historial para la grafica
    gbest_score_history = []

    # Hiperparámetros de PSO
    w = 0.9  # Inercia inicial
    w_min = 0.4  # Inercia mínima
    c1 = 2.5  # Coeficiente cognitivo inicial
    c2 = 0.5  # Coeficiente social inicial

    # Iteraciones del PSO
    no_improvement_counter = 0  # Contador de iteraciones sin mejora
    max_no_improvement = 5  # Máximo número de iteraciones sin mejora antes de aplicar mutación

    for iteration in range(n_iterations):
        for i in range(n_particles):
            # Actualizar la velocidad
            r1 = np.random.rand(n_features)
            r2 = np.random.rand(n_features)
            cognitive = c1 * r1 * (pbest[i] - particles[i])
            social = c2 * r2 * (gbest - particles[i])
            velocities[i] = w * velocities[i] + cognitive + social

            # Actualizar la posición de la partícula usando la función sigmoide
            sigmoid = 1 / (1 + np.exp(-velocities[i]))
            particles[i] = np.where(np.random.rand(n_features) < sigmoid, 1, 0)

            # Asegurar que la partícula cumpla con el número mínimo y máximo de características
            if np.count_nonzero(particles[i]) < min_features:
                selected_indices = np.random.choice(n_features, min_features, replace=False)
                particles[i, selected_indices] = 1
            elif np.count_nonzero(particles[i]) > max_features:
                deselected_indices = np.random.choice(np.where(particles[i] == 1)[0], np.count_nonzero(particles[i]) - max_features, replace=False)
                particles[i, deselected_indices] = 0

            # Evaluar el nuevo subconjunto de características
            score = fitness_function(particles[i], X, y, model, min_features, max_features)

            # Actualizar el mejor personal
            if score > pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score

            # Actualizar el mejor global
            if score > gbest_score:
                gbest = particles[i].copy()
                gbest_score = score
                no_improvement_counter = 0  # Reiniciar contador de iteraciones sin mejora
            else:
                no_improvement_counter += 1

        # Reducir el coeficiente de inercia
        w = max(w_min, w * 0.99)

        # Ajustar los coeficientes cognitivo y social
        c1 = max(0.5, c1 * 0.99)
        c2 = min(2.5, c2 * 1.01)

        # Aplicar mutación si no hay mejora
        if no_improvement_counter >= max_no_improvement:
            mutation_idx = np.random.randint(0, n_particles)
            particles[mutation_idx] = np.random.choice([0, 1], size=n_features)  # Reinicializar la partícula aleatoriamente
            if np.count_nonzero(particles[mutation_idx]) < min_features:
                selected_indices = np.random.choice(n_features, min_features, replace=False)
                particles[mutation_idx, selected_indices] = 1
            elif np.count_nonzero(particles[mutation_idx]) > max_features:
                deselected_indices = np.random.choice(np.where(particles[mutation_idx] == 1)[0], np.count_nonzero(particles[mutation_idx]) - max_features, replace=False)
                particles[mutation_idx, deselected_indices] = 0
            no_improvement_counter = 0

        # Guardar el mejor score de esta iteración
        gbest_score_history.append(gbest_score)

    # Imprimir resultados finales
    selected_features = X.columns[gbest == 1]

    return selected_features


def main(csv: str, target: str, model: int):
    # Parámetros de entrada
    dataset_path = csv
    target_variable = target
    seed = 105678

    # Cargar el dataset
    data = pd.read_csv(dataset_path)

    # Definir el modelo
    if model == 0:
        model = DecisionTreeClassifier()
    elif model == 1:
        model = RandomForestClassifier()
    elif model == 2:
        model = SVC()
    elif model == 3:
        model = KNeighborsClassifier()
    elif model == 4:
        model = LogisticRegression()

    ###################################################################################################################
    min_features = 2  # minimal number of features in a subset of features
    max_features = 4  # maximum number of features in a subset of features
    seed = 105678  # seed for reproducibility
    n_particles = 30  # number of particles
    n_iterations = 8  # number of iterations
    ###################################################################################################################
    
    # Ejecutar la selección de características
    selected_features = feature_selection_pso(data, target_variable, model, n_particles, n_iterations, min_features, max_features, seed=seed)
    return selected_features

if __name__ == '__main__':
    print(main("datasets\iris.csv", "Species", 1))