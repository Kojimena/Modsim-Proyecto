from typing import List, Tuple, Any

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

DATASET = None
TARGET = ""
MODEL = None

def _preprocess(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        if data[column].dtype == 'object':
            if len(data[column].unique()) == 2:
                data[column] = pd.Categorical(data[column]).codes
            else:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])

    return data


def _generate_random_individuals(population_size, num_features, min_features, max_features):
    individuals = np.zeros((population_size, num_features))
    for i in range(population_size):
        num_ones = np.random.randint(min_features, max_features+1)
        ones_indices = np.random.choice(num_features, num_ones, replace=False)
        individuals[i, ones_indices] = 1
    return individuals


def _train_model(x_train, x_test, y_train, y_test, predictor_names, multiclass=False):
    x_train = x_train.loc[:, predictor_names]
    x_test = x_test.loc[:, predictor_names]

    # Building the random forest model
    mdl = MODEL
    mdl.fit(x_train, y_train)  # Training the Model with x_train & y_train
    y_hat = mdl.predict(x_test)  # Predicting the x_test

    if multiclass:
        prec = precision_score(y_test, y_hat, average='weighted')
    else:
        prec = precision_score(y_test, y_hat)

    return prec


def _choose_parents(population, accuracy, elite_percent):
    # Get elite of top 2 which doesn't mutate
    elite_num = int(round(((elite_percent * population.shape[0]) // 2) * 2))
    ind_ac = np.argsort(-accuracy)
    top_perc = ind_ac[:elite_num]
    elite_population = population[top_perc, :]  # We should keep this elite

    # Normalize accuracy to obtain weights for roulette wheel selection
    weight_norm = accuracy / accuracy.sum()  # calculate normalised weight from accuracy
    weight_comu = weight_norm.cumsum()  # calc cumulative weight from accuracy

    # Roulette wheel selection
    num_parents_wo_elite = population.shape[0] - elite_num
    parents_wo_elite = np.empty([num_parents_wo_elite, population.shape[1]])
    for count in range(num_parents_wo_elite):
        b = weight_comu[-1]  # current last element of weight_comu
        rand_num = np.random.uniform(0, b)  # random foating-point number btw 0 and current max weight_comu

        indices = np.searchsorted(weight_comu, rand_num)  # get indices of the number in weight_comu greater than rand_num
        parents_wo_elite[count, :] = population[indices, :]

    parents = np.concatenate((elite_population, parents_wo_elite), axis=0)  # Concatenate elite and parents_wo_elite to get all parents
    return parents


def _one_point_crossover(parents, elite_percent, mutation_probability, min_features, max_features, population):
    elite_num = int(round(((elite_percent * population.shape[0]) // 2) * 2))
    crossover_population = np.zeros((parents.shape[0], parents.shape[1]))  # first two are elite
    crossover_population[0:elite_num, :] = parents[0:elite_num, :]

    for ii in range(int((parents.shape[0] - elite_num) / 2)):
        n = 2 * ii + elite_num  # gives even number
        parents_couple = parents[n:n + 2, :]  # comb of parents
        b2 = parents.shape[1]  # num of features
        rand_n = np.random.randint(1, b2 - 1)  # generate rand number from 1 to num_of_features-1
        crossover_population[n, :] = np.concatenate([parents_couple[0, :rand_n], parents_couple[1, rand_n:]])
        crossover_population[n + 1, :] = np.concatenate([parents_couple[1, :rand_n], parents_couple[0, rand_n:]])

    # check if every child has minimum number of features or all true values
    for kk in range(crossover_population.shape[0]):
        Sum = np.sum(crossover_population[kk, :])
        if Sum > max_features:
            # if the number of 1s is bigger than max number of features
            excess = int(Sum - max_features)
            indices = np.where(crossover_population[kk, :] == 1)[0]
            position1 = np.random.choice(indices, size=excess, replace=False)
            crossover_population[kk, position1] = 0  # put 0s in random positions
        elif Sum < min_features:
            # if the number of 1s is smaller than min number of features
            missing = int(min_features - Sum)
            indices = np.where(crossover_population[kk, :] == 0)[0]
            position2 = np.random.choice(indices, size=missing, replace=False)
            crossover_population[kk, position2] = 1  # put 1s in random positions

    # mutation
    child_row = crossover_population.shape[0]
    child_col = crossover_population.shape[1]
    num_mutations = round(child_row * child_col * mutation_probability)
    for jj in range(num_mutations):
        ind_row = np.random.randint(0, child_row)  # random number btw 0 and num of rows
        ind_col = np.random.randint(0, child_col)  # random number btw 0 and num of colmns
        if (crossover_population[ind_row, ind_col] == 0 and
                np.sum(crossover_population[ind_row, :]) < max_features):
            crossover_population[ind_row, ind_col] = 1
        elif (crossover_population[ind_row, ind_col] == 1 and
              np.sum(crossover_population[ind_row, :]) >= min_features + 1):
            crossover_population[ind_row, ind_col] = 0

    return crossover_population


def main(csv: str, target: str, model: int) -> tuple[list[str], float]:
    global DATASET, TARGET, MODEL
    DATASET = pd.read_csv(csv)
    TARGET = target

    if model == 0:
        MODEL = DecisionTreeClassifier()
    elif model == 1:
        MODEL = RandomForestClassifier()
    elif model == 2:
        MODEL = SVC()
    elif model == 3:
        MODEL = KNeighborsClassifier()
    elif model == 4:
        MODEL = LogisticRegression()
    else:
        return ["Invalid model"]

    DATASET = _preprocess(DATASET)

    target = DATASET[TARGET]
    predictors = DATASET.drop(TARGET, axis=1)
    predictor_names = predictors.columns

    ###################################################################################################################
    num_features = predictors.shape[1]
    min_features = 2  # minimal number of features in a subset of features
    population_size = 8  # size of population (number of instances)
    max_iterations = 8  # maximum number of iterations
    elite_percent = 0.4  # percentage of elite population which doesn't mutate
    mutation_probability = 0.2  # percentage of total genes that mutate
    max_features = 4  # maximum number of features in a subset of features
    ###################################################################################################################

    x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=42)

    population = _generate_random_individuals(population_size, num_features, min_features, max_features)
    accuracy = np.zeros(population_size)
    response_name = target.name

    for i in range(max_iterations):
        for j in range(population_size):
            accuracy[j] = _train_model(x_train, x_test, y_train, y_test, predictor_names, multiclass=True)

        parents = _choose_parents(population, accuracy, elite_percent)
        population = _one_point_crossover(parents, elite_percent, mutation_probability, min_features, max_features, population)

    gen = 0
    best_acc_i = np.zeros(max_iterations)
    best_acc_i[gen] = max(accuracy)  # keep best accuracy from 1st gen

    is_multiclass = True if len(target.unique()) > 2 else False

    while gen < max_iterations - 1:
        print('Begin iteration num {}/{}'.format(gen + 2, max_iterations))
        gen += 1
        parents = _choose_parents(population, accuracy, elite_percent)
        children = _one_point_crossover(parents, elite_percent, mutation_probability, min_features, max_features, population)
        population = children
        for ind in range(population_size):
            predictor_names_ind = predictor_names[population[ind, :] == 1]
            accuracy_ind = _train_model(x_train, x_test, y_train, y_test, predictor_names_ind, multiclass=is_multiclass)
            accuracy[ind] = accuracy_ind
        best_acc_i[gen] = max(accuracy)

    ind_max_acc = np.argmax(accuracy)
    best_features = population[ind_max_acc, :]

    print(f"Best accuracy: {best_acc_i[-1]}")
    print(f"Best features: {predictor_names[best_features == 1].tolist()}")

    # Return the best features
    return predictor_names[best_features == 1].tolist(), float(best_acc_i[-1])


if __name__ == "__main__":
    print(main("../datasets/iris.csv", "Species", 1))
