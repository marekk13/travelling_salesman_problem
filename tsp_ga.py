import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, List


class DataHandler:
    """Handles geospatial data loading and processing"""

    def __init__(self, data_dir: str = r"C:\Users\Marek\PycharmProjects\problem_komiwojazera"):
        self.data_dir = Path(data_dir)
        self.raw_data_path = self.data_dir / "Miejscowosci.zip"
        self.distances_matrix_path = self.data_dir / "distances_matrix.csv"
        self.cities_coordinates_path = self.data_dir / "cities.shp"
        self.cities = ["Warszawa", "Kraków", "Wrocław", "Łódź", "Poznań", "Gdańsk",
                       "Szczecin", "Lublin", "Bydgoszcz", "Białystok", "Katowice",
                       "Gdynia", "Częstochowa", "Radom", "Rzeszów", "Toruń", "Sosnowiec",
                       "Kielce", "Gliwice", "Olsztyn", "Bielsko-Biała", "Zabrze", "Bytom",
                       "Zielona Góra", "Rybnik", "Ruda Śląska", "Opole", "Tychy", "Gorzów Wielkopolski",
                       "Dąbrowa Górnicza", "Elbląg", "Płock", "Koszalin", "Tarnów", "Włocławek",
                       "Chorzów", "Wałbrzych", "Piaseczno", "Kalisz", "Legnica", "Grudziądz", "Jaworzno",
                       "Słupsk", "Jastrzębie-Zdrój", "Nowy Sącz", "Jelenia Góra", "Siedlce", "Mysłowice",
                       "Piła", "Ostrów Wielkopolski"]
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_geospatial_data(self) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Load processed distance matrix and city coordinates"""
        return (
            pd.read_csv(self.distances_matrix_path, index_col=0),
            gpd.read_file(self.cities_coordinates_path).set_index("nazwa")
        )

    def generate_distance_matrix(self, cities: list) -> None:
        """Generate and save distance matrix from shapefile data"""
        cities_gdf = gpd.read_file(self.cities_coordinates_path)
        dist_matrix = pd.DataFrame(index=cities, columns=cities)

        for city in cities:
            distances = cities_gdf.geometry.distance(
                cities_gdf.loc[cities_gdf['nazwa'] == city, 'geometry'].iloc[0]
            )
            dist_matrix[city] = (distances.values / 1000).round(0).astype(int)

        dist_matrix.to_csv(self.distances_matrix_path)


class TSPOptimizer:
    """Genetic algorithm implementation for TSP problem"""

    DEFAULT_PARAMS = {
        'max_iter': 400,
        'n_cities': 50,
        'n_pop': 100,
        'selection_size': 90,
        'crossover_prob': 0.95,
        'mutation_prob': 0.05,
        'elitist_succession_prct': 0.15,
        'satisfying_result': 4000,
        'selection_method': 'ranking'
    }

    def __init__(self, data_handler: DataHandler, **kwargs):
        self.dh = data_handler
        self.params = {**self.DEFAULT_PARAMS, **kwargs}

        self._validate_params()
        self._initialize_city_list()

    def _validate_params(self) -> None:
        """Validate input parameters"""
        if self.params['n_cities'] > 50:
            raise ValueError("Maximum number of cities is 50")

        if self.params['selection_method'] not in ['ranking', 'roulette']:
            raise ValueError("Invalid selection method")

        if self.params['selection_size'] > self.params['n_pop']:
            raise ValueError("Number of individuals for ranking-based selection exceeds size of population")

    def _initialize_city_list(self) -> None:
        """Initialize list of Polish cities"""
        self.cities = ["Warszawa", "Kraków", "Wrocław", "Łódź", "Poznań", "Gdańsk",
                       "Szczecin", "Lublin", "Bydgoszcz", "Białystok", "Katowice",
                       "Gdynia", "Częstochowa", "Radom", "Rzeszów", "Toruń", "Sosnowiec",
                       "Kielce", "Gliwice", "Olsztyn", "Bielsko-Biała", "Zabrze", "Bytom",
                       "Zielona Góra", "Rybnik", "Ruda Śląska", "Opole", "Tychy", "Gorzów Wielkopolski",
                       "Dąbrowa Górnicza", "Elbląg", "Płock", "Koszalin", "Tarnów", "Włocławek",
                       "Chorzów", "Wałbrzych", "Piaseczno", "Kalisz", "Legnica", "Grudziądz", "Jaworzno",
                       "Słupsk", "Jastrzębie-Zdrój", "Nowy Sącz", "Jelenia Góra", "Siedlce", "Mysłowice",
                       "Piła", "Ostrów Wielkopolski"]

    def _generate_initial_population(self) -> np.ndarray:
        """Generate initial population of random permutations"""
        return np.apply_along_axis(
            lambda x: np.random.permutation(self.params['n_cities']),
            axis=1,
            arr=np.zeros((self.params['n_pop'], self.params['n_cities']), dtype=int))

    def _calculate_fitness(self, individual: np.ndarray,
                           distances: pd.DataFrame) -> int:
        """Calculate fitness for single individual"""
        indices = individual[:-1]
        next_indices = individual[1:]
        distances_array = distances.values[indices, next_indices]
        return np.sum(distances_array)

    def _evaluate_population(self, population: np.ndarray,
                             distances: pd.DataFrame) -> pd.DataFrame:
        """Evaluate and sort population by fitness"""
        fitness = np.apply_along_axis(self._calculate_fitness, 1, population, distances)
        population_fitness = pd.DataFrame(data=population) \
            .assign(fitness=fitness) \
            .sort_values(by='fitness') \
            .reset_index(drop=True)
        return population_fitness

    def _selection_ranking(self, population: pd.DataFrame) -> pd.DataFrame:
        """Ranking-based selection"""
        return population.head(self.params['selection_size'])

    def _selection_roulette(self, population: pd.DataFrame) -> pd.DataFrame:
        """Roulette wheel selection"""
        probabilities = np.flip(population['fitness'].values) / population['fitness'].sum()
        return population.iloc[np.random.choice(np.arange(population.shape[0]),
                                                size=self.selection_size,
                                                replace=False,
                                                p=probabilities)] \
            .reset_index(drop=True)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Order crossover with repair mechanism"""
        cross_point_1, cross_point_2 = np.sort(np.random.randint(0, np.shape(parent1)[0], 2))

        offspring1 = np.concatenate(
            (parent1[:cross_point_1],
             parent2[cross_point_1:cross_point_2 + 1],
             parent1[cross_point_2 + 1:]))
        offspring2 = np.concatenate(
            (parent2[:cross_point_1],
             parent1[cross_point_1:cross_point_2 + 1],
             parent2[cross_point_2 + 1:]))

        return self._repair_duplicates_offspring(offspring1, offspring2, parent1, cross_point_1, cross_point_2)

    def _repair_duplicates_offspring(self, offspring1: np.ndarray, offspring2: np.ndarray,
                                     parent1: np.ndarray, point1: int, point2: int) -> Tuple[np.ndarray, np.ndarray]:
        """Repair duplicate cities in offspring chromosomes after crossover operation."""
        crossover_indices = np.arange(point1, point2 + 1)
        common_values = np.intersect1d(offspring1, offspring2, return_indices=True)[1:]

        individual_ix_range = np.arange(len(parent1))
        to_change_1 = np.setdiff1d(np.setdiff1d(individual_ix_range, common_values[0]), crossover_indices)
        to_change_2 = np.setdiff1d(np.setdiff1d(individual_ix_range, common_values[1]), crossover_indices)

        offspring1[to_change_1], offspring2[to_change_2] = offspring2[to_change_2], offspring1[to_change_1]
        return offspring1, offspring2

    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        """Swap mutation operator"""
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def _genetic_operators(self, population: np.ndarray, iteration: int) -> np.ndarray:
        """Apply crossover and mutation"""
        # Adaptive mutation probability
        if iteration in [self.params['max_iter'] // 3,
                         2 * self.params['max_iter'] // 3,
                         3 * self.params['max_iter'] // 4]:
            self.params['mutation_prob'] += 0.08

        # Crossover
        crossover_indices = np.where(np.random.rand(len(population)) <
                                     self.params['crossover_prob'])[0]
        if len(crossover_indices) % 2 != 0:
            crossover_indices = crossover_indices[:-1]

        new_population = []
        for i in range(0, len(crossover_indices), 2):
            parent1 = population[crossover_indices[i]]
            parent2 = population[crossover_indices[i + 1]]
            child1, child2 = self._crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # Mutation
        mutation_indices = np.where(np.random.rand(len(population)) <
                                    self.params['mutation_prob'])[0]
        for idx in mutation_indices:
            new_population.append(self._mutation(population[idx]))

        return np.array(new_population)

    def _succession(self, old_population: np.ndarray,
                    new_generation: np.ndarray) -> np.ndarray:
        """Create next generation with elitism"""
        elite_size = int(self.params['n_pop'] * self.params['elitist_succession_prct'])
        elite = old_population.head(elite_size).values[:, :-1]

        remaining_size = self.params['n_pop'] - elite_size
        if remaining_size > len(new_generation):
            raise ValueError("Cannot proceed to next generation due to number of genoms available to progress being "
                             "too small. Please raise n_pop, selection_size or elitist_selection_prct parameters.")
        selected_new = new_generation[np.random.choice(np.arange(new_generation.shape[0]),
                                                       remaining_size,
                                                       replace=False)]

        combined_pop = np.vstack([elite, selected_new])
        unique_pop = np.unique(combined_pop, axis=0)

        # Uzupełnianie brakujących osobników
        if unique_pop.shape[0] < self.params['n_pop']:
            n_needed = self.params['n_pop'] - unique_pop.shape[0]
            additional = new_generation[np.random.choice(len(new_generation), n_needed, replace=True)]
            unique_pop = np.vstack([unique_pop, additional])

        return unique_pop[:self.params['n_pop']]

    def _stopping_condition_met(self, population: pd.DataFrame) -> bool:
        """Check early stopping condition"""
        if self.params['satisfying_result'] is None:
            return False
        return population['fitness'].iloc[0] < self.params['satisfying_result']

    def _visualize_results(self, cities_coordinates: np.ndarray,
                           genoms_for_stats: np.ndarray,
                           fitness_values_stats: np.ndarray, iteration: int) -> None:
        """Visualize optimization results"""
        country = gpd.read_file(self.dh.data_dir / "Kraj.zip")
        fig, ax = plt.subplots(1, 2, figsize=(25, 10))
        fig.suptitle('Podsumowanie wykorzystania algorytmu genetycznego do rozwiązania problemu komiwojażera',
                     fontsize=30)
        fig.patch.set_facecolor('white')

        # left - plot kraju z miastami i trasami
        country.boundary.plot(ax=ax[0], edgecolor='black')
        ax[0].set_axis_off()
        ax[0].set_title(f'Najlepsze znalezione rozwiązanie dla {self.params["n_cities"]} największych miast w Polsce',
                        fontsize=18)

        colors = ['#000000', '#A9A9A9', '#D3D3D3']
        thickness = [3, 2, 1]
        visibility = [1, 0.7, 0.4]
        zorders = [3, 2, 1]
        lines = []
        for ix, row in enumerate(genoms_for_stats):
            for j in range(row.shape[0] - 1):
                point1 = cities_coordinates.iloc[row[j]]
                point2 = cities_coordinates.iloc[row[j + 1]]
                line, = ax[0].plot([point1.iloc[0].x, point2.iloc[0].x],
                                   [point1.iloc[0].y, point2.iloc[0].y],
                                   color=colors[ix],
                                   linewidth=thickness[ix],
                                   alpha=visibility[ix],
                                   zorder=zorders[ix])
            lines.append(line)

        ax[0].legend(lines, [f'Najlepsza droga po {iteration} iteracjach',
                             f'Najlepsza droga po {iteration // 2} iteracjach',
                             f'Najlepsza droga po {iteration // 4} iteracjach'],
                     loc='lower left')
        cities_coordinates.plot(ax=ax[0], markersize=130, color='red', edgecolor='black', zorder=10)

        # right - plot optymalizacji
        bests, prct_75, medians = fitness_values_stats[:, 0], fitness_values_stats[:, 1], fitness_values_stats[:, 2]
        ax[1].plot(bests, color='green', label='Najlepszy osobnik', marker='o', markersize=5, zorder=10)
        ax[1].plot(prct_75, color='lightgreen', label='75 procentyl', marker='o', markersize=5, zorder=5)
        ax[1].plot(medians, color='blue', label='Mediana', marker='o', markersize=5)
        ax[1].legend()
        ax[1].set_xlabel("Pokolenia", fontsize=12)
        ax[1].set_ylabel("Wartości funkcji przystosowania", fontsize=12)

        min_index = bests.shape[0] - 1
        min_value = bests[min_index]
        ax[1].annotate('Najmniejsza wartość: {}'.format(min_value.astype(int)), xy=(min_index, min_value),
                       xytext=(min_index + 5, min_value + 10),
                       arrowprops=dict(facecolor='black', arrowstyle='->'),
                       fontsize=10)

        plt.show()

    def optimize(self) -> None:
        """Main optimization loop"""
        distances, cities_coords = self.dh.load_geospatial_data()
        population = self._evaluate_population(self._generate_initial_population(), distances)

        fitness_stats = np.empty((0, 3)).astype(int)
        routes_plot = np.empty((0, self.params["n_cities"])).astype(int)

        selection = self._selection_roulette if self.params['selection_method'] == 'roulette' \
            else self._selection_ranking
        for iteration in range(self.params['max_iter']):
            selected = selection(population)

            # Genetic operations called 2 times
            # to make sure there's enough individuals in succession
            new_generation = np.vstack(
                (self._genetic_operators(selected.values[:, :-1], iteration),
                 self._genetic_operators(selected.values[:, :-1], iteration)))

            # Succession
            population = self._evaluate_population(
                self._succession(population, new_generation),
                distances
            )


            if iteration in [self.params['max_iter'] // 4, self.params['max_iter'] // 2, self.params['max_iter'] - 1]:
                routes_plot = np.vstack([routes_plot, population.iloc[0].values[:-1]])
            current_stats = np.array([
                population['fitness'].min(),
                population['fitness'].quantile(0.75),
                population['fitness'].median()
            ])
            fitness_stats = np.vstack((fitness_stats, current_stats))

            if self._stopping_condition_met(population):
                break

        self._visualize_results(cities_coords, routes_plot, fitness_stats, iteration + 1)


if __name__ == "__main__":
    dh = DataHandler()
    optimizer = TSPOptimizer(dh)
    optimizer.optimize()
