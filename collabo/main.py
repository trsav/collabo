from utils import *


class Collaboration:
    def __init__(
        self, data_location=None, bounds=None, aquisition_function=None, fun=None
    ):
        try:
            self.data_location = data_location  # location of data
        except:
            warnings.warn(
                "No data location provided. Need this to either load existing data, or as a place to save new data. Nothing will be saved to or loaded from disk."
            )

        self.bounds = np.array(bounds)  # bounds of the problem
        if fun:
            self.fun = fun
        else:
            warnings.warn(
                "No function provided, you will be prompted to input objective values..."
            )
        self.aq = aquisition_function  # acquisition function
        self.d = len(bounds)  # dimensions
        return

    def load_data(self):
        try:
            with open(self.data_location) as f:
                self.data = json.load(f)
            f.close()
            self.data_location = self.data_location
        except:
            raise FileNotFoundError(f"File at {self.data_location} not found")

    def save_data(self):
        print("Saving data at ", self.data_location)
        with open(self.data_location, "w") as f:
            json.dump(self.data, f)
        f.close()
        return

    def design_experiments(self, n_experiments, fixed_solutions=None):
        # n_experiments is the total number of experiments to distrubute (including fixed solutions)
        try:
            _ = self.data
            warnings.warn("Data already exists within the object and will be replaced")
        except:
            pass

        if fixed_solutions:
            for solution in fixed_solutions:
                if len(solution) != self.d:
                    raise ValueError(
                        f"Specified solution {solution} does not have correct dimensions. (Expected {self.d}, got {len(solution)})"
                    )
            designed_solutions = distribute_solutions(
                fixed_solutions, self.bounds, n_experiments - len(fixed_solutions)
            )
        else:
            # latin hypercube sampling
            designed_solutions = []
            for i in range(0, self.d):
                s = np.linspace(self.bounds[i][0], self.bounds[i][1], n_experiments)
                np.random.shuffle(s)
                designed_solutions.append(s)
            designed_solutions = np.array(designed_solutions).T.tolist()

        data = {"experiments": []}
        for solution in designed_solutions:
            data["experiments"].append({"solution": solution})

        self.data = data
        self.save_data()

        for experiment in self.data["experiments"]:
            solution = experiment["solution"]
            try:
                _ = self.fun
                objective_value = self.fun(solution)
            except:
                warnings.warn(
                    "Either no function was provided or the function is not callable."
                )

                while True:
                    objective_value = input(
                        f"Please input the objective value for the solution {solution}: "
                    )
                    try:
                        objective_value = float(objective_value)
                        break
                    except:
                        print(f"Objective value {objective_value} is not a number.")

            experiment["objective"] = objective_value

        self.data = data
        self.save_data()
        return

    @staticmethod
    def parse_data(data):
        # reads data from datastore into convenient arrays
        solutions = []
        objectives = []
        # iterate over data
        for experiment in data["experiments"]:
            solutions += [list(experiment["solution"])]
            objectives += [experiment["objective"]]

        solutions = np.array(solutions)
        objectives = np.array(objectives).reshape(-1, 1)

        return solutions, objectives

    def propose_solutions(self, n_solutions):
        inputs, outputs = self.parse_data(self.data)
        mean_outputs = np.mean(outputs)
        std_outputs = np.std(outputs)
        if std_outputs != 0:
            outputs = (outputs - mean_outputs) / std_outputs

        mean_inputs = np.mean(inputs, axis=0)
        std_inputs = np.std(inputs, axis=0)
        inputs = (inputs - mean_inputs) / std_inputs

        normalized_bounds = []
        for i in range(len(self.bounds)):
            lb = float((self.bounds[i][0] - mean_inputs[i]) / std_inputs[i])
            ub = float((self.bounds[i][1] - mean_inputs[i]) / std_inputs[i])
            normalized_bounds.append([lb, ub])

        gp = train_gp(inputs, outputs, noise=False)

        def aq(x, args):
            f_best = args[0]
            x = np.array([x])
            x = torch.tensor(x)
            dist = gp(x)
            m = dist.mean
            sigma = torch.sqrt(dist.variance)
            diff = m - f_best
            p_z = torch.distributions.Normal(0, 1)
            Z = diff / sigma
            expected_improvement = diff * p_z.cdf(Z) + sigma * torch.exp(
                p_z.log_prob(Z)
            )
            # convert to numpy
            expected_improvement = expected_improvement.detach().numpy()
            return -expected_improvement

        evolutionary_upper_bounds = [b[1] for b in normalized_bounds]
        evolutionary_lower_bounds = [b[0] for b in normalized_bounds]

        termination = DefaultMultiObjectiveTermination(
            xtol=0.001, ftol=0.005, period=30, n_max_gen=10000, n_max_evals=10000
        )

        algorithm = NSGA2(
            pop_size=50,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        class MO_aq(Problem):
            def __init__(self):
                super().__init__(
                    n_var=len(evolutionary_lower_bounds),
                    n_obj=1,
                    n_ieq_constr=0,
                    xl=np.array(evolutionary_lower_bounds),
                    xu=np.array(evolutionary_upper_bounds),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = [aq(x, args=(max(outputs)))]

        problem = MO_aq()
        res = minimize_mo(
            problem, algorithm, termination, seed=1, save_history=False, verbose=False
        )
        optimal_solution = res.X

        mo_upper_bounds = np.array(
            [b[1] for b in normalized_bounds] * (n_solutions - 1)
        )
        mo_lower_bounds = np.array(
            [b[0] for b in normalized_bounds] * (n_solutions - 1)
        )

        termination = DefaultMultiObjectiveTermination(
            xtol=0.001, ftol=0.005, period=30, n_max_gen=10000, n_max_evals=100000
        )

        algorithm = NSGA2(
            pop_size=50,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        class MO_aq(Problem):
            def __init__(self):
                super().__init__(
                    n_var=len(mo_lower_bounds),
                    n_obj=2,
                    n_ieq_constr=0,
                    xl=np.array(mo_lower_bounds),
                    xu=np.array(mo_upper_bounds),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                x_sols = np.array(np.split(x, n_solutions - 1, axis=1))
                d = x_sols.shape[0]
                aq_list = np.sum([aq(x_i, (max(outputs))) for x_i in x_sols], axis=0)[0]
                if d == 1:
                    app = np.array(
                        [[optimal_solution for i in range(len(x_sols[0, :, 0]))]]
                    ).T
                else:
                    app = np.array(
                        [[optimal_solution for i in range(len(x_sols[0, :, 0]))]]
                    )

                x_sols = np.append(x_sols, app, axis=0)

                K_list = []
                for i in range(len(x_sols[0])):
                    set = np.array([np.array(x_sols[j][i]) for j in range(n_solutions)])
                    kernel = gp.covar_module
                    K = kernel(torch.tensor(set)).detach().numpy()
                    K = np.linalg.det(K)
                    K_list.append(K)
                K_list = np.array(K_list)

                out["F"] = [aq_list, -K_list]

        problem = MO_aq()
        res = minimize_mo(
            problem, algorithm, termination, seed=1, save_history=True, verbose=True
        )

        F = res.F
        X = res.X

        AQ = F[:, 0]
        D = F[:, 1]

        aq_norm = (AQ - np.min(AQ)) / (np.max(AQ) - np.min(AQ))
        d_norm = (D - np.min(D)) / (np.max(D) - np.min(D))
        distances = np.sqrt(aq_norm**2 + d_norm**2)

        knee_solutions = np.append(X[np.argmin(distances)], optimal_solution)

        alternate_solutions = list(np.split(knee_solutions, n_solutions))
        alternate_solutions = [
            alternate_solutions[i].tolist() for i in range(n_solutions)
        ]

        for i in range(n_solutions):
            alternate_solutions[i] = list(
                (np.array(alternate_solutions[i]) * std_inputs) + mean_inputs
            )

        self.choices = alternate_solutions
        return alternate_solutions

    def view_choices(self):
        try:
            _ = self.choices
        except:
            raise ValueError(
                "No choices have been proposed. Please call propose_solutions() first."
            )
        for i, choice in enumerate(self.choices):
            print(f"Choice {i+1}: {choice}")
        return

    def make_choice(self, choice):
        try:
            _ = self.choices
        except:
            raise ValueError(
                "No choices have been proposed. Please call propose_solutions() first."
            )

        choice = choice - 1
        solution = self.choices[choice]
        experiment = {"solution": solution}

        try:
            _ = self.fun
            objective_value = self.fun(solution)
        except:
            warnings.warn(
                "Either no function was provided or the function is not callable."
            )

            while True:
                objective_value = input(
                    f"Please input the objective value for the solution {solution}: "
                )
                try:
                    objective_value = float(objective_value)
                    break
                except:
                    print(f"Objective value {objective_value} is not a number.")

        experiment["objective"] = objective_value

        self.data["experiments"].append(experiment)
        self.save_data()
        self.choices = None
        return


# f = lambda x: x[0] + x[1] + x[2] + x[3]
# collab = Collaboration('./data.json',bounds=[(0,5),(0,5),(0,5),(0,5)],fun=f)
# collab.design_experiments(4)
# # collab.load_data()
# collab.propose_solutions(3)
# collab.view_choices()
# collab.make_choice(0)


# collab.design_experiments(10,fixed_solutions=[[1,1,1,1],[2,2,2,2],[3,3,3,3]])
# collab.load_data()

# collab.load_data('data.json')
