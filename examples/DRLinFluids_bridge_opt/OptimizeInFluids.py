from bayes_opt import BayesianOptimization
from blackBoxFunction import black_box_function





#black_box_function=black_box_function()


# Bounded region of parameter space
pbounds = {'omega': (40.0, 160.0), }


optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)


optimizer.maximize(
    init_points=3,
    n_iter=150,
)



#
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
# print(enumerate(optimizer.res))

