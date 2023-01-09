from synthetic_dataset import generate_gaussians

# [can choose from "test","train","all"]

data_name = "train"

gauss_params = {
    'n_samples': 2500,
    'dim': 20,
    'n_clusters': 10,
    'distance_to_center': 5,
    'test_size': 0.25,
    'upper_weight': 1,
    'lower_weight': -1,
    'seed': 564,
    'sigma': None,
    'sparsity': 0.5
}

if gauss_params is None:
    gauss_params = {
        'n_samples': 2500,
        'dim': 20,
        'n_clusters': 10,
        'distance_to_center': 5,
        'test_size': 0.25,
        'upper_weight': 1,
        'lower_weight': -1,
        'seed': 564,
        'sigma': None,
        'sparsity': 0.25
    }


dataset,probs,masks,weights,masked_weights,cluster_idx = generate_gaussians(gauss_params['n_samples'],
                                                                gauss_params['dim'],
                                                                gauss_params['n_clusters'],
                                                                gauss_params['distance_to_center'],
                                                                gauss_params['test_size'],
                                                                gauss_params['upper_weight'],
                                                                gauss_params['lower_weight'],
                                                                gauss_params['seed'],
                                                                gauss_params['sigma'],
                                                                gauss_params['sparsity']).dgp_vars(data_name=data_name)
print(dataset)
print("masks",masks[2490:2510,:])
print("weights",weights[2490:2510,:])
print("masked_weights",masked_weights[2490:2510,:])