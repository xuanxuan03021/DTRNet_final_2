import numpy as np
import torch
from scipy.stats import pearsonr, rankdata
import itertools
from scipy.special import comb
import pandas as pd
from new_version import feature_importance_CXAI
from perturbation_methods import NormalPerturbation
from Generate_Dataset import generate_new_dataset


# reference form XAI https://github.com/AI4LIFE-GROUP/OpenXAI/blob/f9fd545bce73657521122c879a3fa70b1b8e2c6b/openxai/evaluator.p
class Evaluator():
    """ Metrics to evaluate an explanation method.
    """

    def __init__(self, inputs, W, con, cat, bi, type_y, model, explaination, y_pred, ground_truth_explaination,
                 perturbation,
                 feature_types, explainer=None, explainer_name="CXAI", perturb_max_distance=0.4, top_k=2,
                 num_perturbations=1000, p_norm=2):
        self.inputs = inputs
        self.labels = W
        self.con = con
        self.cat = cat
        self.bi = bi
        self.type_y = type_y
        self.model = model
        # self.explainer = feature_importance_CXAI
        self.explainer_name = explainer_name
        self.explainer = explainer
        self.gt_feature_importances = ground_truth_explaination
        self.explanation_x_f = explaination
        self.y_pred = y_pred
        self.perturbation = perturbation
        self.perturb_max_distance = perturb_max_distance
        self.feature_types = feature_types
        self.k = top_k
        self.num_perturbations = num_perturbations
        self.p_norm = p_norm

    def pairwise_comp(self):
        '''
        inputs
        attrA: np.array, n x p
        attrB: np.array, n x p
        outputs:
        pairwise_distr: 1D numpy array (dimensions=(n,)) of pairwise comparison agreement for each data point
        pairwise_avg: mean of pairwise_distr
        '''

        attrA = self.gt_feature_importances
        attrA = list(map(float, attrA))

        attrB = self.explanation_x_f
        attrB = list(map(float, attrB))

        # n_datapoints = attrA.shape[0]
        n_feat = len(attrA)

        # rank of all features --> manually calculate rankings (instead of using 0, 1, ..., k ranking based on argsort output) to account for ties
        # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
        all_feat_ranksA = rankdata(-np.abs(attrA), method='dense')
        all_feat_ranksB = rankdata(-np.abs(attrB), method='dense')

        # count # of pairs of features with same relative ranking
        # feat_pairs_w_same_rel_rankings = np.zeros(n_datapoints)
        feat_pairs_w_same_rel_rankings = 0
        for feat1, feat2 in itertools.combinations_with_replacement(range(n_feat), 2):
            if feat1 != feat2:
                rel_rankingA = all_feat_ranksA[feat1] < all_feat_ranksA[feat2]
                rel_rankingB = all_feat_ranksB[feat1] < all_feat_ranksB[feat2]
                feat_pairs_w_same_rel_rankings += rel_rankingA == rel_rankingB

        pairwise_distr = feat_pairs_w_same_rel_rankings / comb(n_feat, 2)

        return pairwise_distr, np.mean(pairwise_distr)

    def _arr(self, x) -> np.ndarray:
        """ Converts x to a numpy array.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    def rankcorr(self):
        '''
        attrA: np.array, n x p
        attrB: np.array, n x p
        '''
        attrA = self.gt_feature_importances
        attrA = list(map(float, attrA))

        attrB = self.explanation_x_f
        attrB = list(map(float, attrB))

        # rank features (accounting for ties)
        # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
        all_feat_ranksA = rankdata(-np.abs(attrA), method='dense')
        all_feat_ranksB = rankdata(-np.abs(attrB), method='dense')

        # Calculate correlation on ranks (iterate through rows: https://stackoverflow.com/questions/44947030/how-to-get-scipy-stats-spearmanra-b-compute-correlation-only-between-variable)
        rho, _ = pearsonr(all_feat_ranksA, all_feat_ranksB)

        # return metric's distribution and average
        return np.array(rho), np.mean(rho)

    def evaluate(self, metric: str, i=0, N_treatment_grid=100, min_bin_points=20):
        """Explanation evaluation of a given metric
        # """
        # if not hasattr(self.model, 'return_ground_truth_importance') and metric in ['PRA', 'RC', 'FA', 'RA', 'SA', 'SRA']:
        #     raise ValueError("This chosen metric is incompatible with non-linear models.")

        # Pairwise rank agreement

        if metric == 'PRA':
            scores, average_score = self.pairwise_comp()
            return scores, average_score
        # Rank correlation
        elif metric == 'RC':
            scores, average_score = self.rankcorr()
            return scores, average_score
        # Feature Agreement
        elif metric == 'FA':
            scores, average_score = self.agreement_fraction(metric='overlap')
            return scores, average_score
        # Rank Agreement
        elif metric == 'RA':
            scores, average_score = self.agreement_fraction(metric='rank')
            return scores, average_score
        # Sign Agreement
        elif metric == 'SA':
            scores, average_score = self.agreement_fraction(metric='sign')
            return scores, average_score
        # Signed Rank Agreement
        elif metric == 'SRA':
            scores, average_score = self.agreement_fraction(metric='ranksign')
            return scores, average_score
        # Prediction Gap on Important Features
        elif metric == 'PGI':
            scores = self.eval_pred_faithfulness(invert=True)
            return scores
        # Prediction Gap on Unimportant Features
        elif metric == 'PGU':
            scores = self.eval_pred_faithfulness(invert=False)
            return scores
        # Relative Input Stability
        elif metric == 'RIS':
            scores = self.eval_relative_stability(num_perturbations=self.num_perturbations, i=i,
                                                  N_treatment_grid=N_treatment_grid, min_bin_points=min_bin_points)
            return scores
        else:
            raise NotImplementedError("This metric is not implemented in this version.")

    def agreement_fraction(self, metric=None):

        attrA = self.gt_feature_importances
        attrA = list(map(float, attrA))

        attrB = self.explanation_x_f
        attrB = list(map(float, attrB))

        print("Ground Truth", attrA)
        print("Real Attribute", attrB)

        k = self.k

        if metric is None:
            raise NotImplementedError(
                "Please make sure that have chosen one of the following metrics: {ranksign, rank, overlap, sign}.")
        else:
            metric_type = metric

        # id of top-k features
        topk_idA = np.argsort(-np.abs(attrA))[0:k]
        topk_idB = np.argsort(-np.abs(attrB))[0:k]

        #         print("ID of the topk ground truth", topk_idA)
        #         print("ID of the topk explanation", topk_idB)

        # rank of top-k features --> manually calculate rankings (instead of using 0, 1, ..., k ranking based on argsort output) to account for ties
        all_feat_ranksA = rankdata(-np.abs(attrA),
                                   method='dense')  # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
        all_feat_ranksB = rankdata(-np.abs(attrB), method='dense')
        print(all_feat_ranksA)
        print(all_feat_ranksB)
        topk_ranksA = all_feat_ranksA[topk_idA]
        topk_ranksB = all_feat_ranksB[topk_idB]

        #         print("Rank of the topk ground truth", topk_ranksA)
        #         print("Rank of the topk explanation", topk_ranksB)

        # sign of top-k features
        topk_signA = np.sign(attrA)[topk_idA]
        topk_signB = np.sign(attrB)[topk_idB]

        #         print("The sign of topk ground truth is", topk_signA)
        #         print("The sign of topk explanation is", topk_signB)

        # overlap agreement = (# topk features in common)/k
        if metric_type == 'overlap':
            topk_setsA = set(topk_idA)
            topk_setsB = set(topk_idB)
            # check if: same id
            metric_distr = np.array(len(topk_setsA.intersection(topk_setsB))) / k

        # rank agreement
        elif metric_type == 'rank':
            topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id
            topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
            topk_ranksA_df = pd.DataFrame(topk_ranksA).applymap(str)  # rank (accounting for ties)
            topk_ranksB_df = pd.DataFrame(topk_ranksB).applymap(str)

            # check if: same id + rank
            topk_id_ranksA_df = ('feat' + topk_idA_df) + ('rank' + topk_ranksA_df)
            topk_id_ranksB_df = ('feat' + topk_idB_df) + ('rank' + topk_ranksB_df)
            #             print("rank agreement", topk_id_ranksA_df)
            #             print("rank agreement", topk_id_ranksB_df)

            metric_distr = (topk_id_ranksA_df == topk_id_ranksB_df).sum().to_numpy() / k

        # sign agreement
        elif metric_type == 'sign':
            topk_idA_df = pd.DataFrame(topk_idA).applymap(
                str)  # id (contains rank info --> order of features in columns)
            topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
            topk_signA_df = pd.DataFrame(topk_signA).applymap(str)  # sign
            topk_signB_df = pd.DataFrame(topk_signB).applymap(str)

            # check if: same id + sign
            topk_id_signA_df = ('feat' + topk_idA_df) + (
                    'sign' + topk_signA_df)  # id + sign (contains rank info --> order of features in columns)
            topk_id_signB_df = ('feat' + topk_idB_df) + ('sign' + topk_signB_df)
            topk_id_signA_sets = set(
                topk_id_signA_df.values.T.tolist()[0])  # id + sign (remove order info --> by converting to sets)
            topk_id_signB_sets = set(topk_id_signB_df.values.T.tolist()[0])

            # print(type(topk_id_signA_df.values), list(topk_id_signA_df.values))
            # print(topk_id_signB_df)
            # print(list(topk_id_signA_df))
            # print(list(topk_id_signB_df))
            #             print("sign agreement", topk_id_signA_sets)
            #             print("sign agreement", topk_id_signB_sets)

            metric_distr = np.array(len(topk_id_signA_sets.intersection(topk_id_signB_sets))) / k

        # rank and sign agreement
        elif metric_type == 'ranksign':
            topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id
            topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
            topk_ranksA_df = pd.DataFrame(topk_ranksA).applymap(str)  # rank (accounting for ties)
            topk_ranksB_df = pd.DataFrame(topk_ranksB).applymap(str)
            topk_signA_df = pd.DataFrame(topk_signA).applymap(str)  # sign
            topk_signB_df = pd.DataFrame(topk_signB).applymap(str)

            # check if: same id + rank + sign
            topk_id_ranks_signA_df = ('feat' + topk_idA_df) + ('rank' + topk_ranksA_df) + ('sign' + topk_signA_df)
            topk_id_ranks_signB_df = ('feat' + topk_idB_df) + ('rank' + topk_ranksB_df) + ('sign' + topk_signB_df)
            #             print("sr agreement", topk_id_ranks_signA_df)
            #             print("sr agreement", topk_id_ranks_signB_df)

            metric_distr = (topk_id_ranks_signA_df == topk_id_ranks_signB_df).sum().to_numpy() / k

        else:
            raise NotImplementedError(
                "Please make sure that have chosen one of the following metrics: {ranksign, rank, overlap, sign}.")

        return metric_distr, np.mean(metric_distr)

    def generate_mask(self, explanation, top_k):
        mask_indices = torch.topk(torch.from_numpy(explanation), top_k).indices
        print("mask indices", mask_indices)
        mask = torch.zeros(torch.from_numpy(explanation).shape) > 10
        for i in mask_indices:
            mask[i] = True
        # print("mask", mask)
        return mask

    def _compute_Lp_norm_diff(self, vec1, vec2, normalize_to_relative_change: bool = True, eps: np.float = 0.001):
        """ Returns the Lp norm of the difference between vec1 and vec2.
        Args:
            normalize_by_vec1: when true, normalizes the difference between vec1 and vec2 by vec1
        """

        # arrays can be flattened, so long as ordering is preserved
        flat_diff = self._arr(vec1).flatten() - self._arr(vec2).flatten()

        if normalize_to_relative_change:
            vec1_arr = self._arr(vec1.flatten())
            vec1_arr = np.clip(vec1_arr, eps, None)
            flat_diff = np.divide(flat_diff, vec1_arr, where=vec1_arr != 0)

        return np.linalg.norm(flat_diff, ord=self.p_norm)

    def eval_pred_faithfulness(self, invert: bool = False):
        """ Approximates the expected local faithfulness of the explanation
            in a neighborhood around input x.
        Args:
            num_perturbations: number of perturbations used for Monte Carlo expectation estimate
        """
        # self._parse_and_check_input(eval_metric='eval_pred_faithfulness')

        self.top_k_mask = self.generate_mask(self.explanation_x_f, self.k)
        if invert:
            self.top_k_mask = torch.logical_not(self.top_k_mask)
        print(self.top_k_mask)

        # get perturbations of instance x
        x_perturbed = self.perturbation.get_perturbed_inputs(original_sample=self.inputs,
                                                             feature_mask=self.top_k_mask,
                                                             num_samples=self.inputs.shape[0],
                                                             max_distance=self.perturb_max_distance,
                                                             feature_metadata=self.feature_types)

        # Average the expected absolute difference.
        y = self._arr(self.model(self.inputs))
        y_perturbed = self._arr(self.model(x_perturbed.cuda()))
        # print(y.shape)
        # print(y_perturbed.shape)
        # print("origin",np.max(np.abs(y - y_perturbed), axis=0))

        # for categorical data sum the difference across all outputs
        #    print("sum all category", np.sum(np.abs(y - y_perturbed), axis=1).shape)

        return np.mean(np.sum(np.abs(y - y_perturbed), axis=1))

    def eval_relative_stability(self,
                                x_prime_samples=None,
                                num_perturbations: int = 1000, i=0, N_treatment_grid=100, min_bin_points=20):
        """ Approximates the maximum L-p distance between explanations in a neighborhood around
            input x.
        Args:
            rep_denominator_flag: when true, normalizes the stability metric by the L-p distance
                between representations (instead of features).
        """
        exp_at_input = self.explanation_x_f
        # self._parse_and_check_input(eval_metric='eval_relative_stability')
        self.top_k_mask = self.generate_mask(self.explanation_x_f, self.k)

        stability_ratios = []
        exp_diffs = []

        # get perturbations of instance x, and for each perturbed instance compute an explanation
        if x_prime_samples is None:
            # Perturb input
            x_prime_samples = self.perturbation.get_perturbed_inputs(original_sample=self.inputs,
                                                                     feature_mask=self.top_k_mask,
                                                                     num_samples=self.inputs.shape[0],
                                                                     max_distance=self.perturb_max_distance,
                                                                     feature_metadata=self.feature_types)

            # Take the first num_perturbations points that have the same predicted class label
            if self.type_y == 2:
                y_prime_preds = self.model(x_prime_samples.cuda()) > 0.5
            #                 print(y_prime_preds)
            #                 print(y_prime_preds.shape)
            elif self.type_y == 3:
                y_prime_preds = torch.argmax(self.model(x_prime_samples.cuda()), dim=1)
            #                 print(y_prime_preds)
            #                 print(y_prime_preds.shape)

            #             print(self.y_pred.shape)

            #             print(y_prime_preds ==  self.y_pred)

            ind_same_class = (y_prime_preds == self.y_pred).squeeze().nonzero()[: num_perturbations].squeeze()
            x_prime_samples = torch.index_select(input=x_prime_samples.cuda(),
                                                 dim=0,
                                                 index=ind_same_class)

            x_new = torch.index_select(input=self.inputs,
                                       dim=0,
                                       index=ind_same_class)
            y_prime_preds = torch.argmax(self.model(x_prime_samples.cuda()), dim=1).unsqueeze(dim=1)
            print("Selecting... points", y_prime_preds.shape)

            if self.type_y == 2:
                if self.explainer_name == "CXAI":
                    #  calculate the explanation
                    exp_prime_samples = feature_importance_CXAI(x_prime_samples, y_prime_preds.float(), self.labels,
                                                                self.con,
                                                                self.cat, self.bi, self.type_y, len(ind_same_class))
                elif self.explainer_name == "ALE":
                    print("Use ALE to explain")
                    ALE_exp = self.explainer.explain(X=x_prime_samples.cpu().detach().numpy(),
                                                     min_bin_points=min_bin_points)
                    exp_prime_samples = self.feature_importance_baseline(ALE_exp.ale_values, self.type_y)

                elif self.explainer_name == "PD":
                    print("Use Partial Dependence to explain")
                    # compute explanations
                    PD_exp = self.explainer.explain(X=x_prime_samples.cpu().detach().numpy(),
                                                    grid_resolution=N_treatment_grid, kind='average')
                    exp_prime_samples = self.feature_importance_baseline(PD_exp.pd_values, self.type_y)

                elif self.explainer_name == "Shapley":
                    print("Use Shapley to explain")
                    self.explainer.fit(x_prime_samples.cpu().detach().numpy()[:100])
                    Shap_explanation = self.explainer.explain(x_prime_samples.cpu().detach().numpy())
                    exp_prime_samples = np.mean(abs(Shap_explanation.data["shap_values"][0]), axis=0)

            if self.type_y == 3:
                if self.explainer_name == "CXAI":

                    exp_prime_samples_all = feature_importance_CXAI(x_prime_samples, y_prime_preds.squeeze(),
                                                                    self.labels, self.con,
                                                                    self.cat, self.bi, self.type_y, len(ind_same_class))
                    step = len(torch.unique(y_prime_preds))
                    # print("step=",step)
                    exp_prime_samples = exp_prime_samples_all[i::step]
                    # print("value=",exp_prime_samples)

                elif self.explainer_name == "ALE":
                    print("Use ALE to explain")
                    ALE_exp = self.explainer.explain(X=x_prime_samples.cpu().detach().numpy(),
                                                     min_bin_points=min_bin_points)
                    exp_prime_samples = self.feature_importance_baseline([i.T for i in ALE_exp.ale_values],
                                                                         type_y=self.type_y)[:, i]
                    print(exp_prime_samples)
                elif self.explainer_name == "PD":
                    print("Use Partial Dependence to explain")
                    PD_exp = self.explainer.explain(X=x_prime_samples.cpu().detach().numpy(),
                                                    grid_resolution=N_treatment_grid, kind='average')
                    exp_prime_samples = self.feature_importance_baseline(PD_exp.pd_values, type_y=self.type_y)[:, i]
                    print(exp_prime_samples)
                elif self.explainer_name == "Shapley":
                    print("Use Shapley to explain")
                    self.explainer.fit(x_prime_samples.cpu().detach().numpy()[:100])
                    Shap_explanation = self.explainer.explain(x_prime_samples.cpu().detach().numpy())
                    Shap_explanation_global = np.mean(abs(Shap_explanation.data["shap_values"]), axis=1).T
                    exp_prime_samples = Shap_explanation_global[:, i]
                    print(exp_prime_samples)
            # for predictions per perturbation
            explanation_diff = self._compute_Lp_norm_diff(exp_at_input,
                                                          exp_prime_samples,
                                                          normalize_to_relative_change=True)

            feature_difference = self._compute_Lp_norm_diff(x_new, x_prime_samples)
            stability_measure = np.divide(explanation_diff, feature_difference)

        return stability_measure  # , stability_ratios, rep_diffs, x_diffs, exp_diffs, ind_max

    # #
    #     def feature_importance_baseline(self,initial_exp):
    #         explanation=[]
    #         for i in initial_exp:
    #             min_effect=np.min(i)
    #             feature_effect_diff=i-min_effect
    #             feature_importance=np.mean(feature_effect_diff)
    #             explanation.append(feature_importance)
    #         return explanation
    def feature_importance_baseline(self, initial_exp, type_y=1):
        explanation = []
        if type_y == 3:
            for i in initial_exp:
                explanation_cate = []
                for cat in i:
                    min_effect = np.min(cat)
                    feature_effect_diff = cat - min_effect
                    feature_importance = np.mean(feature_effect_diff)
                    explanation_cate.append(feature_importance)
                explanation.append(explanation_cate)
            return np.array(explanation)
        else:
            for i in initial_exp:
                min_effect = np.min(i)
                feature_effect_diff = i - min_effect
                feature_importance = np.mean(feature_effect_diff)
                explanation.append(feature_importance)
            return explanation


# # Perturbation class parameters
# perturbation_mean = 0.0
# perturbation_std = 0.05
# perturbation_flip_percentage = 0.3
#
# # if use real_data, change the data name to real data
# data_name = 'synthetic'
# # c represents continuous
# # d represents discrete
#
# feature_types = []
# if data_name == 'synthetic':
#     feature_types = ['c', 'c', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd']
#
# # change the data type for real data!
# if data_name == 'real_data':
#     feature_types = ['c', 'd', 'c', 'c', 'd', 'd', 'd']
#
# perturbation = NormalPerturbation("tabular",
#                                   mean=perturbation_mean,
#                                   std_dev=perturbation_std,
#                                   flip_percentage=perturbation_flip_percentage)
# Evaluator_g = Evaluator(inputs=None, labels=None, model=None, explaination=[1, 2, -3, 1, 2, 0, 1, 0.5], y_pred=None,
#                         ground_truth_explaination=[1, 0, 3, 1, -2, 0, 1, 0],
#                         perturbation=perturbation,
#                         feature_types=feature_types,
#                         top_k=2, )
# FA_score = Evaluator_g.evaluate('FA')
# print("Feature_agreement is", FA_score)
# SA_score = Evaluator_g.evaluate('SA')
# print("Sign_agreement is", SA_score)
# RA_score = Evaluator_g.evaluate('RA')
# print("rank agreement is", RA_score)
# SRA_score = Evaluator_g.evaluate('SRA')
# print("signed rank agreement is", SRA_score)
# RC_score = Evaluator_g.evaluate('RC')
# print("corraltion agreement is", RC_score)
# PW_score = Evaluator_g.evaluate('PRA')
# print("pairwise agreement is", PW_score)
#
#
# class Layer():
#     def __init__(self, IN, OUT, activation=None):
#         self.IN = IN
#         self.OUT = OUT
#         self.w = torch.randn(IN, OUT, device=device, requires_grad=True)
#         self.b = torch.randn(1, OUT, device=device, requires_grad=True)
#         self.activation = activation
#
#     def forward(self, x):
#         if self.activation is None:
#             y = x.mm(self.w) + self.b
#         else:
#             y = self.activation(x.mm(self.w) + self.b)
#         return y
#
#     def reset(self):
#         self.w.data = torch.randn(self.IN, self.OUT, device=device, requires_grad=False)
#         self.b.data = torch.randn(1, self.OUT, device=device, requires_grad=False)
#
#
# def initialize_network(D_in, H, D_out, type_y):
#     # type_y=1 continous
#     # type_y=2 binary
#     # type_y=3 categorical
#
#     Layers = []
#     Layer_in = Layer(D_in, H, torch.tanh)
#     Layers.append(Layer_in)
#     for l in range(Layer_num - 2):
#         Layers.append(Layer(H, H, torch.tanh))
#     if type_y == 1 or type_y == 3:
#         Layer_out = Layer(H, D_out)
#     elif type_y == 2:
#         Layer_out = Layer(H, D_out, torch.sigmoid)
#     Layers.append(Layer_out)
#     return Layers
#
#
# def layer_forward(Layers, x):
#     result = x
#     for layer in Layers:
#         result = layer.forward(result)
#     return result
#
#
# N, D_in, H, D_out = 2000, 10, 5, 1
# # reg_factors = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2]
#
#
# # some big reg_factors will lead many node to 0 and the continuous variable tend to select big regressor
# # big reg_factors->binary y no diversity (binary also want to select big reg)
# reg_factors = [0.001, 0.005, 0.01, 0.03, 0.05, ]
#
# N_runs = 1
# N_splits = 3
# epoch = 2
# eta = 1e-2
#
# # Add number
# Layer_num = 4
# top_feature = 2
# num_gaussians = 20
# N_treatment_grid = 100
# batch_size = 32
# shuffle = True
# loss_batch = 50
#
# X, y, W, con, cat, bi, type_y = generate_new_dataset()
# if type_y == 1 or type_y == 2:
#     out_dim = 1
# elif type_y == 3:
#     print(torch.unique(y))
#     out_dim = len(torch.unique(y))
#
# Layers = initialize_network(D_in, H, out_dim, type_y)
# y_pred = layer_forward(Layers, X.cuda())
#
# Evaluator_p = Evaluator(inputs=X.cuda(), labels=None, model=Layers,
#                         explaination=np.array([0, 0, 1, 2, -3, 1, 2, 0, 1, 0.5]),
#                         y_pred=None,
#                         ground_truth_explaination=np.array([0, 0, 1, 0, 3, 1, -2, 0, 1, 0]),
#                         perturbation=perturbation,
#                         feature_types=feature_types,
#                         top_k=2, )
# PGI_score = Evaluator_p.evaluate('PGI')
# print("Predictive_agreement is", PGI_score)
#
# PGU_score = Evaluator_p.evaluate('PGU')
# print("Predictive_agreement_UNIMPORTANT is", PGU_score)
