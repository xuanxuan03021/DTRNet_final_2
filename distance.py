
from util import *
import torch
def calculate_disc(self, h_rep_norm, coef, FLAGS):
    t = self.t

    if FLAGS.use_p_correction:
        p_ipm = self.p_t
    else:
        p_ipm = 0.5

    if FLAGS.imb_fun == 'mmd2_rbf':
        imb_dist = mmd2_rbf(h_rep_norm, t, p_ipm, FLAGS.rbf_sigma)
        imb_error = coef * imb_dist
    elif FLAGS.imb_fun == 'mmd2_lin':
        imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
        imb_error = coef * mmd2_lin(h_rep_norm, t, p_ipm)
    elif FLAGS.imb_fun == 'mmd_rbf':
        imb_dist = abs(mmd2_rbf(h_rep_norm, t, p_ipm, FLAGS.rbf_sigma))
        imb_error = safe_sqrt(torch.square(coef) * imb_dist)
    elif FLAGS.imb_fun == 'mmd_lin':
        imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
        imb_error = safe_sqrt(torch.square(coef) * imb_dist)
    elif FLAGS.imb_fun == 'wass':
        imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                        sq=False, backpropT=FLAGS.wass_bpt)
        imb_error = coef * imb_dist
        self.imb_mat = imb_mat  # FOR DEBUG
    elif FLAGS.imb_fun == 'wass2':
        imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations, sq=True,
                                        backpropT=FLAGS.wass_bpt)
        imb_error = coef * imb_dist
        self.imb_mat = imb_mat  # FOR DEBUG
    else:
        imb_dist = lindisc(h_rep_norm, t, p_ipm)
        imb_error = coef * imb_dist

    return imb_error, imb_dist