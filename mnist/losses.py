import torch 
import torch.nn as nn
import math 
from scipy.special import binom

BCEloss = torch.nn.BCEWithLogitsLoss()

clf_loss = nn.NLLLoss()

def nll_loss_neg(y_pred, y_true):  # # #
    out = torch.sum(y_true * y_pred, dim=1)
    return torch.mean(- torch.log((1 - out) + 1e-6))

def inverted_cross_entropy(y_pred, y_true):
    out = - torch.mean(y_true * torch.log(1-y_pred + 1e-6) + 1e-6)
    return out

def d_loss_cosine_margin(real, fake, y, m=0.15, s=10.0):
    real = real - m
    return BCEloss(s*(real - fake) + 1e-6 , y)

def g_loss_cosine_margin(real, fake, y, m=0.15, s=10.0):
    fake = fake + m
    return BCEloss(s*(fake - real) + 1e-6, y)

def calculate_phi_theta(cos_theta, m=4): # m: margin
    C_m_2n = torch.Tensor(binom(m, range(0, m + 1, 2))).cuda()  # C_m^{2n}
    cos_powers = torch.Tensor(range(m, -1, -2)).cuda()  # m - 2n
    sin2_powers = torch.Tensor(range(len(cos_powers))).cuda()  # n
    signs = torch.ones(m // 2 + 1).cuda()  # 1, -1, 1, -1, ...
    signs[1::2] = -1

    sin2_theta = 1 - cos_theta**2
    cos_terms = cos_theta ** cos_powers # cos^{m - 2n}
    sin2_terms = sin2_theta ** sin2_powers # sin2^{n}

    cos_m_theta = (signs *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                    C_m_2n *
                    cos_terms *
                    sin2_terms).sum(1)  # summation of all terms # shape: batch_size

    k = find_k(cos_m_theta, m)
    phi_theta = (-1) ** k * cos_m_theta - 2 * k
    return phi_theta.unsqueeze(1)

def find_k(cos_theta, m):
    divisor = math.pi / m  # pi/m
    # to account for acos numerical errors
    eps = 1e-7
    cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
    theta = cos_theta.acos() # arccos
    k = (theta / divisor).floor().detach()
    return k

def d_loss_multi_angular_2k(real, fake, y, m=4, s = 10.0):
    m = int(m)
    real = calculate_phi_theta(real, m)
    return BCEloss(s * (real - torch.mean(fake)) + 1e-6, y)

def g_loss_multi_angular_2k(real, fake, y, m=4, s = 10.0):
    m = int(m)
    fake = calculate_phi_theta(fake, m)
    return BCEloss(s * (fake - torch.mean(real)) + 1e-6, y)

def d_loss_additive_angular_arccos(real, fake, y, m=1.35, s = 10.0):
    real.arccos_()
    real = real + m
    real.cos_()
    return BCEloss(s * (real - torch.mean(fake)) + 1e-6, y)

def g_loss_additive_angular_arccos(real, fake, y, m=1.35, s = 10.0):
    fake.arccos_()
    fake = fake + m
    fake.cos_()
    return BCEloss(s * (fake - torch.mean(real)) + 1e-6, y)
