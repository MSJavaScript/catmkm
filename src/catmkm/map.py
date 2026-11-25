"""Linear scaling relations used by the activity-map calculation."""

from __future__ import annotations

import numpy as np

from .model import MKM


def regress(X_in, y_in):
    index = np.logical_not(np.isnan(y_in))
    X = X_in[index, :]
    y = y_in[index]
    Q, R = np.linalg.qr(X)
    b = np.linalg.solve(R, np.matmul(Q.T, y))
    r = y - np.matmul(X, b)
    SSE = np.linalg.norm(r) ** 2
    TSS = np.linalg.norm(y - np.mean(y)) ** 2
    r2 = 1 - SSE / TSS
    r_out = np.zeros(y_in.shape) * np.nan
    r_out[index] = r
    return b, r_out, r2


def fit_linear_relationship(mkm: MKM, barrier_data, ads_dict, params: dict):
    """Fit barrier/energy data against one or two descriptors."""
    ndes = params["ndes"]
    ndes_data = params["ndes_data"]
    des_datas = params["des_datas"]

    if ndes == 1:
        X = np.ones((ndes_data, 2))
        X[:, 1] = des_datas[0]
    else:
        X = np.ones((ndes_data, 3))
        X[:, 1] = des_datas[0]
        X[:, 2] = des_datas[1]

    if barrier_data is not None:
        nreactions = mkm.nreactions
        residuals = np.zeros((nreactions * 2, ndes_data))
        r2 = np.zeros((nreactions * 2,))
        coeffs = np.zeros((nreactions * 2, ndes + 1))
        for ir in range(nreactions):
            y = barrier_data[ir, 0:ndes_data]
            b, r, rsq = regress(X, y)
            coeffs[ir * 2, :] = b
            residuals[ir * 2, :] = r
            r2[ir * 2] = rsq

            y = barrier_data[ir, ndes_data:]
            b, r, rsq = regress(X, y)
            coeffs[ir * 2 + 1, :] = b
            residuals[ir * 2 + 1, :] = r
            r2[ir * 2 + 1] = rsq
        return coeffs, residuals, r2

    coeffs, residuals, r2 = {}, {}, {}
    for key in ads_dict.keys():
        y = ads_dict[key]
        if abs(np.std(y)) < 1.0e-10:
            u = y[np.logical_not(np.isnan(y))]
            coeffs[key] = np.zeros((ndes + 1,))
            coeffs[key][0] = u[0]
            residuals[key] = np.zeros(y.shape)
            r2[key] = 1.0
            continue
        b, r, rsq = regress(X, y)
        coeffs[key] = b
        residuals[key] = r
        r2[key] = rsq
    return coeffs, residuals, r2


def get_data_from_linear_relationship(des1: float, des2, mkm: MKM, coeffs):
    """Evaluate the fitted relation at one descriptor point."""
    if des2 is None:
        if isinstance(coeffs, np.ndarray):
            data = coeffs[:, 1] * des1 + coeffs[:, 0]
            data = np.reshape(data, (mkm.nreactions, 2))
            data[data[:, 0] < 0, 0] = 0
        else:
            data = {key: coeffs[key][0] + coeffs[key][1] * des1
                    for key in coeffs.keys()}
    else:
        if isinstance(coeffs, np.ndarray):
            data = coeffs[:, 2] * des2 + coeffs[:, 1] * des1 + coeffs[:, 0]
            data = np.reshape(data, (mkm.nreactions, 2))
            data[data[:, 0] < 0, 0] = 0
        else:
            data = {key: coeffs[key][0] + coeffs[key][1] * des1
                    + coeffs[key][2] * des2 for key in coeffs.keys()}
    return data
