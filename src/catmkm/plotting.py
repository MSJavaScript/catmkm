"""Plotting helpers for scaling relations, activity maps and reaction paths.

Matplotlib is an optional dependency; importing :mod:`catmkm` does not import
it.  The functions here raise a clear error only when plotting is actually
requested.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install it with "
            "`pip install catmkm[plot]`.") from exc
    return plt


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


def regress_check(X: np.ndarray, y: np.ndarray, outliers_threshold=-1.0):
    """Linear regression with optional Cook's-distance outlier removal."""
    ndata, ncoeff = X.shape
    index = np.logical_not(np.isnan(y))
    if np.std(y[index]) < 1.0e-6:
        b = np.zeros((ncoeff,))
        b[0] = np.mean(y[index])
        res = np.zeros((ndata,))
        return b, res, 1.0, np.ones((ndata,), dtype=bool)

    XX = X[index, :]
    b, r_out, r2 = regress(XX, y[index])

    if outliers_threshold > 0:
        n = XX.shape[0]
        p = ncoeff
        s2 = np.sum(r_out ** 2) / (n - p)
        A = np.linalg.inv(np.matmul(XX.T, XX))
        cooks_dist = np.zeros(n,)
        for j in range(n):
            hii = np.matmul(np.matmul(XX[j, :], A), XX[j, :].T)
            cooks_dist[j] = r_out[j] * r_out[j] / (p * s2) * hii / ((1 - hii) ** 2)
        threshold = outliers_threshold * np.mean(cooks_dist)
        flag = cooks_dist < threshold
        if np.sum(flag) > ncoeff:
            flag = cooks_dist > threshold
            index[np.where(index)[0][flag]] = False
            XX = X[index, :]
            b, r_out, r2 = regress(XX, y[index])
    return b, r_out, r2, index


def fit_scaling_relationship(mkm, fit_scaling_setting: dict, barrier_data: dict):
    """Fit 1D/2D scaling relations for every surface species and reaction."""
    outliers_threshold = fit_scaling_setting.get("outliers_threshold", -1.0)
    des_1d_name = fit_scaling_setting["descriptor_1d"]
    des_2d_name_x, des_2d_name_y = fit_scaling_setting["descriptor_2d"]

    x = barrier_data[des_1d_name]
    X_1d = np.column_stack((np.ones_like(x), x))

    xx = barrier_data[des_2d_name_x]
    yy = barrier_data[des_2d_name_y]
    X_2d = np.column_stack((np.ones_like(xx), xx, yy))

    n_data = len(barrier_data["#"])

    ads_scaling_dict = dict()
    for key in mkm.surface_species:
        y = barrier_data[key]
        res_1d = np.zeros((n_data,))
        b_1d, r_out_1d, r2_1d, index_1d = regress_check(X_1d, y, outliers_threshold)
        res_1d[index_1d] = r_out_1d

        res_2d = np.zeros((n_data,))
        b_2d, r_out_2d, r2_2d, index_2d = regress_check(X_2d, y, outliers_threshold)
        res_2d[index_2d] = r_out_2d

        ads_scaling_dict[key] = {
            "coeff": [b_1d, b_2d],
            "res": [res_1d, res_2d],
            "R2": [r2_1d, r2_2d],
        }

    n_reactions = len(mkm.reaction_list)
    scaling_coeff_1d = np.zeros((n_reactions, 2))
    scaling_coeff_2d = np.zeros((n_reactions, 3))

    for ir, reaction in enumerate(mkm.reaction_list):
        H = np.zeros((n_data,))
        for r in reaction.reactants:
            H += barrier_data[r.name] * (-r.num)
        for p in reaction.products:
            H += barrier_data[p.name] * p.num

        b_1d, _, _, _ = regress_check(X_1d, H, outliers_threshold)
        b_2d, _, _, _ = regress_check(X_2d, H, outliers_threshold)
        scaling_coeff_1d[ir, :] = b_1d
        scaling_coeff_2d[ir, :] = b_2d

    return {
        "ads_scaling_dict": ads_scaling_dict,
        "scaling_coeff_1d": scaling_coeff_1d,
        "scaling_coeff_2d": scaling_coeff_2d,
    }


def scaling_relationship_2d_modify(scaling_data, mkm, fit_scaling_setting: dict,
                                   barrier_data: dict):
    """Modify 2D scaling fits according to ``fit_with_y`` / ``fit_with_xy``."""
    fit_with_y = fit_scaling_setting["fit_2d_setting"]["fit_with_y"]
    fit_with_xy = fit_scaling_setting["fit_2d_setting"]["fit_with_xy"]
    _, des_2d_name_y = fit_scaling_setting["descriptor_2d"]
    outliers_threshold = fit_scaling_setting.get("outliers_threshold", -1.0)

    if len(fit_with_y) == 0 and len(fit_with_xy) == 0:
        return

    fit_with_x = [s for s in mkm.surface_species
                  if s not in (fit_with_xy + fit_with_y)]
    for key in fit_with_x:
        item = scaling_data["ads_scaling_dict"][key]
        item["R2"][1] = item["R2"][0]
        item["res"][1] = item["res"][0]
        item["coeff"][1][0:2] = item["coeff"][0]
        item["coeff"][1][2] = 0

    n_data = len(barrier_data["#"])
    x = barrier_data[des_2d_name_y]
    X_1d = np.column_stack((np.ones_like(x), x))
    for key in fit_with_y:
        y = barrier_data[key]
        res_1d = np.zeros((n_data,))
        b_1d, r_out_1d, r2_1d, index_1d = regress_check(X_1d, y, outliers_threshold)
        res_1d[index_1d] = r_out_1d

        item = scaling_data["ads_scaling_dict"][key]
        item["R2"][1] = r2_1d
        item["res"][1] = res_1d
        item["coeff"][1] = np.array([b_1d[0], 0, b_1d[1]])

    for ir, reaction in enumerate(mkm.reaction_list):
        coeffs_2d = np.zeros((3,))
        for r in reaction.reactants:
            if '#' in r.name:
                coeffs_2d += scaling_data["ads_scaling_dict"][r.name]["coeff"][1] * (-r.num)
            else:
                coeffs_2d[0] += barrier_data[r.name][0] * (-r.num)
        for p in reaction.products:
            if '#' in p.name:
                coeffs_2d += scaling_data["ads_scaling_dict"][p.name]["coeff"][1] * p.num
            else:
                coeffs_2d[0] += barrier_data[p.name][0] * p.num
        scaling_data["scaling_coeff_2d"][ir, :] = coeffs_2d


def draw_scaling_1d(data, surface_species, setting):
    plt = _get_plt()
    ncol = setting.get("ncol", 5)
    with_label = setting.get("with_label", True)
    label_size = setting.get("label_size", 10)
    show_formula = setting.get("show_formula", True)
    marker_size = setting.get("marker_size", 10)
    marker_color = setting.get("marker_color", "red")
    save_name = setting.get("save_name", "scaling_1d.png")
    fig_size = setting.get("fig_size", [15, 15])
    dpi = setting.get("dpi", 72)
    title_size = setting.get("title_size", 8)

    res_1d = data["res_1d"]
    R2_1d = data["R2_1d"]
    des_1d = data["1d_descriptor"]
    scaling_coeff_1d = data["scaling_coeff_1d"]
    data_ori = data["data_ori"]
    surf_name_list = np.array(data["surf_name_list"], dtype=str)

    x_name = des_1d["name"]
    x = np.array(des_1d["data"])
    min_x, max_x = np.min(x), np.max(x)

    n = len(surface_species)
    nrow = int(float(n) / ncol)
    if nrow * ncol < n:
        nrow += 1

    fig, axes = plt.subplots(nrow, ncol, figsize=fig_size, squeeze=False)
    for i in range(n):
        irow, icol = int(i / ncol), int(i - int(i / ncol) * ncol)
        c0, c1 = scaling_coeff_1d[i, :]
        good_index = np.logical_not(np.isnan(res_1d[i, :]))
        s_x = x[good_index]
        s_y = data_ori[surface_species[i]][good_index]

        axes[irow, icol].plot([min_x, max_x], [c0 + c1 * min_x, c0 + c1 * max_x])
        axes[irow, icol].scatter(s_x, s_y, s=marker_size, color=marker_color)
        axes[irow, icol].set_xlabel(r"$E_\text{{{:s}}}$/eV".format(x_name))
        axes[irow, icol].set_ylabel(r"$E_\text{{{:s}}}$/eV".format(surface_species[i]))
        if with_label:
            good_name = surf_name_list[good_index]
            for j in range(len(good_name)):
                axes[irow, icol].text(s_x[j], s_y[j], good_name[j], fontsize=label_size)
        if show_formula:
            formula = (r"$E_\text{{{:s}}} = {:.2f} {:+.2f}*E_\text{{{:s}}}$"
                       "\n" r"$R^2={{{:.2f}}}$").format(
                surface_species[i], c0, c1, x_name, R2_1d[i])
            axes[irow, icol].set_title(formula, fontsize=title_size)

    for i in range(n, ncol * nrow):
        irow, icol = int(i / ncol), int(i - int(i / ncol) * ncol)
        axes[irow, icol].remove()

    plt.tight_layout()
    plt.savefig(save_name, dpi=dpi)


def draw_scaling_2d(data, surface_species, setting):
    plt = _get_plt()
    ncol = setting.get("ncol", 5)
    with_label = setting.get("with_label", True)
    label_size = setting.get("label_size", 10)
    show_formula = setting.get("show_formula", True)
    marker_size = setting.get("marker_size", 10)
    save_name = setting.get("save_name", "scaling_2d.png")
    fig_size = setting.get("fig_size", [15, 15])
    dpi = setting.get("dpi", 72)
    draw_true_vs_fitting = setting.get("draw_true_vs_fitting", False)
    marker_color = setting.get("marker_color", "red")
    title_size = setting.get("title_size", 8)

    res_2d = data["res_2d"]
    surf_name_list = np.array(data["surf_name_list"], dtype=str)
    des_2d = data["2d_descriptor"]
    R2_1d = data["R2_2d"]
    scaling_coeff_2d = data["scaling_coeff_2d"]
    data_ori = data["data_ori"]

    x_name = des_2d[0]["name"]
    x = np.array(des_2d[0]["data"])
    min_x, max_x = np.min(x), np.max(x)
    y_name = des_2d[1]["name"]
    y = np.array(des_2d[1]["data"])
    min_y, max_y = np.min(y), np.max(y)

    xx, yy = np.meshgrid([min_x, 0.5 * (min_x + max_x), max_x],
                         [min_y, 0.5 * (min_y + max_y), max_y])

    n = len(surface_species)
    nrow = int(float(n) / ncol)
    if nrow * ncol < n:
        nrow += 1

    if not draw_true_vs_fitting:
        fig = plt.figure(figsize=fig_size)
        for i in range(n):
            c0, c1, c2 = scaling_coeff_2d[i, :]
            good_index = np.logical_not(np.isnan(res_2d[i, :]))
            s_x, s_y = x[good_index], y[good_index]
            s_z = data_ori[surface_species[i]][good_index]
            zz = c0 + c1 * xx + c2 * yy

            ax = fig.add_subplot(nrow, ncol, i + 1, projection='3d')
            ax.plot_surface(xx, yy, zz, alpha=0.7)
            ax.scatter(s_x, s_y, s_z, s=marker_size, color=marker_color)
            ax.set_xlabel(r"$E_\text{{{:s}}}$/eV".format(x_name))
            ax.set_ylabel(r"$E_\text{{{:s}}}$/eV".format(y_name))
            ax.set_zlabel(r"$E_\text{{{:s}}}$/eV".format(surface_species[i]))
            if with_label:
                good_name = surf_name_list[good_index]
                for j in range(len(good_name)):
                    ax.text(s_x[j], s_y[j], s_z[j], good_name[j], fontsize=label_size)
            if show_formula:
                formula = (r"$E_\text{{{:s}}} = {:.2f} {:+.2f}*E_\text{{{:s}}} "
                           r"{:+.2f}*E_\text{{{:s}}} $"
                           "\n" r"$R^2={{{:.2f}}}$").format(
                    surface_species[i], c0, c1, x_name, c2, y_name, R2_1d[i])
                ax.set_title(formula, fontsize=title_size)
    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=fig_size, squeeze=False)
        for i in range(n):
            irow, icol = int(i / ncol), int(i - int(i / ncol) * ncol)
            c0, c1, c2 = scaling_coeff_2d[i, :]
            good_index = np.logical_not(np.isnan(res_2d[i, :]))
            true_value = data_ori[surface_species[i]][good_index]
            fitting_value = c0 + c1 * x[good_index] + c2 * y[good_index]
            aa = np.hstack((true_value, fitting_value))
            aa_max, aa_min = np.max(aa), np.min(aa)
            axes[irow, icol].plot([aa_min, aa_max], [aa_min, aa_max])
            axes[irow, icol].scatter(true_value, fitting_value, s=marker_size,
                                     color=marker_color)
            axes[irow, icol].set_xlabel(r"$E_\text{{{:s}}}$/eV".format(surface_species[i]))
            axes[irow, icol].set_ylabel(r"$\hat{{E}}_\text{{{:s}}}$/eV".format(surface_species[i]))
            if with_label:
                good_name = surf_name_list[good_index]
                for j in range(len(good_name)):
                    axes[irow, icol].text(true_value[j], fitting_value[j],
                                          good_name[j], fontsize=label_size)
            if show_formula:
                formula = (r"$E_\text{{{:s}}} = {:.2f} {:+.2f}*E_\text{{{:s}}} "
                           r"{:+.2f}*E_\text{{{:s}}} $"
                           "\n" r"$R^2={{{:.2f}}}$").format(
                    surface_species[i], c0, c1, x_name, c2, y_name, R2_1d[i])
                axes[irow, icol].set_title(formula, fontsize=title_size)

        for i in range(n, ncol * nrow):
            irow, icol = int(i / ncol), int(i - int(i / ncol) * ncol)
            axes[irow, icol].remove()

    plt.tight_layout()
    plt.savefig(save_name, dpi=dpi)


def find_best_path(catrap_path_data, reac_scaling_1d, map_grid_1d,
                   reac_scaling_2d, map_grid_2d, save_n_best):
    """Select the lowest-barrier paths on 1D and 2D descriptor grids."""
    path_array, path_index = catrap_path_data

    n_tot_reactions = len(path_index)
    npath, nreac = path_array.shape
    pos_flag = path_array > 0.1
    neg_flag = path_array < -0.1
    col_pos = 2 * np.arange(nreac)
    col_neg = 2 * np.arange(nreac) + 1

    reaction_pathes_2_flag = np.zeros((npath, nreac * 2), dtype=bool)
    reaction_pathes_2_flag[:, col_pos] = pos_flag
    reaction_pathes_2_flag[:, col_neg] = neg_flag

    xmin, xmax, nx = map_grid_1d
    x = np.linspace(xmin, xmax, nx)
    X_1d = np.column_stack((np.ones_like(x), x))
    reaction_energy_1d = X_1d @ reac_scaling_1d.T

    reaction_energy_1d_2 = np.empty((reaction_energy_1d.shape[0],
                                     reaction_energy_1d.shape[1] * 2))
    reaction_energy_1d_2[:, ::2] = reaction_energy_1d
    reaction_energy_1d_2[:, 1::2] = -reaction_energy_1d

    collect_result_1d = np.zeros((nx, n_tot_reactions + 1, 4, save_n_best))
    index = np.argsort(-reaction_energy_1d_2, axis=1)
    for i, i_index in enumerate(index):
        ordered_flag = reaction_pathes_2_flag[:, i_index]
        path_order = np.lexsort(ordered_flag.T[::-1])
        order_order = np.argsort(path_order)
        first_true_indices = np.argmax(ordered_flag, axis=1)
        energy = reaction_energy_1d_2[i][i_index[first_true_indices]]

        for itr in range(n_tot_reactions):
            i_start, i_end = path_index[itr][0], path_index[itr][-1]
            sub_index = np.arange(i_start, i_end + 1)
            partial_order_order = order_order[i_start:(i_end + 1)]
            order_index = np.argsort(partial_order_order)
            select_path_index = sub_index[order_index][0:save_n_best]
            optimal_reac_index = first_true_indices[select_path_index]
            dG = energy[select_path_index]
            collect_result_1d[i, itr, 0, :] = itr
            collect_result_1d[i, itr, 1, :] = select_path_index
            collect_result_1d[i, itr, 2, :] = optimal_reac_index
            collect_result_1d[i, itr, 3, :] = dG

        all_path_index = np.arange(npath)
        select_path_index = all_path_index[path_order[0:save_n_best]]
        optimal_reac_index = first_true_indices[select_path_index]
        dG = energy[select_path_index]
        belong_to_tot_index = []
        for ii in select_path_index:
            for itr in range(n_tot_reactions):
                if path_index[itr][0] <= ii <= path_index[itr][-1]:
                    belong_to_tot_index.append(itr)
                    break
        collect_result_1d[i, n_tot_reactions, 0, :] = belong_to_tot_index
        collect_result_1d[i, n_tot_reactions, 1, :] = select_path_index
        collect_result_1d[i, n_tot_reactions, 2, :] = optimal_reac_index
        collect_result_1d[i, n_tot_reactions, 3, :] = dG
        print("At point %6.2f, %12d, %5d, %5d, dG = %6.3f" % (
            X_1d[i, 1], select_path_index[0], optimal_reac_index[0],
            belong_to_tot_index[0], dG[0]))

    xmin, xmax, nx = map_grid_2d[0]
    ymin, ymax, ny = map_grid_2d[1]
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, nx),
                         np.linspace(ymin, ymax, ny))
    xx = xx.flatten()
    yy = yy.flatten()
    X_2d = np.column_stack((np.ones_like(xx), xx, yy))
    reaction_energy_2d = X_2d @ reac_scaling_2d.T

    reaction_energy_2d_2 = np.empty((reaction_energy_2d.shape[0],
                                     reaction_energy_2d.shape[1] * 2))
    reaction_energy_2d_2[:, ::2] = reaction_energy_2d
    reaction_energy_2d_2[:, 1::2] = -reaction_energy_2d
    nxy = nx * ny

    collect_result_2d = np.zeros((nxy, n_tot_reactions + 1, 4, save_n_best))
    index = np.argsort(-reaction_energy_2d_2, axis=1)
    for i, i_index in enumerate(index):
        ordered_flag = reaction_pathes_2_flag[:, i_index]
        path_order = np.lexsort(ordered_flag.T[::-1])
        order_order = np.argsort(path_order)
        first_true_indices = np.argmax(ordered_flag, axis=1)
        energy = reaction_energy_2d_2[i][i_index[first_true_indices]]

        for itr in range(n_tot_reactions):
            i_start, i_end = path_index[itr][0], path_index[itr][-1]
            sub_index = np.arange(i_start, i_end + 1)
            partial_order_order = order_order[i_start:(i_end + 1)]
            order_index = np.argsort(partial_order_order)
            select_path_index = sub_index[order_index][0:save_n_best]
            optimal_reac_index = first_true_indices[select_path_index]
            dG = energy[select_path_index]
            collect_result_2d[i, itr, 0, :] = itr
            collect_result_2d[i, itr, 1, :] = select_path_index
            collect_result_2d[i, itr, 2, :] = optimal_reac_index
            collect_result_2d[i, itr, 3, :] = dG

        all_path_index = np.arange(npath)
        select_path_index = all_path_index[path_order[0:save_n_best]]
        optimal_reac_index = first_true_indices[select_path_index]
        dG = energy[select_path_index]
        belong_to_tot_index = []
        for ii in select_path_index:
            for itr in range(n_tot_reactions):
                if path_index[itr][0] <= ii <= path_index[itr][-1]:
                    belong_to_tot_index.append(itr)
                    break
        collect_result_2d[i, n_tot_reactions, 0, :] = belong_to_tot_index
        collect_result_2d[i, n_tot_reactions, 1, :] = select_path_index
        collect_result_2d[i, n_tot_reactions, 2, :] = optimal_reac_index
        collect_result_2d[i, n_tot_reactions, 3, :] = dG
        print("At point (%6.2f,%6.2f), %12d, %5d, %5d, dG = %6.3f" % (
            X_2d[i, 1], X_2d[i, 2], select_path_index[0],
            optimal_reac_index[0], belong_to_tot_index[0], dG[0]))

    return (x, xx, yy, reaction_pathes_2_flag, collect_result_1d,
            collect_result_2d)


def draw_1d_map(mkm, mkm_gas, reac_scaling_1d, x, path_index,
                reaction_pathes_2_flag, collect_result_1d, map_1d_setting,
                descriptor_1d_name):
    plt = _get_plt()
    linestyle_sequence = map_1d_setting["linestyle_sequence"]
    linecolor_sequence = map_1d_setting["linecolor_sequence"]
    color_and_style = [(color, style) for color in linecolor_sequence
                       for style in linestyle_sequence]
    n_color_and_style = len(color_and_style)
    xlabel = descriptor_1d_name + " / eV"

    save_name = map_1d_setting["save_name"]
    base_name, ext_name = save_name.split('.')
    xmin, xmax, nx = map_1d_setting["grid"]

    nx, nreaction, nitem, save_n_best = collect_result_1d.shape
    assert nreaction == len(path_index) + 1

    for i_tot_reac in range(nreaction):
        if i_tot_reac < nreaction - 1:
            i_start = path_index[i_tot_reac][0]
            i_end = path_index[i_tot_reac][-1]
            path_flag = reaction_pathes_2_flag[i_start:(i_end + 1), :]
        else:
            path_flag = reaction_pathes_2_flag
        reaction_flag = np.any(path_flag, axis=0)
        plt.figure(figsize=map_1d_setting["fig_size"])
        plt.plot(x, collect_result_1d[:, i_tot_reac, 3, 0],
                 linestyle="solid",
                 linewidth=map_1d_setting["activity_linewidth"],
                 color=map_1d_setting["activity_linecolor"])
        i_line_setting = 0
        for irc, frc in enumerate(reaction_flag):
            if not frc:
                continue
            ir = int(irc / 2)
            r_name = str(mkm.reaction_list[ir]).replace(' ', '')
            b, a = reac_scaling_1d[ir, :]
            linelabel = (f"({ir}):{r_name}"
                         if map_1d_setting["show_reaction_in_lengend"] else f"({ir})")
            if irc % 2 == 1:
                b = -b
                a = -a
                reverse_reac = deepcopy(mkm.reaction_list[ir])
                reverse_reac.reactants, reverse_reac.products = (
                    reverse_reac.products, reverse_reac.reactants)
                r_name = str(reverse_reac).replace(' ', '')
                linelabel = (f"({-ir}):{r_name}"
                             if map_1d_setting["show_reaction_in_lengend"] else f"({-ir})")
            linecolor, linestyle = color_and_style[i_line_setting % n_color_and_style]
            plt.plot([xmin, xmax], [b + xmin * a, b + xmax * a],
                     linestyle=linestyle,
                     linewidth=map_1d_setting["linewidth"],
                     color=linecolor, label=linelabel)
            i_line_setting += 1
        plt.legend(ncol=map_1d_setting["legend_columns"],
                   loc="center left", bbox_to_anchor=(1, 0.5))
        if i_tot_reac < nreaction - 1:
            plt.title(str(mkm_gas.reaction_list[i_tot_reac]).replace(' ', ''))
        else:
            plt.title("All total reactions")
        plt.xlim([xmin, xmax])
        plt.xlabel(xlabel)
        plt.tight_layout()
        if i_tot_reac < nreaction - 1:
            plt.savefig(f"{base_name}_{i_tot_reac}_.{ext_name}",
                        dpi=map_1d_setting["dpi"])
        else:
            plt.savefig(f"{base_name}_all_.{ext_name}",
                        dpi=map_1d_setting["dpi"])


def draw_2d_map(xx, yy, collect_result_2d, map_2d_setting, descriptor_2d_name):
    plt = _get_plt()
    from matplotlib import colors

    nxy, nreaction, nitem, save_n_best = collect_result_2d.shape
    dG_tot = collect_result_2d[:, nreaction - 1, 3, 0]
    mech_tot = np.array(collect_result_2d[:, nreaction - 1, 2, 0], dtype=int)

    grid = map_2d_setting["grid"]
    xmin, xmax, nx = grid[0]
    ymin, ymax, ny = grid[1]
    xlabel, ylabel = descriptor_2d_name

    p_data = map_2d_setting["scatter_points"]["data"]
    p_x = [p[0] for p in p_data]
    p_y = [p[1] for p in p_data]
    p_label = [p[2] for p in p_data]
    dp_x, dp_y = map_2d_setting["scatter_points"]["text_shift"]

    XX = xx.reshape((nx, ny))
    YY = yy.reshape((nx, ny))
    ZZ = -dG_tot.reshape((nx, ny))

    plt.figure(figsize=map_2d_setting["activity_map_setting"]["fig_size"])
    heatmap = plt.pcolormesh(
        XX, YY, ZZ, cmap=map_2d_setting["activity_map_setting"]["color_map"])
    if (map_2d_setting["activity_map_setting"]["show_scatter_points"]
            and len(p_x) > 0):
        plt.scatter(p_x, p_y,
                    s=map_2d_setting["scatter_points"]["point_size"],
                    c=map_2d_setting["scatter_points"]["point_color"])
        for i in range(len(p_x)):
            plt.text(p_x[i] + dp_x, p_y[i] + dp_y, p_label[i],
                     color=map_2d_setting["scatter_points"]["text_color"],
                     fontsize=map_2d_setting["scatter_points"]["text_size"])
    plt.colorbar(heatmap)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel(xlabel + " / eV")
    plt.ylabel(ylabel + " / eV")
    plt.tight_layout()
    plt.savefig(map_2d_setting["activity_map_setting"]["save_name"],
                dpi=map_2d_setting["activity_map_setting"]["dpi"])

    ZZ = mech_tot.reshape((nx, ny))
    unique_values = np.unique(ZZ)
    num_regions = len(unique_values)
    cmap = plt.get_cmap(map_2d_setting["mechanism_map_setting"]["color_map"],
                        num_regions)
    bounds = np.arange(num_regions + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    value_to_index = {v: i for i, v in enumerate(unique_values)}
    D_indexed = np.vectorize(value_to_index.get)(ZZ)

    plt.figure(figsize=map_2d_setting["mechanism_map_setting"]["fig_size"])
    img = plt.pcolormesh(XX, YY, D_indexed, cmap=cmap, norm=norm, shading='auto')
    cb = plt.colorbar(img, ticks=bounds[:-1] + 0.5)
    cb.ax.set_yticklabels(sorted(unique_values))
    cb.set_label('Pathway number')
    if (map_2d_setting["mechanism_map_setting"]["show_scatter_points"]
            and len(p_x) > 0):
        plt.scatter(p_x, p_y,
                    s=map_2d_setting["scatter_points"]["point_size"],
                    c=map_2d_setting["scatter_points"]["point_color"])
        for i in range(len(p_x)):
            plt.text(p_x[i] + dp_x, p_y[i] + dp_y, p_label[i],
                     color=map_2d_setting["scatter_points"]["text_color"],
                     fontsize=map_2d_setting["scatter_points"]["text_size"])
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel(xlabel + " / eV")
    plt.ylabel(ylabel + " / eV")
    plt.tight_layout()
    plt.savefig(map_2d_setting["mechanism_map_setting"]["save_name"],
                dpi=map_2d_setting["mechanism_map_setting"]["dpi"])


def draw_scaling_and_map(data: dict, settings: dict):
    """End-to-end scaling/map workflow (former ``draw_scaling_and_map.py``).

    Parameters
    ----------
    data:
        Dictionary with ``reactions``, ``total_reactions``, ``gas_info``,
        ``barriers`` and ``surf_name_list``.
    settings:
        The former plot-setting JSON.  It must contain
        ``fit_scaling_setting`` and ``draw_map_setting``; if
        ``catrap_path_pkl`` is provided it is read, otherwise path data is
        taken from ``catrap_path_data`` in *data*.
    """
    import pickle

    from .model import GasSpecies, MKM, parse_network

    reactions = data["reactions"]
    total_reactions = data.get(
        "total_reactions",
        (data.get("find_reaction_path") or {}).get("total_reactions"))
    if not total_reactions:
        raise ValueError(
            "The data must contain 'total_reactions' (top level or inside "
            "'find_reaction_path').")
    gas_info = data.get("gas_info", [])
    barriers = data["barriers"]
    surf_name_list = data["surf_name_list"]

    mkm = MKM()
    mkm_gas = MKM()
    gas_list = []
    for gas in gas_info:
        obj = GasSpecies(gas["name"], gas.get("pressure", [1.0]))
        gas_list.append(obj)
    mkm.gas_list = gas_list
    mkm_gas.gas_list = gas_list
    parse_network(mkm, reactions, True, [])
    parse_network(mkm_gas, total_reactions, True, [])

    barrier_data = {item["name"]: np.array(item["data"], dtype=float)
                    for item in barriers}

    fit_scaling_setting = settings["fit_scaling_setting"]
    scaling_data = fit_scaling_relationship(mkm, fit_scaling_setting, barrier_data)
    scaling_relationship_2d_modify(scaling_data, mkm, fit_scaling_setting,
                                   barrier_data)

    ads_scaling_dict = scaling_data["ads_scaling_dict"]
    scaling_coeff_1d = scaling_data["scaling_coeff_1d"]
    scaling_coeff_2d = scaling_data["scaling_coeff_2d"]

    surface_species = [s for s in mkm.surface_species if s[0] != '#']
    n_surf_spec = len(surface_species)
    n_data = len(barrier_data["#"])
    ads_coeff_1d = np.zeros((n_surf_spec, 2))
    R2_1d = np.zeros((n_surf_spec,))
    res_1d = np.zeros((n_surf_spec, n_data)) * np.nan
    ads_coeff_2d = np.zeros((n_surf_spec, 3))
    R2_2d = np.zeros((n_surf_spec,))
    res_2d = np.zeros((n_surf_spec, n_data)) * np.nan
    for i, key in enumerate(surface_species):
        item = ads_scaling_dict[key]
        ads_coeff_1d[i, :] = item["coeff"][0]
        R2_1d[i] = item["R2"][0]
        res_1d[i, :] = item["res"][0]
        ads_coeff_2d[i, :] = item["coeff"][1]
        R2_2d[i] = item["R2"][1]
        res_2d[i, :] = item["res"][1]

    scaling_data_dict = {
        "data_ori": barrier_data,
        "surf_name_list": surf_name_list,
        "1d_descriptor": {
            "name": fit_scaling_setting["descriptor_1d"],
            "data": barrier_data[fit_scaling_setting["descriptor_1d"]],
        },
        "scaling_coeff_1d": ads_coeff_1d,
        "R2_1d": R2_1d,
        "res_1d": res_1d,
        "2d_descriptor": [
            {"name": fit_scaling_setting["descriptor_2d"][0],
             "data": barrier_data[fit_scaling_setting["descriptor_2d"][0]]},
            {"name": fit_scaling_setting["descriptor_2d"][1],
             "data": barrier_data[fit_scaling_setting["descriptor_2d"][1]]},
        ],
        "scaling_coeff_2d": ads_coeff_2d,
        "R2_2d": R2_2d,
        "res_2d": res_2d,
    }

    draw_map_setting = settings["draw_map_setting"]
    if "catrap_path_data" in data:
        catrap_path_data = data["catrap_path_data"]
    else:
        with open(settings["catrap_path_pkl"], "rb") as fh:
            catrap_path_data = pickle.load(fh)

    optimal_path_pkl = settings.get("optimal_path_pkl")
    if optimal_path_pkl:
        try:
            with open(optimal_path_pkl, "rb") as fh:
                optimal = pickle.load(fh)
            x, xx, yy = optimal["x"], optimal["xx"], optimal["yy"]
            reaction_pathes_2_flag = optimal["reaction_pathes_2_flag"]
            collect_result_1d = optimal["collect_result_1d"]
            collect_result_2d = optimal["collect_result_2d"]
        except FileNotFoundError:
            optimal = None
    else:
        optimal = None

    if optimal is None:
        x, xx, yy, reaction_pathes_2_flag, collect_result_1d, collect_result_2d = \
            find_best_path(catrap_path_data, scaling_coeff_1d,
                           draw_map_setting["map_1d"]["grid"],
                           scaling_coeff_2d,
                           draw_map_setting["map_2d"]["grid"],
                           draw_map_setting["save_n_best"])
        if optimal_path_pkl:
            with open(optimal_path_pkl, "wb") as fh:
                pickle.dump({
                    "x": x, "xx": xx, "yy": yy,
                    "reaction_pathes_2_flag": reaction_pathes_2_flag,
                    "collect_result_1d": collect_result_1d,
                    "collect_result_2d": collect_result_2d,
                }, fh)

    draw_1d_map(mkm, mkm_gas, scaling_coeff_1d, x, catrap_path_data[1],
                reaction_pathes_2_flag, collect_result_1d,
                draw_map_setting["map_1d"],
                fit_scaling_setting["descriptor_1d"])
    draw_2d_map(xx, yy, collect_result_2d, draw_map_setting["map_2d"],
                fit_scaling_setting["descriptor_2d"])
