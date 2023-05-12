import numpy as np
import matplotlib.pyplot as plt


def prune_contributions(contributions):

    total_ECP = dict()

    for a in contributions:
        total_ECP[a[0]] = total_ECP.get(a[0], 0) + a[1]

    # remove the contributions that are 0
    to_del = []
    for key in total_ECP:
        if total_ECP[key] == 0:
            to_del.append(key)
    for key in to_del:
        del total_ECP[key]
        
    return sorted(list(total_ECP.items()), key = lambda x: x[0])


def EC_at_bifiltration(contributions, f1, f2):
    return sum([c[1] for c in contributions if (c[0][0] <= f1) and
                                               (c[0][1] <= f2)])


def difference_ECP(ecp_1, ecp_2, dims, return_contributions = False):
    f1min, f1max, f2min, f2max = dims
    
    contributions = [((f1min, f2min), 0), ((f1max, f2max), 0)]
    
    contributions += ecp_1
    contributions += [(c[0], -1*c[1]) for c in ecp_2]
    
    contributions = [((f1min, f2min), 0)]+prune_contributions(contributions)+[((f1max, f2max), 0)]
    
    f1_list = sorted(set([c[0][0] for c in contributions]))
    f2_list = sorted(set([c[0][1] for c in contributions]))
    
    difference = 0
    
    for i, f1 in enumerate(f1_list[:-1]):
        delta_i = f1_list[i+1] - f1_list[i]
        for j, f2 in enumerate(f2_list[:-1]):
            delta_j = f2_list[j+1] - f2_list[j]
            
            difference += EC_at_bifiltration(contributions, f1, f2) * delta_i * delta_j
    
    if return_contributions:
        return difference, contributions
    else:
        return difference



def plot_ECP(contributions, dims,
             this_ax=None, 
             colorbar=False, **kwargs):
    
    f1min, f1max, f2min, f2max = dims
    
    if this_ax == None:
        this_ax = plt.gca()
    
    f1_list = [f1min] + sorted(set([c[0][0] for c in contributions])) + [f1max]
    f2_list = [f2min] + sorted(set([c[0][1] for c in contributions])) + [f2max]
    
    Z = np.zeros((len(f2_list)-1, len(f1_list)-1))

    for i, f1 in enumerate(f1_list[:-1]):
        for j, f2 in enumerate(f2_list[:-1]):
            Z[j,i] = EC_at_bifiltration(contributions, f1, f2)
    
    # Plotting
    im = this_ax.pcolormesh(f1_list, f2_list, Z, **kwargs)
    
    if colorbar:
        plt.colorbar(im, ax=this_ax)
    
    return this_ax



# given the ordered list of local contributions
# returns a list of tuples (filtration, euler characteristic)
def euler_characteristic_list_from_all(local_contributions):

    local_contributions = [(c[0][0], c[1]) for c in local_contributions]

    euler_characteristic = []
    old_f, current_characteristic = local_contributions[0]

    for filtration, contribution in local_contributions[1:]:
        if filtration > old_f:
            euler_characteristic.append([old_f, current_characteristic])
            old_f = filtration

        current_characteristic += contribution

    # add last contribution
    if len(local_contributions) > 1:
        euler_characteristic.append([filtration, current_characteristic])
        
    if len(local_contributions) == 1:
        euler_characteristic.append(local_contributions[0])

    return euler_characteristic



# WARNING
# when plotting a lot of points, drawing the lines can take some time
def plot_euler_curve(e_list, this_ax=None, with_lines=False, **kwargs):

    if this_ax == None:
        this_ax = plt.gca()

    # Plotting
    this_ax.scatter([f[0] for f in e_list], [f[1] for f in e_list])
    # draw horizontal and vertical lines b/w points
    if with_lines:
        for i in range(1, len(e_list)):
            this_ax.vlines(
                x=e_list[i][0],
                ymin=min(e_list[i - 1][1], e_list[i][1]),
                ymax=max(e_list[i - 1][1], e_list[i][1]),
            )
            this_ax.hlines(y=e_list[i - 1][1], xmin=e_list[i - 1][0], xmax=e_list[i][0])

    this_ax.set_xlabel("Filtration")
    this_ax.set_ylabel("Euler Characteristic")
    return this_ax





# given the list of changes to the EC and a filtration value
# returns the EC at that filtration value
def EC_at_filtration(ecc_list, f):

    ec = ecc_list[0][1]

    for current_ec in ecc_list:
        if current_ec[0] > f:
            break
        ec = current_ec[1]

    return ec


# computes the difference between two ECC from 0 to a max filtration value
def difference_ECC(ecc1, ecc2, max_f):
    # find full list of filtration points
    filtration_steps = list(set(([f[0] for f in ecc1] + [f[0] for f in ecc2] + [max_f])))
    filtration_steps.sort()

    difference = 0

    for i in range(1, len(filtration_steps)):
        if filtration_steps[i] > max_f:
            break

        ec_1 = EC_at_filtration(ecc1, filtration_steps[i-1])
        ec_2 = EC_at_filtration(ecc2, filtration_steps[i-1])

        difference += abs(ec_1 - ec_2) * (filtration_steps[i] - filtration_steps[i-1])

    return difference



import warnings

def graded_rank_at_value(betti, x,y):
    for i, this_x in enumerate(betti.dimensions.x_grades):
        if this_x > x:
            i -= 1
            break

    for j, this_y in enumerate(betti.dimensions.y_grades):
        if this_y > y:
            j -= 1
            break

    if j < 0 or i < 0:
        return 0
    return betti.graded_rank[j,i]

def discretize_graded_rank(betti, x_grid, y_grid, idx=None):
    betti_grid = np.zeros((len(x_grid), len(y_grid)))

    try:
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                betti_grid[i,j] = graded_rank_at_value(betti, x, y)
    except:
        warnings.warn('the graded rank is empty for graph {}'.format(idx))

    # just to be consistent with pyRivet
    return betti_grid.T