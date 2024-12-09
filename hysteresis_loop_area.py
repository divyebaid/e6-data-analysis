import pandas as pd
import numpy as np

# constants and experimental parameters
MILD_STEEL_AREA = 7.45e-6  # cross-sectional area of mild steel rod (m^2)
TRANSFORMER_AREA = 1.65e-6  # cross-sectional area of transformer iron rod (m^2)
CUNI_AREA = 1.90e-5  # cross-sectional area of Cu/Ni alloy rod (m^2)
L_P = 4.2e-2  # length of primary coil (m)
R_P = 2  # resistance in series with primary coil (ohm)
N_P = 400  # number of turns of primary coil
N_S = 500  # number of turns of secondary coil
C = 1.02e-6  # capacitance used in integrator circuit (F)
R_I = 1e3  # input resistance of integrator circuit (ohm)
MU_0 = (4e-7) * np.pi  # permittivity of vacuum (NA^-2)

# bounds and tolerances (hardcoded)
TOL_MILD_STEEL = [0.02, 0.1]
BOUNDS_MILD_STEEL = [[-21500, 21500], [-2.05, 2.05]]
TOL_TRANSFORMER = [0.02, 0.1]
BOUNDS_TRANSFORMER = [[-3000, 3000], [-2.8, 2.8]]
TOL_CUNI = [0.09, 0.01]
BOUNDS_CUNI = [[-4500, 5000], [-0.11, 0.11]]


def hysteresis_area(filepath: str, area: float, tol: list, bounds: list):
    # filepath: path of data in csv
    # area: cross-sectional area of sample
    # tol: tolerance in [V_X, V_Y] for separating loops: endpoints are identified to within tolerance
    # bounds: [[H lower, upper], [B lower, upper]]
    #         to manually select out the tails of the hysteresis curve because of instrument error causing tails to have nonzero area
    #         i.e. all (H, B) removed with both H, B below or above their lower or upper bounds
    # returns: [mean, standard deviation] of the list of areas of all isolated individual loops
    data = pd.read_csv(filepath, header=None)
    data = data.fillna(0)
    data = data.rename(columns={0:'V_X', 1:'V_Y'})
    V_X = data['V_X'].to_numpy()
    V_Y = data['V_Y'].to_numpy()

    # separating the data into individual hysteresis loops
    v_x_0, v_y_0 = V_X[0], V_Y[0]
    tol_X, tol_Y = tol
    distinct_loops = [[], []]  # data from distinct V_X, V_Y loops
    current_loop = [[], []]  # V_X, V_Y data for current loop
    for i, v_x in enumerate(V_X):
        v_y = V_Y[i]
        if np.abs(v_y - v_y_0) <= tol_Y and np.abs (v_x - v_x_0) <= tol_X:
            # new loop reached within tolerances
            if len(current_loop[0]) != 0:
                distinct_loops[0].append(current_loop[0])
                distinct_loops[1].append(current_loop[1])
                current_loop = [[], []]
                v_x_0, v_y_0 = v_x, v_y  
            else:
                pass
        else:
            current_loop[0].append(v_x)
            current_loop[1].append(v_y)

    # manually removing the few unseparated multi-loops
    min_len = len(distinct_loops[0][0])
    for loop in distinct_loops[0]:
        if len(loop) < min_len:
            min_len = len(loop)
    rem_loops = []
    for i, loop in enumerate(distinct_loops[0]):
        if np.round(len(loop) / min_len) > 1:
            rem_loops.append(i)
    rem_loops = rem_loops[::-1]
    for idx in rem_loops:
        distinct_loops[0].pop(idx)
        distinct_loops[1].pop(idx)  # distinct_loops has V_X and V_Y values for each separate loop

    # computing the areas of each loop using shoelace formula and average them
    areas = []
    bounds_H, bounds_B = bounds
    for i in range(len(distinct_loops[0])):
        B = np.array(distinct_loops[1][i]) / (N_S * area / (C * R_I))
        H = np.array(distinct_loops[0][i]) / (L_P * R_P / N_P)

        B = B.tolist()
        H = H.tolist()
        remove_B_H = []
        if bounds_B and bounds_H:
            for i in range(len(B)):
                if (H[i] < bounds_H[0] and B[i] < bounds_B[0]): 
                    remove_B_H.append(i)
                elif (H[i] > bounds_H[1] and B[i] > bounds_B[1]):
                    remove_B_H.append(i)
                else:
                    pass

        if len(remove_B_H) > 0:
            remove_B_H = remove_B_H[::-1]
            for n in remove_B_H:
                B.pop(n)
                H.pop(n)  

        B = np.array(B)
        H = np.array(H)  # removed all (H, B) out of bounds  
        centroid_H, centroid_B = np.mean(H), np.mean(B)
        angles = np.arctan2(B - centroid_B, H - centroid_H)
        sorted_indices = np.argsort(angles)
        H_sorted = H[sorted_indices]
        B_sorted = B[sorted_indices]  # sorting (H, B) by polar angle about (H mean, B mean)

        if H_sorted[0] != H_sorted[-1] or B_sorted[0] != B_sorted[-1]:
            H_sorted = np.append(H_sorted, H_sorted[-1])
            B_sorted = np.append(B_sorted, B_sorted[-1])  # ensures a closed polygon for shoelace formula
    
        # shoelace formula
        graph_area = 0.5 * np.abs(np.dot(H_sorted, np.roll(B_sorted, -1)) - np.dot(B_sorted, np.roll(H_sorted, -1)))
        areas.append(graph_area)

    areas = np.array(areas)  # areas of all the different loops isolated
    area_mean = np.mean(areas)
    area_stdev = np.sqrt(np.var(areas))  

    return [area_mean, area_stdev]