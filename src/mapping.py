import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la


def boundary_side_count_matrix(squares):
    """
    Construct the diagonal (n, n) matrix Λ counting exposed boundary sides.

    Returns: diagonal matrix whose entries Λ_ii count the amount of boundary sides per square i.
    """
    boundary_side_counts = []

    for x, y, neighbors in squares:
        boundary_side_counts.append(neighbors.count(-1))

    return sp.diags(boundary_side_counts, format="csc", dtype=int)

def graph_laplacian(squares):
    """
    Construct the discrete (n,n) graph Laplacian on the square adjacency graph.

    Each square center is treated as a graph node. Two nodes are adjacent if the corresponding squares share a side. 
    Since each square in the ambient square lattice has degree 4, the discrete operator is Δ = 4I - A, 
    where A is the adjacency matrix.

    Returns: sparse Laplacian matrix on the square centers.
    """
    n = len(squares)
    A = sp.lil_matrix((n, n))
    I = sp.identity(n)
    for i, square in enumerate(squares):
        x,y,neighbors = square
        for neighbor in neighbors:
            if neighbor != -1: #i and j are neighbors
                A[i, neighbor] = 1
                A[neighbor, i] = 1
    return 4*I - A



def inverse_laplacian(laplacian):
    """
    Compute the inverse of the discrete Laplacian.

    Returns: Sparse Δ^{-1}.
    """
    return spla.inv(sp.csc_matrix(laplacian))


def start_vertex(inverse_laplacian):
    """
    Choose a central start square using the inverse Laplacian. The selected square is the one with maximal row sum in Δ^{-1}, 
    which is used as a proxy for the square with largest expected random-walk hitting time to the boundary.

    Returns:
    Index of chosen start square.
    """
    row_sums = inverse_laplacian.sum(axis=1)
    return np.argmax(row_sums)

def harmonic_measure_by_inverse(start_x, lambda_matrix, inv_laplacian):
    """
    Compute the boundary hitting probabilities from a chosen start square.
    (Δ^{-1} Λ)[i] gives the probability the boundary is exited via square i.

    Returns: Square-based hitting probabilities.
    """
    row = inv_laplacian.getrow(start_x)
    product = row @ lambda_matrix
    return product.toarray().ravel()   

def probs_per_edge(squares, probs, lambda_matrix):
    """
    Distribute hitting probabilities per square over the individual boundary edges.
    If a boundary square has k exposed boundary sides, then its total probability
    is divided equally over the k edges.

    Returns: List of harmonic measure assigned to each ordered boundary edge.
    """
    edge_list = lambda_matrix.diagonal()
    edge_probs = []

    for i, (x, y, neighbors) in enumerate(squares):
        n_boundary_sides = int(edge_list[i])
        for _ in range(n_boundary_sides):
            edge_probs.append(probs[i] / n_boundary_sides)

    return edge_probs

def probs_per_edge_ordered(squares, probs, boundary):
    edge_probs = []
    for idx, side in boundary:
        n_boundary_sides = squares[idx][2].count(-1)
        edge_probs.append(probs[idx] / n_boundary_sides)
    return np.array(edge_probs, dtype=float)
    
def boundary_preimages(edge_probs):
    """
    Map ordered boundary edges to points on the unit circle, according to computed harmonic measure of the edges.
    Each boundary edge is represented by the midpoint of its corresponding arc.
    
    Returns: Complex coordinates on the unit circle representing boundary-edge preimages.
    """
    cumulative = np.cumsum(edge_probs)
    angles = 2 * np.pi * cumulative

    return np.exp(1j * angles)

def boundary_midpoint_preimages(edge_probs):
    """
    Compute midpoint preimages on the unit circle for each boundary edge.
    Each boundary edge corresponds to an arc on the unit circle whose angular
    size equals its harmonic measure. This function returns the midpoint of
    each arc.

    Returns: Array of complex coordinates on the unit circle representing midpoints
    """
    edge_probs = np.asarray(edge_probs, dtype=float)

    cumulative = np.cumsum(edge_probs)

    # Arc endpoints
    angles = np.concatenate(([0.0], 2 * np.pi * cumulative))

    # Midpoints of arcs
    mid_angles = 0.5 * (angles[:-1] + angles[1:])

    return np.exp(1j * mid_angles)

def construct_B(squares, boundary):
    """
    Construct the (n,m) matrix B coupling interior square variables to boundary-edge data. Entry B[i, j] equals 1 if 
    boundary edge j is attached to square i, and 0 otherwise.

    Returns: Sparse matrix B.
    """
    n = len(squares)  # Number of interior points
    m = len(boundary)  # Number of boundary points
    B = sp.lil_matrix((n, m))

    #Loop over each interior points
    for i, square in enumerate(squares):
        x, y, neighbors = square
        for j, edge in enumerate(boundary):
            idx, side = edge
            if idx == i:
                B[i,j] = 1
    return B


def solve_dirichlet_problem(laplacian, B, w):
    """
    Solve the discrete Dirichlet problem for the interior harmonic extension: Δ z = -B w
    for the complex coordinates z that are mapped to the square centers, for boundary array w

    Returns: Array of complex interior values z on the disk
    """
    Bw = - B @ w
    return - spla.spsolve(laplacian, Bw)

def square_centers_complex(squares):
    """
    Convert square centers to complex coordinates u = x + i y.

    Returns: Complex coordinate array u
    """
    u = np.zeros(len(squares), dtype=complex)

    for i, (x, y, neighbors) in enumerate(squares):
        u[i] = x + 1j * y

    return u

def inside_radius(z_values, u_values, squares, radius):
    """
    Restrict paired data (z_i, u_i) to points with |z_i| <= radius.

    Returns:
    z_r: Disk points with |z| <= radius.
    u_r: Corresponding complex lattice coordinates.
    squares_r: Corresponding subset of squares.
    index_map: Dictionary that maps original indices to filtered indices.
    """
    mask = np.abs(z_values) <= radius
    kept_indices = np.where(mask)[0]

    z_r = z_values[kept_indices]
    u_r = u_values[kept_indices]
    squares_r = [squares[i] for i in kept_indices]

    index_map = {old: new for new, old in enumerate(kept_indices)}

    return z_r, u_r, squares_r, index_map

def evaluate_polynomial(z, coeffs):
    """
    Evaluate a holomorphic polynomial map at points z:
    p(z) = a_0 + a_1 z + a_2 z^2 + ...

    Returns: Array of polynomial values at z.
    """
    return sum(c * z**k for k, c in enumerate(coeffs))

def fit_holomorphic_polynomial(z_values, u_values, degree):
    """
    Fit a holomorphic polynomial u(z) ≈ a_0 + a_1 z + ... + a_degree z^degree by least squares.

   
    Returns:    
    coeffs : Array of fitted complex coefficients [a_0, ..., a_degree]
    residual : L2 residual of the fit.
    """
    V = np.vander(z_values, N=degree + 1, increasing=True)
    coeffs, _, _, _ = la.lstsq(V, u_values)

    u_fit = V @ coeffs
    residual = la.norm(u_values - u_fit) / la.norm(u_values)

    return coeffs, residual

def discrete_disk_mapping(filename):
    """
    Compute the harmonic-measure based mapping from the unit disk to a lattice disk.

    Returns: dict containing: squares, boundary_coords, boundary_mids, boundary, boundary, start_idx, 
                z_values (on unit disk interior), u_values (on latttice), w (boundary pre-images on circle)
    """

    # Load disk
    squares, boundary_coords, boundary_mids, boundary = extract_disk_from_file(filename)

    # Build matrices
    Lambda = boundary_side_count_matrix(squares)
    Delta = graph_laplacian(squares)
    Delta_inv = inverse_laplacian(Delta)

    # Choose starting square
    start_idx = start_vertex(Delta_inv)

    # Harmonic measure
    square_probs = harmonic_measure_by_inverse(start_idx, Lambda, Delta_inv)
    edge_probs = probs_per_edge_ordered(squares, square_probs, boundary)

    # Normalisation for numerical safety
    edge_probs = np.array(edge_probs, dtype=float)
    edge_probs /= edge_probs.sum()

    # Boundary parametrization
    w = boundary_midpoint_preimages(edge_probs)

    # Solve Dirichlet problem
    B = construct_B(squares, boundary)
    z_values = solve_dirichlet_problem(Delta, B, w)

    # lattice image coordinates
    u_values = square_centers_complex(squares)

    return {
        "squares": squares,
        "boundary_coords": boundary_coords,
        "boundary_mids": boundary_mids,
        "boundary": boundary,
        "start_idx": start_idx,
        "z": z_values,
        "u": u_values,
        "w": w
    }
