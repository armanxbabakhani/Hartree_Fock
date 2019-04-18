import os
import numpy as np
from scipy.linalg import expm
from scipy.linalg import eig
from scipy.linalg import sqrtm
from scipy.special import erf
from pandas import *

# =========================================== Notes on Atomic Units ========================================== #
# Note: Throughout the calculations, atomic units are assumed

# ==================================== STO-3G Calculation of H2 molecule ====================================== #
pi = np.pi
Z_nuclear = 1
number_of_orbitals = 2
number_of_orbitals_per_orbital = 3

atomic_distance = 1.4
atom_1_pos = np.array([0.0,0.0,0.0])
atom_2_pos = np.array([0.0,0.0,atomic_distance])
# ---------------------------------------------- Orbital parameters -------------------------------------------
# ::::::::----for STO_3G----:::::::
orb_factors = np.array([0.168856, 0.623913, 3.42525])				# ------ Gaussian exponents for a single slater
orb_norms = np.array([0.444635, 0.535328, 0.154329])				# ------ Gaussian normalization factor for a single slater : psi_slater = n1*psi_g1 + n2*psi_g2 + n3*psi_g3


def norm_const(a):
	return (2*a/pi)**0.75

def overlaps(m, n, orb_dist):
	if orb_dist > 1e-5:
		gauss_shift = -1*m*n*orb_dist**2/(m+n)
		return ((4*m*n/(m+n)**2)**0.75)*np.exp(gauss_shift)
	else:
		return ((4*m*n/(m+n)**2)**0.75)

def T_kin(m,n, orb_dist):
	# A and B are pos of the neucli
	gauss_shift = -1*m*n*orb_dist**2/(m+n)
	return (2**1.5)*(m*n)**(7.0/4.0)/(m+n)**2.5 *(3+2*gauss_shift)*np.exp(gauss_shift)

def V_core(m, n, orb_dist, reduced_nuc_dist): 
	# reduced_dist = R_p - R_c, where R_p is the reduced Gaussian position
	gauss_shift = -1*m*n*orb_dist**2/(m+n)
	par = (m+n)*reduced_nuc_dist**2
	normalization = norm_const(m)*norm_const(n)
	if par < 1e-8:
		return -1*normalization*(2*pi/(m+n))*Z_nuclear*np.exp(gauss_shift)
	else:
		return -1*normalization*(2*pi/(m+n))*Z_nuclear*np.exp(gauss_shift)*(0.5*(pi/par)**0.5*erf(par**0.5))
		# return (-2.0/pi)*(4*m*n)**0.75/(m+n)**2.5 * Z_nuclear*np.exp(gauss_shift)*(0.5*(pi/par)**0.5*erf(par**0.5))

def H_core(m, n, orb_pos1, orb_pos2, atom_pos1, atom_pos2):
	reduced_pos = (m*orb_pos1 + n*orb_pos2)/(m+n)
	reduced_nuc_dist_1 = np.linalg.norm(reduced_pos - atom_pos1)
	reduced_nuc_dist_2 = np.linalg.norm(reduced_pos - atom_pos2)
	orb_dist = np.linalg.norm(orb_pos1 - orb_pos2)
	T = T_kin(m, n, orb_dist)
	V_1 = V_core(m, n, orb_dist, reduced_nuc_dist_1)
	V_2 = V_core(m, n, orb_dist, reduced_nuc_dist_2)
	return T + V_1 + V_2

def two_body_integrals(m, n, p, q, orb_pos1, orb_pos2, orb_pos3, orb_pos4): #red_orb1_orb2_dist is the distance between the reduced orbitals! |R_p - R_Q|
	orb_dist12 = np.linalg.norm(orb_pos1 - orb_pos2)
	orb_dist34 = np.linalg.norm(orb_pos3 - orb_pos4)
	orb_reduced12 = (m*orb_pos1 + n*orb_pos2)/(m+n)
	orb_reduced34 = (p*orb_pos3 + q*orb_pos4)/(p+q)
	orb_red_dist = np.linalg.norm(orb_reduced12 - orb_reduced34)
	if orb_red_dist < 1e-8:
		return 0.0
	else:
		gauss_shift12 = -1*m*n*orb_dist12**2/(m+n)
		gauss_shift34 = -1*p*q*orb_dist34**2/(p+q)
		gauss_shift = gauss_shift12 + gauss_shift34 
		par = (m+n)*(p+q)/((m+n)*(p+q)*(m+n+p+q)**0.5)*orb_red_dist**2
		normalization = norm_const(m)*norm_const(n)*norm_const(p)*norm_const(q)
		return normalization*(2*pi**(2.5)/((m+n)*(p+q)*(m+n+p+q)**0.5)*np.exp(gauss_shift)*(0.5*(pi/par)**0.5*erf(par**0.5)))

def display_matrix(name, matrix):
	print '******************* '+ name + ' *******************'
	print DataFrame(matrix)
	print ' '

# ==================================== [1] Problem Matrices ===================================== #
# ------------------------------------ [1.1] Wavefunction Overlaps---------------------------------
# ------------------ Diagonal terms
S_fac = np.array([[overlaps(m,n,0.0) for n in orb_factors] for m in orb_factors])
S_norm = np.outer(orb_norms, orb_norms)
S_matrix_diag = sum(sum(S_fac*S_norm))
# ------------------ off-Diagonal terms
S_fac = np.array([[overlaps(m,n,atomic_distance) for n in orb_factors] for m in orb_factors])
S_norm = np.outer(orb_norms, orb_norms)
S_matrix_off = sum(sum(S_fac*S_norm))

# Note: One could save time in calculation of the off-diagonal terms for diatomic (pure element) calculations 
#       by just multiplying out the diagonal terms by the Gaussian shifts
S_matrix = S_matrix_diag*np.eye(number_of_orbitals) + S_matrix_off*np.array([[0.0, 1.0],[1.0, 0.0]])
display_matrix('Overlap', S_matrix)
# ------------------------------------ [1.2] The X_rotation matrix --------------------------
X_matrix = np.linalg.inv(sqrtm(S_matrix))
X_matrix_d = np.conjugate(np.transpose(X_matrix))

# ------------------------------------ [1.3] Core Hamiltonian -------------------------------------
# ------------------ Diagonal terms
H_fac = np.array([[H_core(m, n, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), atom_1_pos, atom_2_pos) for n in orb_factors] for m in orb_factors])
H_norm = np.outer(orb_norms, orb_norms)
H_c_diag = sum(sum(H_fac*H_norm))

# ------------------ off-Diagonal terms
H_fac = np.array([[H_core(m, n, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,atomic_distance]), atom_1_pos, atom_2_pos) for n in orb_factors] for m in orb_factors])
H_norm = np.outer(orb_norms, orb_norms)
H_c_off = sum(sum(H_fac*H_norm))

H_c_matrix = H_c_diag*np.eye(number_of_orbitals) + np.multiply(H_c_off,np.array([[0.0, 1.0],[1.0, 0.0]]))
display_matrix('Hamiltonian core', H_c_matrix)
# ------------------------------------ [1.4] Two-body Integrals -----------------------------------
orb_positions = np.array([[0.0,0.0,0.0], [0.0,0.0,atomic_distance]])
two_body_norm = np.outer(orb_norms, orb_norms)
two_body_norms = np.array([[two_body_norm[i,j]*two_body_norm for i in range(len(two_body_norm))] for j in range(len(two_body_norm))])
two_body_matrix = np.zeros((((number_of_orbitals, number_of_orbitals, number_of_orbitals, number_of_orbitals))))
for n_pos in range(2):
	for m_pos in range(2):
		for p_pos in range(2):
			for q_pos in range(2):
				two_body_term = np.array([[[[two_body_integrals(n, m, p, q, orb_positions[n_pos], orb_positions[m_pos], orb_positions[p_pos], orb_positions[q_pos]) for n in orb_factors] for m in orb_factors]for p in orb_factors]for q in orb_factors])
				two_body_terms = two_body_term*two_body_norms
				two_body_matrix[n_pos, m_pos, p_pos, q_pos] = sum(sum(sum(sum(two_body_terms))))


# ==================================== [2] Computation ========================================== #
# ------------------------------------ [2.0] Guessing the initial occupations ---------------------
P_comp = np.eye(number_of_orbitals, dtype = complex)
P_matrix = np.zeros((number_of_orbitals, number_of_orbitals), dtype = complex)
while np.linalg.norm(P_comp - P_matrix) > 1e-2:
	P_matrix = np.copy(P_comp)
	# ------------------------------------ [2.1] The G matrix -----------------------------------------
	G_matrix = np.zeros((number_of_orbitals, number_of_orbitals), dtype = complex)
	for i in range(number_of_orbitals):
		for j in range(number_of_orbitals):
			tbs_matrix = np.array([[P_matrix[m,n]*(two_body_matrix[i,j,n,m] - 0.5*two_body_matrix[i,m,n,j]) for m in range(number_of_orbitals)] for n in range(number_of_orbitals)])
			G_matrix[i,j] = sum(sum(tbs_matrix))

	# ------------------------------------ [2.2] The Fock matrix --------------------------------------
	Fock_matrix = H_c_matrix + G_matrix
	Fock_mat = np.dot(X_matrix_d, np.dot(Fock_matrix, X_matrix))

	# ------------------------------------ [2.3] Calculating occupations and Energy -------------------
	Energies, C_mat = eig(Fock_mat)
	C_matrix = np.dot(X_matrix, C_mat)
	C_matirx_conj = np.conjugate(C_matrix)

	# ------------------------------------ [2.4] Forming final P matrix (density) ---------------------
	for i in range(number_of_orbitals):
		for j in range(number_of_orbitals):
			P_comp[i, j] = 2*sum([C_matrix[i, a]*C_matirx_conj[j,a] for a in range(int(number_of_orbitals/2.0))])
print '**===================================================================**'
print '   The Final Results of Hartree Fock Calculation for H2 molecule are: \n'
display_matrix('Density', P_comp)
display_matrix('Occupation', C_matrix)
display_matrix('Energies', Energies)