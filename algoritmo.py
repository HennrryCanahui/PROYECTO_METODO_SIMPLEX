from fractions import Fraction
class NumberTypeclass:
    def zero(self): return 0
    def one(self): return 1
    def positive(self,x): return x > 0
    def iszero(self,x): return x == 0
    def nonnegative(self,x): return self.positive(x) or self.iszero(x)
    def coerce(self, x): return x
    def coerce_vec(self, x): return [self.coerce(xi) for xi in x]
    def coerce_mtx(self, x): return [self.coerce_vec(xi) for xi in x]

class RealFiniteTolerance(NumberTypeclass):
    def __init__(self, eps=1e-6):
        super(RealFiniteTolerance, self).__init__()
        self.eps = eps
        assert eps >= 0
        
    def zero(self): return 0.0
    def one(self): return 1.0
    def iszero(self,x): return abs(x) < self.eps
    def coerce(self, x): return float(x)

class RationalNumbers(NumberTypeclass):    
    def __init__(self):
        super(RationalNumbers, self).__init__()
        self._one = Fraction(1)
        self._zero = Fraction(0)
    def one(self): return self._one
    def zero(self): return self._zero
    def nonnegative(self,x): return x >= 0
    def coerce(self, x): return Fraction(x)

def _subtract_scaled_row(row1, row2, k, numclass):
    """row1 -= k*row2"""
    if numclass.iszero(k): return
    for i, row2_i in enumerate(row2):
        row1[i] -= k*row2_i

RESOLUTION_NO = "no"
RESOLUTION_SOLVED = "solved"
RESOLUTION_UNBOUNDED = "unbounded"
RESOLUTION_INCOMPATIBLE = "incompatible"

class SimplexSolver:
    def __init__(self, a, b, c, basis, numclass, clean_c_row=False):
        assert len(a) == len(b)
        for aj in a:
            assert len(aj) == len(c)
            
        self.numclass = numclass
        self.a = a
        self.b = b
        self.c = c
        self.basis = basis
        self.n = len(c)
        self.m = len(b)
        self.resolution = RESOLUTION_NO
        self.iterations = []  # Para guardar el historial
        if clean_c_row: self._diagonalize_c_row()
        self._validate_diagonzlized()
        
    def _save_iteration(self):
        """Guarda el estado actual del tableau"""
        self.iterations.append({
            'a': [row[:] for row in self.a],
            'b': self.b[:],
            'c': self.c[:],
            'basis': self.basis[:]
        })
        
    def _diagonalize_c_row(self):
        c = self.c
        for j, i in enumerate(self.basis):
            if not self.numclass.iszero(c[i]):
                _subtract_scaled_row(c, self.a[j], c[i], self.numclass)
                assert self.numclass.iszero(c[i])
                c[i] = self.numclass.zero()
        
    def vertex(self):
        v = [self.numclass.zero()] * self.n
        for i, val in zip(self.basis, self.b):
            v[i] = val
        return v
        
    def _validate_diagonzlized(self):
        for i in self.basis:
            assert self.numclass.iszero(self.c[i])
        for j, a_j in enumerate(self.a):
            for j1, i in enumerate(self.basis):
                if j1 == j:
                    assert self.numclass.iszero(a_j[i] - self.numclass.one())
                else:
                    if not self.numclass.iszero(a_j[i]): 
                        raise AssertionError(f"A: column {i} row {j} must be 0")
        for bi in self.b:
            assert self.numclass.nonnegative(bi)

    def _find_leading_column(self):
        imin = min(range(self.n), key=lambda i: self.c[i])
        if self.numclass.nonnegative(self.c[imin]):
            return None
        else:
            return imin
                        
    def step(self):
        self._save_iteration()  # Guardar antes del paso
        
        i_lead = self._find_leading_column()
        if i_lead is None:
            self.resolution = RESOLUTION_SOLVED
            return False

        assert i_lead not in self.basis

        best_ratio = None
        best_row = None

        for j, b_j in enumerate(self.b):
            a_ji = self.a[j][i_lead]
            if self.numclass.iszero(a_ji):
                continue
            if self.numclass.iszero(b_j):
                if not self.numclass.positive(a_ji):
                    continue
            ratio = b_j / a_ji
            if not self.numclass.nonnegative(ratio): continue

            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_row = j
                
        if best_row is None:
            self.resolution = RESOLUTION_UNBOUNDED
            return False
            
        self._diagonalize_by_row_col(best_row, i_lead)
        self.basis[best_row] = i_lead
        self._validate_diagonzlized()
        return True
    
    def _diagonalize_by_row_col(self, j, i):
        a_ji = self.a[j][i]
        assert not self.numclass.iszero(a_ji)
        
        self.b[j] /= a_ji
        aj = self.a[j]
        for i1 in range(self.n):
            if i1 != i:
                aj[i1] /= a_ji
            else:
                aj[i1] = self.numclass.one()
        
        _subtract_scaled_row(self.c, aj, self.c[i], self.numclass)
        self.c[i] = self.numclass.zero()

        for j1, a_j1 in enumerate(self.a):
            if j1 == j: continue
            k = a_j1[i]
            _subtract_scaled_row(a_j1, aj, k, self.numclass)
            assert self.numclass.iszero(a_j1[i]) 
            a_j1[i] = self.numclass.zero()
            self.b[j1] -= self.b[j] * k

def simplex_canonical_m(a, b, c, basis, num, verbose=False, do_coerce=True):
    if do_coerce:
        a = num.coerce_mtx(a)
        b = num.coerce_vec(b)
        c = num.coerce_vec(c)
        
    n_artificial = sum(int(bi is None) for bi in basis)
    n = len(c)
    if n_artificial == 0:
        solver = SimplexSolver(a, b, c, basis, numclass=num, clean_c_row=True)
        while solver.resolution == RESOLUTION_NO:
            solver.step()
        return solver.resolution, solver.vertex(), solver
    
    zeros = [num.zero()] * n_artificial
    a = [a_j + zeros for a_j in a]
    
    i_next = n
    m_basis = basis[:]
    for j, bi in enumerate(basis):
        if bi is None:
            a[j][i_next] = num.one()
            m_basis[j] = i_next
            i_next += 1
    
    cm = [num.zero()] * n + [num.one()] * n_artificial
    m_solver = SimplexSolver(a, b, cm, m_basis, numclass=num, clean_c_row=True)
    real_vertex_reached = False
    
    while m_solver.resolution == RESOLUTION_NO:
        m_solver.step()
        if all(bi < n for bi in m_solver.basis):
            real_vertex_reached = True
            break
        
    if not real_vertex_reached:
        return RESOLUTION_INCOMPATIBLE, None, m_solver
    
    a = [a_row[:n] for a_row in m_solver.a]
    solver = SimplexSolver(a, m_solver.b, c, m_solver.basis, numclass=num, clean_c_row=True)
    while solver.resolution == RESOLUTION_NO:
        solver.step()
    
    return solver.resolution, solver.vertex(), solver

def linsolve(objective, ineq_left=(), ineq_right=(), eq_left=(), eq_right=(),
             nonneg_variables=(), num=RealFiniteTolerance(), verbose=False, do_coerce=True):
    nonneg_variables = set(nonneg_variables)
    n = len(objective)
    next_variable = n
    negative_part2positive_part = {}

    for var in range(n):
        if var not in nonneg_variables:
            var_neg = next_variable
            next_variable += 1
            negative_part2positive_part[var_neg] = var

    n_nonneg = next_variable

    def positivise_row(row):
        return [row[i] if i not in negative_part2positive_part else -row[negative_part2positive_part[i]]
                for i in range(n_nonneg)]

    c_nonneg = positivise_row(objective)
    ineq_left_nonneg = list(map(positivise_row, ineq_left))
    eq_left_nonneg = list(map(positivise_row, eq_left))
    
    def negate_row(row): return [-ri for ri in row]

    num_inequalities = len(ineq_left)
    a_extended = []
    b_extended = []
    basis = []
    
    for a_row, bi in zip(ineq_left_nonneg, ineq_right):
        a_row_extended = a_row + [num.zero()] * num_inequalities
        artificial_var = next_variable
        next_variable += 1
        a_row_extended[artificial_var] = num.one()

        if not num.nonnegative(bi):
            bi = -bi
            a_row_extended = negate_row(a_row_extended)
            basis.append(None)
        else:
            basis.append(artificial_var)
        a_extended.append(a_row_extended)
        b_extended.append(bi)
        
    for a_row, bi in zip(eq_left_nonneg, eq_right):
        if not num.nonnegative(bi):
            bi = -bi
            a_row = negate_row(a_row)
        a_extended.append(a_row + [num.zero()] * num_inequalities)
        b_extended.append(bi)
        basis.append(None)

    resolution, solution, solver = simplex_canonical_m(
        a_extended, b_extended, c_nonneg + [num.zero()] * num_inequalities,
        basis, num=num, verbose=verbose, do_coerce=do_coerce
    )

    if resolution == RESOLUTION_SOLVED:
        orig_solution = solution[:n]
        for negative_var, positive_var in negative_part2positive_part.items():
            orig_solution[positive_var] -= solution[negative_var]
        return resolution, orig_solution, solver
    else:
        return resolution, solution, solver
