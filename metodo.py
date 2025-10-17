import flet as ft
from fractions import Fraction

# ==================== CLASES DEL SOLVER PROFESIONAL ====================

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

# ==================== INTERFAZ FLET ====================

def main(page: ft.Page):
    page.title = "Método Simplex - Solver Profesional"
    page.scroll = "adaptive"
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT

    num_restrictions = 2
    restriction_rows = []
    result_text = ft.Text("", size=14, weight=ft.FontWeight.BOLD)
    iterations_column = ft.Column([], scroll="auto")

    c1_field = ft.TextField(label="Coeficiente x₁", value="3", width=120)
    c2_field = ft.TextField(label="Coeficiente x₂", value="5", width=120)
    
    # Dropdown para seleccionar signo entre x₁ y x₂
    sign_dropdown = ft.Dropdown(
        label="",
        width=80,
        value="+",
        options=[
            ft.dropdown.Option("+", "➕"),
            ft.dropdown.Option("-", "➖")
        ]
    )

    opt_type = ft.RadioGroup(
        content=ft.Row([
            ft.Radio(value="max", label="Maximizar"),
            ft.Radio(value="min", label="Minimizar")
        ]),
        value="max"
    )

    def create_restriction_row(index, defaults=None):
        default_a1 = defaults[0] if defaults else ("1" if index == 0 else "2")
        default_a2 = defaults[1] if defaults else ("2" if index == 0 else "1")
        default_comp = defaults[2] if defaults else "<="
        default_b = defaults[3] if defaults else ("8" if index == 0 else "6")

        a1 = ft.TextField(label=f"x₁", value=str(default_a1), width=100)
        a2 = ft.TextField(label=f"x₂", value=str(default_a2), width=100)
        comp = ft.Dropdown(
            label="", width=80, value=default_comp,
            options=[
                ft.dropdown.Option("<="),
                ft.dropdown.Option(">="),
                ft.dropdown.Option("=")
            ]
        )
        b = ft.TextField(label=f"b", value=str(default_b), width=100)

        row = ft.Row([
            ft.Text(f"R{index + 1}:", size=16, weight=ft.FontWeight.BOLD),
            a1, ft.Text("x₁ +", size=16), a2, ft.Text("x₂", size=16), comp, b
        ])
        return row, (a1, a2, comp, b)

    restrictions_container = ft.Column([])

    def update_restrictions():
        nonlocal restriction_rows
        current_values = []
        for fields in restriction_rows:
            a1, a2, comp, b = fields
            try:
                current_values.append((a1.value, a2.value, comp.value, b.value))
            except:
                current_values.append((None, None, "<=", None))

        restriction_rows.clear()
        restrictions_container.controls.clear()

        for i in range(num_restrictions):
            defaults = current_values[i] if i < len(current_values) else None
            row, fields = create_restriction_row(i, defaults)
            restrictions_container.controls.append(row)
            restriction_rows.append(fields)
        page.update()

    def add_restriction(e):
        nonlocal num_restrictions
        num_restrictions += 1
        update_restrictions()

    def remove_restriction(e):
        nonlocal num_restrictions
        if num_restrictions > 1:
            num_restrictions -= 1
            update_restrictions()

    def solve_simplex(e):
        try:
            iterations_column.controls.clear()

            # Leer función objetivo
            c1_val = float(c1_field.value)
            c2_val = float(c2_field.value)
            
            # Aplicar el signo seleccionado al segundo coeficiente
            if sign_dropdown.value == "-":
                c2_val = -c2_val
            
            c_original = [c1_val, c2_val]
            maximize = (opt_type.value == "max")
            
            # El solver SIEMPRE minimiza, así que:
            # - Si queremos MAXIMIZAR: minimizamos -c
            # - Si queremos MINIMIZAR: minimizamos c directamente
            if maximize:
                c = [-ci for ci in c_original]
            else:
                c = c_original[:]

            # Leer restricciones
            ineq_left = []
            ineq_right = []
            eq_left = []
            eq_right = []

            for fields in restriction_rows:
                a1, a2, comp, b_val = fields
                row = [float(a1.value), float(a2.value)]
                b_value = float(b_val.value)

                if comp.value == "<=":
                    ineq_left.append(row)
                    ineq_right.append(b_value)
                elif comp.value == ">=":
                    # Convertir >= a <= negando
                    ineq_left.append([-x for x in row])
                    ineq_right.append(-b_value)
                else:  # =
                    eq_left.append(row)
                    eq_right.append(b_value)

            # Resolver con el solver profesional
            num = RealFiniteTolerance()
            resolution, solution, solver = linsolve(
                objective=c,
                ineq_left=ineq_left,
                ineq_right=ineq_right,
                eq_left=eq_left,
                eq_right=eq_right,
                nonneg_variables=(0, 1),  # x₁ y x₂ son no negativas
                num=num,
                verbose=False
            )

            if resolution == RESOLUTION_SOLVED:
                # Calcular valor óptimo
                # c ya tiene los coeficientes correctos (negados si maximize)
                optimal_value = sum(c[i] * solution[i] for i in range(len(solution)))
                
                # Si maximizamos, el valor óptimo real es el negativo
                if maximize:
                    optimal_value = -optimal_value

                result_text.value = f"✅ Solución óptima encontrada:\n\n"
                result_text.value += f"x₁ = {solution[0]:.6f}\n"
                result_text.value += f"x₂ = {solution[1]:.6f}\n\n"
                result_text.value += f"Valor óptimo Z = {optimal_value:.6f}"
                result_text.color = ft.Colors.GREEN_900

                # Mostrar iteraciones
                if hasattr(solver, 'iterations') and solver.iterations:
                    iterations_column.controls.append(
                        ft.Text("📊 Iteraciones del Simplex:", 
                               size=16, weight=ft.FontWeight.BOLD)
                    )

                    for idx, iteration in enumerate(solver.iterations):
                        iterations_column.controls.append(
                            ft.Text(f"\nIteración {idx}:", weight=ft.FontWeight.BOLD)
                        )

                        tableau_str = "Tableau:\n"
                        a = iteration['a']
                        b = iteration['b']
                        c_iter = iteration['c']
                        basis = iteration['basis']

                        # Encabezados
                        n_vars = len(c_iter)
                        headers = [f"x{i+1}" for i in range(n_vars)] + ["RHS"]
                        tableau_str += "     " + "".join([f"{h:>10}" for h in headers]) + "\n"
                        tableau_str += "     " + "-" * (10 * len(headers)) + "\n"

                        # Fila Z
                        tableau_str += "Z    "
                        for val in c_iter:
                            tableau_str += f"{float(val):>10.3f}"
                        tableau_str += f"{0:>10.3f}\n"

                        # Filas de restricciones
                        for i, row in enumerate(a):
                            tableau_str += f"R{i+1}   "
                            for val in row:
                                tableau_str += f"{float(val):>10.3f}"
                            tableau_str += f"{float(b[i]):>10.3f}\n"

                        tableau_str += f"\nBase: {[f'x{i+1}' for i in basis]}"

                        iterations_column.controls.append(
                            ft.Container(
                                content=ft.Text(tableau_str, font_family="Courier New", size=12),
                                bgcolor=ft.Colors.GREY_100,
                                padding=10,
                                border_radius=5,
                                border=ft.border.all(1, ft.Colors.GREY_400)
                            )
                        )

            elif resolution == RESOLUTION_UNBOUNDED:
                result_text.value = "❌ El problema no está acotado (solución infinita)"
                result_text.color = ft.Colors.RED
            elif resolution == RESOLUTION_INCOMPATIBLE:
                result_text.value = "❌ El problema no tiene solución factible"
                result_text.color = ft.Colors.RED
            else:
                result_text.value = "❌ No se pudo resolver el problema"
                result_text.color = ft.Colors.RED

            page.update()

        except ValueError as ve:
            result_text.value = f"❌ Error: Ingrese valores numéricos válidos\n{str(ve)}"
            result_text.color = ft.Colors.RED
            iterations_column.controls.clear()
            page.update()
        except Exception as ex:
            result_text.value = f"❌ Error: {str(ex)}"
            result_text.color = ft.Colors.RED
            iterations_column.controls.clear()
            page.update()

    update_restrictions()

    page.add(
        ft.Container(
            content=ft.Column([
                ft.Text("🎯 Método Simplex Profesional", 
                       size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_900),
                ft.Divider(height=20, color=ft.Colors.GREY_400),

                ft.Container(
                    content=ft.Column([
                        ft.Text("Función Objetivo:", size=18, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            ft.Text("Z =", size=16), 
                            c1_field,
                            ft.Text("x₁", size=16),
                            sign_dropdown,
                            c2_field,
                            ft.Text("x₂", size=16)
                        ]),
                        opt_type
                    ]),
                    bgcolor=ft.Colors.BLUE_50, padding=15, border_radius=10,
                    border=ft.border.all(2, ft.Colors.BLUE_200)
                ),

                ft.Divider(height=20, color=ft.Colors.GREY_400),

                ft.Container(
                    content=ft.Column([
                        ft.Text("Restricciones:", size=18, weight=ft.FontWeight.BOLD),
                        restrictions_container,
                        ft.Row([
                            ft.ElevatedButton("➕ Agregar restricción", 
                                            on_click=add_restriction, icon=ft.Icons.ADD,
                                            bgcolor=ft.Colors.GREEN_400, color=ft.Colors.WHITE),
                            ft.ElevatedButton("➖ Quitar restricción", 
                                            on_click=remove_restriction, icon=ft.Icons.REMOVE,
                                            bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE)
                        ])
                    ]),
                    bgcolor=ft.Colors.AMBER_50, padding=15, border_radius=10,
                    border=ft.border.all(2, ft.Colors.AMBER_200)
                ),

                ft.Divider(height=20, color=ft.Colors.GREY_400),

                ft.ElevatedButton("🚀 Resolver", on_click=solve_simplex,
                                bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE,
                                height=50, width=200),

                ft.Divider(height=20, color=ft.Colors.GREY_400),

                ft.Container(content=result_text, bgcolor=ft.Colors.GREEN_50,
                           padding=15, border_radius=10,
                           border=ft.border.all(2, ft.Colors.GREEN_200)),

                iterations_column

            ], scroll="auto"),
            padding=20
        )
    )

ft.app(target=main)