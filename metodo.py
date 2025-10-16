import flet as ft
import numpy as np

class SimplexSolver:
    def __init__(self):
        self.tableau = None
        self.iterations = []
    
    def solve(self, c, A, b, maximize=True):
        self.iterations = []

        # Trabajar con copias para no mutar los argumentos externos
        A = [list(row) for row in A]
        b = list(b)
        c = list(c)

        # Asegurar b >= 0: si b[i] < 0 invertimos la fila (esto es estándar)
        for i in range(len(b)):
            if b[i] < 0:
                A[i] = [-x for x in A[i]]
                b[i] = -b[i]

        # Convertir maximización a minimización internamente
        # Convención: la fila objetivo en el tableau será c (problema de minimización).
        # Si queremos maximizar, trabajamos con -c (equiv. convertir a minimización).
        if maximize:
            c_eval = [-ci for ci in c]
        else:
            c_eval = [ci for ci in c]

        m, n = len(A), len(A[0])

        # Construir tableau: filas = m+1, cols = n + m (slacks) + 1 (RHS)
        tableau = np.zeros((m + 1, n + m + 1), dtype=float)

        # Fila objetivo (fila 0) contiene los coeficientes c_eval sobre las variables originales
        tableau[0, :n] = c_eval
        # RHS de la fila objetivo inicialmente 0 (ya por defecto)

        # Restricciones: filas 1..m
        for i in range(m):
            tableau[i + 1, :n] = A[i]
            tableau[i + 1, n + i] = 1.0  # variable de holgura
            tableau[i + 1, -1] = b[i]

        self.iterations.append(tableau.copy())

        # Simplex iterativo (primal)
        iteration = 0
        max_iterations = 200
        tol = 1e-10

        while iteration < max_iterations:
            # condición de optimalidad: en minimización todos los coeficientes de la fila 0 >= 0
            if all(tableau[0, j] >= -tol for j in range(n + m)):
                break

            # elegir variable entrante: la más negativa en fila 0
            col_pivot = int(np.argmin(tableau[0, :n + m]))
            if tableau[0, col_pivot] >= -tol:
                break  # óptimo

            # test del cociente mínimo para fila pivote (solo donde coef > 0)
            ratios = []
            rows = []
            for i in range(1, m + 1):
                aij = tableau[i, col_pivot]
                if aij > tol:
                    ratio = tableau[i, -1] / aij
                    if ratio >= -tol:
                        ratios.append(ratio)
                        rows.append(i)

            if len(ratios) == 0:
                return None, "Solución no acotada"

            # seleccionar fila con menor ratio
            min_idx = int(np.argmin(ratios))
            row_pivot = rows[min_idx]

            # pivoteo
            pivot = tableau[row_pivot, col_pivot]
            if abs(pivot) < tol:
                return None, "Pivote cero inesperado"

            tableau[row_pivot, :] = tableau[row_pivot, :] / pivot

            for i in range(m + 1):
                if i == row_pivot:
                    continue
                factor = tableau[i, col_pivot]
                if abs(factor) > tol:
                    tableau[i, :] = tableau[i, :] - factor * tableau[row_pivot, :]

            self.iterations.append(tableau.copy())
            iteration += 1

        if iteration >= max_iterations:
            return None, "Se alcanzó el máximo de iteraciones"

        # Extraer solución: buscar columnas básicas entre las primeras n variables
        solution = np.zeros(n)
        rows_total = m + 1
        for j in range(n):
            col = tableau[:, j]
            # buscar índice del 1 (si existe) y verificar que el resto sean ~0
            ones = np.where(np.abs(col - 1.0) < 1e-8)[0]
            if len(ones) == 1:
                one_row = ones[0]
                # comprobar que el resto sean ~0
                other_indices = [r for r in range(rows_total) if r != one_row]
                if all(abs(col[r]) < 1e-8 for r in other_indices):
                    if one_row > 0:  # si corresponde a una fila de restricción
                        solution[j] = tableau[one_row, -1]
                    else:
                        solution[j] = 0.0
                else:
                    solution[j] = 0.0
            else:
                solution[j] = 0.0

        # Determinar valor óptimo:
        # Recordar: trabajamos con c_eval. Si maximize=True entonces c_eval = -c original.
        # El valor óptimo del problema (usando la convención) es -tableau[0,-1] si max, pero
        # con nuestra convención mejor razonada:
        # - Si maximize: original Z* = - tableau[0, -1]
        # - Si minimize: original Z* = tableau[0, -1]
        optimal_value = -tableau[0, -1] if maximize else tableau[0, -1]

        return solution, optimal_value

def main(page: ft.Page):
    page.title = "Método Simplex - Solver"
    page.scroll = "adaptive"
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT

    # Estado
    num_restrictions = 2
    restriction_rows = []  # lista de tuplas (a1_field, a2_field, comp, b_field)
    result_text = ft.Text("", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK)
    iterations_column = ft.Column([], scroll="auto")

    # Función objetivo
    c1_field = ft.TextField(label="Coeficiente x₁", value="3", width=150)
    c2_field = ft.TextField(label="Coeficiente x₂", value="5", width=150)

    # Tipo de optimización
    opt_type = ft.RadioGroup(
        content=ft.Row([
            ft.Radio(value="max", label="Maximizar"),
            ft.Radio(value="min", label="Minimizar")
        ]),
        value="max"
    )

    def create_restriction_row(index, defaults=None):
        """Crea una fila de restricción. `defaults` es (a1,a2,comp,b) si se quiere preservar valores."""
        default_a1 = defaults[0] if defaults else ("1" if index == 0 else "2")
        default_a2 = defaults[1] if defaults else ("2" if index == 0 else "1")
        default_comp = defaults[2] if defaults else "<="
        default_b = defaults[3] if defaults else ("8" if index == 0 else "6")

        a1 = ft.TextField(label=f"x₁", value=str(default_a1), width=100)
        a2 = ft.TextField(label=f"x₂", value=str(default_a2), width=100)
        comp = ft.Dropdown(
            label="",
            width=80,
            value=default_comp,
            options=[
                ft.dropdown.Option("<="),
                ft.dropdown.Option(">="),
                ft.dropdown.Option("=")
            ]
        )
        b = ft.TextField(label=f"b", value=str(default_b), width=100)

        row = ft.Row([
            ft.Text(f"R{index + 1}:", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK),
            a1,
            ft.Text("x₁ +", size=16, color=ft.Colors.BLACK),
            a2,
            ft.Text("x₂", size=16, color=ft.Colors.BLACK),
            comp,
            b
        ])
        return row, (a1, a2, comp, b)

    # Contenedor
    restrictions_container = ft.Column([])

    def update_restrictions():
        """Reconstruye la UI de restricciones preservando los valores previos."""
        nonlocal restriction_rows
        # Guardar valores actuales como defaults
        current_values = []
        for fields in restriction_rows:
            a1, a2, comp, b = fields
            try:
                current_values.append((a1.value, a2.value, comp.value, b.value))
            except Exception:
                current_values.append((None, None, "<=", None))

        # Limpiar
        restriction_rows.clear()
        restrictions_container.controls.clear()

        # Recrear filas usando defaults cuando existan
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
            c = [float(c1_field.value), float(c2_field.value)]

            # Leer restricciones
            A = []
            b = []
            for fields in restriction_rows:
                a1, a2, comp, b_val = fields
                row = [float(a1.value), float(a2.value)]
                b_value = float(b_val.value)

                # convertir >= a <= multiplicando por -1
                if comp.value == ">=":
                    row = [-x for x in row]
                    b_value = -b_value
                elif comp.value == "=":
                    # Para igualdad lo dejamos tal cual (no estamos añadiendo variables artificiales aquí).
                    pass

                A.append(row)
                b.append(b_value)

            solver = SimplexSolver()
            maximize = (opt_type.value == "max")
            solution, optimal = solver.solve(c, A, b, maximize)

            if solution is None:
                result_text.value = f"❌ {optimal}"
                result_text.color = ft.Colors.RED
                iterations_column.controls.clear()
            else:
                result_text.value = f"✅ Solución óptima encontrada:\n\n"
                result_text.value += f"x₁ = {solution[0]:.4f}\n"
                result_text.value += f"x₂ = {solution[1]:.4f}\n\n"
                result_text.value += f"Valor óptimo Z = {optimal:.4f}"
                result_text.color = ft.Colors.GREEN_900

                # Mostrar iteraciones
                iterations_column.controls.append(
                    ft.Text("📊 Iteraciones del Simplex:", 
                           size=16, 
                           weight=ft.FontWeight.BOLD,
                           color=ft.Colors.BLACK)
                )

                for idx, tableau in enumerate(solver.iterations):
                    iterations_column.controls.append(
                        ft.Text(f"\nIteración {idx}:", 
                               weight=ft.FontWeight.BOLD,
                               color=ft.Colors.BLACK)
                    )

                    tableau_str = ""
                    rows, cols = tableau.shape

                    # encabezados
                    headers = []
                    for j in range(cols - 1):
                        if j < len(c):
                            headers.append(f"x{j+1}")
                        else:
                            headers.append(f"s{j - len(c) + 1}")
                    headers.append("RHS")

                    tableau_str += "     " + "".join([f"{h:>10}" for h in headers]) + "\n"
                    tableau_str += "     " + "-" * (10 * len(headers)) + "\n"

                    for i in range(rows):
                        row_label = "Z  " if i == 0 else f"R{i} "
                        tableau_str += row_label + " "
                        for j in range(cols):
                            tableau_str += f"{tableau[i, j]:>10.3f}"
                        tableau_str += "\n"

                    iterations_column.controls.append(
                        ft.Container(
                            content=ft.Text(
                                tableau_str,
                                font_family="Courier New",
                                size=12,
                                color=ft.Colors.BLACK
                            ),
                            bgcolor=ft.Colors.GREY_100,
                            padding=10,
                            border_radius=5,
                            border=ft.border.all(1, ft.Colors.GREY_400)
                        )
                    )

            page.update()

        except ValueError:
            result_text.value = "❌ Error: Ingrese valores numéricos válidos"
            result_text.color = ft.Colors.RED
            iterations_column.controls.clear()
            page.update()
        except Exception as ex:
            result_text.value = f"❌ Error: {str(ex)}"
            result_text.color = ft.Colors.RED
            iterations_column.controls.clear()
            page.update()

    # Inicializar restricciones (conserva valores al agregar/quitar)
    update_restrictions()

    # Layout principal
    page.add(
        ft.Container(
            content=ft.Column([
                ft.Text("🎯 Método Simplex", 
                       size=28, 
                       weight=ft.FontWeight.BOLD,
                       color=ft.Colors.BLUE_900),

                ft.Divider(height=20, color=ft.Colors.GREY_400),

                ft.Container(
                    content=ft.Column([
                        ft.Text("Función Objetivo:", 
                               size=18, 
                               weight=ft.FontWeight.BOLD,
                               color=ft.Colors.BLACK),
                        ft.Row([
                            ft.Text("Z =", size=16, color=ft.Colors.BLACK),
                            c1_field,
                            ft.Text("x₁ +", size=16, color=ft.Colors.BLACK),
                            c2_field,
                            ft.Text("x₂", size=16, color=ft.Colors.BLACK)
                        ]),
                        opt_type
                    ]),
                    bgcolor=ft.Colors.BLUE_50,
                    padding=15,
                    border_radius=10,
                    border=ft.border.all(2, ft.Colors.BLUE_200)
                ),

                ft.Divider(height=20, color=ft.Colors.GREY_400),

                ft.Container(
                    content=ft.Column([
                        ft.Text("Restricciones:", 
                               size=18, 
                               weight=ft.FontWeight.BOLD,
                               color=ft.Colors.BLACK),
                        restrictions_container,
                        ft.Row([
                            ft.ElevatedButton("➕ Agregar restricción", 
                                            on_click=add_restriction,
                                            icon=ft.Icons.ADD,
                                            bgcolor=ft.Colors.GREEN_400,
                                            color=ft.Colors.WHITE),
                            ft.ElevatedButton("➖ Quitar restricción", 
                                            on_click=remove_restriction,
                                            icon=ft.Icons.REMOVE,
                                            bgcolor=ft.Colors.RED_400,
                                            color=ft.Colors.WHITE)
                        ])
                    ]),
                    bgcolor=ft.Colors.AMBER_50,
                    padding=15,
                    border_radius=10,
                    border=ft.border.all(2, ft.Colors.AMBER_200)
                ),

                ft.Divider(height=20, color=ft.Colors.GREY_400),

                ft.ElevatedButton(
                    "🚀 Resolver",
                    on_click=solve_simplex,
                    bgcolor=ft.Colors.BLUE_700,
                    color=ft.Colors.WHITE,
                    height=50,
                    width=200
                ),

                ft.Divider(height=20, color=ft.Colors.GREY_400),

                ft.Container(
                    content=result_text,
                    bgcolor=ft.Colors.GREEN_50,
                    padding=15,
                    border_radius=10,
                    border=ft.border.all(2, ft.Colors.GREEN_200)
                ),

                iterations_column

            ], scroll="auto"),
            padding=20
        )
    )

ft.app(target=main)
