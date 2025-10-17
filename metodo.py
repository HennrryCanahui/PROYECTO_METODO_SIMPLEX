import flet as ft
from algoritmo import *

def main(page: ft.Page):
    page.title = "Método Simplex - Solver Profesional"
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT

    num_restrictions = 2
    restriction_rows = []
    result_text = ft.Text("", size=14, weight=ft.FontWeight.BOLD)
    iterations_column = ft.Column([])

    c1_field = ft.TextField(label="Coeficiente x₁", value="3", width=120)
    c2_field = ft.TextField(label="Coeficiente x₂", value="5", width=120)
    
    # Dropdown para seleccionar signo entre x₁ y x₂
    sign_dropdown = ft.Dropdown(
        label="",
        width=90,
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

                result_text.value = f"Solución óptima encontrada:\n\n"
                result_text.value += f"x₁ = {solution[0]:.2f}\n"
                result_text.value += f"x₂ = {solution[1]:.2f}\n\n"
                result_text.value += f"Valor óptimo Z = {optimal_value:.2f}"
                result_text.color = ft.Colors.GREEN_900

                # Mostrar iteraciones
                if hasattr(solver, 'iterations') and solver.iterations:
                    iterations_column.controls.append(
                        ft.Text(" Iteraciones del Simplex:", 
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
                            ft.ElevatedButton("Agregar restricción", 
                                            on_click=add_restriction, icon=ft.Icons.ADD,
                                            bgcolor=ft.Colors.GREEN_400, color=ft.Colors.WHITE),
                            ft.ElevatedButton("Quitar restricción", 
                                            on_click=remove_restriction, icon=ft.Icons.REMOVE,
                                            bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE)
                        ])
                    ]),
                    bgcolor=ft.Colors.AMBER_50, padding=15, border_radius=10,
                    border=ft.border.all(2, ft.Colors.AMBER_200)
                ),

                ft.Divider(height=20, color=ft.Colors.GREY_400),

                ft.ElevatedButton("Resolver", on_click=solve_simplex,
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