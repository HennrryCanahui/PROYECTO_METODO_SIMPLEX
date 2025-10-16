import flet as ft
import numpy as np

class SimplexSolver:
    def __init__(self):
        self.tableau = None
        self.iterations = []
    
    def solve(self, c, A, b, maximize=True):
        """
        Resuelve el problema de programación lineal usando el método Simplex
        c: coeficientes de la función objetivo
        A: matriz de coeficientes de las restricciones
        b: términos independientes de las restricciones
        maximize: True para maximizar, False para minimizar
        """
        self.iterations = []
        
        # Convertir a minimización si es necesario
        if maximize:
            c = [-x for x in c]
        
        m, n = len(A), len(A[0])
        
        # Crear tableau inicial
        tableau = np.zeros((m + 1, n + m + 1))
        
        # Función objetivo
        tableau[0, :n] = c
        
        # Restricciones
        for i in range(m):
            tableau[i + 1, :n] = A[i]
            tableau[i + 1, n + i] = 1  # Variables de holgura
            tableau[i + 1, -1] = b[i]
        
        self.iterations.append(tableau.copy())
        
        # Método Simplex
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            # Verificar optimalidad
            if all(tableau[0, :-1] >= -1e-10):
                break
            
            # Encontrar columna pivote (variable entrante)
            col_pivot = np.argmin(tableau[0, :-1])
            
            # Encontrar fila pivote (variable saliente)
            ratios = []
            for i in range(1, m + 1):
                if tableau[i, col_pivot] > 1e-10:
                    ratios.append(tableau[i, -1] / tableau[i, col_pivot])
                else:
                    ratios.append(float('inf'))
            
            if all(r == float('inf') for r in ratios):
                return None, "Solución no acotada"
            
            row_pivot = ratios.index(min(ratios)) + 1
            
            # Operación de pivoteo
            pivot = tableau[row_pivot, col_pivot]
            tableau[row_pivot] = tableau[row_pivot] / pivot
            
            for i in range(m + 1):
                if i != row_pivot:
                    factor = tableau[i, col_pivot]
                    tableau[i] = tableau[i] - factor * tableau[row_pivot]
            
            self.iterations.append(tableau.copy())
            iteration += 1
        
        # Extraer solución
        solution = np.zeros(n)
        for j in range(n):
            col = tableau[:, j]
            if np.sum(col == 1) == 1 and np.sum(col == 0) == m:
                row = np.where(col == 1)[0][0]
                solution[j] = tableau[row, -1]
        
        optimal_value = -tableau[0, -1] if maximize else tableau[0, -1]
        
        return solution, optimal_value

def main(page: ft.Page):
    page.title = "Método Simplex - Solver"
    page.scroll = "adaptive"
    page.padding = 20
    
    # Variables de estado
    num_restrictions = 2
    restriction_rows = []
    result_text = ft.Text("", size=14, weight=ft.FontWeight.BOLD)
    iterations_column = ft.Column([], scroll="auto")
    
    # Campos de función objetivo
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
    
    def create_restriction_row(index):
        """Crea una fila de restricción"""
        a1 = ft.TextField(label=f"x₁", value="1" if index == 0 else "2", width=100)
        a2 = ft.TextField(label=f"x₂", value="2" if index == 0 else "1", width=100)
        comp = ft.Dropdown(
            label="",
            width=80,
            value="<=",
            options=[
                ft.dropdown.Option("<="),
                ft.dropdown.Option(">="),
                ft.dropdown.Option("=")
            ]
        )
        b = ft.TextField(label=f"b", value="8" if index == 0 else "6", width=100)
        
        return ft.Row([
            ft.Text(f"R{index + 1}:", size=16, weight=ft.FontWeight.BOLD),
            a1,
            ft.Text("x₁ +", size=16),
            a2,
            ft.Text("x₂", size=16),
            comp,
            b
        ]), (a1, a2, comp, b)
    
    # Contenedor de restricciones
    restrictions_container = ft.Column([])
    
    def update_restrictions():
        """Actualiza las restricciones mostradas"""
        restriction_rows.clear()
        restrictions_container.controls.clear()
        
        for i in range(num_restrictions):
            row, fields = create_restriction_row(i)
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
            # Obtener coeficientes de función objetivo
            c = [float(c1_field.value), float(c2_field.value)]
            
            # Obtener restricciones
            A = []
            b = []
            
            for fields in restriction_rows:
                a1, a2, comp, b_val = fields
                
                row = [float(a1.value), float(a2.value)]
                b_value = float(b_val.value)
                
                # Convertir >= a <= multiplicando por -1
                if comp.value == ">=":
                    row = [-x for x in row]
                    b_value = -b_value
                
                A.append(row)
                b.append(b_value)
            
            # Resolver
            solver = SimplexSolver()
            maximize = (opt_type.value == "max")
            solution, optimal = solver.solve(c, A, b, maximize)
            
            # Mostrar resultado
            if solution is None:
                result_text.value = f"❌ {optimal}"
                result_text.color = ft.Colors.RED
            else:
                result_text.value = f"✅ Solución óptima encontrada:\n"
                result_text.value += f"x₁ = {solution[0]:.4f}\n"
                result_text.value += f"x₂ = {solution[1]:.4f}\n"
                result_text.value += f"Valor óptimo Z = {optimal:.4f}"
                result_text.color = ft.Colors.GREEN
                
                # Mostrar iteraciones
                iterations_column.controls.clear()
                iterations_column.controls.append(
                    ft.Text("📊 Iteraciones del Simplex:", 
                           size=16, 
                           weight=ft.FontWeight.BOLD)
                )
                
                for idx, tableau in enumerate(solver.iterations):
                    iterations_column.controls.append(
                        ft.Text(f"\nIteración {idx}:", weight=ft.FontWeight.BOLD)
                    )
                    
                    tableau_str = "Tableau:\n"
                    for row in tableau:
                        tableau_str += "  " + "  ".join([f"{val:8.3f}" for val in row]) + "\n"
                    
                    iterations_column.controls.append(
                        ft.Container(
                            content=ft.Text(tableau_str, font_family="Courier New", size=12),
                            bgcolor=ft.Colors.GREY_300,
                            padding=10,
                            border_radius=5
                        )
                    )
            
            page.update()
            
        except ValueError as e:
            result_text.value = f"❌ Error: Ingrese valores numéricos válidos"
            result_text.color = ft.Colors.RED
            page.update()
        except Exception as e:
            result_text.value = f"❌ Error: {str(e)}"
            result_text.color = ft.Colors.RED
            page.update()
    
    # Inicializar restricciones
    update_restrictions()
    
    # Layout principal
    page.add(
        ft.Container(
            content=ft.Column([
                ft.Text("🎯 Método Simplex", 
                       size=28, 
                       weight=ft.FontWeight.BOLD,
                       color=ft.Colors.BLUE),
                
                ft.Divider(height=20),
                
                # Función objetivo
                ft.Container(
                    content=ft.Column([
                        ft.Text("Función Objetivo:", size=18, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            ft.Text("Z =", size=16),
                            c1_field,
                            ft.Text("x₁ +", size=16),
                            c2_field,
                            ft.Text("x₂", size=16)
                        ]),
                        opt_type
                    ]),
                    bgcolor=ft.Colors.BLUE_GREY_100,
                    padding=15,
                    border_radius=10
                ),
                
                ft.Divider(height=20),
                
                # Restricciones
                ft.Container(
                    content=ft.Column([
                        ft.Text("Restricciones:", size=18, weight=ft.FontWeight.BOLD),
                        restrictions_container,
                        ft.Row([
                            ft.ElevatedButton("➕ Agregar restricción", 
                                            on_click=add_restriction,
                                            icon=ft.Icons.ADD),
                            ft.ElevatedButton("➖ Quitar restricción", 
                                            on_click=remove_restriction,
                                            icon=ft.Icons.REMOVE_CIRCLE)
                        ])
                    ]),
                    bgcolor=ft.Colors.BLUE_GREY_100,
                    padding=15,
                    border_radius=10
                ),
                
                ft.Divider(height=20),
                
                # Botón resolver
                ft.ElevatedButton(
                    "🚀 Resolver",
                    on_click=solve_simplex,
                    style=ft.ButtonStyle(
                        color=ft.Colors.WHITE,
                        bgcolor=ft.Colors.BLUE
                    ),
                    height=50,
                    width=200
                ),
                
                ft.Divider(height=20),
                
                # Resultado
                ft.Container(
                    content=result_text,
                    bgcolor=ft.Colors.GREY_200,
                    padding=15,
                    border_radius=10
                ),
                
                # Iteraciones
                iterations_column
                
            ], scroll="auto"),
            padding=20
        )
    )

ft.app(target=main)