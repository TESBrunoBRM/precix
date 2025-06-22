from sympy import symbols

def get_gain_function(demand_a, demand_b, unit_cost_variable, fixed_cost):
    """
    Define la función de ganancia G(p) = I(p) - C(p).
    La función de demanda es Q(p) = demand_a - demand_b * p
    La función de costo total es C(Q) = fixed_cost + unit_cost_variable * Q
    """
    p = symbols('p') # Define 'p' como un símbolo para SymPy

    # 1. Función de Demanda
    # Q(p) = a - b*p
    # demand_a: cantidad demandada cuando el precio es 0 (intercepto en el eje Q)
    # demand_b: sensibilidad de la demanda al precio (pendiente)
    Q_p = demand_a - demand_b * p

    # Asegurarse de que la cantidad no sea negativa (no tiene sentido económico)
    # Aunque SymPy trabajará con la expresión, para evaluaciones reales, esto es importante
    # Sin embargo, para la derivación simbólica, la expresión matemática es suficiente.

    # 2. Función de Ingreso Total I(p) = p * Q(p)
    I_p = p * Q_p

    # 3. Función de Costo Total C(p) = Costo Fijo + Costo Variable * Q(p)
    C_p = fixed_cost + unit_cost_variable * Q_p

    # 4. Función de Ganancia G(p) = I(p) - C(p)
    G_p = I_p - C_p

    return p, G_p # Retorna el símbolo 'p' y la expresión de la ganancia