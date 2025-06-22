from sympy import diff, solveset, Reals, Eq, N
from economic_models import get_gain_function

def find_optimal_price(demand_a, demand_b, unit_cost_variable, fixed_cost):
    """
    Encuentra el precio óptimo que maximiza la ganancia usando derivadas.
    Retorna el precio óptimo, la ganancia máxima y un mensaje de estado.
    """
    p, G_p = get_gain_function(demand_a, demand_b, unit_cost_variable, fixed_cost)

    # 1. Calcular la primera derivada de la función de ganancia con respecto a 'p'
    first_derivative = diff(G_p, p)

    # 2. Resolver para 'p' donde la primera derivada es igual a cero
    # solveset busca las raíces de la ecuación en el conjunto de los números reales (Reals)
    critical_points = solveset(Eq(first_derivative, 0), p, domain=Reals)

    optimal_price = None
    max_gain = None
    status_message = "No se pudo encontrar un precio óptimo válido."

    if critical_points:
        # Convertir el conjunto de soluciones a una lista y tomar el primer (y esperado único) punto
        # N() se usa para obtener una aproximación numérica si la solución es simbólica/racional
        potential_optimal_prices = [N(point) for point in critical_points]

        for price_candidate in potential_optimal_prices:
            # Asegurarse de que el precio sea positivo y la cantidad demandada sea no negativa
            # (no tiene sentido vender a precio negativo o tener demanda negativa)
            if price_candidate >= 0:
                Q_at_price = demand_a - demand_b * price_candidate
                if Q_at_price >= 0:
                    # 3. Calcular la segunda derivada para verificar si es un máximo
                    second_derivative = diff(G_p, p, 2)
                    second_derivative_value = second_derivative.subs(p, price_candidate)

                    # Si la segunda derivada es negativa, es un máximo local
                    if second_derivative_value < 0:
                        optimal_price = price_candidate
                        max_gain = G_p.subs(p, optimal_price)
                        status_message = "Precio óptimo encontrado con éxito."
                        break # Encontramos un máximo válido, salimos del bucle
                    elif second_derivative_value == 0:
                        status_message = "Punto crítico encontrado, pero la segunda derivada es cero (posible punto de inflexión o más complejos)."
                    else:
                        status_message = "Punto crítico encontrado, pero es un mínimo de ganancia."
                else:
                    status_message = "El precio candidato resulta en una cantidad demandada negativa."
            else:
                status_message = "El precio candidato es negativo."

    return optimal_price, max_gain, status_message