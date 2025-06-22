import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, solveset, Reals, Eq, lambdify, N

# --- Funciones de Modelado Económico y Optimización ---

def get_gain_function_sympy(demand_a, demand_b, unit_cost_variable, fixed_cost):
    """
    Define la función de ganancia G(p) = I(p) - C(p) usando SymPy.
    Q(p) = demand_a - demand_b * p
    C(Q) = fixed_cost + unit_cost_variable * Q
    """
    p = symbols('p') # Define 'p' como un símbolo para SymPy

    Q_p = demand_a - demand_b * p
    I_p = p * Q_p
    C_p = fixed_cost + unit_cost_variable * Q_p
    G_p = I_p - C_p
    
    return p, G_p

def find_optimal_price_and_gain(demand_a, demand_b, unit_cost_variable, fixed_cost):
    """
    Encuentra el precio óptimo que maximiza la ganancia usando derivadas con SymPy.
    Retorna el precio óptimo, la ganancia máxima y un mensaje de estado.
    """
    p, G_p = get_gain_function_sympy(demand_a, demand_b, unit_cost_variable, fixed_cost)

    first_derivative = diff(G_p, p)
    critical_points = solveset(Eq(first_derivative, 0), p, domain=Reals)

    optimal_price = None
    max_gain = None
    status_message = "No se pudo encontrar un precio óptimo válido."

    if critical_points:
        potential_optimal_prices = [N(point) for point in critical_points]

        for price_candidate in potential_optimal_prices:
            # Asegurarse de que el precio sea positivo y la cantidad demandada sea no negativa
            if price_candidate >= 0:
                Q_at_price = demand_a - demand_b * price_candidate
                if Q_at_price >= 0:
                    second_derivative = diff(G_p, p, 2)
                    second_derivative_value = second_derivative.subs(p, price_candidate)

                    # Si la segunda derivada es negativa, es un máximo local
                    # Usamos un pequeño umbral para la comparación con cero, por si hay problemas de flotantes
                    if second_derivative_value < -1e-9: # Es claramente negativo
                        optimal_price = price_candidate
                        max_gain = G_p.subs(p, optimal_price)
                        status_message = "Precio óptimo encontrado con éxito."
                        break
                    elif abs(second_derivative_value) < 1e-9: # Es aproximadamente cero
                        status_message = "Punto crítico encontrado, pero la segunda derivada es cero (posiblemente la función de ganancia es lineal o más compleja)."
                    else: # Es positivo
                        status_message = "Punto crítico encontrado, pero es un mínimo de ganancia. Por favor, revise sus parámetros de demanda y costo."
                else:
                    status_message = "El precio candidato resulta en una cantidad demandada negativa. Ajuste el rango de precios o los parámetros de demanda."
            else:
                status_message = "El precio candidato es negativo. Ajuste el rango de precios o los parámetros de demanda."

    return optimal_price, max_gain, status_message

# --- Interfaz de Usuario con Streamlit ---

st.set_page_config(
    page_title="Precix: Optimizador de Ganancias",
    page_icon="📈",
    layout="centered", # O 'wide' si quieres que ocupe todo el ancho
    initial_sidebar_state="expanded" # O 'collapsed'
)

st.title("📈 Precix: Optimizador de Ganancias")
st.markdown("---")

st.write(
    """
    Bienvenido a **Precix**, tu herramienta inteligente para encontrar el **precio de venta óptimo** que maximiza las ganancias de tu producto. 
    Simplemente ajusta los parámetros de demanda y costos, y Precix hará los cálculos por ti.
    """
)

# Sidebar para parámetros de entrada
st.sidebar.header("📊 Parámetros del Producto")
st.sidebar.markdown("Define cómo la **demanda** de tu producto varía con el precio y cuáles son tus **costos**.")

with st.sidebar:
    st.subheader("Demanda (Modelo: Q = a - b * Precio)")
    demand_a = st.number_input(
        "Coeficiente 'a' (Cantidad a precio cero)",
        min_value=0.1, value=1000.0, step=10.0, format="%.2f",
        help="La cantidad de unidades que se venderían si el precio fuera cero (intercepto en el eje de cantidad)."
    )
    demand_b = st.number_input(
        "Coeficiente 'b' (Sensibilidad al precio)",
        min_value=0.01, value=2.0, step=0.1, format="%.2f",
        help="Cuánto disminuye la demanda por cada unidad de aumento en el precio (pendiente de la curva de demanda)."
    )

    st.subheader("Costos (Modelo: Costo Total = Fijo + Variable * Q)")
    unit_cost_variable = st.number_input(
        "Costo Variable Unitario",
        min_value=0.0, value=50.0, step=5.0, format="%.2f",
        help="El costo de producir una sola unidad del producto (materia prima, mano de obra directa)."
    )
    fixed_cost = st.number_input(
        "Costo Fijo Total",
        min_value=0.0, value=10000.0, step=100.0, format="%.2f",
        help="Costos que no varían con la cantidad producida (alquiler, salarios administrativos)."
    )

st.markdown("---")

# Botón para calcular
if st.button("🚀 Calcular Precio Óptimo y Simular"):
    try:
        # Validaciones adicionales de entrada para evitar cálculos sin sentido
        if demand_a <= 0:
            st.error("El coeficiente 'a' de la demanda debe ser positivo para que haya alguna demanda.")
            st.stop() # Detiene la ejecución para que el usuario corrija

        if demand_b <= 0:
            st.error("El coeficiente 'b' de la demanda debe ser positivo para que la demanda disminuya con el precio (comportamiento normal).")
            st.stop()
            
        if unit_cost_variable < 0 or fixed_cost < 0:
            st.error("Los costos no pueden ser negativos. Por favor, ingrese valores válidos.")
            st.stop()


        # Calcular los resultados
        optimal_price, max_gain, status_message = find_optimal_price_and_gain(
            demand_a, demand_b, unit_cost_variable, fixed_cost
        )

        st.subheader("Resultado de la Optimización")
        if optimal_price is not None:
            st.success(f"**Precio Óptimo Sugerido:** **${optimal_price:,.2f}**")
            st.info(f"**Ganancia Máxima Estimada:** **${max_gain:,.2f}**")
            st.write(status_message)
        else:
            st.error(status_message)
            # No intentar graficar si no hay un precio óptimo válido
            st.stop()

        # Simular y Graficar la Función de Ganancia
        st.subheader("Simulación de Ganancia por Precio")
        p_sym, G_p_sym = get_gain_function_sympy(demand_a, demand_b, unit_cost_variable, fixed_cost)
        gain_func_numeric = lambdify(p_sym, G_p_sym, 'numpy')

        # Definir un rango de precios para la gráfica
        # Es importante que el rango incluya el precio óptimo y tenga sentido económico
        # El precio máximo de demanda (Q=0) es a/b. El precio mínimo para cubrir costos es unit_cost_variable.
        
        # Considerar el rango donde la cantidad demandada es positiva
        price_at_zero_demand = demand_a / demand_b if demand_b > 0 else optimal_price + 100 # Evitar división por cero
        
        # El inicio del rango debe ser al menos 0 o ligeramente por debajo del costo variable si es relevante
        min_price_plot = max(0.0, unit_cost_variable * 0.8) # Empezar un poco antes del costo variable
        
        # El final del rango debe extenderse un poco más allá del óptimo o del precio donde la demanda es cero
        max_price_plot = max(optimal_price * 1.5, price_at_zero_demand * 1.2) # Asegurar que cubre bien
        
        # Ajustar el rango para asegurar que el óptimo esté dentro y haya suficiente contexto
        if optimal_price < min_price_plot or optimal_price > max_price_plot:
             min_price_plot = max(0.0, optimal_price - (optimal_price * 0.5 + 20))
             max_price_plot = optimal_price + (optimal_price * 0.5 + 20)
        
        # Asegurarse de que min_price_plot no sea mayor que max_price_plot
        if min_price_plot >= max_price_plot:
            min_price_plot = max(0.0, optimal_price - 50)
            max_price_plot = optimal_price + 50
            if min_price_plot >= max_price_plot: # Ultimo recurso, asegurar un rango
                min_price_plot = 0.0
                max_price_plot = optimal_price * 2 + 10 if optimal_price > 0 else 100

        prices = np.linspace(min_price_plot, max_price_plot, 400)
        
        # --- Aplicación de la corrección aquí ---
        raw_gains = gain_func_numeric(prices)
        quantities_at_prices = demand_a - demand_b * prices
        valid_indices = quantities_at_prices >= 0
        gains = np.where(valid_indices, raw_gains, np.nan)

        # Manejo de casos donde no hay datos válidos para graficar
        if np.all(np.isnan(gains)) or np.any(np.isinf(gains)):
            st.warning("No se pudieron generar datos válidos para el gráfico en el rango de precios calculado. Intenta ajustar los parámetros o el rango de precios.")
            st.stop() # Detiene la ejecución de graficación si no hay datos válidos.
        # --- Fin de la corrección ---

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(prices, gains, label='Función de Ganancia', color='#1f77b4', linewidth=2)
        
        # Marcar el punto óptimo
        ax.scatter([optimal_price], [max_gain], color='red', s=150, zorder=5, label=f'Óptimo: ${optimal_price:,.2f} / G: ${max_gain:,.2f}')
        ax.axvline(x=optimal_price, color='gray', linestyle='--', linewidth=0.8)
        ax.axhline(y=max_gain, color='gray', linestyle=':', linewidth=0.8)

        ax.set_title('Ganancia en Función del Precio de Venta', fontsize=16)
        ax.set_xlabel('Precio de Venta ($)', fontsize=12)
        ax.set_ylabel('Ganancia ($)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        # Ajustar los límites del eje Y para que sean más legibles
        # Buscar el mínimo de ganancia solo en los puntos válidos
        valid_gains_min = np.nanmin(gains) if not np.all(np.isnan(gains)) else 0
        y_min_plot = min(0, valid_gains_min)
        y_max_plot = np.nanmax(gains) if not np.all(np.isnan(gains)) else (max_gain * 1.2 if max_gain is not None else 100)
        
        # Un pequeño padding para el eje Y
        y_range = y_max_plot - y_min_plot
        ax.set_ylim(bottom=y_min_plot - y_range * 0.1, top=y_max_plot + y_range * 0.1)

        st.pyplot(fig)
        st.markdown(
            """
            **Interpretación del Gráfico:**
            * La curva azul muestra cómo cambia la ganancia a medida que ajustas el precio de venta.
            * El punto rojo indica el **precio óptimo** donde la ganancia es máxima.
            * Si la curva desciende después del punto rojo, significa que aumentar el precio más allá del óptimo reduce la demanda y, por lo tanto, la ganancia.
            * Si la curva es baja o negativa antes del punto rojo, significa que el precio es demasiado bajo para cubrir los costos o no es suficientemente rentable.
            """
        )

    except ValueError:
        st.error("Por favor, asegúrate de ingresar números válidos en todos los campos.")
    except Exception as e:
        st.error(f"Ha ocurrido un error inesperado: {e}. Por favor, revise los parámetros de entrada.")

st.markdown("---")
st.markdown("Creado con Python, SymPy, NumPy, Matplotlib y Streamlit por tu equipo de Precix. ¡Optimiza tus precios inteligentemente!")