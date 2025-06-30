import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, solveset, Reals, Eq, lambdify, N, sympify, sqrt

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
            if price_candidate >= -1e-9: # Considerar 0 o muy ligeramente negativo como válido
                Q_at_price = demand_a - demand_b * price_candidate
                if Q_at_price >= -1e-9: # La cantidad demandada debe ser no negativa
                    second_derivative = diff(G_p, p, 2)
                    second_derivative_value = N(second_derivative.subs(p, price_candidate)) # Evaluar numéricamente

                    # Si la segunda derivada es negativa, es un máximo local
                    if second_derivative_value < -1e-6: # Un umbral más claro para 'negativo'
                        optimal_price = float(price_candidate) # Convertir a float nativo aquí
                        max_gain = float(N(G_p.subs(p, optimal_price))) # Convertir a float nativo aquí
                        status_message = "¡Éxito! Hemos encontrado el precio que te dará la mayor ganancia. ¡Es un punto ideal!"
                        break
                    elif abs(second_derivative_value) < 1e-6: # Es aproximadamente cero (caso lineal o punto de inflexión)
                        status_message = "Encontramos un punto interesante, pero la ganancia es casi recta en esa zona. Revisa tus números, quizás no haya un único precio 'ideal' para maximizar."
                        optimal_price = None
                        max_gain = None
                        break 
                    else: # Es positivo (mínimo)
                        status_message = "¡Atención! Este precio es un punto de ganancia mínima, no máxima. Si fijas este precio, podrías perder más. Revisa tus datos."
                else:
                    status_message = "Este precio haría que nadie comprara tu producto (demanda cero o negativa). Prueba con precios más bajos."
            else:
                status_message = "El precio no puede ser negativo. Ajusta el precio o los parámetros de demanda para que sea un valor real."

    return optimal_price, max_gain, status_message

def calculate_scenario_metrics(price, demand_a, demand_b, unit_cost_variable, fixed_cost):
    """
    Calcula la cantidad, ingreso y ganancia para un precio dado.
    """
    Q = demand_a - demand_b * price
    if Q < 0: # La cantidad no puede ser negativa en la realidad
        Q = 0
    
    I = price * Q
    C = fixed_cost + unit_cost_variable * Q
    G = I - C
    
    return Q, I, G

# --- Interfaz de Usuario con Streamlit ---

st.set_page_config(
    page_title="Precix: Optimizador de Ganancias",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📈 Precix: Tu Asesor de Precios para Máximas Ganancias")
st.markdown("---")

st.write(
    """
    Bienvenido a **Precix**, tu aliado inteligente para descubrir el **precio de venta perfecto** que disparará las ganancias de tu producto. 
    Solo tienes que indicarnos algunos datos clave sobre cómo se vende tu producto y cuánto te cuesta. ¡Precix hará el resto!
    """
)

# Sidebar para parámetros de entrada
st.sidebar.header("📊 Tus Datos Clave")
st.sidebar.markdown("Ayúdanos a entender cómo la gente **compra** tu producto y cuánto te cuesta **producirlo**.")

with st.sidebar:
    st.subheader("Cómo se Vende tu Producto (Demanda)")
    st.markdown("**(Modelo simple: Cantidad = 'a' - 'b' x Precio)**")
    demand_a = st.number_input(
        "Cantidad si fuera Gratis ('a')",
        min_value=0.1, value=1000.0, step=10.0, format="%.2f",
        help="""
        Imagina que tu producto es **gratis**. ¿Cuántas unidades crees que podrías vender? 
        Este número ('a') representa la demanda máxima cuando el precio es cero. 
        Un número más alto aquí significa que más gente querría tu producto a precios muy bajos.
        """
    )
    demand_b = st.number_input(
        "Sensibilidad al Precio ('b')",
        min_value=0.0, value=2.0, step=0.1, format="%.2f", 
        help="""
        ¿Cuánto deja de comprar la gente si subes el precio? Este número ('b') mide cómo reacciona la demanda a los cambios de precio.
        * Si 'b' es **grande**, la gente deja de comprar mucho si subes un poco el precio (son muy sensibles).
        * Si 'b' es **pequeño** (cercano a cero), la gente sigue comprando casi lo mismo aunque subas el precio (no son muy sensibles).
        * Si 'b' es **cero**, ¡la demanda es siempre la misma sin importar el precio! (Poco común en la realidad).
        """
    )

    st.subheader("Cuánto Cuesta Producir (Costos)")
    st.markdown("**(Modelo simple: Costo Total = Fijo + Variable x Cantidad)**")
    unit_cost_variable = st.number_input(
        "Costo por Cada Unidad Producida",
        min_value=0.0, value=50.0, step=5.0, format="%.2f",
        help="""
        Es el dinero que gastas por hacer **una sola unidad** de tu producto. 
        Por ejemplo, el costo de los materiales, o lo que le pagas al trabajador por hacer una unidad. 
        Este costo **aumenta** si produces más.
        """
    )
    fixed_cost = st.number_input(
        "Costos Fijos Mensuales/Anuales",
        min_value=0.0, value=10000.0, step=100.0, format="%.2f",
        help="""
        Son los gastos que tienes **sí o sí**, sin importar cuántos productos vendas o produzcas. 
        Por ejemplo, el alquiler de tu local, los salarios de administración, la electricidad básica. 
        Estos costos son los mismos cada mes, vendas mucho o vendas nada.
        """
    )

    st.subheader("¿Qué Moneda Usamos?")
    selected_currency = st.selectbox(
        "Elige tu moneda:",
        ("USD - Dólar Estadounidense ($)", "CLP - Peso Chileno ($)"),
        help="""
        Aquí puedes elegir la moneda para ver todos los resultados, como el precio, 
        las ganancias y los costos.
        """
    )
    
    st.markdown("---")
    st.subheader("🎯 Tu Precio Actual o de Referencia")
    current_price = st.number_input(
        "Introduce tu Precio Actual (O un precio para comparar)",
        min_value=0.0, value=70.0, step=1.0, format="%.2f",
        help="""
        Escribe el precio al que vendes tu producto actualmente, o cualquier precio que te gustaría comparar. 
        Veremos cómo se comporta tu negocio con este precio frente al precio óptimo que calculamos.
        """
    )

if "USD" in selected_currency:
    currency_symbol = "USD "
elif "CLP" in selected_currency:
    currency_symbol = "CLP "
else:
    currency_symbol = "$ "

st.markdown("---")

if st.button("🚀 Calcular Mi Precio Ideal y Ver la Simulación"):
    try:
        if demand_a <= 0:
            st.error("La 'Cantidad si fuera Gratis' (coeficiente 'a') debe ser mayor que cero. ¡Incluso gratis, algo se debería vender!")
            st.stop() 
            
        if unit_cost_variable < 0 or fixed_cost < 0:
            st.error("¡Los costos no pueden ser negativos! Por favor, revisa los valores que ingresaste.")
            st.stop()
            
        if current_price < 0:
            st.error("El 'Precio Actual/de Referencia' no puede ser negativo. Por favor, introduce un valor válido.")
            st.stop()

        if demand_b == 0:
            st.warning("¡Atención! Has indicado que la demanda no cambia con el precio ('b' es cero). Esto es inusual en el mundo real.")
            Q_constant = demand_a
            
            st.info(f"Con una demanda fija de {Q_constant:,.2f} unidades (porque 'b' es cero):")
            
            p_sym_temp, G_p_sym_temp = get_gain_function_sympy(demand_a, demand_b, unit_cost_variable, fixed_cost)
            first_derivative_value = N(diff(G_p_sym_temp, p_sym_temp))
            
            if first_derivative_value > 1e-6:
                st.success(f"¡Buenas noticias! Si la demanda es fija ({Q_constant:,.2f} unidades), tu ganancia aumentará cuanto más alto sea el precio que fijes. Teóricamente, no hay límite. En la práctica, la demanda no será infinita y eventualmente caerá a precios muy altos.")
                st.markdown(f"Tu ganancia sería: $G(p) = ({Q_constant:.2f})p - ({unit_cost_variable * Q_constant + fixed_cost:.2f})$")
                st.markdown(f"Para maximizar, deberías buscar el precio más alto que tus clientes estén dispuestos a pagar por tus {Q_constant:,.2f} unidades.")
            elif first_derivative_value < -1e-6:
                st.warning(f"¡Alerta! Si la demanda es fija, tu ganancia disminuye al subir el precio. Esto pasa si tus costos por unidad son muy altos o tus costos fijos son enormes. Lo mejor sería vender a un precio igual a tu costo variable ({currency_symbol}{unit_cost_variable:.2f}) o incluso a cero para minimizar pérdidas, ya que la demanda fija es irreal a precios muy bajos.")
                st.markdown(f"Tu ganancia sería: $G(p) = ({Q_constant:.2f})p - ({unit_cost_variable * Q_constant + fixed_cost:.2f})$")
                st.markdown("Revisa si tu producto es viable con tus costos y precios actuales.")
            else:
                gain_at_zero_price = (0 - unit_cost_variable) * Q_constant - fixed_cost
                st.info(f"¡Interesante! Tu ganancia es siempre la misma, {currency_symbol}{gain_at_zero_price:,.2f}, sin importar el precio. Esto puede ocurrir si el dinero que ganas por cada unidad es exactamente igual a lo que te cuesta producirla. En este caso, cualquier precio que te permita vender tus {Q_constant:,.2f} unidades es 'óptimo'.")
                st.markdown(f"Tu ganancia sería: $G(p) = {gain_at_zero_price:,.2f}$")

            st.markdown("---")
            st.stop()
        
        optimal_price, max_gain, status_message = find_optimal_price_and_gain(
            demand_a, demand_b, unit_cost_variable, fixed_cost
        )

        st.subheader("🎉 ¡Tus Resultados Clave!")
        if optimal_price is not None:
            st.success(f"**🌟 Precio Ideal Sugerido:** **{currency_symbol}{optimal_price:,.2f}**")
            st.info(f"**💰 Ganancia Máxima Esperada:** **{currency_symbol}{max_gain:,.2f}**")
            st.write(status_message)
        else:
            st.error(status_message)
            st.stop()

        # --- Comparación con el Precio Actual/Referencia ---
        st.markdown("---")
        st.subheader("📊 Comparación: Tu Precio Actual vs. El Precio Ideal")
        st.write("Mira cuánto podrías mejorar tu negocio si ajustas tu precio.")

        Q_current, I_current, G_current = calculate_scenario_metrics(current_price, demand_a, demand_b, unit_cost_variable, fixed_cost)
        
        Q_optimal, I_optimal, G_optimal = 0, 0, 0
        if optimal_price is not None: # Asegurarse de que tenemos un precio óptimo para comparar
             Q_optimal, I_optimal, G_optimal = calculate_scenario_metrics(optimal_price, demand_a, demand_b, unit_cost_variable, fixed_cost)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Tu Escenario Actual (Precio: {currency_symbol}{current_price:,.2f})**")
            st.metric("Cantidad de Unidades Vendidas", f"{Q_current:,.2f}")
            st.metric("Ingreso Total (Ventas)", f"{currency_symbol}{I_current:,.2f}")
            st.metric("Ganancia Neta", f"{currency_symbol}{G_current:,.2f}")

        with col2:
            if optimal_price is not None:
                st.markdown(f"**Escenario Óptimo (Precio: {currency_symbol}{optimal_price:,.2f})**")
                st.metric("Cantidad de Unidades Vendidas", f"{Q_optimal:,.2f}")
                st.metric("Ingreso Total (Ventas)", f"{currency_symbol}{I_optimal:,.2f}")
                st.metric("Ganancia Neta", f"{currency_symbol}{G_optimal:,.2f}")
            else:
                st.warning("No se pudo calcular el escenario óptimo para comparar.")

        if optimal_price is not None:
            gain_difference = G_optimal - G_current
            if gain_difference > 0:
                st.success(f"¡Increíble! Si cambias de tu precio actual de {currency_symbol}{current_price:,.2f} al precio ideal de {currency_symbol}{optimal_price:,.2f}, tu ganancia podría **aumentar en {currency_symbol}{gain_difference:,.2f}**.")
            elif gain_difference < 0:
                st.warning(f"¡Atención! Si cambias de tu precio actual de {currency_symbol}{current_price:,.2f} al precio ideal de {currency_symbol}{optimal_price:,.2f}, tu ganancia podría **disminuir en {currency_symbol}{abs(gain_difference):,.2f}**. Esto puede ocurrir si tu precio actual ya es muy cercano al óptimo o si tus datos indican una situación particular.")
            else:
                st.info("Tu precio actual ya está muy cerca del precio ideal. ¡Bien hecho!")

        st.subheader("📈 ¿Cómo se ve tu Ganancia con Diferentes Precios?")
        st.write("Mira este gráfico para entender cómo cambia tu ganancia si varías el precio de tu producto.")
        
        p_sym, G_p_sym = get_gain_function_sympy(demand_a, demand_b, unit_cost_variable, fixed_cost)
        gain_func_numeric = lambdify(p_sym, G_p_sym, 'numpy')

        # --- Cálculo de Puntos de Equilibrio (BEP) ---
        try:
            A_bep = float(N(G_p_sym.coeff(p_sym, 2)))
        except (TypeError, AttributeError):
            A_bep = 0.0
        try:
            B_bep = float(N(G_p_sym.coeff(p_sym, 1)))
        except (TypeError, AttributeError):
            B_bep = 0.0
        try:
            C_bep = float(N(G_p_sym.coeff(p_sym, 0)))
        except (TypeError, AttributeError):
            C_bep = 0.0

        break_even_points = []
        discriminant = B_bep**2 - 4 * A_bep * C_bep

        if discriminant >= -1e-9: 
            if abs(A_bep) > 1e-9: 
                p_bep1 = (-B_bep + np.sqrt(max(0, discriminant))) / (2 * A_bep)
                p_bep2 = (-B_bep - np.sqrt(max(0, discriminant))) / (2 * A_bep)

                if p_bep1 >= -1e-9 and (demand_a - demand_b * p_bep1) >= -1e-9:
                    break_even_points.append(p_bep1)
                if p_bep2 >= -1e-9 and (demand_a - demand_b * p_bep2) >= -1e-9 and abs(p_bep1 - p_bep2) > 1e-9:
                    break_even_points.append(p_bep2)
                break_even_points.sort()
            elif abs(B_bep) > 1e-9: 
                p_bep_linear = -C_bep / B_bep
                if p_bep_linear >= -1e-9 and (demand_a - demand_b * p_bep_linear) >= -1e-9:
                    break_even_points.append(p_bep_linear)
        # --- Fin Cálculo de Puntos de Equilibrio ---

        price_at_zero_demand = demand_a / demand_b if demand_b > 1e-9 else optimal_price + 100 
        
        # Ajustar rango del gráfico para incluir el precio actual si no está en el rango inicial
        all_prices_to_consider = [optimal_price, current_price] + break_even_points
        all_prices_to_consider = [p for p in all_prices_to_consider if p is not None and np.isfinite(p) and p >= 0]
        
        if not all_prices_to_consider: # Fallback if no valid prices
            min_price_plot = 0.0
            max_price_plot = 100.0 # Default range
        else:
            min_price_plot = max(0.0, min(all_prices_to_consider) * 0.8)
            max_price_plot = max(max(all_prices_to_consider) * 1.2, price_at_zero_demand * 1.1)

        prices = np.linspace(min_price_plot, max_price_plot, 400)
        
        raw_gains = gain_func_numeric(prices)
        raw_gains = np.array(raw_gains, dtype=float)
        
        quantities_at_prices = demand_a - demand_b * prices
        valid_indices = quantities_at_prices >= -1e-9 # Consider demand near zero as valid
        gains = np.where(valid_indices, raw_gains, np.nan) # Set invalid gains to NaN

        valid_plot_indices = np.isfinite(gains) # Filter out NaN for plotting
        gains_for_plot = gains[valid_plot_indices]
        prices_for_plot = prices[valid_plot_indices]

        if gains_for_plot.size == 0:
            st.warning("No pudimos generar el gráfico con estos datos. Prueba a ajustar los números o el rango de precios. ¡Estamos trabajando en ello!")
            st.stop() 

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(prices_for_plot, gains_for_plot, label='Ganancia (Cuánto ganas)', color='#1f77b4', linewidth=2) 
        
        ax.scatter([optimal_price], [max_gain], color='red', s=150, zorder=5, label=f'Tu Precio Ideal: {currency_symbol}{optimal_price:,.2f} (Ganancia: {currency_symbol}{max_gain:,.2f})')
        ax.axvline(x=optimal_price, color='gray', linestyle='--', linewidth=0.8)
        ax.axhline(y=max_gain, color='gray', linestyle=':', linewidth=0.8)

        # Plot current price point if valid
        if Q_current > -1e-9: # Only plot if demand is non-negative at current price
            ax.scatter([current_price], [G_current], color='purple', s=150, zorder=5, label=f'Tu Precio Actual: {currency_symbol}{current_price:,.2f} (Ganancia: {currency_symbol}{G_current:,.2f})')
            ax.axvline(x=current_price, color='purple', linestyle=':', linewidth=0.8, alpha=0.7)


        display_beps = [p for p in break_even_points if min_price_plot <= p <= max_price_plot]
        if display_beps:
            for i, p_bep in enumerate(display_beps):
                ax.scatter([float(p_bep)], [0], color='green', marker='o', s=100, zorder=6, label=f'Punto de Equilibrio {i+1}: {currency_symbol}{p_bep:,.2f}')
                ax.axvline(x=float(p_bep), color='green', linestyle=':', linewidth=0.6, alpha=0.7)
        
        ax.set_title('Cómo Varía tu Ganancia con el Precio de Venta', fontsize=16)
        ax.set_xlabel(f'Precio de Venta ({currency_symbol})', fontsize=12)
        ax.set_ylabel(f'Ganancia ({currency_symbol})', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        y_min_plot = np.nanmin(gains_for_plot) if gains_for_plot.size > 0 else 0
        y_max_plot = np.nanmax(gains_for_plot) if gains_for_plot.size > 0 else (max_gain * 1.2 if max_gain is not None else 100)
        
        y_min_plot = min(y_min_plot, 0)
        y_max_plot = max(y_max_plot, 0)

        y_range = y_max_plot - y_min_plot
        ax.set_ylim(bottom=y_min_plot - y_range * 0.1, top=y_max_plot + y_range * 0.1)

        st.pyplot(fig) # <-- El gráfico se muestra aquí

        # --- Mensajes de Puntos de Equilibrio DESPUÉS del gráfico ---
        if display_beps:
            st.markdown(f"**Puntos donde la Ganancia es Cero (¡Ni ganas ni pierdes!):** " + 
                        ", ".join([f"{currency_symbol}{p:.2f}" for p in break_even_points]) + 
                        ". En el gráfico, son los puntos verdes. Si tu precio está **entre** estos puntos, ¡estás ganando dinero!")
        else:
            if max_gain is not None and max_gain > 1e-9:
                st.info("¡Buena señal! Parece que estás ganando dinero en casi todos los precios lógicos. No hay puntos donde tu ganancia sea cero en este rango.")
            else:
                st.info("Parece que, por ahora, tu negocio está teniendo pérdidas. No se encontraron puntos donde la ganancia sea cero. Revisa tus costos o precios.")

        st.markdown(
            """
            **Qué Nos Dice Este Gráfico:**
            * La **línea azul** te muestra tu **ganancia** para cada posible precio de venta.
            * El **punto rojo** es el **precio ideal** que calculamos, donde tu ganancia es la más alta. ¡Es tu punto dulce!
            * El **punto morado** (si aparece) es el **precio actual** que ingresaste y la ganancia que te da.
            * Los **puntos verdes** (si aparecen) son los **precios de equilibrio**: si vendes a esos precios, no ganas ni pierdes dinero (ganancia cero).
            * Si la línea azul baja después del punto rojo, significa que si subes demasiado el precio, venderás menos y tu ganancia bajará.
            * Si la línea azul está muy baja o en números negativos antes del punto rojo, significa que tu precio es demasiado bajo para cubrir tus costos.
            """
        )

    except ValueError:
        st.error("Por favor, asegúrate de escribir solo números válidos en todos los campos.")
    except Exception as e:
        st.error(f"¡Ups! Ocurrió un problema inesperado: {e}. Por favor, revisa tus datos de entrada. A veces, si los números no tienen sentido en la vida real (por ejemplo, costos muy altos para una demanda muy baja), el cálculo puede fallar.")

st.markdown("---")

# --- Sección de Fórmulas Matemáticas (AHORA SÚPER SIMPLIFICADAS) ---
st.subheader("💡 La Magia Detrás de los Números (Las Fórmulas Simplificadas)")
st.write("Precix usa un poco de matemáticas para encontrar ese precio ideal. Aquí te explicamos, paso a paso, cómo lo hace, ¡pero sin enredos!")

a_val = f"{demand_a:.2f}"
b_val = f"{demand_b:.2f}"
cv_val = f"{unit_cost_variable:.2f}"
cf_val = f"{fixed_cost:.2f}"

p_sym_temp, G_p_sym_temp = get_gain_function_sympy(demand_a, demand_b, unit_cost_variable, fixed_cost)

try:
    coeff_p2 = float(N(G_p_sym_temp.coeff(p_sym_temp, 2)))
except (TypeError, AttributeError):
    coeff_p2 = 0.0
try:
    coeff_p1 = float(N(G_p_sym_temp.coeff(p_sym_temp, 1)))
except (TypeError, AttributeError):
    coeff_p1 = 0.0
try:
    coeff_p0 = float(N(G_p_sym_temp.coeff(p_sym_temp, 0)))
except (TypeError, AttributeError):
    coeff_p0 = 0.0

st.markdown(fr"""
1.  **¿Cuántos Productos Se Venden? (Función de Demanda)**
    Esta fórmula nos ayuda a predecir cuántas unidades (Q) de tu producto se venderían si fijas un **precio (p)**.
    Según tus datos, la fórmula se ve así:
    $$Q(p) = {a_val} - {b_val} \cdot p$$
""")
with st.expander("Haz clic para entender mejor"):
    st.markdown("""
    Esta fórmula nos dice **cuántas unidades (Q)** de tu producto la gente comprará a un determinado **precio (p)**.
    * **'{a_val}'** es la cantidad de personas interesadas si tu producto fuera gratis (el máximo posible).
    * **'{b_val}'** nos dice cuánto menos se vende por cada peso/dólar que subes el precio. ¡Es la sensibilidad de tus clientes!
    * La **'p'** es el **precio que tú podrías elegir**, y al cambiarlo, verás cómo varía la cantidad de productos que se venderían (Q).
    """)

st.markdown(fr"""
2.  **¿Cuánto Dinero Entra? (Función de Ingreso Total)**
    $$I(p) = p \cdot Q(p)$$
    Sustituyendo cuánto se vende (Q(p)):
    $$I(p) = p \cdot ({a_val} - {b_val} \cdot p)$$
    $$I(p) = {a_val}p - {b_val}p^2$$
""")
with st.expander("Haz clic para entender mejor"):
    st.markdown("""
    Esto es el **dinero total que recibes (I)** por vender tus productos.
    Simplemente multiplicamos el **Precio (p)** por la **Cantidad vendida (Q)**.
    Como la cantidad vendida depende del precio, el dinero que entra también cambia con el precio.
    """)

st.markdown(fr"""
3.  **¿Cuánto Dinero Sale? (Función de Costo Total)**
    $$C(p) = {cf_val} + {cv_val} \cdot Q(p)$$
    Sustituyendo cuánto se vende (Q(p)):
    $$C(p) = {cf_val} + {cv_val} \cdot ({a_val} - {b_val} \cdot p)$$
    $$C(p) = {cf_val} + {unit_cost_variable * demand_a:.2f} - {unit_cost_variable * demand_b:.2f}p$$
""")
with st.expander("Haz clic para entender mejor"):
    st.markdown("""
    Estos son todos los **gastos (C)** que tienes para producir y vender.
    * **Costos Fijos:** Son gastos que pagas siempre, vendas o no (como el alquiler).
    * **Costos Variables:** Son gastos que aumentan con cada producto que haces (como los materiales).
    También vemos cómo estos gastos cambian según el precio que fijes, porque el precio afecta cuántas unidades produces.
    """)

sign_p1 = '+' if coeff_p1 >= 0 else ''
sign_p0 = '+' if coeff_p0 >= 0 else ''

st.markdown(fr"""
4.  **¿Cuánto Te Queda? (Función de Ganancia)**
    $$G(p) = I(p) - C(p)$$
    Restando los gastos (C(p)) al dinero que entra (I(p)), ¡con tus números!:
    $$G(p) = ({a_val}p - {b_val}p^2) - ({cf_val} + {cv_val} \cdot ({a_val} - {b_val} \cdot p))$$
    $$G(p) = {coeff_p2:.2f}p^2 {sign_p1}{coeff_p1:.2f}p {sign_p0}{coeff_p0:.2f}$$
    (Esta fórmula te muestra exactamente cómo cambia tu ganancia según el precio 'p'. Lo que queremos es encontrar el 'p' que te dé el número más grande para G(p)!)
""")
with st.expander("Haz clic para entender mejor"):
    st.markdown("""
    La **Ganancia (G)** es el dinero que te queda después de restar tus gastos a tus ingresos. 
    ¡Es lo que realmente te embolsas! Esta es la fórmula mágica que queremos hacer que dé el valor más alto posible.
    Generalmente, esta función tiene forma de montaña, y buscamos la cima de esa montaña.
    """)

if abs(coeff_p2) > 1e-9:
    st.markdown(fr"""
    5.  **Encontrando la "Cima" de la Ganancia (Derivada)**
        Para encontrar el precio que te da la máxima ganancia, usamos una herramienta matemática llamada **derivada**.
        Piensa en ella como un detector de la "pendiente" de la curva de ganancia. Cuando la pendiente es cero, ¡estás en la cima!
        $$\frac{{dG}}{{dp}} = {2*coeff_p2:.2f}p {sign_p1}{coeff_p1:.2f}$$
        Igualamos a cero para encontrar ese punto:
        $${2*coeff_p2:.2f}p {sign_p1}{coeff_p1:.2f} = 0$$
    """)
    with st.expander("Haz clic para entender mejor"):
        st.markdown("""
        Imagina que tu ganancia es una montaña. Queremos encontrar la **cima** de esa montaña. 
        Matemáticamente, la "derivada" nos ayuda a saber si la montaña está subiendo, bajando, o si ya llegamos a la cima (donde la pendiente es plana, es decir, cero). 
        Al igualar la derivada a cero, le decimos a la calculadora: "¡Busca el punto donde la ganancia deja de subir y empieza a bajar!".
        """)

    st.markdown(fr"""
    6.  **¡Tu Precio Ideal! (Resolviendo para 'p')**
        Ahora, solo necesitamos despejar 'p' de la ecuación de arriba, ¡y listo! Ese es tu precio óptimo.
        $${2*coeff_p2:.2f}p = -({coeff_p1:.2f})$$
        $$p^* = \frac{{-({coeff_p1:.2f})}}{{{2*coeff_p2:.2f}}}$$
        Y el resultado final es:
        $$p^* = {optimal_price if optimal_price is not None else 'N/A' :.2f}$$
        ¡Este 'p*' es el precio que, según tus datos, te dará la mayor ganancia posible!
    """)
    with st.expander("Haz clic para entender mejor"):
        st.markdown("""
        Una vez que encontramos la ecuación que describe dónde la ganancia está en su punto más alto, 
        simplemente la resolvemos para 'p' (el precio). 
        El número que obtenemos es **el precio perfecto** que, según tus datos, te dará la máxima ganancia. 
        ¡Es el número que buscabas!
        """)
else:
    st.markdown("""
    5.  **¿Por qué no hay un "Precio Ideal" Claro aquí?**
        Como el número de 'Sensibilidad al Precio' ('b') es muy pequeño (cero o casi cero), ¡la demanda de tu producto no cambia mucho con el precio!
        Esto hace que la fórmula de ganancia sea como una línea recta, no una montaña con una cima clara.
        """)
    with st.expander("Haz clic para entender mejor"):
        st.markdown("""
        Cuando la "Sensibilidad al Precio" ('b') es cero, significa que la gente compra la misma cantidad de tu producto, 
        ¡sin importar cuánto lo vendas! (Aunque esto es raro en la vida real). 
        Si la demanda no cambia, tu ganancia no tendrá un "pico" claro como una montaña. 
        Será una línea recta que o sube sin parar (si ganas dinero por cada venta), o baja sin parar (si pierdes), o se queda plana (si ganas siempre lo mismo). 
        Por eso, en este caso, no hay un único "precio ideal" que maximice la ganancia de forma matemática simple.
        """)

st.markdown("---")
st.markdown("¡Esperamos que Precix te ayude a tomar decisiones de precios más inteligentes! Si tienes dudas, consulta a un experto en negocios.")
st.markdown("Creado con Python, SymPy, NumPy, Matplotlib y Streamlit por tu equipo de Precix. ¡Optimiza tus precios inteligentemente!")