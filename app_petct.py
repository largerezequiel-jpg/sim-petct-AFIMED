"""
Interfaz Streamlit para el Simulador PET/CT v0.2 — AFIMED
Correr con:  streamlit run app_petct.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import pet_sim_v02 as sim

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PET/CT Simulator — AFIMED",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 Simulador PET/CT — v0.2")
st.caption("AFIMED · Modelo estocástico educativo · SimPy")


def _hhmm(m: float) -> str:
    t = int(8 * 60 + m)
    return f"{t // 60:02d}:{t % 60:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Parámetros
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Parámetros")

    # ── Agenda y recursos ────────────────────────────────────────────────────
    with st.expander("📋 Agenda y recursos", expanded=True):
        num_injectors = st.slider("Inyectorios", 1, 6, 3)
        num_patients  = st.slider("Pacientes agendados", 1, 20, 10)
        inter_arrival = st.slider("Inter-arrival (min)", 5, 90, 30, step=5)

    # ── Tiempos de servicio ──────────────────────────────────────────────────
    with st.expander("⏱ Tiempos de servicio"):

        def tri_inputs(label: str, lo_def, mo_def, hi_def):
            st.markdown(f"**{label}** — triangular (mín · moda · máx)")
            c1, c2, c3 = st.columns(3)
            lo = c1.number_input("mín",  value=float(lo_def), step=0.5,
                                  key=f"{label}_lo", format="%.1f")
            mo = c2.number_input("moda", value=float(mo_def), step=0.5,
                                  key=f"{label}_mo", format="%.1f")
            hi = c3.number_input("máx",  value=float(hi_def), step=0.5,
                                  key=f"{label}_hi", format="%.1f")
            return lo, mo, hi

        rec_lo, rec_mo, rec_hi = tri_inputs("Recepción", 3,  5,  8)
        hot_lo, hot_mo, hot_hi = tri_inputs("Hot room",  3,  5,  8)
        inj_lo, inj_mo, inj_hi = tri_inputs("Inyección", 3,  5,  7)
        bat_lo, bat_mo, bat_hi = tri_inputs("Baño",      3,  5, 10)
        tra_lo, tra_mo, tra_hi = tri_inputs("Traslado",  1,  2,  4)

        st.markdown("**Captación** — Normal truncada")
        c1, c2, c3, c4 = st.columns(4)
        upt_mean = c1.number_input("Media", value=60.0, step=1.0, key="upt_mean")
        upt_sd   = c2.number_input("DE",    value=5.0,  step=0.5, key="upt_sd")
        upt_lo   = c3.number_input("Mín",   value=50.0, step=1.0, key="upt_lo")
        upt_hi   = c4.number_input("Máx",   value=75.0, step=1.0, key="upt_hi")

    # ── Puntualidad de llegadas ───────────────────────────────────────────────
    with st.expander("🚶 Puntualidad de llegadas"):
        st.caption("Triangular: adelanto (−) / retraso (+) respecto del turno programado")
        c1, c2, c3 = st.columns(3)
        arr_lo = c1.number_input("Mín",  value=-10.0, step=1.0, key="arr_lo")
        arr_mo = c2.number_input("Moda", value=0.0,   step=1.0, key="arr_mo")
        arr_hi = c3.number_input("Máx",  value=25.0,  step=1.0, key="arr_hi")

    # ── Protocolos PET/CT ────────────────────────────────────────────────────
    with st.expander("📡 Protocolos PET/CT"):
        st.caption("Las probabilidades deben sumar exactamente 1.00")
        protocols  = []
        prob_total = 0.0
        for pname, p_def, lo_def, mo_def, hi_def in [
            ("Estándar",  0.75, 25, 30, 35),
            ("Extendido", 0.20, 35, 40, 50),
            ("Corto",     0.05, 20, 25, 30),
        ]:
            st.markdown(f"**{pname}**")
            c1, c2, c3, c4 = st.columns(4)
            prob = c1.number_input("Prob", value=p_def, min_value=0.0, max_value=1.0,
                                    step=0.05, key=f"pp_{pname}", format="%.2f")
            lo   = c2.number_input("mín", value=float(lo_def), step=1.0, key=f"pd_{pname}_lo")
            mo   = c3.number_input("mod", value=float(mo_def), step=1.0, key=f"pd_{pname}_mo")
            hi   = c4.number_input("máx", value=float(hi_def), step=1.0, key=f"pd_{pname}_hi")
            prob_total += prob
            protocols.append({
                "name": pname,
                "prob": prob,
                "dist": {"type": "triangular", "low": lo, "mode": mo, "high": hi},
            })
        if abs(prob_total - 1.0) > 0.001:
            st.warning(f"⚠ Probabilidades suman {prob_total:.3f}")
            probs_ok = False
        else:
            st.success(f"✓ Suma: {prob_total:.3f}")
            probs_ok = True

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    with st.expander("🎲 Monte Carlo"):
        random_seed  = int(st.number_input("Semilla aleatoria", value=20261027, step=1))
        replications = st.slider("Réplicas", 50, 2000, 500, step=50)

# ─────────────────────────────────────────────────────────────────────────────
# Construir PARAMS desde la sidebar
# ─────────────────────────────────────────────────────────────────────────────

params = {
    "num_injectors":    num_injectors,
    "num_patients":     num_patients,
    "inter_arrival_time": inter_arrival,
    "random_seed":      random_seed,
    "replications":     replications,
    "distributions": {
        "arrival_delay": {"type": "triangular", "low": arr_lo, "mode": arr_mo, "high": arr_hi},
        "reception":     {"type": "triangular", "low": rec_lo, "mode": rec_mo, "high": rec_hi},
        "hot_room":      {"type": "triangular", "low": hot_lo, "mode": hot_mo, "high": hot_hi},
        "injection":     {"type": "triangular", "low": inj_lo, "mode": inj_mo, "high": inj_hi},
        "uptake":        {"type": "truncnorm",  "mean": upt_mean, "sd": upt_sd,
                          "low": upt_lo, "high": upt_hi},
        "bathroom":      {"type": "triangular", "low": bat_lo, "mode": bat_mo, "high": bat_hi},
        "transfer":      {"type": "triangular", "low": tra_lo, "mode": tra_mo, "high": tra_hi},
    },
    "pet_protocols": protocols,
}
params_key = json.dumps(params, sort_keys=True)  # huella para detectar cambios

# ─────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["▶  Jornada única", "🔁  Monte Carlo"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Jornada única
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if not probs_ok:
        st.error("Corregí las probabilidades de los protocolos PET antes de correr.")
    else:
        if st.button("▶  Correr jornada única", type="primary", use_container_width=True):
            with st.spinner("Simulando jornada..."):
                records, mon, sim_duration = sim.run_simulation(params, seed=random_seed)
                df_run  = sim.build_dataframe(records)
                sim.validate_simulation(df_run, params, verbose=False)
                summary = sim.compute_summary(df_run, mon, sim_duration, params)
                ev_log  = sim.build_event_log(records)
            st.session_state.update({
                "df":              df_run,
                "summary":         summary,
                "event_log":       ev_log,
                "params_key_used": params_key,
            })

        # Aviso de parámetros cambiados
        if ("params_key_used" in st.session_state
                and st.session_state["params_key_used"] != params_key):
            st.warning("⚠ Los parámetros cambiaron. Volvé a correr la simulación.")

        if "df" in st.session_state:
            df      = st.session_state["df"]
            summary = st.session_state["summary"]
            ev_log  = st.session_state["event_log"]

            # ── KPIs ─────────────────────────────────────────────────────────
            st.subheader("Indicadores de la jornada")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Fin último paciente",      _hhmm(summary["last_patient_out_min"]))
            c2.metric("Throughput",               f"{summary['throughput_per_hour']:.2f} pac/h")
            c3.metric("Espera PET/CT (media)",    f"{summary['mean_wait_petct_min']:.1f} min")
            c4.metric("Espera inyect. (media)",   f"{summary['mean_wait_injector_min']:.1f} min")
            c5.metric("Tiempo en sistema (media)",f"{summary['mean_total_time_min']:.0f} min")
            c6.metric("P90 tiempo en sistema",    f"{summary['p90_total_time_min']:.0f} min")

            # ── Gantt ─────────────────────────────────────────────────────────
            st.subheader("Diagrama de Gantt")
            fig = sim.plot_gantt(df, save=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Línea de tiempo de recursos ───────────────────────────────────
            st.subheader("Línea de tiempo de recursos")
            fig = sim.plot_timeline_resources(df, params, save=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Fila de gráficos secundarios ──────────────────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Ocupación de recursos")
                fig = sim.plot_utilization(summary, save=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with col2:
                st.subheader("Tiempos de espera")
                fig = sim.plot_wait_times(df, save=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with col3:
                st.subheader("Sala de espera")
                fig = sim.plot_waiting_room_occupancy(df, summary, save=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # ── Detalles expandibles ─────────────────────────────────────────
            with st.expander("📋 Tabla completa de pacientes"):
                st.dataframe(df, use_container_width=True)

            with st.expander("📝 Log de eventos — P01"):
                p01 = ev_log[ev_log["Paciente"] == "P01"].copy()
                p01["Inicio"]   = p01["Inicio"].apply(_hhmm)
                p01["Fin"]      = p01["Fin"].apply(_hhmm)
                p01["Duración"] = p01["Duración"].apply(lambda x: f"{x:.1f} min")
                st.dataframe(p01, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    if not probs_ok:
        st.error("Corregí las probabilidades de los protocolos PET antes de correr.")
    else:
        st.info(
            f"Se ejecutarán **{replications} réplicas** independientes, "
            f"cada una con una semilla distinta. "
            f"Muestra la variabilidad natural entre jornadas."
        )

        if st.button("▶  Ejecutar Monte Carlo", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Réplica 0 / …")
            rep_df = sim.run_replications(
                params,
                n_replications=replications,
                progress_callback=lambda p: progress_bar.progress(
                    p, text=f"Réplica {int(p * replications)} / {replications}"
                ),
            )
            progress_bar.empty()
            rep_summary = sim.summarize_replications(rep_df)
            st.session_state.update({
                "rep_df":         rep_df,
                "rep_summary":    rep_summary,
                "rep_params_key": params_key,
            })

        if ("rep_params_key" in st.session_state
                and st.session_state["rep_params_key"] != params_key):
            st.warning("⚠ Los parámetros cambiaron. Volvé a ejecutar Monte Carlo.")

        if "rep_df" in st.session_state:
            rep_df      = st.session_state["rep_df"]
            rep_summary = st.session_state["rep_summary"]

            # ── Tabla resumen ─────────────────────────────────────────────────
            st.subheader("Resumen estadístico entre jornadas")
            st.dataframe(
                rep_summary.set_index("Métrica").style.format("{:.2f}"),
                use_container_width=True,
            )

            # ── Histogramas ───────────────────────────────────────────────────
            st.subheader("Distribuciones entre jornadas")
            fig = sim.plot_replication_distributions(rep_df, save=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Descarga ──────────────────────────────────────────────────────
            st.subheader("Descargar datos")
            col1, col2 = st.columns(2)
            col1.download_button(
                "⬇ Réplicas (CSV)",
                data=rep_df.to_csv(index=False),
                file_name="replicaciones_petct.csv",
                mime="text/csv",
            )
            col2.download_button(
                "⬇ Resumen estadístico (CSV)",
                data=rep_summary.to_csv(index=False),
                file_name="resumen_estadistico_petct.csv",
                mime="text/csv",
            )
