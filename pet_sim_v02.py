"""
╔══════════════════════════════════════════════════════════════════╗
║         SIMULACIÓN CENTRO DE IMÁGENES PET/CT — v0.2             ║
║         SimPy · Modelo estocástico · Fines educativos           ║
╚══════════════════════════════════════════════════════════════════╝

Flujo del paciente:
  Llegada → Recepción → Espera → Inyectorio {hot room + inyección +
  captación + espera PET/CT + baño} → Traslado → PET/CT → Egreso

t = 0  corresponde a las 08:00 del día.
Todas las unidades de tiempo están en minutos.

SUPUESTOS DE MODELO
  [A1] Preparación de dosis: la preparación/fraccionamiento de la dosis comienza
       una vez que el paciente ha sido asignado a un inyectorio. No se realiza
       preparación anticipada de dosis. Consecuencia: si el cuarto caliente está
       congestionado, el paciente ocupa un inyectorio mientras espera su turno
       en hot room, reduciendo la capacidad efectiva disponible para nuevos pacientes.
  [A2] Los turnos siguen siendo programados cada 30 min. La llegada real se obtiene
       sumando un adelanto/retraso aleatorio a la hora programada.
  [A3] Las distribuciones de v0.2 son DIDÁCTICAS: deben calibrarse con datos reales
       del servicio antes de utilizar el modelo para decisiones operativas.
  [A4] Cada paciente recibe sus propios sorteos aleatorios pre-generados. Esto hace
       que sus tiempos no dependan del orden interno de eventos de SimPy y permite
       comparar escenarios usando números aleatorios comunes.
"""

import simpy
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # renderizar a archivo sin abrir ventana interactiva
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random
import math
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = {
    # ── Recursos ──────────────────────────────────────────────────────────────
    "num_injectors": 3,

    # ── Agenda ────────────────────────────────────────────────────────────────
    "num_patients":       10,
    "inter_arrival_time": 30,    # minutos entre turnos programados

    # ── Aleatoriedad ──────────────────────────────────────────────────────────
    "random_seed": 20261027,      # reproducibilidad de la jornada mostrada
    "replications": 500,          # jornadas para el análisis Monte Carlo

    # Distribuciones DIDÁCTICAS. Ajustar con datos del servicio real.
    # triangular: mínimo, moda, máximo
    # truncnorm : media, desvío estándar, mínimo, máximo
    "distributions": {
        # Adelanto/retraso respecto del turno. Negativo = llega antes.
        "arrival_delay": {"type": "triangular", "low": -10, "mode": 0, "high": 25},

        "reception": {"type": "triangular", "low": 3,  "mode": 5,  "high": 8},
        "hot_room":  {"type": "triangular", "low": 3,  "mode": 5,  "high": 8},
        "injection": {"type": "triangular", "low": 3,  "mode": 5,  "high": 7},
        "uptake":    {"type": "truncnorm",  "mean": 60, "sd": 5, "low": 50, "high": 75},
        "bathroom":  {"type": "triangular", "low": 3,  "mode": 5,  "high": 10},
        "transfer":  {"type": "triangular", "low": 1,  "mode": 2,  "high": 4},
    },

    # Heterogeneidad de duración del estudio PET/CT.
    # La probabilidad total debe sumar 1.0.
    "pet_protocols": [
        {"name": "Estándar",  "prob": 0.75,
         "dist": {"type": "triangular", "low": 25, "mode": 30, "high": 35}},
        {"name": "Extendido", "prob": 0.20,
         "dist": {"type": "triangular", "low": 35, "mode": 40, "high": 50}},
        {"name": "Corto",     "prob": 0.05,
         "dist": {"type": "triangular", "low": 20, "mode": 25, "high": 30}},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientRecord:
    """Almacena todos los timestamps del recorrido de un paciente."""
    patient_id: int

    injector_id:         str   = ""    # "I1" | "I2" | "I3"

    # Agenda y variables aleatorias pre-generadas
    scheduled_time:        float = 0.0
    sampled_arrival_delay: float = 0.0   # sorteo original; negativo = adelanto
    pet_protocol:          str   = ""

    reception_duration: float = 0.0
    hot_room_duration:  float = 0.0
    injection_duration: float = 0.0
    uptake_duration:    float = 0.0
    bathroom_duration:  float = 0.0
    transfer_duration:  float = 0.0
    petct_duration:     float = 0.0

    # Timestamps
    arrival_time:        float = 0.0
    reception_start:     float = 0.0
    reception_end:       float = 0.0
    waiting_room_enter:  float = 0.0   # = reception_end  (estado físico explícito)
    injector_wait_start: float = 0.0
    injector_assigned:   float = 0.0
    waiting_room_exit:   float = 0.0   # = injector_assigned
    hot_room_start:     float = 0.0
    hot_room_end:       float = 0.0
    injection_start:    float = 0.0
    injection_end:      float = 0.0
    uptake_end:         float = 0.0
    petct_wait_start:   float = 0.0
    petct_reserved:     float = 0.0   # recurso SimPy bloqueado
    bathroom_end:       float = 0.0
    transfer_end:       float = 0.0   # = liberación del inyectorio
    petct_scan_start:   float = 0.0   # paciente en camilla, adquisición comienza
    petct_end:          float = 0.0   # = egreso del sistema


@dataclass
class EventRecord:
    """Una fila del log de eventos: estado de un paciente en un intervalo."""
    patient_id: int
    state:      str    # estado clínico (ej. "Captación")
    location:   str    # ubicación física (ej. "I1", "PET/CT")
    start:      float  # minutos desde las 08:00
    end:        float  # minutos desde las 08:00

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SystemMonitor:
    """Acumula tiempos de ocupación para calcular utilización."""
    reception_busy:      float = 0.0   # minutos que recepción estuvo ocupada
    hot_room_busy:       float = 0.0   # minutos que cuarto caliente estuvo ocupado
    injectors_busy:      float = 0.0   # Σ (tiempo de ocupación × paciente)
    petct_scan_busy:     float = 0.0   # Σ t_scan  (solo adquisición)
    petct_reserved_busy: float = 0.0   # Σ (t_end − t_reserved)  (recurso comprometido)
    peak_concurrent: int   = 0     # máx pacientes simultáneos en el sistema
    _concurrent:     int   = 0     # contador interno (no exportar directamente)

    def patient_enters(self):
        self._concurrent += 1
        if self._concurrent > self.peak_concurrent:
            self.peak_concurrent = self._concurrent

    def patient_exits(self):
        self._concurrent -= 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERACIÓN DE VARIABLES ALEATORIAS
# ─────────────────────────────────────────────────────────────────────────────

def sample_distribution(rng: random.Random, spec: dict) -> float:
    """Sortea una duración/variable según una especificación simple y auditable."""
    kind = spec["type"].lower()

    if kind == "fixed":
        return float(spec["value"])

    if kind == "triangular":
        return float(rng.triangular(spec["low"], spec["high"], spec["mode"]))

    if kind == "truncnorm":
        mean, sd = float(spec["mean"]), float(spec["sd"])
        low, high = float(spec["low"]), float(spec["high"])
        # Rejection sampling. Con los parámetros usados aquí la aceptación es alta.
        for _ in range(1000):
            x = rng.gauss(mean, sd)
            if low <= x <= high:
                return float(x)
        # Fallback extremadamente improbable: recortar al rango permitido.
        return float(min(high, max(low, mean)))

    raise ValueError(f"Distribución no soportada: {kind}")


def sample_pet_protocol(rng: random.Random, protocols: list) -> Tuple[str, float]:
    """Sortea tipo de estudio PET/CT y su duración."""
    total_prob = sum(float(x["prob"]) for x in protocols)
    if not math.isclose(total_prob, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Las probabilidades de pet_protocols deben sumar 1.0 (suman {total_prob})")

    u = rng.random()
    cumulative = 0.0
    for item in protocols:
        cumulative += float(item["prob"])
        if u <= cumulative:
            return item["name"], sample_distribution(rng, item["dist"])

    # Protección frente a redondeo flotante.
    item = protocols[-1]
    return item["name"], sample_distribution(rng, item["dist"])


def make_patient_record(pid: int, p: dict, replication_seed: int) -> PatientRecord:
    """
    Pre-genera agenda y todos los sorteos del paciente.

    La semilla depende de la réplica y del ID del paciente, no del orden de ejecución
    de SimPy. Esto facilita comparar escenarios con números aleatorios comunes.
    """
    rng = random.Random(replication_seed + pid * 1009)
    d = p["distributions"]

    scheduled = (pid - 1) * p["inter_arrival_time"]
    sampled_delay = sample_distribution(rng, d["arrival_delay"])
    arrival = max(0.0, scheduled + sampled_delay)  # el servicio abre en t=0

    protocol, pet_duration = sample_pet_protocol(rng, p["pet_protocols"])

    return PatientRecord(
        patient_id=pid,
        scheduled_time=scheduled,
        sampled_arrival_delay=sampled_delay,
        arrival_time=arrival,
        pet_protocol=protocol,
        reception_duration=sample_distribution(rng, d["reception"]),
        hot_room_duration=sample_distribution(rng, d["hot_room"]),
        injection_duration=sample_distribution(rng, d["injection"]),
        uptake_duration=sample_distribution(rng, d["uptake"]),
        bathroom_duration=sample_distribution(rng, d["bathroom"]),
        transfer_duration=sample_distribution(rng, d["transfer"]),
        petct_duration=pet_duration,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROCESO DEL PACIENTE
# ─────────────────────────────────────────────────────────────────────────────

def patient_process(env: simpy.Environment,
                    rec: PatientRecord,
                    resources: dict,
                    mon: SystemMonitor,
                    p: dict):
    """
    Proceso SimPy que modela el recorrido completo de un paciente.

    Uso de recursos:
      - Recepción      : con 'with' (liberación automática)
      - Cuarto caliente: con 'with' (liberación automática)
      - Inyectorio     : manual request/release (se libera al traslado, no al
                         final del proceso)
      - PET/CT         : manual request/release (se pide estando en inyectorio,
                         pero se usa después de liberarlo)
    """
    mon.patient_enters()

    # ── ETAPA 1: RECEPCIÓN ────────────────────────────────────────────────────
    with resources["reception"].request() as req:
        yield req
        rec.reception_start = env.now
        yield env.timeout(rec.reception_duration)
        rec.reception_end = env.now
        mon.reception_busy += rec.reception_duration

    rec.waiting_room_enter = env.now      # entra a sala de espera al salir de recepción

    # ── ETAPA 2: ESPERA POR INYECTORIO ───────────────────────────────────────
    # Store.get() bloquea hasta que haya un token disponible.
    # El token obtenido es el ID del inyectorio ("I1", "I2" o "I3").
    rec.injector_wait_start = env.now
    rec.injector_id = yield resources["injectors"].get()
    rec.injector_assigned = env.now
    rec.waiting_room_exit = env.now       # sale de sala de espera al entrar al inyectorio

    # ── ETAPA 3: PREPARACIÓN EN CUARTO CALIENTE ──────────────────────────────
    # [A1] La dosis se prepara DESPUÉS de asignar inyectorio (no hay preparación
    # anticipada). Si hot room está ocupado, el inyectorio queda bloqueado en espera.
    with resources["hot_room"].request() as req:
        yield req
        rec.hot_room_start = env.now
        yield env.timeout(rec.hot_room_duration)
        rec.hot_room_end = env.now
        mon.hot_room_busy += rec.hot_room_duration

    # ── ETAPA 4: INYECCIÓN ───────────────────────────────────────────────────
    rec.injection_start = env.now
    yield env.timeout(rec.injection_duration)
    rec.injection_end = env.now

    # ── ETAPA 5: CAPTACIÓN (uptake) ──────────────────────────────────────────
    yield env.timeout(rec.uptake_duration)
    rec.uptake_end = env.now

    # ── ETAPA 6: ESPERA POR PET/CT (paciente aún en el inyectorio) ──────────
    # El recurso se reserva aquí; el paciente sólo se mueve cuando PET/CT
    # está libre, evitando que deambule radioactivo por pasillos.
    rec.petct_wait_start = env.now
    petct_req = resources["petct"].request()
    yield petct_req
    rec.petct_reserved = env.now          # ← recurso SimPy bloqueado desde aquí

    # ── ETAPA 7: MICCIÓN EN BAÑO PRIVADO DEL INYECTORIO ─────────────────────
    yield env.timeout(rec.bathroom_duration)
    rec.bathroom_end = env.now

    # ── ETAPA 8: TRASLADO ────────────────────────────────────────────────────
    yield env.timeout(rec.transfer_duration)
    rec.transfer_end = env.now

    # Liberación del inyectorio: devuelve el token al Store.
    resources["injectors"].put(rec.injector_id)
    mon.injectors_busy += (rec.transfer_end - rec.injector_assigned)

    # ── ETAPA 9: ADQUISICIÓN PET/CT ──────────────────────────────────────────
    rec.petct_scan_start = env.now        # ← adquisición comienza aquí
    yield env.timeout(rec.petct_duration)
    rec.petct_end = env.now

    resources["petct"].release(petct_req)
    mon.petct_scan_busy     += rec.petct_duration
    mon.petct_reserved_busy += (rec.petct_end - rec.petct_reserved)

    mon.patient_exits()


# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERADOR DE LLEGADAS
# ─────────────────────────────────────────────────────────────────────────────

def arrival_generator(env: simpy.Environment,
                      resources: dict,
                      mon: SystemMonitor,
                      p: dict,
                      records: list,
                      replication_seed: int):
    """
    Mantiene una agenda fija, pero cada paciente tiene adelanto/retraso aleatorio.
    Los pacientes se crean en orden de llegada REAL, que puede diferir del orden de turnos.
    """
    planned = [
        make_patient_record(pid, p, replication_seed)
        for pid in range(1, p["num_patients"] + 1)
    ]
    planned.sort(key=lambda r: (r.arrival_time, r.patient_id))

    for rec in planned:
        wait = max(0.0, rec.arrival_time - env.now)
        if wait > 0:
            yield env.timeout(wait)
        records.append(rec)
        env.process(patient_process(env, rec, resources, mon, p))


# ─────────────────────────────────────────────────────────────────────────────
# 6. MOTOR DE SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(params: dict = PARAMS, seed: Optional[int] = None):
    """
    Instancia el entorno SimPy, los recursos y ejecuta la simulación.
    Retorna (records, monitor, sim_duration).
    """
    env = simpy.Environment()
    replication_seed = params["random_seed"] if seed is None else int(seed)

    # Inyectorios: Store de tokens nombrados en lugar de Resource anónimo.
    # Cada token es el ID de la habitación ("I1", "I2", "I3").
    # Store.get() bloquea si no hay token disponible (= inyectorio ocupado).
    # Store.put() devuelve el token al pool (= inyectorio liberado).
    injector_store = simpy.Store(env)
    for i in range(1, params["num_injectors"] + 1):
        injector_store.put(f"I{i}")

    resources = {
        "reception": simpy.Resource(env, capacity=1),
        "hot_room":  simpy.Resource(env, capacity=1),
        "injectors": injector_store,
        "petct":     simpy.Resource(env, capacity=1),
    }

    mon = SystemMonitor()
    records: List[PatientRecord] = []

    env.process(arrival_generator(env, resources, mon, params, records, replication_seed))
    env.run()   # avanza hasta que no quedan eventos pendientes

    return records, mon, env.now


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONSTRUCCIÓN DEL DATAFRAME Y MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(records: List[PatientRecord]) -> pd.DataFrame:
    """Convierte la lista de PatientRecord a un DataFrame de pandas."""
    rows = []
    for r in records:
        rows.append({
            "ID":               r.patient_id,
            "Inyectorio":       r.injector_id,
            "Turno":            r.scheduled_time,
            "Retraso sorteado": r.sampled_arrival_delay,
            "Retraso efectivo": r.arrival_time - r.scheduled_time,
            "Protocolo PET":    r.pet_protocol,
            "Llegada":          r.arrival_time,
            "Recepción inicio": r.reception_start,  # cuando recepción realmente atiende
            "Fin recepción":     r.reception_end,
            "Sala espera entra": r.waiting_room_enter,
            "Sala espera sale":  r.waiting_room_exit,
            "Inyectorio OK":     r.injector_assigned,
            "Hot room inicio":  r.hot_room_start,   # cuando hot room realmente inicia
            "Inyección":        r.injection_start,  # = hot_room_end (sin gap)
            "Fin inyección":    r.injection_end,
            "Fin captación":    r.uptake_end,
            "PET/CT reservado": r.petct_reserved,   # recurso bloqueado
            "Fin baño":         r.bathroom_end,
            "Fin traslado":     r.transfer_end,
            "PET/CT scan":      r.petct_scan_start, # adquisición comienza
            "Fin PET/CT":       r.petct_end,
            # Duraciones sorteadas
            "t recepción":      r.reception_duration,
            "t hot room":       r.hot_room_duration,
            "t inyección":      r.injection_duration,
            "t captación":      r.uptake_duration,
            "t baño":           r.bathroom_duration,
            "t traslado":       r.transfer_duration,
            "t PET/CT":         r.petct_duration,
            # Duraciones derivadas
            "Espera recep.":    r.reception_start   - r.arrival_time,
            "Espera hot room":  r.hot_room_start     - r.injector_assigned,
            "Espera inyect.":   r.injector_assigned  - r.injector_wait_start,
            "Espera PET/CT":    r.petct_reserved     - r.petct_wait_start,
            "Tiempo total":     r.petct_end          - r.arrival_time,
        })
    return pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)


def compute_summary(df: pd.DataFrame,
                    mon: SystemMonitor,
                    sim_duration: float,
                    params: dict) -> dict:
    """Calcula métricas del sistema a partir del DataFrame y el monitor."""
    n = params["num_injectors"]

    # Minutos crudos ocupados (denominador = sim_duration, o sim_duration×n para pool)
    # Nota: para comparar escenarios usar busy_minutes directamente, ya que
    # sim_duration varía entre escenarios y hace que los % no sean comparables.
    busy_minutes = {
        "Recepcion":          mon.reception_busy,
        "Cuarto caliente":    mon.hot_room_busy,
        "Inyectorios":        mon.injectors_busy,       # Σ acumulado (pool de 3)
        "PET/CT (adq.)":      mon.petct_scan_busy,
        "PET/CT (reservado)": mon.petct_reserved_busy,
    }
    # Ocupación = busy_minutes / denominador × 100
    # Semántica: "qué fracción del período simulado estuvo ocupado el recurso"
    utilization = {
        "Recepcion":          mon.reception_busy      / sim_duration * 100,
        "Cuarto caliente":    mon.hot_room_busy        / sim_duration * 100,
        "Inyectorios":        mon.injectors_busy       / (sim_duration * n) * 100,
        "PET/CT (adq.)":      mon.petct_scan_busy      / sim_duration * 100,
        "PET/CT (reservado)": mon.petct_reserved_busy  / sim_duration * 100,
    }

    # Pico de ocupación de sala de espera por método de eventos
    # Ignorar permanencias de duración cero: el paciente pasa directamente al inyectorio.
    events = []
    for _, r in df.iterrows():
        if r["Sala espera sale"] > r["Sala espera entra"] + 1e-9:
            events.append((r["Sala espera entra"], +1))
            events.append((r["Sala espera sale"], -1))
    events.sort(key=lambda x: (x[0], x[1]))   # salidas antes que entradas en t igual
    _n, peak_wr = 0, 0
    for _, d in events:
        _n += d
        peak_wr = max(peak_wr, _n)

    return {
        "utilization":              utilization,
        "busy_minutes":             busy_minutes,
        "mean_wait_injector_min":   df["Espera inyect."].mean(),
        "mean_wait_petct_min":      df["Espera PET/CT"].mean(),
        "peak_waiting_room":        peak_wr,
        "peak_concurrent":          mon.peak_concurrent,
        "throughput":               len(df),
        "throughput_per_hour":      len(df) / (sim_duration / 60),
        "last_patient_out_min":     df["Fin PET/CT"].max(),
        "mean_total_time_min":      df["Tiempo total"].mean(),
        "p90_total_time_min":       df["Tiempo total"].quantile(0.90),
        "p90_wait_injector_min":    df["Espera inyect."].quantile(0.90),
        "p90_wait_petct_min":       df["Espera PET/CT"].quantile(0.90),
        "pct_late":                 (df["Retraso efectivo"] > 0).mean() * 100,
        "mean_lateness_min":        (df.loc[df["Retraso efectivo"] > 0, "Retraso efectivo"].mean()
                                      if (df["Retraso efectivo"] > 0).any() else 0.0),
        "mean_petct_duration_min":  df["t PET/CT"].mean(),
        "sim_duration_min":         sim_duration,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOG DE EVENTOS
# ─────────────────────────────────────────────────────────────────────────────

def build_event_log(records: List[PatientRecord]) -> pd.DataFrame:
    """
    Genera el log de eventos paciente-estado-ubicación-tiempo.

    Una fila por etapa del recorrido; cada fila es atómica (un paciente,
    un estado, un intervalo continuo).  Base para análisis de actividad
    radiactiva, carga de personal y flujo en versiones futuras.
    """
    rows = []

    for rec in records:
        pid = rec.patient_id
        inj = rec.injector_id

        def ev(state: str, location: str, start: float, end: float):
            rows.append({
                "Paciente":  f"P{pid:02d}",
                "Estado":    state,
                "Ubicación": location,
                "Inicio":    start,
                "Fin":       end,
                "Duración":  end - start,
            })

        # Espera antes de recepción (puede ser >0 en v0.2 si recepción está ocupada)
        if rec.reception_start > rec.arrival_time:
            ev("Espera recepción", "Entrada",    rec.arrival_time,      rec.reception_start)

        ev("Recepción",          "Recepción",   rec.reception_start,   rec.reception_end)
        ev("Espera inyectorio",  "Sala espera", rec.reception_end,     rec.injector_assigned)

        # Preinyección: paciente en inyectorio, cubre [A1] espera hot room + preparación de dosis
        ev("Preinyección",       inj,           rec.injector_assigned, rec.injection_start)

        ev("Inyección",          inj,           rec.injection_start,   rec.injection_end)
        ev("Captación",          inj,           rec.injection_end,     rec.uptake_end)

        # Espera PET/CT: paciente listo pero scanner ocupado, aún en inyectorio
        ev("Espera PET/CT",      inj,           rec.uptake_end,        rec.petct_reserved)

        bath = f"B{inj[1:]}" if inj.startswith("I") else "Baño"
        ev("Micción",            bath,          rec.petct_reserved,    rec.bathroom_end)
        ev("Traslado",           "Hall",        rec.bathroom_end,      rec.transfer_end)
        ev("Imagen PET/CT",      "PET/CT",      rec.petct_scan_start,  rec.petct_end)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 9. VALIDACIÓN DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def _check_concurrent(starts, ends, cap: int, label: str) -> list:
    """Devuelve errores si la concurrencia máxima supera cap."""
    events = [(t, +1) for t in starts] + [(t, -1) for t in ends]
    events.sort(key=lambda x: (x[0], x[1]))   # salidas antes que entradas en t igual
    n, peak = 0, 0
    for _, d in events:
        n += d
        peak = max(peak, n)
    if peak > cap:
        return [f"{label}: concurrencia máxima {peak} > capacidad {cap}"]
    return []


def validate_simulation(df: pd.DataFrame, params: dict, verbose: bool = True) -> None:
    """
    Verifica invariantes conceptuales del modelo tras cada corrida.

    Una simulación puede ejecutar sin errores de Python y estar conceptualmente
    mal (ej: dos pacientes en el mismo inyectorio, adquisiciones solapadas).
    Si alguna condición falla lanza AssertionError y detiene la ejecución.
    """
    errors: list = []
    n_inj = params["num_injectors"]

    # ── 1. Todos los timestamps son ≥ 0 ──────────────────────────────────────
    ts_cols = [
        "Llegada", "Recepción inicio", "Fin recepción",
        "Sala espera entra", "Sala espera sale", "Inyectorio OK",
        "Hot room inicio", "Inyección", "Fin inyección", "Fin captación",
        "PET/CT reservado", "Fin baño", "Fin traslado", "PET/CT scan", "Fin PET/CT",
    ]
    for col in ts_cols:
        bad = df[df[col] < 0]["ID"].astype(int).tolist()
        if bad:
            errors.append(f"Timestamps negativos en '{col}': pacientes {bad}")

    # ── 2. Orden temporal por paciente ────────────────────────────────────────
    ORDER_CHECKS = [
        ("Llegada",          "Recepción inicio",  "Llegada > Recepción inicio"),
        ("Recepción inicio", "Fin recepción",     "Recepción inicio > Fin recepción"),
        ("Fin recepción",    "Inyectorio OK",     "Fin recepción > Inyectorio OK"),
        ("Inyectorio OK",    "Hot room inicio",   "Inyectorio OK > Hot room inicio"),
        ("Hot room inicio",  "Inyección",         "Hot room inicio > Inyección"),
        ("Inyección",        "Fin inyección",     "Inyección > Fin inyección"),
        ("Fin inyección",    "Fin captación",     "Fin inyección > Fin captación"),
        ("Fin captación",    "PET/CT reservado",  "Fin captación > PET/CT reservado"),
        ("PET/CT reservado", "Fin baño",           "PET/CT reservado > Fin baño"),
        ("Fin baño",         "Fin traslado",       "Fin baño > Fin traslado"),
        ("Fin traslado",     "PET/CT scan",        "Fin traslado > PET/CT scan"),
        ("PET/CT scan",      "Fin PET/CT",         "PET/CT scan > Fin PET/CT"),
    ]
    _EPS = 1e-9  # tolerancia flotante: 1 nanosegundo en unidades de minutos
    for _, r in df.iterrows():
        pid = int(r["ID"])
        for col_a, col_b, msg in ORDER_CHECKS:
            if r[col_a] > r[col_b] + _EPS:
                errors.append(f"P{pid:02d}: {msg} ({r[col_a]:.6f} > {r[col_b]:.6f})")

    # ── 3. Concurrencia en inyectorios nunca supera capacidad ─────────────────
    errors += _check_concurrent(
        df["Inyectorio OK"], df["Fin traslado"], n_inj, "Inyectorios"
    )

    # ── 4. Concurrencia en PET/CT nunca supera 1 ─────────────────────────────
    errors += _check_concurrent(
        df["PET/CT reservado"], df["Fin PET/CT"], 1, "PET/CT (reservado)"
    )
    errors += _check_concurrent(
        df["PET/CT scan"], df["Fin PET/CT"], 1, "PET/CT (scan)"
    )

    # ── 5. Ningún inyectorio asignado simultáneamente a dos pacientes ─────────
    for inj_id in df["Inyectorio"].unique():
        pats = df[df["Inyectorio"] == inj_id].sort_values("Inyectorio OK")
        intervals = list(zip(
            pats["Inyectorio OK"], pats["Fin traslado"], pats["ID"].astype(int)
        ))
        for i in range(len(intervals) - 1):
            s1, e1, p1 = intervals[i]
            s2, _,  p2 = intervals[i + 1]
            if s2 < e1:
                errors.append(
                    f"{inj_id}: solapamiento P{p1:02d} (libera {e1:.1f}) "
                    f"y P{p2:02d} (entra {s2:.1f})"
                )

    # ── 6. Ninguna adquisición PET/CT se superpone ────────────────────────────
    pet_s = df.sort_values("PET/CT scan")
    scans = list(zip(
        pet_s["PET/CT scan"], pet_s["Fin PET/CT"], pet_s["ID"].astype(int)
    ))
    for i in range(len(scans) - 1):
        s1, e1, p1 = scans[i]
        s2, _,  p2 = scans[i + 1]
        if s2 < e1:
            errors.append(
                f"PET/CT scan: solapamiento P{p1:02d} (hasta {e1:.1f}) "
                f"y P{p2:02d} (desde {s2:.1f})"
            )

    # ── Resultado ─────────────────────────────────────────────────────────────
    if errors:
        sep = "=" * 62
        msg = "\n  ".join(errors)
        raise AssertionError(
            f"\n{sep}\n  VALIDACION FALLIDA — {len(errors)} error(es):\n  {msg}\n{sep}"
        )
    if verbose:
        print(f"  ✓ Validacion OK — {len(ORDER_CHECKS)} checks de orden, "
              f"concurrencia, solapamientos")


# ─────────────────────────────────────────────────────────────────────────────
# 10. REPORTE EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def _min_to_hhmm(minutes: float) -> str:
    """Convierte minutos desde las 08:00 a formato HH:MM."""
    total = int(8 * 60 + minutes)
    return f"{total // 60:02d}:{total % 60:02d}"


def print_report(df: pd.DataFrame, summary: dict):
    import sys
    # Usar stdout con UTF-8 para caracteres especiales en Windows
    out = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)

    W = 70
    out.write("\n" + "=" * W + "\n")
    out.write("  PET/CT CENTER SIMULATION - v0.2\n")
    out.write("=" * W + "\n")

    out.write("\n  RECORRIDO POR PACIENTE\n\n")
    display_cols = [
        "ID", "Inyectorio", "Protocolo PET", "Turno", "Llegada", "Retraso efectivo", "Fin recepcion", "Inyectorio OK",
        "Inyeccion", "Fin captacion", "PET/CT reservado", "PET/CT scan", "Fin PET/CT",
        "Espera inyect.", "Espera PET/CT", "Tiempo total",
    ]
    # Renombrar columnas para evitar tildes en la tabla
    rename = {
        "Fin recepción":  "Fin recepcion",
        "Inyección":      "Inyeccion",
        "Fin captación":  "Fin captacion",
    }
    timestamp_cols = [
        "Turno", "Llegada", "Fin recepcion", "Inyectorio OK",
        "Inyeccion", "Fin captacion", "PET/CT reservado", "PET/CT scan", "Fin PET/CT",
    ]
    display = df.rename(columns=rename)[display_cols].copy()
    for col in timestamp_cols:
        display[col] = display[col].apply(_min_to_hhmm)
    display["Retraso efectivo"] = display["Retraso efectivo"].apply(lambda x: f"{x:+.0f} min")
    for col in ["Espera inyect.", "Espera PET/CT", "Tiempo total"]:
        display[col] = display[col].apply(lambda x: f"{x:.0f} min")
    out.write(display.to_string(index=False) + "\n")

    out.write(f"\n  OCUPACION DURANTE PERIODO SIMULADO  "
              f"({summary['sim_duration_min']:.0f} min = "
              f"{_min_to_hhmm(0)} – {_min_to_hhmm(summary['sim_duration_min'])})\n\n")
    busy = summary["busy_minutes"]
    for recurso, pct in summary["utilization"].items():
        filled   = int(pct / 5)
        bar      = "#" * filled + "-" * (20 - filled)
        raw_min  = busy[recurso]
        pool_tag = " (pool)" if recurso == "Inyectorios" else ""
        out.write(f"  {recurso:<18} [{bar}]  {pct:5.1f}%   {raw_min:5.0f} min{pool_tag}\n")

    out.write(f"\n  Espera media inyectorio :  {summary['mean_wait_injector_min']:5.1f} min\n")
    out.write(f"  Espera media PET/CT     :  {summary['mean_wait_petct_min']:5.1f} min\n")
    out.write(f"  N max sala de espera    :  {summary['peak_waiting_room']} pac. simultaneos\n")
    out.write(f"  Pico de concurrencia    :  {summary['peak_concurrent']} pac. simultaneos\n")
    out.write(f"  Pacientes completados   :  {summary['throughput']}\n")
    out.write(f"  Throughput horario      :  {summary['throughput_per_hour']:.2f} pac/hora\n")
    out.write(f"  Fin ultimo paciente     :  {_min_to_hhmm(summary['last_patient_out_min'])}\n")
    out.write(f"  Tiempo medio en sistema :  {summary['mean_total_time_min']:5.1f} min\n")
    out.write(f"  P90 tiempo en sistema   :  {summary['p90_total_time_min']:5.1f} min\n")
    out.write(f"  Pacientes tarde         :  {summary['pct_late']:5.1f}%\n")
    out.write(f"  Retraso medio si tarde       :  {summary['mean_lateness_min']:5.1f} min\n")
    out.write(f"  Duracion PET/CT media   :  {summary['mean_petct_duration_min']:5.1f} min\n")
    out.write(f"  Duracion simulacion     :  {summary['sim_duration_min']:5.1f} min  "
              f"(aprox {_min_to_hhmm(summary['sim_duration_min'])})\n")

    out.write("\n  ASIGNACION DE INYECTORIOS\n\n")
    out.write(f"  {'Paciente':<10} {'Inyectorio':<12} {'Ingreso':<10} {'Egreso':<10} {'Duracion':>10}\n")
    out.write(f"  {'-'*8:<10} {'-'*10:<12} {'-'*7:<10} {'-'*7:<10} {'-'*8:>10}\n")
    for _, r in df.iterrows():
        duracion = r["Fin traslado"] - r["Inyectorio OK"]
        out.write(f"  P{int(r['ID']):02d}       {r['Inyectorio']:<12} "
                  f"{_min_to_hhmm(r['Inyectorio OK']):<10} "
                  f"{_min_to_hhmm(r['Fin traslado']):<10} "
                  f"{duracion:>7.0f} min\n")
    out.write("=" * W + "\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# 11. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

# Paleta de colores por etapa
STAGE_COLORS = {
    # ── Recepción ──────────────────────────────────────────────────────────────
    "Espera recep.":      "#a8c4e0",   # azul claro  — espera por recepción
    "Recepción":          "#4e79a7",   # azul        — servicio de recepción
    # ── Sala de espera ─────────────────────────────────────────────────────────
    "Sala espera":        "#f28e2b",   # naranja     — en sala de espera, aguardando inyectorio
    "Espera hot room":    "#f0a8a9",   # rojo claro  — en inyectorio, hot room ocupado
    "Hot room":           "#e15759",   # rojo        — preparación de dosis
    "Inyección":          "#76b7b2",   # teal        — administración
    "Captación":          "#59a14f",   # verde       — uptake (60 min)
    # ── PET/CT ─────────────────────────────────────────────────────────────────
    "Espera PET/CT":      "#edc948",   # amarillo    — en inyectorio, scanner ocupado
    "Baño+traslado":      "#b07aa1",   # violeta     — PET/CT reservado, aún en inyectorio
    "PET/CT scan":        "#ff9da7",   # rosa        — adquisición pura
}


def _fmt_xaxis(ax):
    """Formatea el eje X en HH:MM (base 08:00)."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: _min_to_hhmm(x)
    ))
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.tick_params(axis="x", rotation=45)


def plot_gantt(df: pd.DataFrame, save: bool = True):
    """
    Diagrama de Gantt: recorrido temporal de cada paciente.
    Cada barra de color representa una etapa del flujo.
    """
    n = len(df)
    fig, ax = plt.subplots(figsize=(15, max(5, n * 0.7 + 2)))

    def hbar(y, start, end, stage):
        ax.barh(y, end - start, left=start, height=0.55,
                color=STAGE_COLORS[stage], edgecolor="white", linewidth=0.5)

    for _, r in df.iterrows():
        pid = r["ID"]
        # Recepción: espera (puede ser 0) + servicio
        hbar(pid, r["Llegada"],          r["Recepción inicio"], "Espera recep.")
        hbar(pid, r["Recepción inicio"], r["Fin recepción"],    "Recepción")
        # Sala de espera: desde fin de recepción hasta inyectorio asignado
        hbar(pid, r["Fin recepción"],    r["Inyectorio OK"],    "Sala espera")
        # Hot room: espera en inyectorio (puede ser 0) + preparación de dosis
        hbar(pid, r["Inyectorio OK"],    r["Hot room inicio"],  "Espera hot room")
        hbar(pid, r["Hot room inicio"],  r["Inyección"],        "Hot room")
        # Resto del recorrido
        hbar(pid, r["Inyección"],        r["Fin inyección"],    "Inyección")
        hbar(pid, r["Fin inyección"],    r["Fin captación"],    "Captación")
        hbar(pid, r["Fin captación"],    r["PET/CT reservado"], "Espera PET/CT")
        hbar(pid, r["PET/CT reservado"], r["PET/CT scan"],      "Baño+traslado")
        hbar(pid, r["PET/CT scan"],      r["Fin PET/CT"],       "PET/CT scan")
        # Etiqueta con ID del inyectorio centrada en la barra de captación
        mid = (r["Fin inyección"] + r["Fin captación"]) / 2
        ax.text(mid, pid, r["Inyectorio"], ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")

    _fmt_xaxis(ax)
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([f"P{int(r['ID']):02d}" for _, r in df.iterrows()])
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Paciente")
    ax.set_title("PET/CT v0.2 — Diagrama de Gantt", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=c, label=l) for l, c in STAGE_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    if save:
        fig.savefig("gantt_petct.png", dpi=150, bbox_inches="tight")
        print("  ✓ Gantt guardado: gantt_petct.png")
        plt.close(fig)
        return None
    return fig


def plot_utilization(summary: dict, save: bool = True):
    """Gráfico de barras de ocupación de cada recurso durante el período simulado."""
    util   = summary["utilization"]
    busy   = summary["busy_minutes"]
    labels = list(util.keys())
    values = list(util.values())
    # Los dos últimos son ambos PET/CT: tono pleno y tono claro
    colors = ["#4e79a7", "#e15759", "#59a14f", "#ff9da7", "#ffd1d8"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.55)

    for bar, label in zip(bars, labels):
        h       = bar.get_height()
        raw_min = busy[label]
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, h / 2,
                f"{raw_min:.0f} min", ha="center", va="center",
                fontsize=8, color="white", alpha=0.85)

    ax.axhline(80, color="red", linestyle="--", alpha=0.5, label="80% ref.")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Ocupación durante período simulado (%)")
    ax.set_title("Ocupación de Recursos durante Período Simulado — PET/CT v0.2",
                 fontsize=11, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    if save:
        fig.savefig("utilizacion_petct.png", dpi=150, bbox_inches="tight")
        print("  ✓ Utilización guardada: utilizacion_petct.png")
        plt.close(fig)
        return None
    return fig


def plot_wait_times(df: pd.DataFrame, save: bool = True):
    """Tiempos de espera por paciente para inyectorio y PET/CT."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, title, color in [
        (axes[0], "Espera inyect.", "Espera por Inyectorio",  "#f28e2b"),
        (axes[1], "Espera PET/CT",  "Espera por PET/CT",      "#edc948"),
    ]:
        ax.bar(df["ID"], df[col], color=color, edgecolor="white")
        ax.set_xlabel("Paciente")
        ax.set_ylabel("Minutos")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(df["ID"])
        ax.set_xticklabels([f"P{int(i):02d}" for i in df["ID"]])
        mean_val = df[col].mean()
        ax.axhline(mean_val, color="red", linestyle="--", alpha=0.7,
                   label=f"Media: {mean_val:.1f} min")
        ax.legend(fontsize=9)

    fig.suptitle("Tiempos de Espera — PET/CT v0.2",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig("esperas_petct.png", dpi=150, bbox_inches="tight")
        print("  ✓ Esperas guardadas: esperas_petct.png")
        plt.close(fig)
        return None
    return fig


def plot_waiting_room_occupancy(df: pd.DataFrame, summary: dict, save: bool = True):
    """
    Serie temporal de la cantidad de pacientes en sala de espera.

    Construcción de la curva escalonada:
      - Evento +1 en cada 'Sala espera entra'
      - Evento -1 en cada 'Sala espera sale'
      - Se ordenan por tiempo (salidas antes que entradas en t igual)
      - La suma acumulada da N(t)
    """
    events = []
    for _, r in df.iterrows():
        if r["Sala espera sale"] > r["Sala espera entra"] + 1e-9:
            events.append((r["Sala espera entra"], +1))
            events.append((r["Sala espera sale"], -1))
    events.sort(key=lambda x: (x[0], x[1]))

    # Construir función escalón: dos puntos por evento (antes y después)
    t_curve = [0]
    n_curve = [0]
    n = 0
    for t, delta in events:
        t_curve.append(t)
        n_curve.append(n)       # valor justo antes del evento
        n += delta
        t_curve.append(t)
        n_curve.append(n)       # valor justo después del evento

    # Extender hasta el fin de la simulación
    t_end = df["Fin PET/CT"].max()
    t_curve.append(t_end)
    n_curve.append(n)

    peak = summary["peak_waiting_room"]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(t_curve, n_curve, step="post", alpha=0.25, color="#f28e2b")
    ax.step(t_curve, n_curve, where="post", color="#f28e2b", linewidth=2,
            label="N pacientes en sala espera")
    ax.axhline(peak, color="red", linestyle="--", alpha=0.6,
               label=f"Máximo: {peak} pac.")

    # Marcar cada ingreso/egreso con una línea vertical tenue
    for _, r in df.iterrows():
        ax.axvline(r["Sala espera entra"], color="#4e79a7", alpha=0.2, linewidth=0.8)
        ax.axvline(r["Sala espera sale"],  color="#e15759", alpha=0.2, linewidth=0.8)

    _fmt_xaxis(ax)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("N pacientes")
    ax.set_title("Ocupación de Sala de Espera PET/CT — v0.2",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(axis="both", alpha=0.2)

    fig.tight_layout()
    if save:
        fig.savefig("sala_espera_petct.png", dpi=150, bbox_inches="tight")
        print("  ✓ Sala espera guardada: sala_espera_petct.png")
        plt.close(fig)
        return None
    return fig


def plot_timeline_resources(df: pd.DataFrame, params: dict, save: bool = True):
    """
    Línea de tiempo de ocupación de cada recurso.
    Cada inyectorio tiene su propia fila: permite ver disponibilidad individual,
    bloqueos esperando PET/CT, huecos del scanner y sincronización uptake–PET.
    """
    injectors = [f"I{i}" for i in range(1, params["num_injectors"] + 1)]
    # Filas: Recepcion | Hot room | I1 … In | PET/CT
    n_rows = 2 + len(injectors) + 1
    row_height = 1.5
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(14, 1.5 + n_rows * row_height),
                             sharex=True)
    ax_recep = axes[0]
    ax_hot   = axes[1]
    ax_inj   = {inj: axes[2 + i] for i, inj in enumerate(injectors)}
    ax_petct = axes[-1]

    BAR_H = 0.65

    # ── Recepción ─────────────────────────────────────────────────────────────
    for _, r in df.iterrows():
        ax_recep.barh(0, r["Fin recepción"] - r["Recepción inicio"],
                      left=r["Recepción inicio"], height=BAR_H,
                      color="#4e79a7", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax_recep.set_ylabel("Recepcion", fontsize=9, labelpad=4)
    ax_recep.set_yticks([])
    ax_recep.text(0.005, 0.72, "cap 1", transform=ax_recep.transAxes, fontsize=7, color="gray")
    ax_recep.grid(axis="x", alpha=0.2)

    # ── Cuarto caliente ───────────────────────────────────────────────────────
    for _, r in df.iterrows():
        ax_hot.barh(0, r["Inyección"] - r["Hot room inicio"],
                    left=r["Hot room inicio"], height=BAR_H,
                    color="#e15759", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax_hot.set_ylabel("Hot room", fontsize=9, labelpad=4)
    ax_hot.set_yticks([])
    ax_hot.text(0.005, 0.72, "cap 1", transform=ax_hot.transAxes, fontsize=7, color="gray")
    ax_hot.grid(axis="x", alpha=0.2)

    # ── Inyectorios individuales ───────────────────────────────────────────────
    # Sub-segmentos dentro del tiempo de ocupación del inyectorio
    INJ_SEGMENTS = [
        # (col_inicio, col_fin, clave en STAGE_COLORS)
        ("Inyectorio OK",    "Hot room inicio",   "Espera hot room"),
        ("Hot room inicio",  "Inyección",         "Hot room"),
        ("Inyección",        "Fin inyección",     "Inyección"),
        ("Fin inyección",    "Fin captación",     "Captación"),
        ("Fin captación",    "PET/CT reservado",  "Espera PET/CT"),
        ("PET/CT reservado", "Fin traslado",      "Baño+traslado"),
    ]
    for inj_id, ax in ax_inj.items():
        pats = df[df["Inyectorio"] == inj_id]
        for _, r in pats.iterrows():
            for t0_col, t1_col, stage in INJ_SEGMENTS:
                width = r[t1_col] - r[t0_col]
                if width > 0:
                    ax.barh(0, width, left=r[t0_col], height=BAR_H,
                            color=STAGE_COLORS[stage], alpha=0.88,
                            edgecolor="white", linewidth=0.4)
            # Etiqueta de paciente centrada en la barra de captación
            mid = (r["Fin inyección"] + r["Fin captación"]) / 2
            ax.text(mid, 0, f"P{int(r['ID']):02d}",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white")
        ax.set_ylabel(inj_id, fontsize=9, labelpad=4)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.2)

    # ── PET/CT ────────────────────────────────────────────────────────────────
    # Segmento claro = baño+traslado (reservado pero aún en inyectorio)
    # Segmento pleno = adquisición pura
    for _, r in df.iterrows():
        ax_petct.barh(0, r["PET/CT scan"] - r["PET/CT reservado"],
                      left=r["PET/CT reservado"], height=BAR_H,
                      color="#ffd1d8", alpha=0.9, edgecolor="white", linewidth=0.4)
        ax_petct.barh(0, r["Fin PET/CT"] - r["PET/CT scan"],
                      left=r["PET/CT scan"], height=BAR_H,
                      color="#ff9da7", alpha=0.9, edgecolor="white", linewidth=0.4)
    ax_petct.set_ylabel("PET/CT", fontsize=9, labelpad=4)
    ax_petct.set_yticks([])
    ax_petct.text(0.005, 0.72, "cap 1", transform=ax_petct.transAxes, fontsize=7, color="gray")
    ax_petct.grid(axis="x", alpha=0.2)

    # ── Eje X y leyenda ───────────────────────────────────────────────────────
    _fmt_xaxis(axes[-1])
    axes[-1].set_xlabel("Hora del día")

    legend_stages = ["Espera hot room", "Hot room", "Inyección",
                     "Captación", "Espera PET/CT", "Baño+traslado"]
    patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in legend_stages]
    patches += [
        mpatches.Patch(color="#ffd1d8", label="PET/CT reservado (pre-scan)"),
        mpatches.Patch(color="#ff9da7", label="PET/CT scan"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Ocupación de Recursos a lo largo del día — PET/CT v0.2",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    if save:
        fig.savefig("ocupacion_recursos.png", dpi=150, bbox_inches="tight")
        print("  ✓ Ocupación guardada: ocupacion_recursos.png")
        plt.close(fig)
        return None
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12. REPETICIONES / ANÁLISIS MONTE CARLO DE JORNADAS
# ─────────────────────────────────────────────────────────────────────────────

def run_replications(params: dict = PARAMS, n_replications: Optional[int] = None,
                     progress_callback=None) -> pd.DataFrame:
    """
    Ejecuta muchas jornadas independientes y devuelve una fila por jornada.

    Una única corrida estocástica sirve para ilustrar el flujo. Las conclusiones sobre
    capacidad y esperas deben basarse en la distribución de muchas corridas.
    """
    n_rep = params["replications"] if n_replications is None else int(n_replications)
    base_seed = int(params["random_seed"])
    rows = []

    for rep in range(n_rep):
        seed = base_seed + rep * 1_000_003
        records, mon, sim_duration = run_simulation(params, seed=seed)
        df = build_dataframe(records)
        validate_simulation(df, params, verbose=False)
        summary = compute_summary(df, mon, sim_duration, params)

        if progress_callback is not None:
            progress_callback((rep + 1) / n_rep)
        rows.append({
            "Replicacion": rep + 1,
            "Seed": seed,
            "Espera media inyectorio": summary["mean_wait_injector_min"],
            "Espera media PET/CT": summary["mean_wait_petct_min"],
            "P90 espera PET/CT": summary["p90_wait_petct_min"],
            "Tiempo medio sistema": summary["mean_total_time_min"],
            "Fin ultimo paciente": summary["last_patient_out_min"],
            "Max sala espera": summary["peak_waiting_room"],
            "Max concurrencia": summary["peak_concurrent"],
            "PET scan busy min": summary["busy_minutes"]["PET/CT (adq.)"],
            "PET reservado min": summary["busy_minutes"]["PET/CT (reservado)"],
            "% pacientes tarde": summary["pct_late"],
        })

    return pd.DataFrame(rows)


def summarize_replications(rep_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen descriptivo: media, DE y percentiles entre jornadas."""
    metrics = [
        "Espera media inyectorio",
        "Espera media PET/CT",
        "P90 espera PET/CT",
        "Tiempo medio sistema",
        "Fin ultimo paciente",
        "Max sala espera",
        "Max concurrencia",
        "PET scan busy min",
        "PET reservado min",
        "% pacientes tarde",
    ]
    rows = []
    for col in metrics:
        x = rep_df[col]
        rows.append({
            "Métrica": col,
            "Media": x.mean(),
            "DE": x.std(ddof=1),
            "P05": x.quantile(0.05),
            "P50": x.quantile(0.50),
            "P95": x.quantile(0.95),
        })
    return pd.DataFrame(rows)


def plot_replication_distributions(rep_df: pd.DataFrame, save: bool = True):
    """Distribuciones entre jornadas de cuatro indicadores operativos principales."""
    metrics = [
        ("Espera media PET/CT", "Espera media PET/CT (min)"),
        ("Tiempo medio sistema", "Tiempo medio en sistema (min)"),
        ("Fin ultimo paciente", "Fin del último paciente (min desde 08:00)"),
        ("Max sala espera", "Máximo en sala de espera (pacientes)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.hist(rep_df[col], bins=20, edgecolor="white", alpha=0.85)
        ax.axvline(rep_df[col].median(), linestyle="--", linewidth=1.5,
                   label=f"Mediana {rep_df[col].median():.1f}")
        ax.axvline(rep_df[col].quantile(0.95), linestyle=":", linewidth=1.5,
                   label=f"P95 {rep_df[col].quantile(0.95):.1f}")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Jornadas")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("PET/CT v0.2 — Variabilidad entre jornadas", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig("estadistica_jornadas_petct.png", dpi=150, bbox_inches="tight")
        print("  ✓ Estadística de jornadas guardada: estadistica_jornadas_petct.png")
        plt.close(fig)
        return None
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 13. PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("PET/CT Simulation v0.2 — iniciando...\n")

    # Correr simulación
    records, mon, sim_duration = run_simulation(PARAMS, seed=PARAMS["random_seed"])

    # Construir DataFrame y validar invariantes del modelo
    df = build_dataframe(records)
    validate_simulation(df, PARAMS)
    df.to_csv("patient_results_petct_v02.csv", index=False, float_format="%.2f")
    print("  ✓ Resultados de la jornada guardados: patient_results_petct_v02.csv")

    # Log de eventos
    event_log = build_event_log(records)
    event_log.to_csv("event_log_petct.csv", index=False, float_format="%.1f")
    print(f"  ✓ Log de eventos guardado: event_log_petct.csv  "
          f"({len(event_log)} filas, {event_log['Paciente'].nunique()} pacientes)\n")

    # Muestra del log: primer paciente
    import sys
    _out = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
    _out.write("  LOG DE EVENTOS — P01\n\n")
    p01 = event_log[event_log["Paciente"] == "P01"][
        ["Estado", "Ubicación", "Inicio", "Fin", "Duración"]
    ].copy()
    p01["Inicio"] = p01["Inicio"].apply(_min_to_hhmm)
    p01["Fin"]    = p01["Fin"].apply(_min_to_hhmm)
    p01["Duración"] = p01["Duración"].apply(lambda x: f"{x:.0f} min")
    _out.write(p01.to_string(index=False) + "\n\n")

    # Métricas y reporte
    summary = compute_summary(df, mon, sim_duration, PARAMS)
    print_report(df, summary)

    # Gráficos
    print("Generando visualizaciones...")
    plot_gantt(df)
    plot_utilization(summary)
    plot_wait_times(df)
    plot_waiting_room_occupancy(df, summary)
    plot_timeline_resources(df, PARAMS)

    # Muchas jornadas: base correcta para inferencia estadística
    print(f"\nEjecutando {PARAMS['replications']} jornadas estocásticas...")
    rep_df = run_replications(PARAMS)
    rep_summary = summarize_replications(rep_df)
    rep_df.to_csv("replicaciones_petct_v02.csv", index=False, float_format="%.3f")
    rep_summary.to_csv("resumen_estadistico_petct_v02.csv", index=False, float_format="%.3f")
    plot_replication_distributions(rep_df)

    _out.write("\n  RESUMEN ENTRE JORNADAS (Monte Carlo)\n\n")
    _out.write(rep_summary.to_string(index=False, float_format=lambda x: f"{x:6.2f}") + "\n")
    _out.write(f"\n  Archivos: replicaciones_petct_v02.csv / resumen_estadistico_petct_v02.csv\n")

    print("\nSimulación completada.")
