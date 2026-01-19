import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone, date
import hashlib
import math
import random
from IPython.display import display

# =========================================================
# PARÁMETROS EDITABLES
# =========================================================
N = 3000
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 6 posiciones y sus parámetros de salario: μ (mean) y σ (std)
salary_params = {
    "Data Scientist": {"mean": 48000, "std": 8000},
    "Data Engineer": {"mean": 52000, "std": 9000},
    "Data Analyst": {"mean": 42000, "std": 7000},
    "Machine Learning Engineer": {"mean": 58000, "std": 10000},
    "Business Intelligence Analyst": {"mean": 40000, "std": 6500},
    "AI Engineer": {"mean": 60000, "std": 10000},
}

# Distribución (probabilidades) para asignar posiciones (usar claves exactas de salary_params)
position_probs = {
    "Data Scientist": 0.24,
    "Data Engineer":  0.22,
    "Data Analyst":   0.20,
    "Machine Learning Engineer": 0.18,
    "Business Intelligence Analyst": 0.16,
}
assert abs(sum(position_probs.values()) - 1.0) < 1e-9, "Las probabilidades de posición deben sumar 1.0"

# Alias opcionales (para mostrar en corto en la columna 'posicion')
position_display_alias = {
    "Machine Learning Engineer": "ML Engineer",
    "Business Intelligence Analyst": "BI Analyst",
}

# Experiencia
experience_levels = ["Intern", "Junior", "Mid", "Senior", "Lead"]
experience_probs  = [0.08, 0.28, 0.34, 0.22, 0.08]

# Rangos de edad (años) por experiencia 
experience_age_ranges = {
    "Intern": (17, 24),
    "Junior": (20, 30),
    "Mid":    (27, 45),
    "Senior": (40, 60),
    "Lead":   (55, 70),
}

# País/ciudad
countries = {
    "España": {
        "cities": {"Madrid": 0.4, "Barcelona": 0.35, "Valencia": 0.1, "Sevilla": 0.08, "Bilbao": 0.07},
    }
}
country_probs = {"España": 1}

# Salario (límites globales)
MIN_SALARY = 12_000
MAX_SALARY = 350_000
CLIP_SIGMAS = 4
ROUNDING_MODE = "trunc"

# Empresas ficticias
company_roots = [
    "TechNova", "DataWorks", "InsightHub", "BlueCloud", "Quantica", "NeoMetrics",
    "CleverByte", "OpenOrbit", "VectorAI", "PlexData", "Synthetica", "DeltaMind"
]
company_suffix = ["SL", "SA", "GmbH", "Ltd.", "Inc.", "LLC", "SAS", "SRL"]

# =========================================================
# FUNCIONES RANDOMIZERS
# =========================================================
def normalized_probs(d: dict):
    keys = list(d.keys())
    vals = np.array(list(d.values()), dtype=float)
    vals = vals / vals.sum()
    return keys, vals

def random_name():
    first_names = ["Carlos", "María", "Lucía", "David", "Jorge", "Ana", "Marta", "Pablo", "Sofía", "Diego",
                   "Sara", "Elena", "Laura", "Andrés", "Raúl", "Carmen", "Paula", "Alberto", "Nuria", "Víctor"]
    last_names  = ["García", "López", "Martínez", "Sánchez", "Pérez", "Gómez", "Fernández", "Díaz", "Ruiz", "Hernández",
                   "Jiménez", "Moreno", "Muñoz", "Álvarez", "Romero", "Alonso", "Gutiérrez", "Navarro", "Torres", "Domínguez"]
    fn = np.random.choice(first_names)
    ln = np.random.choice(last_names)
    return f"{fn} {ln}"

def normalize_email_localpart(name):
    mapping = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ ", "aeiouunAEIOUUN_")
    base = name.translate(mapping).lower().replace("__", "_")
    base = base.replace(".", "_")
    return base

def unique_emails_from_names(names):
    domains = ["gmail.com", "outlook.com", "yahoo.com", "hotmail.com", "proton.me"]
    seen = {}
    emails = []
    for n in names:
        local = normalize_email_localpart(n).replace(" ", "_")
        salt = hashlib.md5(n.encode()).hexdigest()[:4]
        candidate = f"{local}.{salt}@{np.random.choice(domains)}"
        if candidate in seen:
            cnt = seen[candidate] + 1
            seen[candidate] = cnt
            candidate = candidate.replace(f".{salt}@", f".{salt}{cnt}@")
        else:
            seen[candidate] = 1
        emails.append(candidate)
    return emails

def to_500_trunc(arr):
    return (np.floor(np.asarray(arr, dtype=float) / 500.0) * 500.0).astype(int)

def to_500_round(arr):
    return (np.round(np.asarray(arr, dtype=float) / 500.0) * 500.0).astype(int)

def gen_gaussian(size, mean, std, min_val, max_val, clip_k):
    s = np.random.normal(loc=mean, scale=std, size=size)
    s = np.clip(s, mean - clip_k*std, mean + clip_k*std)
    s = np.clip(s, min_val, max_val)
    return s

# =========================================================
# CONFIG FECHAS: Semana fija (12–18 enero 2026)
# =========================================================
WEEK_START_DATE = date(2026, 1, 12)  # lunes 12-01-2026
DAYS_IN_WEEK = 7
DAY_STARTS = [
    datetime(year=WEEK_START_DATE.year, month=WEEK_START_DATE.month, day=WEEK_START_DATE.day, tzinfo=timezone.utc)
    + timedelta(days=i)
    for i in range(DAYS_IN_WEEK)
]

# =========================================================
# GENERACIÓN DE CAMPOS
# =========================================================
# Nombres y emails
names = [random_name() for _ in range(N)]
emails = unique_emails_from_names(names)

# País/ciudad
country_keys, country_weights = normalized_probs(country_probs)
selected_countries = np.random.choice(country_keys, size=N, p=country_weights)
selected_cities = []
for c in selected_countries:
    city_keys, city_weights = normalized_probs(countries[c]["cities"])
    selected_cities.append(np.random.choice(city_keys, p=city_weights))

# Posición (coincidiendo con salary_params) y experiencia
pos_keys, pos_weights = normalized_probs(position_probs)
positions_sel = np.random.choice(pos_keys, size=N, p=pos_weights)  # nombres largos (claves salary_params)
positions_display = [position_display_alias.get(p, p) for p in positions_sel]

# Experiencia con orden natural
experience_levels = ["Intern", "Junior", "Mid", "Senior", "Lead"]
experience_probs  = [0.08, 0.28, 0.34, 0.22, 0.08]
experiences = np.random.choice(experience_levels, size=N, p=experience_probs)
exp_cat = pd.Categorical(experiences, categories=experience_levels, ordered=True)

# Empresa
companies = [f"{np.random.choice(company_roots)} {np.random.choice(company_suffix)}" for _ in range(N)]

# =========================================================
# FECHAS: created_at dentro de la semana con volumen diario aleatorio
# =========================================================
day_probs = np.random.dirichlet(alpha=np.ones(DAYS_IN_WEEK))
day_counts = np.random.multinomial(N, day_probs)

created_at = []
for day_start, count in zip(DAY_STARTS, day_counts):
    secs = np.random.rand(count) * 86400.0  # segundos aleatorios del día
    for s in secs:
        created_at.append(day_start + timedelta(seconds=float(s)))
random.shuffle(created_at)
assert len(created_at) == N

# ✅ consent y updated_at iguales a created_at; consent_accepted siempre 1
updated_at = created_at[:] 
consent_accepted = np.ones(N, dtype=int)
consent_ts = created_at[:]

# =========================================================
# FECHA DE NACIMIENTO COHERENTE CON LA EXPERIENCIA
# =========================================================
TODAY_DATE = datetime.now(timezone.utc).date()

def birthdate_for_experience(exp_name: str, today: date) -> date:
    amin, amax = experience_age_ranges[exp_name]
    min_days = int(amin * 365.2425)
    max_days = int(amax * 365.2425)
    age_days = np.random.randint(min_days, max_days + 1)
    return today - timedelta(days=int(age_days))

birthdates = [birthdate_for_experience(exp, TODAY_DATE) for exp in exp_cat.astype(str)]

# Validación edad-experiencia
ages_years = ((pd.Timestamp(TODAY_DATE) - pd.to_datetime(birthdates)).days / 365.2425)
tmp_df_check = pd.DataFrame({"exp": exp_cat.astype(str), "age": ages_years})
for e, (amin, amax) in experience_age_ranges.items():
    s = tmp_df_check.loc[tmp_df_check["exp"] == e, "age"]
    if len(s) > 0:
        assert (s >= amin - 0.05).all() and (s <= amax + 0.05).all(), f"Edades fuera de rango para {e}"

# =========================================================
# SALARIO BRUTO por cuantiles según experiencia
# =========================================================
experience_quantile_bands = {
    "Intern": (0.00, 0.25),
    "Junior": (0.15, 0.40),
    "Mid":    (0.40, 0.60),
    "Senior": (0.60, 0.85),
    "Lead":   (0.80, 0.95),
}

def sample_by_quantile_band(n, mean, std, qlow, qhigh, pool_factor=6):
    pool_size = max(n * pool_factor, 10_000)
    pool = gen_gaussian(pool_size, mean, std, MIN_SALARY, MAX_SALARY, CLIP_SIGMAS)
    lo = np.quantile(pool, qlow)
    hi = np.quantile(pool, qhigh)
    sliced = pool[(pool >= lo) & (pool <= hi)]

    tries = 0
    while len(sliced) < n and tries < 5:
        extra = gen_gaussian(pool_size, mean, std, MIN_SALARY, MAX_SALARY, CLIP_SIGMAS)
        pool = np.concatenate([pool, extra])
        sliced = pool[(pool >= lo) & (pool <= hi)]
        tries += 1

    if len(sliced) == 0:
        qlow2 = max(0.0, qlow * 0.9)
        qhigh2 = min(1.0, qhigh * 1.1)
        lo = np.quantile(pool, qlow2)
        hi = np.quantile(pool, qhigh2)
        sliced = pool[(pool >= lo) & (pool <= hi)]
        if len(sliced) == 0:
            sliced = pool

    idx = np.random.randint(0, len(sliced), size=n)
    return sliced[idx]

new_salaries = np.empty(N, dtype=float)
pos_series = pd.Series(positions_sel)  # nombres "largos"
exp_series = pd.Series(exp_cat.astype(str))

for pos_name, idx_pos in pos_series.groupby(pos_series).groups.items():
    if pos_name not in salary_params:
        raise ValueError(f"Posición '{pos_name}' no está en salary_params.")
    mu = salary_params[pos_name]["mean"]
    sd = salary_params[pos_name]["std"]

    exp_sub = exp_series.loc[idx_pos]
    for exp_name, idx_exp in exp_sub.groupby(exp_sub).groups.items():
        qlow, qhigh = experience_quantile_bands.get(exp_name, (0.40, 0.60))
        n = len(idx_exp)
        sampled = sample_by_quantile_band(n, mu, sd, qlow, qhigh)
        sampled = to_500_round(sampled) if ROUNDING_MODE == "round" else to_500_trunc(sampled)
        sampled = np.clip(sampled, MIN_SALARY, MAX_SALARY)
        new_salaries[list(idx_exp)] = sampled

if np.isnan(new_salaries).any():
    raise ValueError("Quedaron salarios sin asignar; revisa bandas o posiciones/experiencias.")

# =========================================================
# CONSTRUIR DATAFRAME
# =========================================================
df = pd.DataFrame({
    "nombre": names,
    "email": emails,
    "created_at": pd.to_datetime(created_at, utc=True),
    "fecha_nacimiento": pd.to_datetime(birthdates).date,
    "pais": selected_countries,
    "ciudad": selected_cities,
    "experiencia": exp_series,
    "empresa": companies,
    "posicion": [position_display_alias.get(p, p) for p in positions_sel],
    "salario_bruto": new_salaries.astype(int),
    "consent_accepted": consent_accepted,
    "consent_ts": pd.to_datetime(consent_ts, utc=True),
    "updated_at": pd.to_datetime(updated_at, utc=True),
})

# =========================================================
# VALIDACIONES
# =========================================================
assert df["email"].is_unique, "Emails deben ser únicos por el constraint UNIQUE."
assert (df["salario_bruto"] >= MIN_SALARY).all(), "Salario por debajo del mínimo."
assert (df["salario_bruto"] <= MAX_SALARY).all(), "Salario por encima del máximo."
# created_at dentro de la semana y updated/consent iguales a created
assert df["created_at"].dt.date.min() >= WEEK_START_DATE
assert df["created_at"].dt.date.max() <= (WEEK_START_DATE + timedelta(days=DAYS_IN_WEEK-1))
assert (df["updated_at"] == df["created_at"]).all(), "updated_at debe ser == created_at"
assert (df.loc[df["consent_accepted"]==1, "consent_ts"] == df.loc[df["consent_accepted"]==1, "created_at"]).all(), "consent_ts debe ser == created_at"
assert (df["consent_accepted"] == 1).all(), "consent_accepted debe ser siempre 1"

# =========================================================
# EXPORTACIÓN
# =========================================================
csv_path = "salaries_sintetico.csv"
df.to_csv(csv_path, index=False)
print(f"[OK] CSV generado: {csv_path} ({len(df)} filas).")

def escape_sql(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NULL"
    if pd.isna(value):
        return "NULL"
    if isinstance(value, (pd.Timestamp, datetime)):
        if isinstance(value, pd.Timestamp) and value.tz is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        elif isinstance(value, datetime) and value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return f"'{value.isoformat()}'"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.2f}"
    s = str(value).replace("'", "''")
    return f"'{s}'"

batch_size = 1000
sql_file = "insert_salaries_sintetico.sql"
with open(sql_file, "w", encoding="utf-8") as f:
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start+batch_size]
        values = []
        for _, r in chunk.iterrows():
            row = (
                escape_sql(r["nombre"]),
                escape_sql(r["email"]),
                escape_sql(r["created_at"]),
                escape_sql(r["fecha_nacimiento"]),
                escape_sql(r["pais"]),
                escape_sql(r["ciudad"]),
                escape_sql(r["experiencia"]),
                escape_sql(r["empresa"]),
                escape_sql(r["posicion"]),
                escape_sql(int(r["salario_bruto"])),
                escape_sql(int(r["consent_accepted"])),
                escape_sql(r["consent_ts"]),
                escape_sql(r["updated_at"])
            )
            values.append("(" + ", ".join(map(str, row)) + ")")
        stmt = (
            "INSERT INTO salaries "
            "(nombre, email, created_at, fecha_nacimiento, pais, ciudad, experiencia, empresa, posicion, salario_bruto, consent_accepted, consent_ts, updated_at) VALUES\n"
            + ",\n".join(values) + ";\n"
        )
        f.write(stmt)
print(f"[OK] Script SQL generado: {sql_file}")

# =========================================================
# RESUMEN POR POSICIÓN Y EXPERIENCIA
# =========================================================
summary = (
    df.groupby(["posicion", "experiencia"])["salario_bruto"]
      .agg(n="count", mean_emp="mean", std_emp="std",
           p5=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95))
      .reset_index()
)


summary2 = (
    df.groupby("experiencia", observed=True)["fecha_nacimiento"]
      .agg(n="count", min="min", max="max")
      .reset_index()
)

print("\nResumen salarios por posición y experiencia (empírico):")
display(summary.to_string(index=False))
print("--------------------------------------------")
display(summary2)
print("--------------------------------------------")
display(df.head(10))
print("--------------------------------------------")
display(df["created_at"].min())
display(df["created_at"].max())
