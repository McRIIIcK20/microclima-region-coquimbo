# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Microclimas de la Región de Coquimbo: costa vs. valle (2024–2025)
#
# ¿Cuánto cambia el clima en 80 km, desde la costa de Tongoy hasta el valle de Elqui? Este análisis compara
# temperatura, amplitud térmica, lluvia, radiación y humedad en 8 localidades de la región durante dos años completos.
#
# **Origen.** Es la versión ampliada del Laboratorio 1 de Machine Learning (2025), que hacía *web scraping* de CEAZA
# y consultaba la API de OpenWeather para 5 localidades entre enero y agosto de 2025 (notebook original en
# `originales/`). Aquí se usa la API histórica de **Open-Meteo** (gratuita, sin API key) para tener 2 años y 8 puntos.

# %%
import json, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

sns.set_theme(style="whitegrid")
FIG = Path("../figures"); FIG.mkdir(exist_ok=True)
DATA = Path("../data/clima_coquimbo_2024_2025.csv")

# %% [markdown]
# ## 1. Datos
#
# La celda siguiente descarga los datos si el CSV no existe (en Colab funciona tal cual). Variables diarias:
# temperatura máxima, mínima y media (°C), precipitación (mm), viento máximo (km/h), radiación solar (MJ/m²) y
# humedad relativa media (%).

# %%
LOCALIDADES = {
    "Coquimbo": (-29.9533, -71.3436), "La Serena": (-29.9027, -71.2519),
    "Tongoy-Guanaqueros": (-30.2567, -71.4950), "Punta de Choros": (-29.2453, -71.4636),
    "La Higuera": (-29.5097, -71.2611), "Andacollo": (-30.2316, -71.0838),
    "Vicuña": (-30.0319, -70.7081), "Paihuano (Gabriela Mistral)": (-30.0283, -70.5181),
}

def descargar(path=DATA, start="2024-01-01", end="2025-12-31"):
    vars_ = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum",
             "wind_speed_10m_max", "shortwave_radiation_sum", "relative_humidity_2m_mean"]
    nombres = ["tmax", "tmin", "tmean", "precip_mm", "viento_max_kmh", "radiacion_MJm2", "hr_media"]
    frames = []
    for loc, (la, lo) in LOCALIDADES.items():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}"
               f"&start_date={start}&end_date={end}&daily={','.join(vars_)}&timezone=America%2FSantiago")
        js = json.load(urllib.request.urlopen(url, timeout=60))
        d = pd.DataFrame(js["daily"]).rename(columns=dict(zip(vars_, nombres))).rename(columns={"time": "fecha"})
        d.insert(0, "localidad", loc); d.insert(1, "lat", js["latitude"]); d.insert(2, "lon", js["longitude"])
        d.insert(3, "elevacion_m", js["elevation"]); frames.append(d)
    pd.concat(frames).to_csv(path, index=False)

if not DATA.exists():
    descargar()
df = pd.read_csv(DATA, parse_dates=["fecha"])
df["amplitud"] = df.tmax - df.tmin
df["mes"] = df.fecha.dt.month
df["anio"] = df.fecha.dt.year
ZONA = {"Coquimbo": "Costa", "La Serena": "Costa", "Tongoy-Guanaqueros": "Costa", "Punta de Choros": "Costa",
        "La Higuera": "Interior", "Andacollo": "Interior", "Vicuña": "Valle de Elqui",
        "Paihuano (Gabriela Mistral)": "Valle de Elqui"}
df["zona"] = df.localidad.map(ZONA)
print(df.shape, df.fecha.min().date(), "→", df.fecha.max().date())
df.head()

# %% [markdown]
# ## 2. Resumen por localidad

# %%
orden = df.groupby("localidad").tmean.mean().sort_values().index
resumen = df.groupby("localidad").agg(
    zona=("zona", "first"), elevacion_m=("elevacion_m", "first"),
    t_media=("tmean", "mean"), t_min_abs=("tmin", "min"), t_max_abs=("tmax", "max"),
    amplitud_media=("amplitud", "mean"), lluvia_anual_mm=("precip_mm", lambda s: s.sum() / 2),
    radiacion_media=("radiacion_MJm2", "mean"), humedad_media=("hr_media", "mean"),
).loc[orden].round(1)
resumen.to_csv("../data/resumen_localidades.csv")
resumen

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=df, x="tmean", y="localidad", order=orden, hue="zona", dodge=False, ax=axes[0])
axes[0].set_title("Temperatura media diaria (2024–2025)"); axes[0].set_xlabel("°C"); axes[0].set_ylabel(""); axes[0].legend(loc="lower right", fontsize=8)
sns.boxplot(data=df, x="amplitud", y="localidad", order=orden, hue="zona", dodge=False, ax=axes[1], legend=False)
axes[1].set_title("Amplitud térmica diaria (máx − mín)"); axes[1].set_xlabel("°C"); axes[1].set_ylabel("")
plt.tight_layout(); plt.savefig(FIG / "boxplots.png", dpi=120); plt.show()

# %% [markdown]
# ## 3. Ciclo anual

# %%
mens = df.groupby(["localidad", "mes"]).tmean.mean().unstack()
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.heatmap(mens.loc[orden[::-1]], annot=True, fmt=".1f", cmap="RdYlBu_r", ax=ax, cbar_kws={"label": "°C"})
ax.set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
ax.set_title("Temperatura media mensual"); ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(FIG / "heatmap_mensual.png", dpi=120); plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=False)
for loc in ["Coquimbo", "Punta de Choros", "Vicuña", "Paihuano (Gabriela Mistral)"]:
    d = df[df.localidad == loc].set_index("fecha")
    axes[0].plot(d.tmean.rolling(15, center=True).mean(), label=loc)
    axes[1].plot(d.amplitud.rolling(15, center=True).mean(), label=loc)
axes[0].set_title("Temperatura media (promedio móvil 15 días)"); axes[0].set_ylabel("°C")
axes[1].set_title("Amplitud térmica (promedio móvil 15 días)"); axes[1].set_ylabel("°C")
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "series.png", dpi=120); plt.show()

# %% [markdown]
# ## 4. ¿Qué explica las diferencias? Elevación y distancia a la costa

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for z, g in resumen.groupby("zona"):
    ax.scatter(g.elevacion_m, g.amplitud_media, s=120, label=z)
for loc, r in resumen.iterrows():
    ax.annotate(loc.split(" (")[0], (r.elevacion_m, r.amplitud_media), xytext=(6, 3), textcoords="offset points", fontsize=8)
r = np.corrcoef(resumen.elevacion_m, resumen.amplitud_media)[0, 1]
ax.set_xlabel("Elevación (m)"); ax.set_ylabel("Amplitud térmica media (°C)")
ax.set_title(f"Más altura y más lejos del mar → más amplitud (r = {r:.2f})"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "elevacion_amplitud.png", dpi=120); plt.show()

# %% [markdown]
# ## 5. Lluvia: pocos días, mucho impacto

# %%
lluvia = df[df.precip_mm >= 10].groupby(["fecha"]).agg(localidades=("localidad", "count"),
                                                        max_mm=("precip_mm", "max")).sort_values("max_mm", ascending=False)
dias_lluvia = df.assign(llueve=df.precip_mm >= 1).groupby("localidad").llueve.sum() / 2
print("Días al año con lluvia ≥ 1 mm:", dias_lluvia.round(0).astype(int).to_dict())
lluvia.head(8)

# %% [markdown]
# ## 6. Agrupación de localidades por clima (clustering)

# %%
feats = df.groupby("localidad").agg(t_media=("tmean", "mean"), t_verano=("tmax", lambda s: s[df.loc[s.index, "mes"].isin([12, 1, 2])].mean()),
                                    t_invierno=("tmin", lambda s: s[df.loc[s.index, "mes"].isin([6, 7, 8])].mean()),
                                    amplitud=("amplitud", "mean"), radiacion=("radiacion_MJm2", "mean"),
                                    humedad=("hr_media", "mean"))
Z = StandardScaler().fit_transform(feats)
link = linkage(Z, "ward")
feats["grupo"] = fcluster(link, 3, "maxclust")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
dendrogram(link, labels=[l.split(" (")[0] for l in feats.index], ax=axes[0], leaf_rotation=30)
axes[0].set_title("Dendrograma (Ward, variables estandarizadas)")
pc = PCA(2).fit(Z); P = pc.transform(Z)
axes[1].scatter(P[:, 0], P[:, 1], c=feats.grupo, cmap="Set1", s=120)
for (x, y), loc in zip(P, feats.index):
    axes[1].annotate(loc.split(" (")[0], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
axes[1].set_title(f"PCA: {pc.explained_variance_ratio_.sum():.0%} de la varianza en 2 componentes")
plt.tight_layout(); plt.savefig(FIG / "clusters.png", dpi=120); plt.show()
feats.round(1)

# %% [markdown]
# ## 7. Dashboard interactivo
#
# Se exporta un HTML autocontenido (Plotly) con la serie diaria de cada localidad, para abrir en el navegador.

# %%
import plotly.express as px
fig = px.line(df, x="fecha", y="tmean", color="localidad",
              hover_data={"tmax": True, "tmin": True, "precip_mm": True},
              title="Temperatura media diaria – Región de Coquimbo (Open-Meteo)",
              labels={"tmean": "°C", "fecha": ""})
fig.update_layout(template="plotly_white", height=500, legend_title_text="")
fig.write_html("../figures/dashboard_temperaturas.html", include_plotlyjs="cdn")
print("Guardado figures/dashboard_temperaturas.html")

# %% [markdown]
# ## 8. Comparación con el laboratorio original (OpenWeather, ene–ago 2025)

# %%
ene_ago = df[(df.anio == 2025) & (df.mes <= 8)]
comp = ene_ago.groupby("localidad").tmean.mean().round(2).sort_values()
comp

# %% [markdown]
# ## Conclusiones
# - El clustering separa **tres grupos**: (1) interior cálido y seco —Vicuña, Paihuano y Andacollo— con 16–20 °C de
#   media, veranos sobre 24 °C, humedad de 40–58 % y la mayor amplitud térmica (9–11 °C); (2) la costa templada
#   —Coquimbo, La Serena, Tongoy y La Higuera, esta última por la neblina costera— con 13–15 °C y humedad sobre 74 %;
#   y (3) **Punta de Choros**, la costa más estable: la menor amplitud (5,3 °C) y los inviernos más templados (10 °C de
#   mínima media), lo mismo que el laboratorio original identificó como "clima anómalo".
# - La amplitud térmica crece con la elevación y la distancia al mar: la camanchaca modera la costa, mientras que en
#   el valle de Elqui el día es cálido y la noche fría.
# - La lluvia se concentra en 12–18 días al año y en pocos frentes de invierno que alcanzan a toda la región el mismo
#   día (13–14 de junio y 2 de agosto de 2024, 1 de agosto de 2025).
# - Diferencia con el laboratorio original (OpenWeather, ene–ago 2025): ahí Punta de Choros salía como la más cálida
#   en promedio; con Open-Meteo lo son Vicuña y Paihuano. Open-Meteo usa reanálisis en una grilla de ~10 km, por lo
#   que los puntos costeros incluyen parte del océano; para un estudio fino conviene contrastar con las estaciones de
#   CEAZA.
