"""
Funciones para crear mapas de México usando datos geográficos.

Este módulo proporciona utilidades para:
- Cargar el `admin1.geojson` de México como `GeoDataFrame`.
- Generar mapas estáticos (base, coropléticos, ciudades principales) con matplotlib.
- Crear mapas interactivos con Folium, incluyendo paletas de color y tooltips modernos.

Dependencias: geopandas, matplotlib, pandas, numpy, seaborn, folium (opcional), branca.

Archivos de entrada:
- `admin1.geojson`: límites administrativos de entidades federativas.

Salida típica:
- Imágenes `.png` con mapas estáticos.
- Archivos `.html` con mapas interactivos (Folium).

Ejemplo mínimo de uso:
    gdf = cargar_geojson_mexico("admin1.geojson")
    fig, ax = crear_mapa_base_mexico(gdf)
    m = crear_mapa_interactivo_folium(gdf, minimal=True)
    guardar_mapa_interactivo(m, "data/mapa_interactivo_mexico.html")
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

def cargar_geojson_mexico(ruta_geojson):
    """
    Carga el GeoJSON administrativo de México y lo prepara para visualización.
    
    Args:
        ruta_geojson (str): Ruta al archivo `admin1.geojson`.
    
    Returns:
        gpd.GeoDataFrame | None: GeoDataFrame con las entidades; `None` si ocurre error.
    
    Raises:
        Muestra un mensaje por consola cuando falla la carga.
    
    Notas:
        - Reproyecta a `EPSG:4326` si el CRS de origen es distinto.
        - Imprime conteo de entidades y columnas disponibles para inspección rápida.
    
    Ejemplo:
        >>> gdf = cargar_geojson_mexico("admin1.geojson")
        >>> gdf.head()
    """
    try:
        # Cargar el GeoJSON
        gdf_mexico = gpd.read_file(ruta_geojson)
        
        # Convertir a un sistema de coordenadas apropiado para México (EPSG:4326 -> EPSG:6372)
        if gdf_mexico.crs != 'EPSG:4326':
            gdf_mexico = gdf_mexico.to_crs('EPSG:4326')
        
        print(f" [OK] GeoJSON cargado exitosamente: {len(gdf_mexico)} estados/regiones")
        print(f" [COLUMNAS] Columnas disponibles: {list(gdf_mexico.columns)}")
        
        return gdf_mexico
    
    except Exception as e:
        print(f" Error al cargar el GeoJSON: {e}")
        return None

def crear_mapa_base_mexico(gdf_mexico, titulo="Mapa de México", figsize=(12, 8)):
    """
    Genera un mapa base estático de México con estilo sobrio.
    
    Args:
        gdf_mexico (gpd.GeoDataFrame): Entidades y geometrías de México.
        titulo (str): Título mostrado en la figura.
        figsize (tuple[int, int]): Tamaño de la figura en pulgadas.
    
    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figura y ejes del mapa.
    
    Notas:
        - Oculta ejes y grilla para estética minimalista.
        - Usa relleno gris claro y bordes suaves.
    
    Ejemplo:
        >>> fig, ax = crear_mapa_base_mexico(gdf, "Mapa base")
        >>> guardar_mapa_como_imagen(fig, "data/mapa_base_mexico.png")
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Dibujar el mapa base con un estilo sobrio
    gdf_mexico.plot(ax=ax, color='#eeeeee', edgecolor='#b0b3b8', linewidth=0.6, alpha=1.0)
    
    # Configuración minimalista: título simple, sin ejes ni grid
    ax.set_title(titulo, fontsize=16)
    ax.set_axis_off()
    ax.set_facecolor('white')
    
    return fig, ax

def crear_mapa_coroplético_ventas(gdf_mexico, datos_ventas, columna_region, columna_valor,
                                  titulo="Ventas por Región", cmap="plasma", figsize=(14, 10)):
    """
    Crea un mapa coroplético por regiones a partir de datos de ventas.
    
    Args:
        gdf_mexico (gpd.GeoDataFrame): Entidades y geometrías.
        datos_ventas (pd.DataFrame): Datos agregados por región.
        columna_region (str): Nombre de columna con identificador de región.
        columna_valor (str): Nombre de columna con el valor numérico a mapear.
        titulo (str): Título del mapa.
        cmap (str): Esquema de color de matplotlib.
        figsize (tuple[int, int]): Tamaño de figura.
    
    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figura y ejes del mapa.
    
    Notas:
        - Mapea `Norte`, `Centro`, `Sur` a conjuntos de estados (acepta alias en inglés: `north_mexico`, `central_mexico`, `south_mexico`).
        - Inserta valores por región y colorea entidades según su grupo.
        - No muestra leyenda para mantener sobriedad.
    
    Ejemplo:
        >>> fig, ax = crear_mapa_coroplético_ventas(gdf, df, "region", "ventas")
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Mapeo de regiones a estados (ES)
    mapeo_regiones = {
        "Norte": [
            "Baja California", "Baja California Sur", "Sonora", "Chihuahua", "Coahuila",
            "Nuevo León", "Tamaulipas", "Sinaloa", "Durango"
        ],
        "Centro": [
            "Aguascalientes", "Zacatecas", "San Luis Potosí", "Guanajuato", "Querétaro",
            "Hidalgo", "Estado de México", "Ciudad de México", "Morelos", "Tlaxcala",
            "Puebla", "Michoacán"
        ],
        "Sur": [
            "Jalisco", "Colima", "Nayarit", "Veracruz", "Guerrero", "Oaxaca",
            "Chiapas", "Tabasco", "Campeche", "Yucatán", "Quintana Roo"
        ],
    }
    
    # Crear una copia del GeoDataFrame
    gdf_plot = gdf_mexico.copy()

    # Asignación de región al GeoDataFrame (ES)
    gdf_plot["region"] = "Otros"
    for region, estados in mapeo_regiones.items():
        mask = gdf_plot["ENTIDAD"].isin(estados)
        gdf_plot.loc[mask, "region"] = region
    
    # Normalizar nombres de región del DataFrame (EN->ES) y construir valores por región
    region_aliases = {
        "north_mexico": "Norte",
        "central_mexico": "Centro",
        "south_mexico": "Sur",
        "other": "Otros",
        "Norte": "Norte",
        "Centro": "Centro",
        "Sur": "Sur",
        "Otros": "Otros",
    }
    regiones_df = datos_ventas[columna_region].map(lambda r: region_aliases.get(r, r))
    valores_region = dict(zip(regiones_df, datos_ventas[columna_valor]))

    # Asignar valores a cada estado basado en su región
    gdf_plot["valor"] = gdf_plot["region"].map(valores_region).fillna(0)
    
    # Crear el mapa coroplético con colorbar para mostrar los datos
    im = gdf_plot.plot(
        column='valor', ax=ax, cmap=cmap, legend=True,
        edgecolor='#b0b3b8', linewidth=0.6, alpha=0.85,
        legend_kwds={'label': f'{columna_valor}', 'orientation': 'horizontal', 
                    'shrink': 0.6, 'aspect': 30, 'pad': 0.1}
    )
    
    # Centroides por región (ES)
    region_centroids = {
        "Norte": (-106.0, 28.0),
        "Centro": (-99.5, 20.5),
        "Sur": (-95.0, 17.0),
    }

    # Mostrar valores de datos en cada región
    for region, valor in valores_region.items():
        if region in region_centroids:
            x, y = region_centroids[region]
            # Determinar si el valor representa un conteo (no moneda)
            es_conteo = any(k in columna_valor.lower() for k in ['num', 'count', 'numero', 'cantidad'])
            # Formatear el valor según su magnitud, sin símbolo "$" para conteos
            if es_conteo:
                if valor >= 1000000:
                    texto_valor = f'{valor/1000000:.1f}M'
                elif valor >= 1000:
                    texto_valor = f'{valor/1000:.0f}K'
                else:
                    texto_valor = f'{valor:.0f}'
            else:
                if valor >= 1000000:
                    texto_valor = f'${valor/1000000:.1f}M'
                elif valor >= 1000:
                    texto_valor = f'${valor/1000:.0f}K'
                else:
                    texto_valor = f'${valor:.0f}'
            
            # Agregar texto con fondo semi-transparente
            ax.text(x, y, texto_valor, fontsize=12, fontweight='bold',
                   ha='center', va='center', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.8, edgecolor='gray', linewidth=0.5),
                   zorder=10)
    
    # Nombres de regiones para leyenda (ES)
    region_names = {
        "Norte": "Norte de México",
        "Centro": "Centro de México",
        "Sur": "Sur de México",
    }

    legend_text = "Regiones:\n"
    es_conteo = any(k in columna_valor.lower() for k in ['num', 'count', 'numero', 'cantidad'])
    for region, valor in valores_region.items():
        if region in region_names:
            if es_conteo:
                if valor >= 1000000:
                    valor_fmt = f'{valor/1000000:.1f}M'
                elif valor >= 1000:
                    valor_fmt = f'{valor/1000:.0f}K'
                else:
                    valor_fmt = f'{valor:.0f}'
            else:
                if valor >= 1000000:
                    valor_fmt = f'${valor/1000000:.1f}M'
                elif valor >= 1000:
                    valor_fmt = f'${valor/1000:.0f}K'
                else:
                    valor_fmt = f'${valor:.0f}'
            legend_text += f"• {region_names[region]}: {valor_fmt}\n"
    
    # Posicionar la leyenda de regiones
    ax.text(0.98, 0.98, legend_text.strip(), transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                    edgecolor='gray', linewidth=0.5))
    
    # Configuración del título y estilo
    ax.set_title(titulo, fontsize=16, pad=20)
    ax.set_axis_off()
    ax.set_facecolor('white')
    
    return fig, ax

def crear_mapa_ciudades_principales(gdf_mexico, datos_ciudades, titulo="Principales Ciudades por Ventas", 
                                  figsize=(14, 10), top_n=10, size_range=(80, 600)):
    """
    Dibuja las principales ciudades con marcadores proporcionalmente escalados.
    
    Args:
        gdf_mexico (gpd.GeoDataFrame): Entidades y geometrías.
        datos_ciudades (pd.DataFrame): Debe contener 'Ciudad' y 'Total_Ventas' o 'Ingresos_Total'.
        titulo (str): Título del mapa.
        figsize (tuple[int, int]): Tamaño de figura.
        top_n (int): Número de ciudades a mostrar.
        size_range (tuple[int, int]): Rango de tamaño de marcadores.
    
    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figura y ejes del mapa.
    
    Notas:
        - Calcula marcador con `numpy.interp` en rango `size_range`.
        - Usa coordenadas aproximadas para ciudades principales.
    
    Ejemplo:
        >>> fig, ax = crear_mapa_ciudades_principales(gdf, df_ciudades, top_n=10)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Dibujar el mapa base con estilo sobrio
    gdf_mexico.plot(ax=ax, color='#eeeeee', edgecolor='#b0b3b8', linewidth=0.6, alpha=1.0)
    
    # Coordenadas aproximadas de ciudades principales de México
    coordenadas_ciudades = {
        'Ciudad de México': (-99.1332, 19.4326),
        'Guadalajara': (-103.3496, 20.6597),
        'Monterrey': (-100.3161, 25.6866),
        'Puebla': (-98.2063, 19.0414),
        'Tijuana': (-117.0382, 32.5149),
        'León': (-101.6804, 21.1619),
        'Juárez': (-106.4245, 31.6904),
        'Torreón': (-103.4344, 25.5428),
        'Querétaro': (-100.3899, 20.5888),
        'San Luis Potosí': (-100.9855, 22.1565),
        'Mérida': (-89.5926, 20.9674),
        'Mexicali': (-115.4683, 32.6245),
        'Aguascalientes': (-102.2916, 21.8853),
        'Hermosillo': (-110.9559, 29.0729),
        'Saltillo': (-101.0053, 25.4260),
        'Culiacán': (-107.3943, 24.7999),
        'Chihuahua': (-106.0691, 28.6353),
        'Morelia': (-101.1949, 19.7006),
        'Toluca': (-99.6832, 19.2926),
        'Veracruz': (-96.1342, 19.1738),
         
         # Ciudades solicitadas para aparecer en el mapa
         'Tuxtla Gutiérrez': (-93.1161, 16.7530),
         'Pachuca': (-98.7333, 20.1167),
         'Villahermosa': (-92.9291, 17.9895),
         'Tampico': (-97.8550, 22.2550),
         'Ensenada': (-116.6070, 31.8650),
         'Tlaxcala': (-98.2386, 19.3186),
         'Cuernavaca': (-99.2340, 18.9242),
         'Reynosa': (-98.2732, 26.0920)
     }
    
    # Filtrar top ciudades
    top_ciudades = datos_ciudades.head(top_n)
    
    # Paleta de colores para cada ciudad (único por punto)
    try:
        colors_ciudades = sns.color_palette("husl", n_colors=len(top_ciudades))
    except Exception:
        colors_ciudades = plt.cm.tab20(np.linspace(0, 1, len(top_ciudades)))
    # Color neutro para la leyenda de tamaños (no representa ciudades)
    color_marcador_leyenda = '#1f77b4'
    
    # Agregar marcadores para las ciudades con información de datos
    # Determinar columna de valor y máximo para escalar tamaños
    col_valor = 'Total_Ventas' if 'Total_Ventas' in top_ciudades.columns else (
        'Ingresos_Total' if 'Ingresos_Total' in top_ciudades.columns else None
    )
    max_valor = 1 if col_valor is None else max(1, float(top_ciudades[col_valor].max()))
    min_valor = 0 if col_valor is None else min(0, float(top_ciudades[col_valor].min()))

    # Lista para almacenar información de ciudades mostradas
    ciudades_mostradas = []
    
    # Preparar control de superposición de etiquetas
    label_bboxes = []

    def colocar_etiqueta(xy, texto):
        # Candidatos de desplazamiento en puntos para evitar solapamiento
        candidatos = [(6, 6), (-6, 6), (6, -6), (-6, -6), (12, 0), (-12, 0), (0, 12), (0, -12),
                      (18, 12), (-18, 12), (18, -12), (-18, -12), (24, 0), (0, 24),
                      (24, 24), (-24, 24), (24, -24), (-24, -24)]
        renderer = None
        mejor_candidato = candidatos[0]
        minimo_solape = float('inf')

        for dx, dy in candidatos:
            ha = 'left' if dx >= 0 else 'right'
            va = 'bottom' if dy >= 0 else 'top'
            ann = ax.annotate(
                texto, xy=xy, xytext=(dx, dy), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                          edgecolor='gray', linewidth=0.5),
                ha=ha, va=va, zorder=10,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
            )
            fig.canvas.draw()
            if renderer is None:
                renderer = fig.canvas.get_renderer()
            bbox = ann.get_window_extent(renderer=renderer)

            # Evaluar solape con etiquetas previas
            solapa = False
            area_solape_total = 0.0
            for old in label_bboxes:
                x0 = max(bbox.x0, old.x0); y0 = max(bbox.y0, old.y0)
                x1 = min(bbox.x1, old.x1); y1 = min(bbox.y1, old.y1)
                if x1 > x0 and y1 > y0:
                    solapa = True
                    area_solape_total += (x1 - x0) * (y1 - y0)
            if not solapa:
                label_bboxes.append(bbox)
                return ann
            # Mantener el mejor candidato con menor área de solape
            if area_solape_total < minimo_solape:
                minimo_solape = area_solape_total
                mejor_candidato = (dx, dy)
            ann.remove()

        # Si todos solapan, usar el mejor candidato (menor solape)
        dx, dy = mejor_candidato
        ha = 'left' if dx >= 0 else 'right'
        va = 'bottom' if dy >= 0 else 'top'
        ann_final = ax.annotate(
            texto, xy=xy, xytext=(dx, dy), textcoords='offset points',
            fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                      edgecolor='gray', linewidth=0.5),
            ha=ha, va=va, zorder=10,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
        )
        fig.canvas.draw()
        bbox_final = ann_final.get_window_extent(renderer=fig.canvas.get_renderer())
        label_bboxes.append(bbox_final)
        return ann_final

    for i, (_, ciudad) in enumerate(top_ciudades.iterrows()):
        nombre_ciudad = ciudad['Ciudad']
        if nombre_ciudad in coordenadas_ciudades:
            lon, lat = coordenadas_ciudades[nombre_ciudad]
            valor = float(ciudad.get('Total_Ventas', ciudad.get('Ingresos_Total', 0)))
            
            # Tamaño del marcador fijo (se elimina la escala por valor)
            size = 300.0
            
            # Marcador con información de datos y color único
            ax.scatter(lon, lat, s=size, color=colors_ciudades[i], alpha=0.8, edgecolors='white', linewidth=1.5, zorder=5)
            
            # Agregar etiqueta con la información completa por ciudad (ventas y total)
            if valor >= 1000000:
                valor_fmt = f'${valor/1000000:.1f}M'
            elif valor >= 1000:
                valor_fmt = f'${valor/1000:.0f}K'
            else:
                valor_fmt = f'${valor:.0f}'

            num_ventas = int(ciudad.get('Num_Ventas', 0))
            etiqueta = f"{nombre_ciudad}\nVentas: {num_ventas}\nTotal: {valor_fmt}"

            # Colocar etiqueta evitando solapes
            colocar_etiqueta((lon, lat), etiqueta)
            
            ciudades_mostradas.append((nombre_ciudad, valor, size))
    
    # Leyenda de tamaños deshabilitada (marcadores con tamaño fijo)
    # Se retiró la escala por valor y su leyenda.

    # Agregar información estadística en la esquina superior derecha
    if ciudades_mostradas:
        stats_text = f"Top {len(ciudades_mostradas)} ciudades mostradas\n"
        stats_text += f"Rango: ${min_valor:,.0f} - ${max_valor:,.0f}\n"
        stats_text += f"Total ciudades: {len(top_ciudades)}"
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                        edgecolor='gray', linewidth=0.5))
    
    # Configuración del título y estilo
    ax.set_title(titulo, fontsize=16, pad=20)
    ax.set_axis_off()
    ax.set_facecolor('white')
    
    return fig, ax


def crear_mapa_ciudades_principales_interactivo(gdf_mexico, datos_ciudades, titulo="Principales Ciudades por Ventas (Interactivo)", top_n=10, tiles='CartoDB Positron'):
    """
    Crea un mapa interactivo HTML con marcadores por ciudad; al hacer clic
    en cada punto se muestra la información de ventas por ciudad.

    Args:
        gdf_mexico (gpd.GeoDataFrame): Entidades y geometrías de México.
        datos_ciudades (pd.DataFrame): Debe contener 'Ciudad', 'Num_Ventas' y 'Total_Ventas'.
        titulo (str): Título del mapa.
        top_n (int): Número de ciudades a mostrar (orden según df de entrada).
        tiles (str): Capa base de Folium (por defecto 'CartoDB Positron').

    Returns:
        folium.Map | None: Mapa interactivo si Folium está disponible, None si no.
    """
    try:
        import folium
        from matplotlib.colors import to_hex
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print(" [AVISO] Folium no está instalado; instale con: pip install folium")
        return None

    # Centro del mapa a partir de límites del GeoDataFrame
    bounds = gdf_mexico.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=tiles, control_scale=True)

    # Capa base de entidades para contexto visual (sobria)
    try:
        for _, row in gdf_mexico.iterrows():
            folium.GeoJson(
                row['geometry'],
                name=row.get('ENTIDAD', 'Entidad'),
                style_function=lambda x: {
                    'fillColor': '#eeeeee',
                    'color': '#b0b3b8',
                    'weight': 0.6,
                    'fillOpacity': 0.15,
                }
            ).add_to(m)
    except Exception:
        pass

    # Coordenadas aproximadas de ciudades principales de México
    coordenadas_ciudades = {
        'Ciudad de México': (-99.1332, 19.4326),
        'Guadalajara': (-103.3496, 20.6597),
        'Monterrey': (-100.3161, 25.6866),
        'Puebla': (-98.2063, 19.0414),
        'Tijuana': (-117.0382, 32.5149),
        'León': (-101.6804, 21.1619),
        'Juárez': (-106.4245, 31.6904),
        'Torreón': (-103.4344, 25.5428),
        'Querétaro': (-100.3899, 20.5888),
        'San Luis Potosí': (-100.9855, 22.1565),
        'Mérida': (-89.5926, 20.9674),
        'Mexicali': (-115.4683, 32.6245),
        'Aguascalientes': (-102.2916, 21.8853),
        'Hermosillo': (-110.9559, 29.0729),
        'Saltillo': (-101.0053, 25.4260),
        'Culiacán': (-107.3943, 24.7999),
        'Chihuahua': (-106.0691, 28.6353),
        'Morelia': (-101.1949, 19.7006),
        'Toluca': (-99.6832, 19.2926),
        'Veracruz': (-96.1342, 19.1738),
        # Ciudades añadidas
        'Tuxtla Gutiérrez': (-93.1161, 16.7530),
        'Pachuca': (-98.7333, 20.1167),
        'Villahermosa': (-92.9291, 17.9895),
        'Tampico': (-97.8550, 22.2550),
        'Ensenada': (-116.6070, 31.8650),
        'Tlaxcala': (-98.2386, 19.3186),
        'Cuernavaca': (-99.2340, 18.9242),
        'Reynosa': (-98.2732, 26.0920)
    }

    # Top N ciudades según el DataFrame de entrada
    top_ciudades = datos_ciudades.head(top_n)

    # Paleta de colores única por ciudad
    try:
        colors_ciudades = [to_hex(c) for c in sns.color_palette("husl", n_colors=len(top_ciudades))]
    except Exception:
        colors_ciudades = [to_hex(c) for c in plt.cm.tab20(np.linspace(0, 1, len(top_ciudades)))]

    # Añadir marcadores con tooltip (hover) y popup (click)
    for i, (_, ciudad) in enumerate(top_ciudades.iterrows()):
        nombre_ciudad = ciudad['Ciudad']
        if nombre_ciudad in coordenadas_ciudades:
            lon, lat = coordenadas_ciudades[nombre_ciudad]
            num_ventas = int(ciudad.get('Num_Ventas', ciudad.get('num_ventas', 0)))
            valor = float(ciudad.get('Total_Ventas', ciudad.get('Ingresos_Total', ciudad.get('total_ventas', 0))))
            valor_fmt = f"${valor:,.0f}"

            popup_html = f"""
            <div style="font-family: Arial; font-size: 13px; padding: 10px; min-width: 220px;">
                <h4 style="color: #2E86AB; margin: 0 0 8px 0;">{nombre_ciudad}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td><b>Ventas:</b></td><td>{num_ventas:,}</td></tr>
                    <tr><td><b>Total:</b></td><td>{valor_fmt}</td></tr>
                </table>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=colors_ciudades[i],
                weight=2,
                fill=True,
                fill_color=colors_ciudades[i],
                fill_opacity=0.9,
                tooltip=folium.Tooltip(nombre_ciudad, sticky=True),
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

    folium.LayerControl(position='topright', collapsed=True).add_to(m)
    # Título en el mapa (overlay simple)
    title_html = f"""
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                     background: rgba(255,255,255,0.95); padding: 6px 12px; border-radius: 8px;
                     border: 1px solid #b0b3b8; font-family: 'Segoe UI', Tahoma, sans-serif;
                     font-size: 14px; color: #2C3E50; z-index: 999;">
            {titulo}
        </div>
    """
    try:
        from folium import Element
        m.get_root().html.add_child(Element(title_html))
    except Exception:
        pass

    return m


def crear_mapa_interactivo_folium(gdf_mexico,
                                  datos_ventas=None,
                                  columna_region=None,
                                  columna_valor=None,
                                  tiles="CartoDB Positron",
                                  zoom_start=5,
                                  center=(-99.1332, 19.4326),
                                  tooltip_col="ENTIDAD",
                                  popup_cols=None,
                                  guardar_path=None,
                                  color_map='viridis',
                                  add_minimap=True,
                                  add_fullscreen=True,
                                  add_measure=True,
                                  add_mousepos=True,
                                  show_circles=False,
                                  minimal=False,
                                  show_legend=True):
    """
    Crea un mapa interactivo con Folium, con paletas y tooltips modernos.
    
    Args:
        gdf_mexico (gpd.GeoDataFrame): Entidades y geometrías de México.
        datos_ventas (pd.DataFrame, opcional): Datos agregados por región.
        columna_region (str, opcional): Columna en `datos_ventas` con la clave de región.
        columna_valor (str, opcional): Columna con valores numéricos para colorear.
        color_map (str): Nombre de paleta ('plasma', 'viridis', 'inferno', 'magma',
            'cividis', 'royal', 'ocean', 'forest', 'warm', 'cool', 'sunset',
            'earth', 'pastel', 'modern', 'elegant'). Por defecto 'plasma'.
        add_minimap (bool): Añade minimapa a la vista (no minimal).
        add_fullscreen (bool): Añade botón de pantalla completa (no minimal).
        add_measure (bool): Añade herramienta de medición (no minimal).
        add_mousepos (bool): Muestra coordenadas del ratón (no minimal).
        show_circles (bool): Dibuja marcadores circulares por centroide (no minimal).
        minimal (bool): Estilo sobrio con 'CartoDB Positron' y UI reducida.
        show_legend (bool): Añade la leyenda (colormap) cuando hay datos y no es minimal.
    
    Returns:
        folium.Map: Objeto de mapa interactivo.
    
    Comportamiento:
        - Si `datos_ventas` y columnas están definidos, colorea por región con colormap continuo.
        - En modo `minimal`, usa paleta discreta `['#F5276C','#F54927','#F5B027']` y UI reducida.
        - En modo no minimal, habilita capas adicionales y complementos opcionales.
        - Tooltips y popups muestran entidad, capital y valor formateado.
    
    Ejemplo:
        >>> m = crear_mapa_interactivo_folium(gdf, df, 'region', 'ventas',
        ...                                   color_map='viridis',
        ...                                   minimal=False, show_legend=True)
        >>> guardar_mapa_interactivo(m, 'data/mapa_interactivo_mexico.html')
    """
    try:
        import folium
        from branca.colormap import LinearColormap
        from folium import plugins
    except ImportError:
        print(" [AVISO] Folium no está instalado; omitiendo mapa interactivo. Instale con: pip install folium")
        return None
    # Centro del mapa
    if center is not None:
        center_lon, center_lat = center
    else:
        bounds = gdf_mexico.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=tiles)
    # Capas base adicionales y control de capas (evitar sobrecarga visual en modo minimal)
    if not minimal:
        folium.TileLayer('CartoDB Positron', name='Positron').add_to(m)
        folium.TileLayer(
            'Stamen Terrain', 
            name='Terrain', 
            attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
        ).add_to(m)
        folium.TileLayer(
            'Stamen Toner', 
            name='Toner', 
            attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
        ).add_to(m)
    
    # Agregar los estados al mapa
    if datos_ventas is not None and columna_region is not None and columna_valor is not None:
        # Normalizar nombres de región del DataFrame (EN->ES)
        region_aliases = {
            "north_mexico": "Norte",
            "central_mexico": "Centro",
            "south_mexico": "Sur",
            "other": "Otros",
            "Norte": "Norte",
            "Centro": "Centro",
            "Sur": "Sur",
        }
        regiones_df = datos_ventas[columna_region].map(lambda r: region_aliases.get(r, r))
        valores_region = dict(zip(regiones_df, datos_ventas[columna_valor]))

        # Mapeo de regiones a estados (ES)
        mapeo_regiones = {
            "Norte": [
                "Baja California", "Baja California Sur", "Sonora", "Chihuahua", "Coahuila",
                "Nuevo León", "Tamaulipas", "Sinaloa", "Durango"
            ],
            "Centro": [
                "Aguascalientes", "Zacatecas", "San Luis Potosí", "Guanajuato", "Querétaro",
                "Hidalgo", "Estado de México", "Ciudad de México", "Morelos", "Tlaxcala",
                "Puebla", "Michoacán"
            ],
            "Sur": [
                "Jalisco", "Colima", "Nayarit", "Veracruz", "Guerrero", "Oaxaca",
                "Chiapas", "Tabasco", "Campeche", "Yucatán", "Quintana Roo"
            ],
        }
        # Colormap continuo basado en rango de valores (vmin→vmax) para gradientes suaves
        vals = list(valores_region.values())
        vmin, vmax = (min(vals) if vals else 0), (max(vals) if vals else 1)
        # Define paletas conocidas (ampliadas) por nombre
        palettes = {
            'plasma': ['#0c0887', '#5601a4', '#8b02a8', '#b5367a', '#e16462', '#f89441', '#fccf2d'],
            'viridis': ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
            'inferno': ['#000004', '#1f0c48', '#741a6d', '#b63679', '#ed6925', '#fcffa4'],
            'magma': ['#000004', '#1c1044', '#5e1f78', '#b63679', '#fb8761', '#fcfdbf'],
            'cividis': ['#00224e', '#2c5c8a', '#3a7f88', '#76a365', '#d7d566'],
            # Paletas estéticas adicionales
            'royal': ['#0b1f5e', '#123b7a', '#1f5da8', '#3a7bd5', '#7aa5e6'],
            'ocean': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a'],
            'forest': ['#0b3d20', '#136a3f', '#4da167', '#7ebc89', '#b7e4c7'],
            'warm': ['#7f1d1d', '#b91c1c', '#ef4444', '#f97316', '#f59e0b', '#fde68a'],
            'cool': ['#0ea5e9', '#22d3ee', '#34d399', '#a7f3d0', '#d1fae5'],
            'sunset': ['#2e1a47', '#5b2a86', '#9c3f8c', '#e36bae', '#f4a261', '#f6bd60'],
            'earth': ['#3b3024', '#6b4f3a', '#8c6e54', '#a68a64', '#c2a387', '#d9c3a2'],
            'pastel': ['#f1e1ff', '#c1d3fe', '#b8e0d2', '#f9f1a5', '#fcd5ce'],
            'modern': ['#2d3748', '#4a5568', '#718096', '#a0aec0', '#e2e8f0'],
            'elegant': ['#2c3e50', '#34495e', '#5d6d7e', '#aeb6bf', '#d6dbdf']
        }
        # Paleta sobria para modo minimal (neutros estéticos)
        colors_minimal = ['#F5276C', '#F54927', '#F5B027']
        colors = colors_minimal if minimal else palettes.get(color_map, palettes.get('viridis'))
        cmap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
        cmap.caption = columna_valor

        # Asigna color según región del estado utilizando el colormap continuo.
        def get_color(entidad):
            for region, estados in mapeo_regiones.items():
                if entidad in estados:
                    valor = float(valores_region.get(region, 0))
                    return cmap(valor)
            return '#95A5A6'
        
        # Capa GeoJson con highlight en hover
        features = []
        for _, row in gdf_mexico.iterrows():
            entidad = row['ENTIDAD']
            color = get_color(entidad)
            
            # Encontrar la región del estado
            region = 'other'
            for reg, estados in mapeo_regiones.items():
                if entidad in estados:
                    region = reg
                    break
            
            valor = float(valores_region.get(region, 0))
            
            # Obtener información adicional del estado
            capital = row.get('CAPITAL', 'N/A')
            clave = row.get('CVE_EDO', 'N/A')
            
            # Calcular estadísticas adicionales para el tooltip
            total_ventas = sum(valores_region.values()) if valores_region else 1
            porcentaje_region = (valor / total_ventas * 100) if total_ventas > 0 else 0
            ranking_region = sorted(valores_region.items(), key=lambda x: x[1], reverse=True)
            posicion_ranking = next((i+1 for i, (r, v) in enumerate(ranking_region) if r == region), 0)
            
            # Tooltip moderno (hover): gradiente, tipografía y valores formateados.
            tooltip_html = f"""
            <div style="
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                font-size: 14px; 
                padding: 12px 16px; 
                background: linear-gradient(135deg, #F5276C 0%, #F54927 50%, #F5B027 100%); 
                border: none; 
                border-radius: 12px; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                min-width: 200px;
                max-width: 250px;
            ">
                <div style="color: #2C3E50; font-size: 18px; font-weight: 600; margin-bottom: 8px;">
                    {entidad}
                </div>
                <div style="color: #7F8C8D; font-size: 12px; margin-bottom: 6px;">Capital: {capital}</div>
                <div style="color: #27AE60; font-size: 16px; font-weight: 500; margin-bottom: 4px;">${valor:,.0f}</div>
                <div style="color: #E67E22; font-size: 12px; margin-bottom: 2px;">
                    Región: {region.replace('_', ' ').title()}
                </div>
                <div style="color: #8E44AD; font-size: 12px; margin-bottom: 2px;">
                    {porcentaje_region:.1f}% del total nacional
                </div>
                <div style="color: #3498DB; font-size: 12px;">
                    Ranking: #{posicion_ranking} de {len(valores_region)} regiones
                </div>
            </div>
            """
            
            # Crear popup más detallado para click (opcional)
            popup_html = f"""
            <div style="font-family: Arial; font-size: 13px; padding: 10px; min-width: 200px;">
                <h4 style="color: #2E86AB; margin: 0 0 10px 0;">{entidad}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td><b>Capital:</b></td><td>{capital}</td></tr>
                    <tr><td><b>Clave:</b></td><td>{clave}</td></tr>
                    <tr><td><b>Región:</b></td><td>{region}</td></tr>
                    <tr><td><b>Ventas Totales:</b></td><td>${valor:,.0f}</td></tr>
                </table>
            </div>
            """
            
            # Estilos más sobrios en modo minimal
            border_color = '#B0B3B8' if minimal else '#2C3E50'
            border_weight = 0.6 if minimal else 1.0
            fill_opacity = 0.7 if minimal else 0.85

            # Capa GeoJson por estado con estilos y highlight (hover).
            gj = folium.GeoJson(
                row['geometry'],
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': border_color,
                    'weight': border_weight,
                    'fillOpacity': fill_opacity,
                },
                highlight_function=lambda x: {
                    'weight': (1.2 if minimal else 2.0),
                    'color': ('#212121' if minimal else '#000000'),
                    'fillOpacity': (0.8 if minimal else 0.95),
                },
                popup=(None if minimal else folium.Popup(popup_html, max_width=300)),
                tooltip=folium.Tooltip(
                    (f"{entidad} — ${valor:,.0f}" if minimal else tooltip_html),
                    sticky=True,
                    style="background-color: transparent; border: none; box-shadow: none;"
                )
            )
            gj.add_to(m)

            # Marcador por centroide proporcional al valor (omitido en minimal)
            effective_show_circles = show_circles and (not minimal)
            if effective_show_circles:
                centroid = row['geometry'].centroid
                radius = 5 + 15 * ((valor - vmin) / (vmax - vmin + 1e-9))
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=float(radius),
                    color='#F54927',
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    tooltip=f"{entidad}: ${valor:,.0f}"
                ).add_to(m)
    else:
        # Mapa simple sin datos de ventas - también con tooltips mejorados
        for _, row in gdf_mexico.iterrows():
            entidad = row['ENTIDAD']
            capital = row.get('CAPITAL', 'N/A')
            clave = row.get('CVE_EDO', 'N/A')
            
            # Tooltip para hover - versión simplificada
            tooltip_html = f"""
            <div style="
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                font-size: 14px; 
                padding: 12px 16px; 
                background: linear-gradient(135deg, #F5276C 0%, #F54927 50%, #F5B027 100%); 
                border: none; 
                border-radius: 12px; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                min-width: 180px;
                max-width: 220px;
            ">
                <div style="color: #2C3E50; font-size: 18px; font-weight: 600; margin-bottom: 8px;">
                    {entidad}
                </div>
                <div style="color: #7F8C8D; font-size: 12px;">{capital}</div>
            </div>
            """
            
            # Popup para click
            popup_html = f"""
            <div style="font-family: Arial; font-size: 13px; padding: 10px; min-width: 200px;">
                <h4 style="color: #2E86AB; margin: 0 0 10px 0;">{entidad}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td><b>Capital:</b></td><td>{capital}</td></tr>
                    <tr><td><b>Clave:</b></td><td>{clave}</td></tr>
                    <tr><td><b>Tipo:</b></td><td>Estado de México</td></tr>
                </table>
            </div>
            """
            
            # Estilos sobrios si es minimal
            border_color = '#B0B3B8' if minimal else '#34495E'
            border_weight = 0.6 if minimal else 0.8
            fill_color = '#F5B027' if minimal else '#85C1E9'

            gj = folium.GeoJson(
                row['geometry'],
                style_function=lambda x: {
                    'fillColor': fill_color,
                    'color': border_color,
                    'weight': border_weight,
                    'fillOpacity': 0.7,
                },
                popup=(None if minimal else folium.Popup(popup_html, max_width=300)),
                tooltip=folium.Tooltip(
                    (entidad if minimal else tooltip_html), 
                    sticky=True, 
                    style="background-color: transparent; border: none; box-shadow: none;"
                )
            )
            gj.add_to(m)

    # Añadir colormap como leyenda si está disponible
    try:
        if (datos_ventas is not None and columna_region and columna_valor) and show_legend and (not minimal):
            cmap.add_to(m)
            
            # Añadir panel de resumen de datos
            total_ventas = sum(valores_region.values())
            promedio_ventas = total_ventas / len(valores_region) if valores_region else 0
            max_region = max(valores_region.items(), key=lambda x: x[1]) if valores_region else ('N/A', 0)
            min_region = min(valores_region.items(), key=lambda x: x[1]) if valores_region else ('N/A', 0)
            
            data_summary_html = f"""
            <div style="
                position: fixed;
                top: 10px;
                left: 10px;
                width: 280px;
                background: rgba(255, 255, 255, 0.95);
                border: 2px solid #3498db;
                border-radius: 10px;
                padding: 15px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 13px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                z-index: 1000;
            ">
                <h4 style="margin: 0 0 12px 0; color: #2c3e50; font-size: 16px; border-bottom: 2px solid #3498db; padding-bottom: 5px;">
                    📊 Resumen de Datos
                </h4>
                <div style="margin-bottom: 8px;">
                    <strong>Total de Ventas:</strong><br>
                    <span style="color: #27ae60; font-size: 15px; font-weight: bold;">${total_ventas:,.0f}</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Promedio por Región:</strong><br>
                    <span style="color: #f39c12; font-size: 14px;">${promedio_ventas:,.0f}</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Región con Mayor Venta:</strong><br>
                    <span style="color: #e74c3c; font-size: 12px;">{max_region[0].replace('_', ' ').title()}</span><br>
                    <span style="color: #e74c3c; font-weight: bold;">${max_region[1]:,.0f}</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Región con Menor Venta:</strong><br>
                    <span style="color: #9b59b6; font-size: 12px;">{min_region[0].replace('_', ' ').title()}</span><br>
                    <span style="color: #9b59b6; font-weight: bold;">${min_region[1]:,.0f}</span>
                </div>
                <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #bdc3c7; font-size: 11px; color: #7f8c8d;">
                    💡 Hover sobre los estados para ver detalles
                </div>
            </div>
            """
            
            # Añadir el panel de resumen como HTML personalizado
            from folium import Element
            data_summary_element = Element(data_summary_html)
            m.get_root().html.add_child(data_summary_element)
            
    except Exception:
        pass

    # Plugins opcionales (evitar exceso visual en modo minimal)
    if not minimal:
        if add_fullscreen:
            plugins.Fullscreen(position='topright').add_to(m)
        if add_minimap:
            plugins.MiniMap(toggle_display=True).add_to(m)
        if add_measure:
            plugins.MeasureControl(position='topleft').add_to(m)
        if add_mousepos:
            plugins.MousePosition(
                position='bottomright',
                separator=' | ',
                prefix='Coords',
                lat_formatter='function(num) {return L.Util.formatNum(num, 5);}',
                lng_formatter='function(num) {return L.Util.formatNum(num, 5);}'
            ).add_to(m)
        # Añadir control de capas
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
    return m

def guardar_mapa_como_imagen(fig, nombre_archivo, dpi=300, bbox_inches='tight'):
    """
    Guarda una figura de matplotlib como imagen en disco.
    
    Args:
        fig (matplotlib.figure.Figure): Figura a guardar.
        nombre_archivo (str): Ruta destino (.png, .jpg).
        dpi (int): Resolución.
        bbox_inches (str): Ajuste del recorte ('tight' recomendado).
    
    Efectos:
        - Escribe el archivo en disco y muestra confirmación en consola.
    
    Errores:
        - Captura excepciones y reporta el mensaje.
    """
    try:
        fig.savefig(nombre_archivo, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
        print(f" [OK] Mapa guardado como: {nombre_archivo}")
    except Exception as e:
        print(f" [ERROR] Error al guardar el mapa: {e}")

def guardar_mapa_interactivo(m, nombre_archivo):
    """
    Guarda un mapa de Folium como archivo HTML.
    
    Args:
        m (folium.Map): Mapa interactivo a guardar.
        nombre_archivo (str): Ruta destino (.html).
    
    Efectos:
        - Escribe el HTML del mapa y confirma por consola.
    
    Errores:
        - Captura y reporta fallos en el guardado.
    """
    try:
        m.save(nombre_archivo)
        print(f" Mapa interactivo guardado como: {nombre_archivo}")
    except Exception as e:
        print(f" Error al guardar el mapa interactivo: {e}")

# Función de ejemplo para demostrar el uso
def ejemplo_uso_mapas():
    """
    Demuestra un flujo básico para generar mapas estáticos e interactivos.
    
    - Carga el GeoJSON.
    - Genera mapa base y coroplético con datos de ejemplo.
    - Guarda resultados bajo `data/`.
    """
    print("  EJEMPLO DE USO DE FUNCIONES DE MAPAS DE MÉXICO")
    print("=" * 60)
    
    # Ruta al archivo GeoJSON
    ruta_geojson = "/Users/sebastian/Downloads/temas c/admin1.geojson"
    
    # Cargar el GeoJSON
    gdf_mexico = cargar_geojson_mexico(ruta_geojson)
    
    if gdf_mexico is not None:
        # Crear mapa base
        fig, ax = crear_mapa_base_mexico(gdf_mexico, "Mapa Base de México")
        guardar_mapa_como_imagen(fig, "data/mapa_base_mexico.png")
        plt.show()
        
        # Datos de ejemplo para el mapa coroplético
        datos_ejemplo = pd.DataFrame({
            'region': ['north_mexico', 'central_mexico', 'south_mexico'],
            'ventas': [25896.71, 24520.91, 26943.64]
        })
        
        # Crear mapa coroplético
        fig, ax = crear_mapa_coroplético_ventas(
            gdf_mexico, datos_ejemplo, 'region', 'ventas',
            "Ventas por Región de México"
        )
        guardar_mapa_como_imagen(fig, "data/mapa_coropletico_ventas.png")
        plt.show()
        
        print(" Mapas de ejemplo creados exitosamente!")

if __name__ == "__main__":
    ejemplo_uso_mapas()
