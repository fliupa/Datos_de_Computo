"""
Funciones para crear mapas de México usando datos geográficos
Utiliza el archivo admin1.geojson para generar visualizaciones geográficas
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

def cargar_geojson_mexico(ruta_geojson):
    """
    Carga el archivo GeoJSON de México y lo prepara para visualización
    
    Args:
        ruta_geojson (str): Ruta al archivo admin1.geojson
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame con los datos de México
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
    Crea un mapa base de México con los estados
    
    Args:
        gdf_mexico (gpd.GeoDataFrame): GeoDataFrame con datos de México
        titulo (str): Título del mapa
        figsize (tuple): Tamaño de la figura
    
    Returns:
        tuple: (fig, ax) - Figura y ejes de matplotlib
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
                                titulo="Ventas por Región", cmap='plasma', figsize=(14, 10)):
    """
    Crea un mapa coroplético de México con datos de ventas por región
    
    Args:
        gdf_mexico (gpd.GeoDataFrame): GeoDataFrame con datos de México
        datos_ventas (pd.DataFrame): DataFrame con datos de ventas
        columna_region (str): Nombre de la columna con las regiones
        columna_valor (str): Nombre de la columna con los valores a mapear
        titulo (str): Título del mapa
        cmap (str): Esquema de colores
        figsize (tuple): Tamaño de la figura
    
    Returns:
        tuple: (fig, ax) - Figura y ejes de matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Crear un diccionario de mapeo de regiones a estados
    mapeo_regiones = {
        'north_mexico': ['BAJA CALIFORNIA', 'SONORA', 'CHIHUAHUA', 'COAHUILA', 'NUEVO LEÓN', 'TAMAULIPAS'],
        'central_mexico': ['DISTRITO FEDERAL', 'MÉXICO', 'MORELOS', 'PUEBLA', 'TLAXCALA', 'HIDALGO', 'QUERÉTARO', 'GUANAJUATO', 'AGUASCALIENTES', 'ZACATECAS', 'SAN LUIS POTOSÍ'],
        'south_mexico': ['GUERRERO', 'OAXACA', 'CHIAPAS', 'VERACRUZ', 'TABASCO', 'CAMPECHE', 'YUCATÁN', 'QUINTANA ROO', 'MICHOACÁN', 'COLIMA', 'JALISCO', 'NAYARIT', 'SINALOA', 'DURANGO']
    }
    
    # Crear una copia del GeoDataFrame
    gdf_plot = gdf_mexico.copy()
    
    # Agregar columna de región basada en el mapeo
    gdf_plot['region'] = 'other'
    for region, estados in mapeo_regiones.items():
        mask = gdf_plot['ENTIDAD'].isin(estados)
        gdf_plot.loc[mask, 'region'] = region
    
    # Crear diccionario de valores por región
    valores_region = dict(zip(datos_ventas[columna_region], datos_ventas[columna_valor]))
    
    # Asignar valores a cada estado basado en su región
    gdf_plot['valor'] = gdf_plot['region'].map(valores_region).fillna(0)
    
    # Crear el mapa coroplético con estilo sobrio
    gdf_plot.plot(
        column='valor', ax=ax, cmap=cmap, legend=False,
        edgecolor='#b0b3b8', linewidth=0.6, alpha=0.85
    )
    
    # Configuración minimalista
    ax.set_title(titulo, fontsize=16)
    ax.set_axis_off()
    ax.set_facecolor('white')
    
    return fig, ax

def crear_mapa_ciudades_principales(gdf_mexico, datos_ciudades, titulo="Principales Ciudades por Ventas", 
                                  figsize=(14, 10), top_n=10, size_range=(80, 600)):
    """
    Crea un mapa de México con marcadores para las principales ciudades
    
    Args:
        gdf_mexico (gpd.GeoDataFrame): GeoDataFrame con datos de México
        datos_ciudades (pd.DataFrame): DataFrame con datos de ciudades (debe tener columnas: Ciudad, Total_Ventas)
        titulo (str): Título del mapa
        figsize (tuple): Tamaño de la figura
        top_n (int): Número de ciudades principales a mostrar
    
    Returns:
        tuple: (fig, ax) - Figura y ejes de matplotlib
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
        'Veracruz': (-96.1342, 19.1738)
    }
    
    # Filtrar top ciudades
    top_ciudades = datos_ciudades.head(top_n)
    
    # Color único para marcadores minimalistas
    color_marcador = '#1f77b4'
    
    # Agregar marcadores para las ciudades con colores más vibrantes
    # Determinar columna de valor y máximo para escalar tamaños
    col_valor = 'Total_Ventas' if 'Total_Ventas' in top_ciudades.columns else (
        'Ingresos_Total' if 'Ingresos_Total' in top_ciudades.columns else None
    )
    max_valor = 1 if col_valor is None else max(1, float(top_ciudades[col_valor].max()))

    for i, (_, ciudad) in enumerate(top_ciudades.iterrows()):
        nombre_ciudad = ciudad['Ciudad']
        if nombre_ciudad in coordenadas_ciudades:
            lon, lat = coordenadas_ciudades[nombre_ciudad]
            valor = float(ciudad.get('Total_Ventas', ciudad.get('Ingresos_Total', 0)))
            
            # Tamaño del marcador proporcional al valor usando interpolación
            size = float(np.interp(valor, [0, max_valor], [size_range[0], size_range[1]]))
            
            # Marcador minimalista sin anotaciones ni cajas
            ax.scatter(lon, lat, s=size, c=color_marcador, alpha=0.7, edgecolors='white', linewidth=1.5, zorder=5)
    
    # Configuración minimalista: título simple, sin ejes, sin leyendas
    ax.set_title(titulo, fontsize=16)
    ax.set_axis_off()
    ax.set_facecolor('white')
    
    return fig, ax

def crear_mapa_interactivo_folium(
    gdf_mexico,
    datos_ventas=None,
    columna_region=None,
    columna_valor=None,
    color_map: str = 'plasma',
    add_minimap: bool = False,
    add_fullscreen: bool = False,
    add_measure: bool = False,
    add_mousepos: bool = False,
    show_circles: bool = False,
    minimal: bool = True,
    show_legend: bool = False
):
    """
    Crea un mapa interactivo de México usando Folium
    
    Args:
        gdf_mexico (gpd.GeoDataFrame): GeoDataFrame con datos de México
        datos_ventas (pd.DataFrame, optional): DataFrame con datos de ventas
        columna_region (str, optional): Nombre de la columna con las regiones
        columna_valor (str, optional): Nombre de la columna con los valores
    
    Returns:
        folium.Map: Mapa interactivo de Folium
    """
    try:
        import folium
        from branca.colormap import LinearColormap
        from folium import plugins
    except ImportError:
        print(" [AVISO] Folium no está instalado; omitiendo mapa interactivo. Instale con: pip install folium")
        return None
    # Calcular el centro de México
    bounds = gdf_mexico.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Crear el mapa base
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=('CartoDB Positron' if minimal else 'OpenStreetMap'))
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
        # Crear mapeo de regiones a valores
        valores_region = dict(zip(datos_ventas[columna_region], datos_ventas[columna_valor]))
        
        # Mapeo de regiones a estados
        mapeo_regiones = {
            'north_mexico': ['BAJA CALIFORNIA', 'SONORA', 'CHIHUAHUA', 'COAHUILA', 'NUEVO LEÓN', 'TAMAULIPAS'],
            'central_mexico': ['DISTRITO FEDERAL', 'MÉXICO', 'MORELOS', 'PUEBLA', 'TLAXCALA', 'HIDALGO', 'QUERÉTARO', 'GUANAJUATO', 'AGUASCALIENTES', 'ZACATECAS', 'SAN LUIS POTOSÍ'],
            'south_mexico': ['GUERRERO', 'OAXACA', 'CHIAPAS', 'VERACRUZ', 'TABASCO', 'CAMPECHE', 'YUCATÁN', 'QUINTANA ROO', 'MICHOACÁN', 'COLIMA', 'JALISCO', 'NAYARIT', 'SINALOA', 'DURANGO']
        }
        
        # Construir colormap continuo para una apariencia más profesional
        vals = list(valores_region.values())
        vmin, vmax = (min(vals) if vals else 0), (max(vals) if vals else 1)
        # Define paletas conocidas (aproximadas) por nombre
        palettes = {
            'plasma': ['#0c0887', '#5601a4', '#8b02a8', '#b5367a', '#e16462', '#f89441', '#fccf2d'],
            'viridis': ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
            'inferno': ['#000004', '#1f0c48', '#741a6d', '#b63679', '#ed6925', '#fcffa4'],
            'magma': ['#000004', '#1c1044', '#5e1f78', '#b63679', '#fb8761', '#fcfdbf'],
            'cividis': ['#00224e', '#2c5c8a', '#3a7f88', '#76a365', '#d7d566']
        }
        # Paleta sobria para modo minimal
        colors_minimal = ['#eeeeee', '#d9d9d9', '#bfbfbf', '#a6a6a6', '#8c8c8c']
        colors = colors_minimal if minimal else palettes.get(color_map, ['#3498DB', '#F1C40F', '#E74C3C', '#8E44AD'])
        cmap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
        cmap.caption = columna_valor

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
            
            # Crear tooltip minimalista y elegante
            tooltip_html = f"""
            <div style="
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                font-size: 14px; 
                padding: 12px 16px; 
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
                border: none; 
                border-radius: 12px; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                min-width: 180px;
                max-width: 220px;
            ">
                <div style="color: #2C3E50; font-size: 18px; font-weight: 600; margin-bottom: 8px;">
                    {entidad}
                </div>
                <div style="color: #7F8C8D; font-size: 12px; margin-bottom: 6px;">{capital}</div>
                <div style="color: #27AE60; font-size: 16px; font-weight: 500;">${valor:,.0f}</div>
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
                    color='#FFFFFF',
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
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
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
            fill_color = '#e5e5e5' if minimal else '#85C1E9'

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
    Guarda un mapa de matplotlib como imagen
    
    Args:
        fig: Figura de matplotlib
        nombre_archivo (str): Nombre del archivo de salida
        dpi (int): Resolución de la imagen
        bbox_inches (str): Configuración de recorte
    """
    try:
        fig.savefig(nombre_archivo, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
        print(f" [OK] Mapa guardado como: {nombre_archivo}")
    except Exception as e:
        print(f" [ERROR] Error al guardar el mapa: {e}")

def guardar_mapa_interactivo(m, nombre_archivo):
    """
    Guarda un mapa interactivo de Folium como archivo HTML
    
    Args:
        m (folium.Map): Mapa interactivo
        nombre_archivo (str): Ruta del archivo de salida (HTML)
    """
    try:
        m.save(nombre_archivo)
        print(f" Mapa interactivo guardado como: {nombre_archivo}")
    except Exception as e:
        print(f" Error al guardar el mapa interactivo: {e}")

# Función de ejemplo para demostrar el uso
def ejemplo_uso_mapas():
    """
    Función de ejemplo que demuestra cómo usar las funciones de mapas
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
