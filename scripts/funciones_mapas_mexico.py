"""
Funciones para crear mapas de México usando datos geográficos
Utiliza el archivo admin1.geojson para generar visualizaciones geográficas
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import folium
import contextily as ctx
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
        
        print(f" GeoJSON cargado exitosamente: {len(gdf_mexico)} estados/regiones")
        print(f" Columnas disponibles: {list(gdf_mexico.columns)}")
        
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
    
    # Dibujar el mapa base con colores más vibrantes
    gdf_mexico.plot(ax=ax, color='#3498DB', edgecolor='#2C3E50', linewidth=1.0, alpha=0.8)
    
    # Configurar el mapa con estilo más vibrante
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_xlabel('Longitud', fontsize=14, fontweight='bold', color='#34495E')
    ax.set_ylabel('Latitud', fontsize=14, fontweight='bold', color='#34495E')
    
    # Remover los ticks para un aspecto más limpio con colores vibrantes
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#2C3E50')
    
    # Agregar grid sutil con color vibrante
    ax.grid(True, alpha=0.4, linestyle='--', color='#7F8C8D')
    ax.set_facecolor('#F8F9FA')  # Fondo ligeramente gris para contraste
    
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
    
    # Crear el mapa coroplético con colores más vibrantes
    gdf_plot.plot(column='valor', ax=ax, cmap=cmap, legend=True, 
                  edgecolor='white', linewidth=1.2, alpha=0.9,
                  legend_kwds={'label': columna_valor, 'orientation': 'horizontal', 'shrink': 0.8})
    
    # Configurar el mapa con estilo más vibrante
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_xlabel('Longitud', fontsize=14, fontweight='bold', color='#34495E')
    ax.set_ylabel('Latitud', fontsize=14, fontweight='bold', color='#34495E')
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#2C3E50')
    ax.grid(True, alpha=0.4, linestyle='--', color='#7F8C8D')
    ax.set_facecolor('#F8F9FA')  # Fondo ligeramente gris para contraste
    
    # Agregar leyenda con estilo vibrante
    ax.text(0.02, 0.98, 'Tamaño del círculo = Volumen de ventas', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFFF', 
                     edgecolor='#3498DB', linewidth=2, alpha=0.95), color='#2C3E50')
    
    return fig, ax

def crear_mapa_ciudades_principales(gdf_mexico, datos_ciudades, titulo="Principales Ciudades por Ventas", 
                                  figsize=(14, 10), top_n=10):
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
    
    # Dibujar el mapa base
    # Dibujar el mapa base con colores más vibrantes
    gdf_mexico.plot(ax=ax, color='#ECF0F1', edgecolor='#34495E', linewidth=1.0, alpha=0.8)
    
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
    
    # Colores vibrantes para los marcadores
    colores_vibrantes = ['#E74C3C', '#9B59B6', '#3498DB', '#1ABC9C', '#F39C12', 
                        '#E67E22', '#2ECC71', '#F1C40F', '#E91E63', '#FF5722']
    
    # Agregar marcadores para las ciudades con colores más vibrantes
    for i, (_, ciudad) in enumerate(top_ciudades.iterrows()):
        nombre_ciudad = ciudad['Ciudad']
        if nombre_ciudad in coordenadas_ciudades:
            lon, lat = coordenadas_ciudades[nombre_ciudad]
            valor = ciudad.get('Total_Ventas', ciudad.get('Ingresos_Total', 0))
            
            # Tamaño del marcador proporcional al valor
            size = max(80, min(600, valor / max(top_ciudades.get('Total_Ventas', top_ciudades.get('Ingresos_Total', [1]))) * 400))
            
            # Color vibrante basado en el ranking
            color = colores_vibrantes[i % len(colores_vibrantes)]
            
            ax.scatter(lon, lat, s=size, c=color, alpha=0.8, edgecolors='white', linewidth=3, zorder=5)
            ax.annotate(f'{nombre_ciudad}\n${valor:,.0f}', xy=(lon, lat), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='#2C3E50',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFFFF', 
                               edgecolor=color, linewidth=2, alpha=0.95))
    
    # Configurar el mapa con estilo más vibrante
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_xlabel('Longitud', fontsize=14, fontweight='bold', color='#34495E')
    ax.set_ylabel('Latitud', fontsize=14, fontweight='bold', color='#34495E')
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#2C3E50')
    ax.grid(True, alpha=0.4, linestyle='--', color='#7F8C8D')
    ax.set_facecolor('#F8F9FA')  # Fondo ligeramente gris para contraste
    
    # Crear leyenda para los marcadores
    legend_elements = []
    for i in range(min(len(top_ciudades), len(colores_vibrantes))):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colores_vibrantes[i], 
                                        markersize=10, label=f'Top {i+1}'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1),
                 frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    return fig, ax

def crear_mapa_interactivo_folium(gdf_mexico, datos_ventas=None, columna_region=None, columna_valor=None, 
                              color_map='elegant', add_layers=True):
    """
    Crea un mapa interactivo de México usando Folium con estilo estético mejorado
    
    Args:
        gdf_mexico (gpd.GeoDataFrame): GeoDataFrame con datos de México
        datos_ventas (pd.DataFrame, optional): DataFrame con datos de ventas
        columna_region (str, optional): Nombre de la columna con las regiones
        columna_valor (str, optional): Nombre de la columna con los valores
        color_map (str): Nombre de la paleta de colores a usar ('elegant', 'sunset', 'modern', etc.)
        add_layers (bool): Si se deben agregar capas adicionales al mapa
    
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
    
    # Crear el mapa base con estilo moderno
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=5, 
        tiles='CartoDB Positron',
        control_scale=True
    )
    
    # Agregar capas base adicionales para mejor estética
    if add_layers:
        folium.TileLayer('CartoDB Dark_Matter', name='Dark Mode').add_to(m)
        folium.TileLayer('CartoDB Voyager', name='Voyager').add_to(m)
        folium.TileLayer(
            'https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}{r}.png',
            name='Voyager Labels',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ).add_to(m)
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            name='Satellite',
            attr='Esri'
        ).add_to(m)
        
        # Agregar control de capas
        folium.LayerControl(position='topright', collapsed=True).add_to(m)
    
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
        
        # Define paletas conocidas (aproximadas) por nombre
        palettes = {
            'plasma': ['#0c0887', '#5601a4', '#8b02a8', '#b5367a', '#e16462', '#f89441', '#fccf2d'],
            'viridis': ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
            'inferno': ['#000004', '#1f0c48', '#741a6d', '#b63679', '#ed6925', '#fcffa4'],
            'magma': ['#000004', '#1c1044', '#5e1f78', '#b63679', '#fb8761', '#fcfdbf'],
            'cividis': ['#00224e', '#2c5c8a', '#3a7f88', '#76a365', '#d7d566'],
            # Nuevas paletas estéticas
            'pastel': ['#a1d6e2', '#1995ad', '#bcbabe', '#f1f1f2', '#a5a5af'],
            'sunset': ['#f8b195', '#f67280', '#c06c84', '#6c5b7b', '#355c7d'],
            'modern': ['#2d4059', '#ea5455', '#f07b3f', '#ffd460', '#a7e9af'],
            'minimal': ['#ececec', '#9fd3c7', '#385170', '#142d4c', '#ececec'],
            'elegant': ['#30475e', '#f2a365', '#ececec', '#222831', '#dddddd']
        }
        
        # Construir colormap continuo para una apariencia más profesional
        vals = list(valores_region.values())
        vmin, vmax = (min(vals) if vals else 0), (max(vals) if vals else 1)
        
        # Usar la paleta seleccionada o elegir una por defecto
        colors = palettes.get(color_map, palettes['elegant'])
        cmap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
        
        # Agregar leyenda de colores
        cmap.caption = f"{columna_valor} por Región"
        m.add_child(cmap)
        
        # Función para obtener color basado en el valor
        def get_color(entidad):
            for region, estados in mapeo_regiones.items():
                if entidad in estados:
                    valor = valores_region.get(region, 0)
                    return cmap(valor)
            return '#95A5A6'  # Gris neutro
        
        # Agregar cada estado al mapa con estilos mejorados
        for _, row in gdf_mexico.iterrows():
            entidad = row['ENTIDAD']
            color = get_color(entidad)
            
            # Encontrar la región del estado
            region = 'other'
            for reg, estados in mapeo_regiones.items():
                if entidad in estados:
                    region = reg
                    break
            
            valor = valores_region.get(region, 0)
            
            # Obtener información adicional del estado
            capital = row.get('CAPITAL', 'N/A')
            clave = row.get('CVE_EDO', 'N/A')
            
            # Crear tooltip con diseño moderno
            tooltip_html = f"""
            <div style="
                font-family: 'Segoe UI', 'Roboto', sans-serif; 
                font-size: 14px; 
                padding: 15px 20px; 
                background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); 
                border: none; 
                border-radius: 16px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.12);
                min-width: 200px;
                max-width: 240px;
                backdrop-filter: blur(5px);
            ">
                <div style="color: #1a1a2e; font-size: 20px; font-weight: 600; margin-bottom: 10px; letter-spacing: -0.5px;">
                    {entidad}
                </div>
                <div style="color: #616161; font-size: 13px; margin-bottom: 8px; font-weight: 500;">{capital}</div>
                <div style="
                    color: #fff; 
                    font-size: 18px; 
                    font-weight: 600; 
                    background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                    padding: 8px 15px;
                    border-radius: 12px;
                    display: inline-block;
                    margin-top: 5px;
                ">${valor:,.0f}</div>
            </div>
            """
            
            # Crear popup más detallado con diseño moderno
            popup_html = f"""
            <div style="
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                padding: 20px;
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                min-width: 250px;
            ">
                <h3 style="
                    color: #1a1a2e;
                    margin: 0 0 15px 0;
                    font-weight: 600;
                    font-size: 22px;
                    border-bottom: 2px solid {color};
                    padding-bottom: 8px;
                ">{entidad}</h3>
                
                <div style="display: flex; margin-bottom: 15px;">
                    <div style="
                        background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                        color: white;
                        padding: 10px 15px;
                        border-radius: 10px;
                        font-weight: 600;
                        font-size: 18px;
                    ">${valor:,.0f}</div>
                </div>
                
                <table style="
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0 8px;
                    font-size: 14px;
                ">
                    <tr>
                        <td style="font-weight: 600; color: #616161; padding: 5px 0;">Capital</td>
                        <td style="color: #1a1a2e; padding: 5px 0;">{capital}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600; color: #616161; padding: 5px 0;">Clave</td>
                        <td style="color: #1a1a2e; padding: 5px 0;">{clave}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600; color: #616161; padding: 5px 0;">Región</td>
                        <td style="color: #1a1a2e; padding: 5px 0;">{region.replace('_', ' ').title()}</td>
                    </tr>
                </table>
            </div>
            """
            
            # Agregar GeoJson con highlight en hover
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': '#34495E',
                    'weight': 0.8,
                    'fillOpacity': 0.8,
                },
                highlight_function=lambda x: {
                    'fillColor': color,
                    'color': '#000',
                    'weight': 2,
                    'fillOpacity': 0.9,
                },
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=folium.Tooltip(
                    tooltip_html, 
                    sticky=True, 
                    style="background-color: transparent; border: none; box-shadow: none;"
                )
            ).add_to(m)
    else:
        # Mapa simple sin datos de ventas con estilo moderno
        for _, row in gdf_mexico.iterrows():
            entidad = row['ENTIDAD']
            capital = row.get('CAPITAL', 'N/A')
            clave = row.get('CVE_EDO', 'N/A')
            
            # Tooltip para hover con diseño moderno
            tooltip_html = f"""
            <div style="
                font-family: 'Segoe UI', 'Roboto', sans-serif; 
                font-size: 14px; 
                padding: 15px 20px; 
                background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); 
                border: none; 
                border-radius: 16px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.12);
                min-width: 200px;
                backdrop-filter: blur(5px);
            ">
                <div style="color: #1a1a2e; font-size: 20px; font-weight: 600; margin-bottom: 10px; letter-spacing: -0.5px;">
                    {entidad}
                </div>
                <div style="color: #616161; font-size: 13px; font-weight: 500;">{capital}</div>
            </div>
            """
            
            # Popup para click con diseño moderno
            popup_html = f"""
            <div style="
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                padding: 20px;
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                min-width: 250px;
            ">
                <h3 style="
                    color: #1a1a2e;
                    margin: 0 0 15px 0;
                    font-weight: 600;
                    font-size: 22px;
                    border-bottom: 2px solid #30475e;
                    padding-bottom: 8px;
                ">{entidad}</h3>
                
                <table style="
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0 8px;
                    font-size: 14px;
                ">
                    <tr>
                        <td style="font-weight: 600; color: #616161; padding: 5px 0;">Capital</td>
                        <td style="color: #1a1a2e; padding: 5px 0;">{capital}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600; color: #616161; padding: 5px 0;">Clave</td>
                        <td style="color: #1a1a2e; padding: 5px 0;">{clave}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600; color: #616161; padding: 5px 0;">Tipo</td>
                        <td style="color: #1a1a2e; padding: 5px 0;">Estado de México</td>
                    </tr>
                </table>
            </div>
            """
            
            # Agregar GeoJson con highlight en hover
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x: {
                    'fillColor': '#9fd3c7',
                    'color': '#385170',
                    'weight': 0.8,
                    'fillOpacity': 0.7,
                },
                highlight_function=lambda x: {
                    'fillColor': '#f2a365',
                    'color': '#30475e',
                    'weight': 2,
                    'fillOpacity': 0.9,
                },
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=folium.Tooltip(
                    tooltip_html, 
                    sticky=True, 
                    style="background-color: transparent; border: none; box-shadow: none;"
                )
            ).add_to(m)
    
    # Agregar controles adicionales para mejor experiencia de usuario
    plugins.Fullscreen(position='topright').add_to(m)
    plugins.MeasureControl(position='bottomright', primary_length_unit='kilometers').add_to(m)
    
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
        print(f" Mapa guardado como: {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar el mapa: {e}")

# Función de ejemplo para demostrar el uso
def ejemplo_uso_mapas():
    """
    Función de ejemplo que demuestra cómo usar las funciones de mapas
    """
    print("EJEMPLO DE USO DE FUNCIONES DE MAPAS DE MÉXICO")
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
