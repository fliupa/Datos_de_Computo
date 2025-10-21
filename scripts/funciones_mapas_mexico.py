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
    Crea un mapa coroplético de México con datos de ventas por región (colores funcionando)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Mapeo de regiones a estados (sinónimos incluidos para mayor cobertura)
    mapeo_regiones = {
        'north_mexico': [
            'BAJA CALIFORNIA', 'BAJA CALIFORNIA SUR', 'SONORA', 'CHIHUAHUA', 'COAHUILA',
            'NUEVO LEÓN', 'TAMAULIPAS', 'SINALOA', 'DURANGO'
        ],
        'central_mexico': [
            'AGUASCALIENTES', 'ZACATECAS', 'SAN LUIS POTOSÍ', 'GUANAJUATO', 'QUERÉTARO',
            'HIDALGO', 'MÉXICO', 'CIUDAD DE MÉXICO', 'MORELOS', 'TLAXCALA', 'PUEBLA', 'MICHOACÁN'
        ],
        'south_mexico': [
            'JALISCO', 'COLIMA', 'NAYARIT', 'VERACRUZ', 'GUERRERO', 'OAXACA',
            'CHIAPAS', 'TABASCO', 'CAMPECHE', 'YUCATÁN', 'QUINTANA ROO'
        ]
    }

    # Normalizar claves de región del DataFrame a las esperadas en el mapeo
    region_aliases = {
        'Norte': 'north_mexico',
        'Centro': 'central_mexico',
        'Sur': 'south_mexico',
        'north_mexico': 'north_mexico',
        'central_mexico': 'central_mexico',
        'south_mexico': 'south_mexico'
    }
    regiones_df = datos_ventas[columna_region].map(lambda r: region_aliases.get(str(r), str(r)))
    valores_region = dict(zip(regiones_df, datos_ventas[columna_valor]))
    
    # Copia del GeoDataFrame y sinónimos de ENTIDAD para mejorar coincidencias
    gdf_plot = gdf_mexico.copy()
    entidad_sinonimos = {
        'DISTRITO FEDERAL': 'CIUDAD DE MÉXICO',
        'ESTADO DE MÉXICO': 'MÉXICO'
    }
    gdf_plot['ENTIDAD'] = gdf_plot['ENTIDAD'].replace(entidad_sinonimos)
    
    # Asignar región a cada estado
    gdf_plot['region'] = 'other'
    for region, estados in mapeo_regiones.items():
        mask = gdf_plot['ENTIDAD'].isin(estados)
        gdf_plot.loc[mask, 'region'] = region
    
    # Asignar valor por región (si no hay valor, 0)
    gdf_plot['valor'] = gdf_plot['region'].map(valores_region).fillna(0)
    
    # Choropleth con leyenda y cmap especificado
    gdf_plot.plot(
        column='valor', ax=ax, cmap=cmap, legend=True,
        edgecolor='white', linewidth=1.2, alpha=0.9,
        legend_kwds={'label': columna_valor, 'orientation': 'horizontal', 'shrink': 0.8}
    )
    
    # Estilo del mapa
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_xlabel('Longitud', fontsize=14, fontweight='bold', color='#34495E')
    ax.set_ylabel('Latitud', fontsize=14, fontweight='bold', color='#34495E')
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#2C3E50')
    ax.grid(True, alpha=0.4, linestyle='--', color='#7F8C8D')
    ax.set_facecolor('#F8F9FA')
    
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
                              color_map='minimal', add_layers=False):
    """
    Crea un mapa interactivo minimalista y estético con Folium.
    - Un solo tile base (CartoDB Positron)
    - Sin popups complejos, solo tooltips simples
    - Estilo suave con bordes discretos y color por valor
    """
    try:
        import folium
        from branca.colormap import LinearColormap
    except ImportError:
        print(" [AVISO] Folium no está instalado; omitiendo mapa interactivo. Instale con: pip install folium")
        return None
        
    # Calcular el centro de México
    bounds = gdf_mexico.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Mapa base minimalista
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=5, 
        tiles='CartoDB Positron',
        control_scale=True
    )

    # Copia y normalización de ENTIDAD para mejores coincidencias
    gdf_plot = gdf_mexico.copy()
    gdf_plot['ENTIDAD'] = gdf_plot['ENTIDAD'].replace({
        'DISTRITO FEDERAL': 'CIUDAD DE MÉXICO',
        'ESTADO DE MÉXICO': 'MÉXICO'
    })

    if datos_ventas is not None and columna_region is not None and columna_valor is not None:
        # Normalizar nombres de región (español e inglés)
        region_aliases = {
            'Norte': 'north_mexico',
            'Centro': 'central_mexico',
            'Sur': 'south_mexico',
            'north_mexico': 'north_mexico',
            'central_mexico': 'central_mexico',
            'south_mexico': 'south_mexico'
        }
        
        # Mapeo de regiones a estados (coherente con el coroplético)
        mapeo_regiones = {
            'north_mexico': [
                'BAJA CALIFORNIA', 'BAJA CALIFORNIA SUR', 'SONORA', 'CHIHUAHUA', 'COAHUILA',
                'NUEVO LEÓN', 'TAMAULIPAS', 'SINALOA', 'DURANGO'
            ],
            'central_mexico': [
                'AGUASCALIENTES', 'ZACATECAS', 'SAN LUIS POTOSÍ', 'GUANAJUATO', 'QUERÉTARO',
                'HIDALGO', 'MÉXICO', 'CIUDAD DE MÉXICO', 'MORELOS', 'TLAXCALA', 'PUEBLA', 'MICHOACÁN'
            ],
            'south_mexico': [
                'JALISCO', 'COLIMA', 'NAYARIT', 'VERACRUZ', 'GUERRERO', 'OAXACA',
                'CHIAPAS', 'TABASCO', 'CAMPECHE', 'YUCATÁN', 'QUINTANA ROO'
            ]
        }

        # Valores por región
        regiones_df = datos_ventas[columna_region].map(lambda r: region_aliases.get(str(r), str(r)))
        valores_region = dict(zip(regiones_df, datos_ventas[columna_valor]))

        vals = list(valores_region.values())
        vmin, vmax = (min(vals) if vals else 0), (max(vals) if vals else 1)

        # Paletas simples y limpias
        palettes = {
            'minimal': ['#f7f7f7', '#cccccc', '#969696', '#525252'],  # Gris neutro
            'viridis': ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
            'pastel': ['#e3f2fd', '#90caf9', '#42a5f5', '#1e88e5'],  # Azules suaves
        }
        colors = palettes.get(color_map, palettes['minimal'])
        cmap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=f"{columna_valor} por región")
        m.add_child(cmap)

        def get_region(entidad):
            for reg, estados in mapeo_regiones.items():
                if entidad in estados:
                    return reg
            return 'other'

        # Añadir estados con estilo minimal y tooltip sencillo
        for _, row in gdf_plot.iterrows():
            entidad = row['ENTIDAD']
            reg = get_region(entidad)
            valor = valores_region.get(reg, None)
            color = cmap(valor if valor is not None else 0)

            folium.GeoJson(
                row['geometry'],
                style_function=lambda x, col=color: {
                    'fillColor': col,
                    'color': '#A0A0A0',
                    'weight': 0.7,
                    'fillOpacity': 0.75,
                },
                highlight_function=lambda x: {
                    'fillColor': color,
                    'color': '#000000',
                    'weight': 1.2,
                    'fillOpacity': 0.9,
                },
                tooltip=folium.Tooltip(
                    f"{entidad} — ${valor:,.0f}" if valor else f"{entidad}",
                    sticky=False
                )
            ).add_to(m)
    else:
        # Sin datos: relleno neutro y tooltip con nombre del estado
        neutral_fill = '#eaeaea'
        for _, row in gdf_plot.iterrows():
            entidad = row['ENTIDAD']
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x: {
                    'fillColor': neutral_fill,
                    'color': '#A0A0A0',
                    'weight': 0.7,
                    'fillOpacity': 0.65,
                },
                highlight_function=lambda x: {
                    'fillColor': '#d5d5d5',
                    'color': '#000000',
                    'weight': 1.2,
                    'fillOpacity': 0.85,
                },
                tooltip=folium.Tooltip(entidad, sticky=False)
            ).add_to(m)

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
