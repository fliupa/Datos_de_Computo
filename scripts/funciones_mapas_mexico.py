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
import os

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
    ax.set_axis_off()
    ax.set_facecolor('#F8F9FA')  # Fondo ligeramente gris para contraste
    
    return fig, ax

def crear_mapa_coroplético_ventas(gdf_mexico, datos_ventas=None, columna_region=None, columna_valor=None, 
                                titulo="Ventas por Región", cmap='plasma', figsize=(14, 10), valores_por_region=None):
    """
    Crea un mapa coroplético de México con datos de ventas por región.
    - Si se proporcionan `datos_ventas` y columnas, se usan directamente.
    - Si no, intenta calcular automáticamente desde ventas/clientes locales.
    - Si falla, usa valores embebidos por región como fallback.
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

    # Normalizar y preparar GeoDataFrame
    gdf_plot = gdf_mexico.copy()
    entidad_sinonimos = {
        'DISTRITO FEDERAL': 'CIUDAD DE MÉXICO',
        'ESTADO DE MÉXICO': 'MÉXICO'
    }
    gdf_plot['ENTIDAD'] = gdf_plot['ENTIDAD'].replace(entidad_sinonimos)

    # Construcción de valores por región
    valores_region = None
    if isinstance(valores_por_region, dict) and valores_por_region:
        valores_region = valores_por_region
    elif datos_ventas is not None and columna_region is not None and columna_valor is not None:
        # Usar DataFrame proporcionado
        region_aliases = {
            'Norte': 'north_mexico',
            'Centro': 'central_mexico',
            'Sur': 'south_mexico',
            'north_mexico': 'north_mexico',
            'central_mexico': 'central_mexico',
            'south_mexico': 'south_mexico'
        }
        regiones_df = datos_ventas[columna_region].map(lambda r: region_aliases.get(str(r), str(r)))
        valores_region = datos_ventas.groupby(regiones_df)[columna_valor].sum().to_dict()
    else:
        # Intentar calcular automáticamente desde CSV locales
        try:
            import os
            import pandas as pd
            base_dir = os.path.dirname(__file__)
            ventas_path = os.path.join(base_dir, 'ventas.csv')
            clientes_path = os.path.join(base_dir, 'clientes.csv')
            ventas = pd.read_csv(ventas_path)
            clientes = pd.read_csv(clientes_path)
            df = ventas.merge(clientes[['ID_Cliente', 'Ciudad']], on='ID_Cliente', how='left')
            
            # Mapeo ciudad -> estado (cobertura amplia)
            ciudad_to_entidad = {
                'Ciudad de México': 'CIUDAD DE MÉXICO', 'Guadalajara': 'JALISCO', 'Monterrey': 'NUEVO LEÓN',
                'Puebla': 'PUEBLA', 'Tijuana': 'BAJA CALIFORNIA', 'León': 'GUANAJUATO', 'Cancún': 'QUINTANA ROO',
                'Querétaro': 'QUERÉTARO', 'Mérida': 'YUCATÁN', 'Aguascalientes': 'AGUASCALIENTES',
                'Hermosillo': 'SONORA', 'Morelia': 'MICHOACÁN', 'Toluca': 'MÉXICO', 'Veracruz': 'VERACRUZ',
                'Tampico': 'TAMAULIPAS', 'Culiacán': 'SINALOA', 'Mazatlán': 'SINALOA', 'Acapulco': 'GUERRERO',
                'Saltillo': 'COAHUILA', 'Durango': 'DURANGO', 'Campeche': 'CAMPECHE', 'Oaxaca': 'OAXACA',
                'Zacatecas': 'ZACATECAS', 'Colima': 'COLIMA', 'Tlaxcala': 'TLAXCALA', 'Ciudad Juárez': 'CHIHUAHUA',
                'Villahermosa': 'TABASCO', 'Chetumal': 'QUINTANA ROO', 'La Paz': 'BAJA CALIFORNIA SUR',
                'Torreón': 'COAHUILA', 'Celaya': 'GUANAJUATO', 'Irapuato': 'GUANAJUATO', 'Manzanillo': 'COLIMA',
                'Ensenada': 'BAJA CALIFORNIA', 'Ciudad Obregón': 'SONORA', 'Nogales': 'SONORA', 'Los Mochis': 'SINALOA',
                'Matamoros': 'TAMAULIPAS', 'Reynosa': 'TAMAULIPAS', 'Coatzacoalcos': 'VERACRUZ', 'Pachuca': 'HIDALGO',
                'Cuernavaca': 'MORELOS', 'Tepic': 'NAYARIT', 'Tuxtla Gutiérrez': 'CHIAPAS', 'Xalapa': 'VERACRUZ',
                'San Luis Potosí': 'SAN LUIS POTOSÍ', 'Gómez Palacio': 'DURANGO', 'Uruapan': 'MICHOACÁN',
                'Ciudad Victoria': 'TAMAULIPAS', 'Mexicali': 'BAJA CALIFORNIA'
            }
            df['ENTIDAD'] = df['Ciudad'].map(ciudad_to_entidad)
            df['ENTIDAD'] = df['ENTIDAD'].replace(entidad_sinonimos)
            
            # Columna de valor
            val_col = 'Total' if 'Total' in df.columns else df.select_dtypes('number').columns[-1]
            
            # Totales por estado y región
            totales_estado = df.groupby('ENTIDAD', dropna=True)[val_col].sum().reset_index()
            region_map = {}
            for region, estados in mapeo_regiones.items():
                for est in estados:
                    region_map[est] = region
            totales_estado['region'] = totales_estado['ENTIDAD'].map(region_map).fillna('other')
            valores_region = totales_estado.groupby('region')[val_col].sum().to_dict()
        except Exception as e:
            print(f" [AVISO] No se pudo calcular automáticamente; usando valores embebidos. Detalle: {e}")
            valores_region = {
                'north_mexico': 25896.71,
                'central_mexico': 24520.91,
                'south_mexico': 26943.64
            }

    # Asignar región por estado en el GeoDataFrame
    gdf_plot['region'] = 'other'
    for region, estados in mapeo_regiones.items():
        mask = gdf_plot['ENTIDAD'].isin(estados)
        gdf_plot.loc[mask, 'region'] = region
    
    # Asignar valor por región (si no hay valor, 0)
    gdf_plot['valor'] = gdf_plot['region'].map(valores_region).fillna(0)
    
    # Dibujar el choropleth
    gdf_plot.plot(
        column='valor', ax=ax, cmap=cmap, legend=True,
        edgecolor='white', linewidth=1.2, alpha=0.9,
        legend_kwds={'label': 'Ventas por región', 'orientation': 'horizontal', 'shrink': 0.8}
    )
    
    # Estilo del mapa
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_axis_off()
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
        'Veracruz': (-96.1342, 19.1738),
        # Ciudades adicionales solicitadas
        'Tuxtla Gutiérrez': (-93.1160, 16.7530),
        'Pachuca': (-98.7333, 20.1167),
        'Villahermosa': (-92.9281, 17.9893),
        'Tampico': (-97.8555, 22.2553),
        'Ensenada': (-116.5964, 31.8667),
        'Tlaxcala': (-98.2372, 19.3183),
    }
    
    # Filtrar top ciudades
    top_ciudades = datos_ciudades.head(top_n)
    
    # Colores vibrantes para los marcadores
    colores_vibrantes = ['#E74C3C', '#9B59B6', '#3498DB', '#1ABC9C', '#F39C12', 
                        '#E67E22', '#2ECC71', '#F1C40F', '#E91E63', '#FF5722']
    
    # Patrón de offsets para distribuir etiquetas y minimizar empalmes
    offset_patterns = [
        (-24, -18), (24, -18), (24, 18), (-24, 18),
        (-32, 0), (32, 0), (0, 28), (0, -28), (20, 12), (-20, -12)
    ]
    annotations = []
    
    # Agregar marcadores y etiquetas con offsets
    for i, (_, ciudad) in enumerate(top_ciudades.iterrows()):
        nombre_ciudad = ciudad['Ciudad']
        if nombre_ciudad in coordenadas_ciudades:
            lon, lat = coordenadas_ciudades[nombre_ciudad]
            valor = ciudad.get('Total_Ventas', ciudad.get('Ingresos_Total', 0))
            
            # Tamaño del marcador proporcional al valor
            max_val = max(top_ciudades.get('Total_Ventas', top_ciudades.get('Ingresos_Total', [1])))
            size = max(80, min(600, (valor / max_val) * 400))
            
            # Color vibrante basado en el ranking
            color = colores_vibrantes[i % len(colores_vibrantes)]
            
            ax.scatter(lon, lat, s=size, c=color, alpha=0.8, edgecolors='white', linewidth=3, zorder=5)
            
            # Offset alternado para etiqueta
            dx, dy = offset_patterns[i % len(offset_patterns)]
            ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
            va = 'bottom' if dy > 0 else ('top' if dy < 0 else 'center')
            ann = ax.annotate(
                f'{nombre_ciudad}\n${valor:,.0f}', xy=(lon, lat), 
                xytext=(dx, dy), textcoords='offset points',
                ha=ha, va=va,
                fontsize=10, fontweight='bold', color='#2C3E50', zorder=6,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFFFF', 
                          edgecolor=color, linewidth=2, alpha=0.95),
                arrowprops=dict(arrowstyle='-', color='#7F8C8D', lw=0.8)
            )
            annotations.append(ann)
    
    # Ajuste automático de etiquetas si la librería adjustText está disponible
    try:
        from adjustText import adjust_text
        adjust_text(
            annotations, ax=ax,
            expand_points=(1.2, 1.2), expand_text=(1.2, 1.2),
            arrowprops=dict(arrowstyle='-', color='#7F8C8D', lw=0.8)
        )
    except ImportError:
        # Si no está instalada, dejamos el posicionamiento por offsets como fallback
        pass
    
    # Configurar el mapa con estilo más vibrante
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_axis_off()
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
                              color_map='pastel_soft', add_layers=False):
    """
    Mapa interactivo que colorea por valor de ventas por estado.
    - Normaliza nombres de estado y soporta entradas por estado o por región (Norte/Centro/Sur).
    - Si se provee `datos_ventas` con regiones, distribuye a estados por conteo de ventas (o equitativo si no hay conteos).
    - Guarda el HTML en `data/mapa_interactivo_mexico.html` y retorna (mapa, ruta_html).
    """
    try:
        import folium
        from branca.colormap import LinearColormap
    except ImportError:
        print(" [AVISO] Folium no está instalado; omitiendo mapa interactivo. Instale con: pip install folium")
        return None

    import os
    import unicodedata
    import pandas as pd

    def _norm(s):
        s = str(s).strip().upper()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        return s

    # Asegurar columna ENTIDAD y normalización
    gdf_plot = gdf_mexico.copy()
    if 'ENTIDAD' not in gdf_plot.columns:
        if 'NOM_ENT' in gdf_plot.columns:
            gdf_plot = gdf_plot.rename(columns={'NOM_ENT': 'ENTIDAD'})
        elif 'name' in gdf_plot.columns:
            gdf_plot = gdf_plot.rename(columns={'name': 'ENTIDAD'})
        else:
            raise ValueError("El GeoDataFrame no contiene una columna de estado ('ENTIDAD', 'NOM_ENT' o 'name').")

    sinonimos = {
        'DISTRITO FEDERAL': 'CIUDAD DE MÉXICO',
        'ESTADO DE MÉXICO': 'MÉXICO'
    }
    gdf_plot['ENTIDAD'] = gdf_plot['ENTIDAD'].astype(str).str.upper().replace(sinonimos)
    gdf_plot['ENTIDAD_UP'] = gdf_plot['ENTIDAD'].map(_norm)
    entidades_set = set(gdf_plot['ENTIDAD_UP'])

    # Centro del mapa
    bounds = gdf_plot.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='CartoDB Positron', control_scale=True)

    # Métricas desde CSV para tooltips (conteo y monto por estado)
    num_ventas_estado = {}
    monto_ventas_estado = {}
    try:
        # Intentar rutas en el proyecto actual
        ventas_candidates = [
            os.path.join(os.getcwd(), 'ventas.csv'),
            os.path.join(os.getcwd(), 'data', 'ventas.csv')
        ]
        clientes_candidates = [
            os.path.join(os.getcwd(), 'clientes.csv'),
            os.path.join(os.getcwd(), 'data', 'clientes.csv')
        ]
        ventas_path = next((p for p in ventas_candidates if os.path.exists(p)), None)
        clientes_path = next((p for p in clientes_candidates if os.path.exists(p)), None)
        if ventas_path and clientes_path:
            v = pd.read_csv(ventas_path)
            c = pd.read_csv(clientes_path)
            merge_col = next((col for col in ['ID_Cliente','cliente_id','id_cliente','Cliente_ID'] if col in v.columns), None)
            df_v = v.merge(c[['ID_Cliente','Ciudad']], left_on=merge_col, right_on='ID_Cliente', how='left') if merge_col else v.copy()
            ciudad_to_entidad = {
                'Ciudad de México': 'CIUDAD DE MÉXICO', 'Guadalajara': 'JALISCO', 'Monterrey': 'NUEVO LEÓN',
                'Puebla': 'PUEBLA', 'Tijuana': 'BAJA CALIFORNIA', 'León': 'GUANAJUATO', 'Cancún': 'QUINTANA ROO',
                'Querétaro': 'QUERÉTARO', 'Mérida': 'YUCATÁN', 'Aguascalientes': 'AGUASCALIENTES',
                'Hermosillo': 'SONORA', 'Morelia': 'MICHOACÁN', 'Toluca': 'MÉXICO', 'Veracruz': 'VERACRUZ',
                'Tampico': 'TAMAULIPAS', 'Culiacán': 'SINALOA', 'Mazatlán': 'SINALOA', 'Acapulco': 'GUERRERO',
                'Saltillo': 'COAHUILA', 'Durango': 'DURANGO', 'Campeche': 'CAMPECHE', 'Oaxaca': 'OAXACA',
                'Zacatecas': 'ZACATECAS', 'Colima': 'COLIMA', 'Tlaxcala': 'TLAXCALA', 'Ciudad Juárez': 'CHIHUAHUA',
                'Villahermosa': 'TABASCO', 'Chetumal': 'QUINTANA ROO', 'La Paz': 'BAJA CALIFORNIA SUR',
                'Torreón': 'COAHUILA', 'Celaya': 'GUANAJUATO', 'Irapuato': 'GUANAJUATO', 'Manzanillo': 'COLIMA',
                'Ensenada': 'BAJA CALIFORNIA', 'Ciudad Obregón': 'SONORA', 'Nogales': 'SONORA', 'Los Mochis': 'SINALOA',
                'Matamoros': 'TAMAULIPAS', 'Reynosa': 'TAMAULIPAS', 'Coatzacoalcos': 'VERACRUZ', 'Pachuca': 'HIDALGO',
                'Cuernavaca': 'MORELOS', 'Tepic': 'NAYARIT', 'Tuxtla Gutiérrez': 'CHIAPAS', 'Xalapa': 'VERACRUZ',
                'San Luis Potosí': 'SAN LUIS POTOSÍ', 'Gómez Palacio': 'DURANGO', 'Uruapan': 'MICHOACÁN',
                'Ciudad Victoria': 'TAMAULIPAS', 'Mexicali': 'BAJA CALIFORNIA'
            }
            if 'Ciudad' in df_v.columns:
                df_v['ENTIDAD'] = df_v['Ciudad'].map(ciudad_to_entidad)
                df_v['ENTIDAD_UP'] = df_v['ENTIDAD'].astype(str).str.upper().replace(sinonimos).map(_norm)
                num_ventas_estado = df_v.dropna(subset=['ENTIDAD_UP'])['ENTIDAD_UP'].value_counts().to_dict()
                value_candidates = ['Total','total','Monto','monto','Importe','importe','Ingresos_Total','ingresos_total','Total_Ventas','total_ventas']
                col_valor = next((col for col in value_candidates if col in df_v.columns), None)
                if col_valor:
                    monto_ventas_estado = df_v.dropna(subset=['ENTIDAD_UP']).groupby('ENTIDAD_UP')[col_valor].sum().to_dict()
    except Exception:
        pass

    # Construir valores por estado
    valores_por_estado = {}

    region_aliases = {
        'NORTE': 'NORTE', 'NORTH': 'NORTE', 'NORTH_MEXICO': 'NORTE', 'NORTH MEXICO': 'NORTE',
        'CENTRO': 'CENTRO', 'CENTRAL': 'CENTRO', 'CENTRAL_MEXICO': 'CENTRO', 'CENTRAL MEXICO': 'CENTRO',
        'SUR': 'SUR', 'SOUTH': 'SUR', 'SOUTH_MEXICO': 'SUR', 'SOUTH MEXICO': 'SUR'
    }
    regiones_a_estados = {
        'NORTE': ['BAJA CALIFORNIA', 'BAJA CALIFORNIA SUR', 'SONORA', 'CHIHUAHUA', 'COAHUILA', 'NUEVO LEÓN', 'TAMAULIPAS', 'SINALOA', 'DURANGO'],
        'CENTRO': ['AGUASCALIENTES', 'ZACATECAS', 'SAN LUIS POTOSÍ', 'GUANAJUATO', 'QUERÉTARO', 'HIDALGO', 'MÉXICO', 'CIUDAD DE MÉXICO', 'MORELOS', 'TLAXCALA', 'PUEBLA', 'MICHOACÁN'],
        'SUR': ['JALISCO', 'COLIMA', 'NAYARIT', 'VERACRUZ', 'GUERRERO', 'OAXACA', 'CHIAPAS', 'TABASCO', 'CAMPECHE', 'YUCATÁN', 'QUINTANA ROO']
    }

    if datos_ventas is not None and columna_region is not None and columna_valor is not None:
        df = datos_ventas.copy()
        df['__KEY'] = df[columna_region].map(_norm).replace(sinonimos)
        df['__VAL'] = pd.to_numeric(df[columna_valor], errors='coerce').fillna(0)
        # ¿Son estados o regiones?
        keys = set(df['__KEY'].unique())
        es_estado = any(k in entidades_set for k in keys)
        if es_estado:
            for _, r in df.iterrows():
                k = r['__KEY']
                if k in entidades_set:
                    valores_por_estado[k] = float(r['__VAL'])
        else:
            # Distribuir por región
            for _, r in df.iterrows():
                rk = region_aliases.get(r['__KEY'], r['__KEY'])
                if rk in regiones_a_estados:
                    estados_r = [ _norm(s) for s in regiones_a_estados[rk] ]
                    # Pesos por conteo de ventas
                    pesos = [ num_ventas_estado.get(_norm(s), 0) for s in regiones_a_estados[rk] ]
                    total_pesos = sum(pesos)
                    if total_pesos == 0:
                        # Reparto equitativo
                        reparto = r['__VAL'] / len(estados_r)
                        for est_up in estados_r:
                            valores_por_estado[est_up] = valores_por_estado.get(est_up, 0) + reparto
                    else:
                        for est_up, p in zip(estados_r, pesos):
                            valores_por_estado[est_up] = valores_por_estado.get(est_up, 0) + (r['__VAL'] * (p / total_pesos))
    else:
        # Fallback: usar montos desde CSV si existen
        valores_por_estado = monto_ventas_estado.copy()

    # Colores pastel suaves
    palettes = {
        'pastel_soft': ['#faf3dd','#ffc6ff','#caffbf','#a0c4ff','#bdb2ff'],
        'pastel_warm': ['#ffe5d9','#ffcad4','#f4acb7','#9d8189','#f7d6e0'],
        'pastel_ocean': ['#e0fbfc','#c2dfe3','#9db4c0','#5c6b73','#253237']
    }
    colors = palettes.get(color_map, palettes['pastel_soft'])

    vals = list(valores_por_estado.values())
    vals = [v for v in vals if pd.notna(v)]
    vmin = min(vals) if vals else 0
    vmax = max(vals) if vals else 1
    if vmin == vmax:
        vmax = vmin + 1
    colormap = LinearColormap(colors, vmin=vmin, vmax=vmax)
    colormap.caption = 'Ventas por estado'
    colormap.add_to(m)

    # Pintar estados con colores por valor
    base_border = '#B0BEC5'
    for _, row in gdf_plot.iterrows():
        entidad = row['ENTIDAD']
        entidad_up = row['ENTIDAD_UP']
        valor = valores_por_estado.get(entidad_up, 0)
        num = num_ventas_estado.get(entidad_up, 0)
        monto = valor if valor else monto_ventas_estado.get(entidad_up, None)
        color = colormap(valor) if (valor is not None) else '#ECEFF4'
        tooltip_text = (
            f"{entidad} — {num} ventas — ${monto:,.0f}" if (monto is not None and num > 0) else
            f"{entidad} — {num} ventas" if (num > 0) else
            f"{entidad} — ${monto:,.0f}" if (monto is not None) else
            entidad
        )
        popup_html = f"<b>{entidad}</b><br>{num} ventas<br>Total: {monto if monto is not None else 'N/A'}"

        folium.GeoJson(
            row['geometry'],
            style_function=lambda x, col=color: {
                'fillColor': col,
                'color': base_border,
                'weight': 0.8,
                'fillOpacity': 0.75,
            },
            highlight_function=lambda x: {
                'fillColor': '#000000',
                'color': '#000000',
                'weight': 1.1,
                'fillOpacity': 0.9,
            },
            tooltip=folium.Tooltip(tooltip_text, sticky=False),
            popup=folium.Popup(popup_html, max_width=300),
            zoom_on_click=True
        ).add_to(m)

    # Añadir control de capas si se solicita
    try:
        if add_layers:
            import folium
            folium.LayerControl(position='topright').add_to(m)
    except Exception:
        pass

    # Guardar HTML en dos ubicaciones: CWD/data y .ipynb_checkpoints/data
    html_out = None
    errores_guardado = []
    try:
        out_dir_cwd = os.path.join(os.getcwd(), 'data')
        os.makedirs(out_dir_cwd, exist_ok=True)
        html_out_cwd = os.path.join(out_dir_cwd, 'mapa_interactivo_mexico.html')
        m.save(html_out_cwd)
        html_out = html_out_cwd
    except Exception as e:
        errores_guardado.append(f"CWD/data: {e}")

    try:
        base_dir_file = os.path.dirname(__file__)
        out_dir_ckpt = os.path.join(base_dir_file, 'data')
        os.makedirs(out_dir_ckpt, exist_ok=True)
        html_out_ckpt = os.path.join(out_dir_ckpt, 'mapa_interactivo_mexico.html')
        m.save(html_out_ckpt)
        if html_out is None:
            html_out = html_out_ckpt
    except Exception as e:
        errores_guardado.append(f".ipynb_checkpoints/data: {e}")

    if errores_guardado:
        print(" [AVISO] No se pudo guardar en todas las rutas:")
        for msg in errores_guardado:
            print("  -", msg)
        if html_out is None:
            print(" [ERROR] No se pudo guardar el mapa en HTML en ninguna ruta.")

    return m, html_out

# ----------------------
# Utilidades integradas
# ----------------------

def construir_df_regiones_desde_csv(base_dir=None):
    """
    Construye un DataFrame de métricas por estado (región) desde ventas.csv y clientes.csv.
    - Detecta la llave de cliente para merge.
    - Determina ENTIDAD por 'Estado' si existe o mapea desde 'Ciudad'.
    - Normaliza nombres y sinónimos de estados.
    - Agrega número de ventas y total de ventas por estado.

    Returns: pd.DataFrame con columnas ['region','num_ventas','total_ventas']
    """
    import os
    import pandas as pd

    base_dir = base_dir or os.path.dirname(__file__)
    ventas_path = os.path.join(base_dir, 'ventas.csv')
    clientes_path = os.path.join(base_dir, 'clientes.csv')

    v = pd.read_csv(ventas_path)
    c = pd.read_csv(clientes_path)

    # Detecta llave de cliente
    merge_col_v = next((col for col in ['ID_Cliente', 'cliente_id', 'id_cliente', 'Cliente_ID'] if col in v.columns), None)
    merge_col_c = next((col for col in ['ID_Cliente', 'cliente_id', 'id_cliente', 'Cliente_ID'] if col in c.columns), None)

    if merge_col_v and merge_col_c:
        df_v = v.merge(c, left_on=merge_col_v, right_on=merge_col_c, how='left')
    else:
        df_v = v.copy()

    # Mapeo Ciudad -> Estado (cobertura amplia)
    ciudad_to_entidad = {
        'Ciudad de México': 'CIUDAD DE MÉXICO', 'Guadalajara': 'JALISCO', 'Monterrey': 'NUEVO LEÓN',
        'Puebla': 'PUEBLA', 'Tijuana': 'BAJA CALIFORNIA', 'León': 'GUANAJUATO', 'Cancún': 'QUINTANA ROO',
        'Querétaro': 'QUERÉTARO', 'Mérida': 'YUCATÁN', 'Aguascalientes': 'AGUASCALIENTES',
        'Hermosillo': 'SONORA', 'Morelia': 'MICHOACÁN', 'Toluca': 'MÉXICO', 'Veracruz': 'VERACRUZ',
        'Tampico': 'TAMAULIPAS', 'Culiacán': 'SINALOA', 'Mazatlán': 'SINALOA', 'Acapulco': 'GUERRERO',
        'Saltillo': 'COAHUILA', 'Durango': 'DURANGO', 'Campeche': 'CAMPECHE', 'Oaxaca': 'OAXACA',
        'Zacatecas': 'ZACATECAS', 'Colima': 'COLIMA', 'Tlaxcala': 'TLAXCALA', 'Ciudad Juárez': 'CHIHUAHUA',
        'Villahermosa': 'TABASCO', 'Chetumal': 'QUINTANA ROO', 'La Paz': 'BAJA CALIFORNIA SUR',
        'Torreón': 'COAHUILA', 'Celaya': 'GUANAJUATO', 'Irapuato': 'GUANAJUATO', 'Manzanillo': 'COLIMA',
        'Ensenada': 'BAJA CALIFORNIA', 'Ciudad Obregón': 'SONORA', 'Nogales': 'SONORA', 'Los Mochis': 'SINALOA',
        'Matamoros': 'TAMAULIPAS', 'Reynosa': 'TAMAULIPAS', 'Coatzacoalcos': 'VERACRUZ', 'Pachuca': 'HIDALGO',
        'Cuernavaca': 'MORELOS', 'Tepic': 'NAYARIT', 'Tuxtla Gutiérrez': 'CHIAPAS', 'Xalapa': 'VERACRUZ',
        'San Luis Potosí': 'SAN LUIS POTOSÍ', 'Gómez Palacio': 'DURANGO', 'Uruapan': 'MICHOACÁN',
        'Ciudad Victoria': 'TAMAULIPAS', 'Mexicali': 'BAJA CALIFORNIA'
    }

    # Determina ENTIDAD: prioriza 'Estado' si existe; si no mapea desde 'Ciudad'
    if 'Estado' in df_v.columns and df_v['Estado'].notna().any():
        df_v['ENTIDAD'] = df_v['Estado']
    elif 'Ciudad' in df_v.columns:
        df_v['ENTIDAD'] = df_v['Ciudad'].map(ciudad_to_entidad)
    else:
        df_v['ENTIDAD'] = None

    # Normaliza y sinónimos
    df_v['ENTIDAD_UP'] = (
        df_v['ENTIDAD'].astype(str).str.strip().str.upper()
          .replace({'DISTRITO FEDERAL': 'CIUDAD DE MÉXICO', 'ESTADO DE MÉXICO': 'MÉXICO'})
    )

    # Detecta columna de monto y convierte a numérico
    value_candidates = ['Total', 'total', 'Monto', 'monto', 'Importe', 'importe',
                        'Ingresos_Total', 'ingresos_total', 'Total_Ventas', 'total_ventas']
    valor_col = next((col for col in value_candidates if col in df_v.columns), None)
    if valor_col is not None:
        df_v[valor_col] = pd.to_numeric(df_v[valor_col], errors='coerce')

    # Agrega por estado
    grp = df_v.dropna(subset=['ENTIDAD_UP']).groupby('ENTIDAD_UP')
    df_regiones = grp.agg(
        num_ventas=('ENTIDAD_UP', 'count'),
        total_ventas=(valor_col, 'sum') if valor_col is not None else ('ENTIDAD_UP', 'size')
    ).reset_index().rename(columns={'ENTIDAD_UP': 'region'})

    return df_regiones


def generar_mapa_interactivo_mexico_desde_csv(geojson_path=None, output_path=None, show_inline=False):
    """
    Genera el mapa interactivo de México a partir de CSVs locales y lo guarda en HTML.
    - Usa color base #8BDDF1 y resaltado #F18BAA.
    - Guarda por defecto en data/mapa_interactivo_mexico.html.

    Args:
        geojson_path: ruta al GeoJSON (por defecto admin1.geojson en la carpeta actual)
        output_path: ruta de salida HTML (por defecto data/mapa_interactivo_mexico.html)
        show_inline: si True e IPython está disponible, muestra el IFrame en notebook
    """
    import os
    from pathlib import Path

    base_dir = os.path.dirname(__file__)
    geojson_path = geojson_path or os.path.join(base_dir, 'admin1.geojson')
    output_path = output_path or os.path.join(base_dir, 'data', 'mapa_interactivo_mexico.html')
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    gdf_mexico = cargar_geojson_mexico(geojson_path)
    df_regiones = construir_df_regiones_desde_csv(base_dir=base_dir)

    m = crear_mapa_interactivo_folium(
        gdf_mexico,
        datos_ventas=df_regiones,
        columna_region='region',
        columna_valor='total_ventas'
    )

    if m is not None:
        m.save(str(output_path))
        print(f"Mapa interactivo de México creado: {output_path}")
        if show_inline:
            try:
                from IPython.display import IFrame, display, HTML
                display(HTML("<h3>Mapa interactivo de México (ventas por estado)</h3>"))
                display(IFrame(src=str(output_path), width="100%", height=650))
            except Exception:
                pass
        return m
    else:
        print("No se pudo crear el mapa interactivo; verifique que Folium y GeoPandas estén instalados.")
        return None


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
