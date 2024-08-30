import numpy as np
import open3d as o3d
import os
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from collada import Collada, geometry, material, scene, source

def leer_nube_y_identificadores(ruta_pcd):
    datos = np.loadtxt(ruta_pcd, delimiter=" ")
    pcd_array = datos[:, :3]  # Coordenadas X, Y, Z
    identificadores = datos[:, 3]  # Identificadores
    return pcd_array, identificadores

def leer_y_trasladar_al_origen(ruta_pcd, ruta_traj):
    pcd = o3d.io.read_point_cloud(ruta_pcd, format='xyz')
    traj = o3d.io.read_point_cloud(ruta_traj, format='xyz')
    pcd_array = np.asarray(pcd.points)
    traj_array = np.asarray(traj.points)
    centroide_pcd = np.mean(pcd_array, axis=0)
    pcd_array_trasladado = pcd_array - centroide_pcd
    traj_array_trasladado = traj_array - centroide_pcd
    return pcd_array_trasladado, traj_array_trasladado

def segmentar_trayectoria(traj_array, num_segmentos):
    segmentos = np.array_split(traj_array, num_segmentos)
    return segmentos

def asignar_puntos_a_segmentos_kdtree(pcd_array, segmentos):
    segmentos_concat = np.concatenate(segmentos, axis=0)
    pcd_traj_segmentada = o3d.geometry.PointCloud()
    pcd_traj_segmentada.points = o3d.utility.Vector3dVector(segmentos_concat)
    kdtree = o3d.geometry.KDTreeFlann(pcd_traj_segmentada)
    asignaciones = np.empty(len(pcd_array), dtype=int)
    indices_segmentos = np.cumsum([len(seg) for seg in segmentos])
    for i, punto in enumerate(pcd_array):
        _, idx, _ = kdtree.search_knn_vector_3d(punto, 1)
        asignaciones[i] = np.searchsorted(indices_segmentos, idx, side='right')
    return asignaciones

def calcular_matriz_rotacion_y_su_inversa(segmento):
    pca = PCA(n_components=2)
    pca.fit(segmento[:, :2])
    direccion_principal = pca.components_[0]
    angulo_rotacion = np.arctan2(direccion_principal[1], direccion_principal[0])
    matriz_rotacion = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -angulo_rotacion))  # Rotar para alinear con el eje X
    matriz_inversa = np.linalg.inv(matriz_rotacion)
    return matriz_rotacion, matriz_inversa

def aplicar_transformacion_inversa_a_vertices(vertices, matriz_inversa, centroide_global):
    # Aplica la matriz de rotación inversa y el desplazamiento a los puntos del plano
    puntos_transformados = []
    for punto in vertices:
        punto_rotado = np.dot(matriz_inversa, punto)
        punto_transformado = punto_rotado + centroide_global
        puntos_transformados.append(punto_transformado)
    return puntos_transformados

def filtrar_y_transformar_segmento(segmento, ids_segmento, matriz_rot, identificador_objetivo):
    # Calcular el centroide del segmento antes de la rotación para usarlo como posición global
    centroide_global = np.mean(segmento, axis=0)
    
    # Aplicar la rotación
    segmento_rotado = np.dot(matriz_rot, (segmento - centroide_global).T).T

    # Filtrar por el identificador objetivo
    indices_objetivo = np.where(ids_segmento == identificador_objetivo)[0]
    if len(indices_objetivo) == 0:
        return None, None
    objetos = segmento_rotado[indices_objetivo]
    
    # Construir el plano sólido para los objetos
    min_x, max_x = np.min(objetos[:, 0]), np.max(objetos[:, 0])
    min_y, max_y = np.min(objetos[:, 1]), np.max(objetos[:, 1])
    min_z, max_z = np.min(objetos[:, 2]), np.max(objetos[:, 2])
    puntos_plano = np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z], 
        [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z], 
        [max_x, max_y, max_z], [min_x, max_y, max_z]
    ])
    return puntos_plano, centroide_global

def guardar_plano(puntos_plano, ruta_archivo):
    if puntos_plano is not None:
        puntos_plano_array = np.array(puntos_plano, dtype=np.float64)
        np.savetxt(ruta_archivo, puntos_plano_array, fmt='%f')

def filtrar_y_transformar_arboles(segmento, ids_segmento, matriz_rot):
    centroide_global = np.mean(segmento, axis=0)
    segmento_rotado = np.dot(matriz_rot, (segmento - centroide_global).T).T
    indices_arboles = np.where(ids_segmento == 33)[0]
    if len(indices_arboles) == 0:
        return None, None
    arboles = segmento_rotado[indices_arboles]

    # Usar DBSCAN para agrupar los puntos de los árboles
    clustering = DBSCAN(eps=1.0, min_samples=2).fit(arboles)
    etiquetas = clustering.labels_

    cilindros = []
    for etiqueta in np.unique(etiquetas):
        if etiqueta == -1:
            continue  # Ignorar ruido
        cluster = arboles[etiquetas == etiqueta]
        centroide = np.mean(cluster, axis=0)
        altura = np.max(cluster[:, 2]) - np.min(cluster[:, 2])
        radio = np.mean([np.linalg.norm(p[:2] - centroide[:2]) for p in cluster])
        cilindros.append((centroide, radio, altura))
    
    return cilindros, centroide_global

def crear_y_guardar_cilindros_collada(cilindros, ruta_archivo, centroide_global, matriz_rot_inversa, color):
    collada_mesh = Collada()

    efecto = material.Effect("efecto0", [], "phong", diffuse=color, specular=(0,0,0))
    mat = material.Material("material0", "MaterialArbol", efecto)
    collada_mesh.effects.append(efecto)
    collada_mesh.materials.append(mat)

    rotacion_x_neg_90 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    for i, (centroide, radio, altura) in enumerate(cilindros):
        segmentos = 20  # Number of segments to approximate the cylinder
        puntos = []
        for j in range(segmentos):
            angulo = 2 * np.pi * j / segmentos
            x = radio * np.cos(angulo)
            y = radio * np.sin(angulo)
            puntos.append(centroide + np.array([x, y, 0]))
            puntos.append(centroide + np.array([x, y, altura]))

        # Apply inverse transformation and rotation
        puntos = aplicar_transformacion_inversa_a_vertices(puntos, matriz_rot_inversa, centroide_global)
        puntos = [np.dot(rotacion_x_neg_90, punto) for punto in puntos]

        vert_src = source.FloatSource(f"cylinder_{i}_array", np.array(puntos).flatten(), ('X', 'Y', 'Z'))
        geom = geometry.Geometry(collada_mesh, f"geometry_{i}", "cylinder", [vert_src])
        input_list = source.InputList()
        input_list.addInput(0, 'VERTEX', f"#cylinder_{i}_array")

        indices = []
        for j in range(segmentos):
            next_idx = (j + 1) % segmentos
            indices.extend([j * 2, next_idx * 2, j * 2 + 1])
            indices.extend([next_idx * 2, next_idx * 2 + 1, j * 2 + 1])

        triset = geom.createTriangleSet(np.array(indices).flatten(), input_list, "material0")
        geom.primitives.append(triset)
        collada_mesh.geometries.append(geom)

        matnode = scene.MaterialNode("materialref", mat, inputs=[])
        geomnode = scene.GeometryNode(geom, [matnode])
        node = scene.Node(f"node_{i}", children=[geomnode])
        myscene = scene.Scene("myscene", [node])
        collada_mesh.scenes.append(myscene)
        collada_mesh.scene = myscene

    collada_mesh.write(ruta_archivo)

def procesar_y_guardar_modelos_collada(segmentos_nube_con_ids, ruta_carpeta, prefijo_archivo="modelo_edificio", identificador_objetivo=20, color=(1, 0, 0)):
    asegurar_directorio(ruta_carpeta)
    for i, (segmento, ids_segmento) in enumerate(segmentos_nube_con_ids):
        matriz_rot, matriz_inversa = calcular_matriz_rotacion_y_su_inversa(segmento)
        if identificador_objetivo == 33:  # Si estamos procesando árboles
            cilindros, centroide_global = filtrar_y_transformar_arboles(segmento, ids_segmento, matriz_rot)
            if cilindros is not None:
                ruta_archivo_dae = os.path.join(ruta_carpeta, f"{prefijo_archivo}_{i}.dae")
                crear_y_guardar_cilindros_collada(cilindros, ruta_archivo_dae, centroide_global, matriz_inversa, color)
        else:
            plano, centroide_global = filtrar_y_transformar_segmento(segmento, ids_segmento, matriz_rot, identificador_objetivo)
            if plano is not None:
                ruta_archivo_dae = os.path.join(ruta_carpeta, f"{prefijo_archivo}_{i}.dae")
                crear_y_guardar_modelo_collada(plano, ruta_archivo_dae, centroide_global, matriz_inversa, color)

def crear_y_guardar_modelo_collada(puntos_plano, ruta_archivo, centroide_global, matriz_rot_inversa, color):
    collada_mesh = Collada()

    efecto = material.Effect("efecto0", [], "phong", diffuse=color, specular=(0,0,0))
    mat = material.Material("material0", "MaterialEdificio", efecto)
    collada_mesh.effects.append(efecto)
    collada_mesh.materials.append(mat)

    puntos_transformados = aplicar_transformacion_inversa_a_vertices(puntos_plano, matriz_rot_inversa, centroide_global)

    rotacion_x_90 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    puntos_transformados = [np.dot(rotacion_x_90, punto) for punto in puntos_transformados]

    vert_src = source.FloatSource("cubeverts-array", np.array(puntos_transformados).flatten(), ('X', 'Y', 'Z'))
    geom = geometry.Geometry(collada_mesh, "geometry0", "mycube", [vert_src])

    input_list = source.InputList()
    input_list.addInput(0, 'VERTEX', "#cubeverts-array")

    indices = [
        0, 1, 2, 0, 2, 3,  # Frente
        4, 5, 7, 4, 7, 6,  # Atrás
        0, 1, 5, 0, 5, 4,  # Izquierda
        1, 2, 6, 1, 6, 5,  # Derecha
        2, 3, 7, 2, 7, 6,  # Superior
        3, 0, 4, 3, 4, 7   # Inferior
    ]

    triset = geom.createTriangleSet(np.array(indices).flatten(), input_list, "material0")
    geom.primitives.append(triset)
    collada_mesh.geometries.append(geom)

    matnode = scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = scene.GeometryNode(geom, [matnode])
    node = scene.Node("node0", children=[geomnode])

    myscene = scene.Scene("myscene", [node])
    collada_mesh.scenes.append(myscene)
    collada_mesh.scene = myscene

    collada_mesh.write(ruta_archivo)

def asegurar_directorio(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

# Configuración inicial
ruta_pcd = 'C:/Users/pablo/OneDrive/Documentos/AWE/pcd/Camelias_C1_1_labelled_subsample2.txt'
ruta_traj = 'C:/Users/pablo/OneDrive/Documentos/AWE/pcd/traj_Camelias_1.txt'
ruta_carpeta_destino_edificios = "C:/Users/pablo/OneDrive/Documentos/AWE/pcd/pcd_trasladada/planos_edificios"
ruta_carpeta_destino_carreteras = "C:/Users/pablo/OneDrive/Documentos/AWE/pcd/pcd_trasladada/planos_carreteras"
ruta_carpeta_destino_aceras = "C:/Users/pablo/OneDrive/Documentos/AWE/pcd/pcd_trasladada/planos_aceras"
ruta_carpeta_destino_arboles = "C:/Users/pablo/OneDrive/Documentos/AWE/pcd/pcd_trasladada/planos_arboles"

pcd = o3d.io.read_point_cloud(ruta_pcd, format='xyz')
o3d.io.write_point_cloud("C:/Users/pablo/OneDrive/Documentos/AWE/pcd/nube_original.ply", pcd)

# Procesamiento principal
pcd_array, identificadores = leer_nube_y_identificadores(ruta_pcd)
pcd_trasladado, traj_trasladado = leer_y_trasladar_al_origen(ruta_pcd, ruta_traj)
nube_trasladada = o3d.geometry.PointCloud()
nube_trasladada.points = o3d.utility.Vector3dVector(pcd_trasladado)
o3d.io.write_point_cloud("C:/Users/pablo/OneDrive/Documentos/AWE/pcd/nube_trasladada.ply", nube_trasladada)

num_segmentos = 10  # Define el número de segmentos deseado
segmentos_trayectoria = segmentar_trayectoria(traj_trasladado, num_segmentos)
asignaciones = asignar_puntos_a_segmentos_kdtree(pcd_trasladado, segmentos_trayectoria)

segmentos_nube_con_ids = [(pcd_trasladado[asignaciones == i], identificadores[asignaciones == i]) for i in range(num_segmentos)]

# Procesar y guardar los planos de edificios por segmento
procesar_y_guardar_modelos_collada(segmentos_nube_con_ids, ruta_carpeta_destino_edificios, identificador_objetivo=20, color=(1, 0, 0))
procesar_y_guardar_modelos_collada(segmentos_nube_con_ids, ruta_carpeta_destino_carreteras, prefijo_archivo="carretera", identificador_objetivo=10, color=(0.5, 1, 0))
procesar_y_guardar_modelos_collada(segmentos_nube_con_ids, ruta_carpeta_destino_aceras, prefijo_archivo="acera", identificador_objetivo=12, color=(0, 0, 1))

# Ajustar las rutas según sea necesario
ruta_carpeta_destino_edificios_collada = "C:/Users/pablo/OneDrive/Documentos/AWE/pcd/pcd_trasladada/modelos_collada_edificios"
ruta_carpeta_destino_carreteras_collada = "C:/Users/pablo/OneDrive/Documentos/AWE/pcd/pcd_trasladada/modelos_collada_carreteras"
ruta_carpeta_destino_aceras_collada = "C:/Users/pablo/OneDrive/Documentos/AWE/pcd/pcd_trasladada/modelos_collada_aceras"
ruta_carpeta_destino_arboles_collada = "C:/Users/pablo/OneDrive/Documentos/AWE/pcd/pcd_trasladada/modelos_collada_arboles"

procesar_y_guardar_modelos_collada(segmentos_nube_con_ids, ruta_carpeta_destino_edificios_collada, identificador_objetivo=20, color=(1, 0, 0))
procesar_y_guardar_modelos_collada(segmentos_nube_con_ids, ruta_carpeta_destino_carreteras_collada, prefijo_archivo="modelo_carretera", identificador_objetivo=10, color=(0.5, 1, 0))
procesar_y_guardar_modelos_collada(segmentos_nube_con_ids, ruta_carpeta_destino_aceras_collada, prefijo_archivo="modelo_acera", identificador_objetivo=12, color=(0, 0, 1))
procesar_y_guardar_modelos_collada(segmentos_nube_con_ids, ruta_carpeta_destino_arboles_collada, prefijo_archivo="modelo_arbol", identificador_objetivo=33, color=(0, 1, 0))
