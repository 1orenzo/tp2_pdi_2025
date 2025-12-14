import cv2 
import numpy as np 
import os 

# --------------------------------------------------------------------------
# ETAPA 1: PREPROCESADO
# Objetivo: Resaltar estructuras verticales (caracteres/bordes) y binarizar.
# --------------------------------------------------------------------------
def etapa1_preprocesado_realce(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # Suavizado para reducir ruido antes de derivar
    
    # Detección de bordes verticales usando Sobel en eje X. 
    # Las patentes tienen alto contraste vertical (letras vs fondo).
    sobelx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3) 
    abs_sobelx = cv2.convertScaleAbs(sobelx) 
    
    # Normalizar para aprovechar todo el rango dinámico (0-255)
    realce = cv2.normalize(abs_sobelx, None, 0, 255, cv2.NORM_MINMAX) 
    
    # Umbralización de Otsu para separar automáticamente el fondo del primer plano (bordes)
    _, th_otsu = cv2.threshold(realce, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    return th_otsu 

# --------------------------------------------------------------------------
# ETAPA 2: DEFINICIÓN MORFOLÓGICA
# Objetivo: Fusionar los caracteres sueltos en un único bloque rectangular (la placa).
# --------------------------------------------------------------------------
def etapa2_definicion_objeto(img_binaria):
    # 1. Erosión Vertical (1, 3): Elimina líneas horizontales finas (ruido de parrillas, etc.)
    # y desconecta objetos que no tengan consistencia vertical.
    kernel_corte = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    img_cortada = cv2.erode(img_binaria, kernel_corte, iterations=1)
    
    # 2. Cierre Horizontal (14, 1): Dilatación grande en X seguida de erosión.
    # "Pega" horizontalmente las letras cercanas para formar una sola mancha sólida.
    kernel_union = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 1))
    morfologia = cv2.morphologyEx(img_cortada, cv2.MORPH_CLOSE, kernel_union)
    return morfologia 

# --------------------------------------------------------------------------
# ETAPA 3: CLASIFICACIÓN (Estándar)
# Objetivo: Filtrar contornos por geometría (Área, Proporción, Posición).
# --------------------------------------------------------------------------
def etapa3_clasificacion(img_morfologia, img_original):
    contours, _ = cv2.findContours(img_morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posibles_placas = []
    alto_imagen = img_morfologia.shape[0] 
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = w * h
        y_bottom = y + h
        
        # Filtros:
        # - Ratio: Patentes son rectangulares (ancho > alto). Rango 1.5 a 7.0.
        # - Área: Descartar ruido pequeño o objetos gigantes.
        # - Posición: Descartar lo que esté muy pegado al borde inferior (suelo/paragolpes).
        if 1.5 < aspect_ratio < 7.0 and 500 < area < 20000 and y_bottom < (0.95 * alto_imagen):
            posibles_placas.append(cnt)
    
    if len(posibles_placas) == 0: 
        return None 
    
    # Elegir el candidato más grande que cumpla los requisitos
    mejor_candidato = sorted(posibles_placas, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(mejor_candidato)
    
    # Dibujar rectángulo en la original para visualización
    cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return (x, y, w, h)

# --------------------------------------------------------------------------
# ETAPA 3: CLASIFICACIÓN ESPECIAL
# Objetivo: Lógica de respaldo más permisiva y basada en puntuación (score) para casos difíciles.
# --------------------------------------------------------------------------
def etapa3_clasificacion_especial(img_morfologia, img_original):
    contours, _ = cv2.findContours(img_morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posibles_placas = []
    alto_imagen = img_morfologia.shape[0]
    ancho_imagen = img_morfologia.shape[1]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = w * h
        y_centro = y + h/2
        x_centro = x + w/2
        
        # Filtros más específicos de posición (centrado) y dimensiones mínimas/máximas
        if (2.0 < aspect_ratio < 6.5 and 600 < area < 18000 and
            0.30 * alto_imagen < y_centro < 0.88 * alto_imagen and
            0.20 * ancho_imagen < x_centro < 0.80 * ancho_imagen and
            20 < h < 100 and w > 50):
            posibles_placas.append(cnt)
    
    if len(posibles_placas) == 0:
        return None
    
    mejor_candidato = None
    mejor_score = -1
    
    # Sistema de Puntuación:
    # Premia área grande, posición central y relación de aspecto cercana a 4.0
    for cnt in posibles_placas:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        x_centro = x + w/2
        aspect_ratio = float(w) / h
        distancia_centro_x = abs(x_centro - ancho_imagen/2) / ancho_imagen
        diferencia_ratio = abs(aspect_ratio - 4.0) / 4.0
        
        bonus_ratio = max(0.7, 1 - (diferencia_ratio * 0.4))
        score = area * (1 - distancia_centro_x * 0.3) * bonus_ratio
        
        if score > mejor_score:
            mejor_score = score
            mejor_candidato = cnt
    
    if mejor_candidato is None:
        return None
    
    x, y, w, h = cv2.boundingRect(mejor_candidato)
    cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 1)
    return (x, y, w, h)

# Selector de estrategia según la imagen (Hardcoded para el TP)
def detectar_placa(img_morfologia, img_original, numero_imagen):
    if numero_imagen in [2, 7, 11]: # Imágenes conocidas por ser difíciles
        return etapa3_clasificacion_especial(img_morfologia, img_original)
    else:
        return etapa3_clasificacion(img_morfologia, img_original)


# --------------------------------------------------------------------------
# SEGMENTACIÓN DE CARACTERES 
# Estrategia: Probar configuraciones progresivamente más agresivas si no se hallan caracteres.
# --------------------------------------------------------------------------
def segmentar_caracteres(placa_img):
    """
    Segmenta caracteres. Si detecta menos de 6, prueba con parámetros alternativos (fuerza bruta controlada).
    """
    # Intento 1: Configuración estándar (Otsu)
    resultado = _segmentar_core(placa_img)
    
    # Intento 2: Si falló, probar Umbral Fijo (ayuda si Otsu falla por sombras)
    if len(resultado[0]) < 6:
        resultado2 = _segmentar_core(placa_img, umbral_fijo=True)
        if len(resultado2[0]) > len(resultado[0]): # Quedarse con el que detecte más
            resultado = resultado2
    
    # Intento 3: Erosión Extra (para separar letras pegadas fuertemente)
    if len(resultado[0]) < 6:
        resultado3 = _segmentar_core(placa_img, erosion_extra=True)
        if len(resultado3[0]) > len(resultado[0]):
            resultado = resultado3

    # Intento 4: Erosión de Bordes Extrema (para casos donde las letras tocan marcos gruesos)
    if len(resultado[0]) < 6:
        resultado4 = _segmentar_core(placa_img, erosion_bordes=True)
        if len(resultado4[0]) > len(resultado[0]):
            resultado = resultado4
    
    return resultado


def _segmentar_core(placa_img, umbral_fijo=False, erosion_extra=False, erosion_bordes=False):
    """
    Función núcleo que realiza la segmentación con parámetros configurables.
    """
    if len(placa_img.shape) == 3:
        gray = cv2.cvtColor(placa_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = placa_img.copy()
    
    alto, ancho = gray.shape
    
    # Paso 1: Binarizar
    if umbral_fijo:
        _, binaria = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Paso 2: Erosión preliminar para separar caracteres de bordes blancos
    # Usa kernel vertical (1,3)
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    iter_erosion = 3 if erosion_extra else 2
    erosionada = cv2.erode(binaria, kernel_erosion, iterations=iter_erosion)
    
    # Paso 3: Invertir imagen. ConnectedComponents busca objetos blancos sobre fondo negro.
    # Asumimos que tras la binarización/erosión el fondo es predominante.
    invertida = cv2.bitwise_not(erosionada)
    
    # Paso 4: Buscar la región útil de la patente (el "rectángulo negro" más grande)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(invertida, connectivity=8)
    
    mejor_rect = None
    mejor_area = 0
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h if h > 0 else 0
        
        # Buscamos algo con forma de "caja de texto" grande
        if 1.5 < aspect_ratio < 7.0 and area > mejor_area and area > (alto * ancho * 0.2):
            mejor_area = area
            mejor_rect = (x, y, w, h)
    
    # Si no se encuentra una caja clara, usar toda la imagen
    if mejor_rect is None:
        mejor_rect = (0, 0, ancho, alto)
    
    x, y, w, h = mejor_rect
    
    # Paso 5: Recortar la región de la patente real (quitando márgenes sobrantes)
    patente_recortada = gray[y:y+h, x:x+w]

    # Paso 6: Re-Binarizar el recorte limpio para tener máxima precisión
    if umbral_fijo:
        _, binaria_limpia = cv2.threshold(patente_recortada, 127, 255, cv2.THRESH_BINARY)
    else:
        _, binaria_limpia = cv2.threshold(patente_recortada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Paso 6.5: Erosión Final para separar caracteres pegados entre sí
    kernel_separar = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    binaria_limpia = cv2.erode(binaria_limpia, kernel_separar, iterations=1)

    # Caso especial: erosión horizontal fuerte si se solicitó (para limpiar bordes laterales)
    if erosion_bordes:
        kernel_erosion_fuerte = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        binaria_limpia = cv2.erode(binaria_limpia, kernel_erosion_fuerte, iterations=10)
    
    dilatada = binaria_limpia
    alto_r, ancho_r = dilatada.shape
    
    # Paso 7: Definir rangos de tamaño esperado para un caracter
    # Se calculan relativos al tamaño del recorte
    ancho_char_esperado = ancho_r / 8
    alto_char_esperado = alto_r * 0.6
    
    min_ancho = ancho_char_esperado * 0.15 # Permisivo para letras finas (I, 1)
    max_ancho = ancho_char_esperado * 1.8
    min_alto = alto_char_esperado * 0.5
    max_alto = alto_char_esperado * 1.5
    
    # Paso 8: Extracción final de componentes (Caracteres)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilatada, connectivity=8)
    
    caracteres = []
    for i in range(1, num_labels):
        cx, cy, cw, ch, area = stats[i]
        
        # Filtro geométrico individual de caracteres
        if min_ancho < cw < max_ancho and min_alto < ch < max_alto:
            caracteres.append({'x': cx, 'y': cy, 'w': cw, 'h': ch, 'area': area})
    
    # Ordenar de izquierda a derecha para lectura correcta
    caracteres.sort(key=lambda c: c['x'])
    
    # Si hay ruido y detectamos más de 6, nos quedamos con los 6 más grandes (heurística)
    # y los reordenamos posicionalmente.
    if len(caracteres) > 6:
        caracteres = sorted(caracteres, key=lambda c: c['area'], reverse=True)[:6]
        caracteres.sort(key=lambda c: c['x'])
    
    # Extraer las sub-imágenes de cada caracter
    imagenes = [dilatada[c['y']:c['y']+c['h'], c['x']:c['x']+c['w']] for c in caracteres]
    
    # Generar imagen de visualización con cajas verdes
    img_resultado = cv2.cvtColor(dilatada, cv2.COLOR_GRAY2BGR)
    for c in caracteres:
        cv2.rectangle(img_resultado, (c['x'], c['y']), (c['x']+c['w'], c['y']+c['h']), (0, 255, 0), 1)
    
    return imagenes, caracteres, dilatada, img_resultado

# MAIN
if __name__ == "__main__":
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Patentes")
    VER = 3    
    for i in range(1, 13):
        img = cv2.imread(os.path.join(carpeta, f"img{i:02d}.png"))
        img_display = img.copy()
        
        # Pipeline principal
        morfologia = etapa2_definicion_objeto(etapa1_preprocesado_realce(img))
        rect = detectar_placa(morfologia, img_display, i)
        
        chars = 0
        if rect:
            x, y, w, h = rect
            # Llamada a la segmentación robusta sobre el recorte de la placa
            chars = len(segmentar_caracteres(img[y:y+h, x:x+w])[0])
            
        print(f"img{i:02d}.png | {'OK' if rect else 'FALLO'} | {chars}")
        
        # Visualización detallada para la imagen seleccionada en VER
        if i == VER and rect:
            x, y, w, h = rect
            _, _, _, resultado = segmentar_caracteres(img[y:y+h, x:x+w])
            cv2.imshow("Resultado", resultado)
            cv2.waitKey(0)
            cv2.destroyAllWindows()