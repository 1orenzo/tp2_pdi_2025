import cv2 
import numpy as np 
import os 

# ETAPA 1: PREPROCESADO

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

# ETAPA 2: DEFINICIÓN MORFOLÓGICA
def etapa2_definicion_objeto(img_binaria):
    kernel_corte = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    img_cortada = cv2.erode(img_binaria, kernel_corte, iterations=1)

    kernel_union = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 1))
    morfologia = cv2.morphologyEx(img_cortada, cv2.MORPH_CLOSE, kernel_union)
    return morfologia 


# ETAPA 3: CLASIFICACIÓN 

def obtener_candidatos_por_score(img_morfologia):
    """
    Devuelve una lista de rectángulos candidatos ordenados por un score geométrico.
    """
    contours, _ = cv2.findContours(img_morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    
    alto_img, ancho_img = img_morfologia.shape[:2]
    centro_x_img = ancho_img / 2
    ratio_ideal = 4.0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = w * h
        
        # Filtro Geométrico "Generoso" (Para no perder ninguna patente difícil)
        if 1.5 < aspect_ratio < 7.5 and 500 < area < 25000:
            if y > alto_img * 0.05: 
                
                centro_x_candidato = x + w / 2
                centro_y_candidato = y + h / 2
                
                # Similitud de forma
                diferencia_ratio = abs(aspect_ratio - ratio_ideal)
                factor_forma = max(0.4, 1.0 - (diferencia_ratio * 0.2))
                
                # Centralidad Horizontal
                distancia_norm = abs(centro_x_candidato - centro_x_img) / (ancho_img / 2)
                factor_centralidad = max(0.4, 1.0 - (distancia_norm * 0.5))
                
                # Posición Vertical (preferimos abajo)
                pos_relativa_y = centro_y_candidato / alto_img
                factor_posicion = 1.0 if pos_relativa_y > 0.3 else 0.6
                
                score = area * factor_forma * factor_centralidad * factor_posicion
                
                candidatos.append({'rect': (x, y, w, h), 'score': score, 'cnt': cnt})

    # Ordenar de mayor score a menor
    candidatos.sort(key=lambda c: c['score'], reverse=True)
    return candidatos


def detectar_placa(img_morfologia, img_original, img_bgr_full):
    """
    Itera sobre los candidatos geométricos y VALIDA el contenido usando segmentación.
    """
    # Obtener lista de posibles patentes ordenadas por probabilidad
    lista_candidatos = obtener_candidatos_por_score(img_morfologia)
    
    max_chars_encontrados = -1
    
    # VALIDACIÓN POR CONTENIDO 
    for candidato in lista_candidatos[:5]: 
        x, y, w, h = candidato['rect']
        
        # Recortamos la imagen original en esa zona
        roi = img_bgr_full[y:y+h, x:x+w]
        
        # Ejecutamos segmentacion para hacer la prueba
        _, lista_chars, _, _ = segmentar_caracteres(roi)
        num_chars = len(lista_chars)
        
        if num_chars >= 5: 
            cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 2) 
            return (x, y, w, h)
            
    return None

# SEGMENTACIÓN DE CARACTERES 

def segmentar_caracteres(placa_img):
    """
    Segmenta caracteres. Si detecta menos de 6, prueba con umbral fijo.
    """
    # Intento 1: Configuración estándar (Otsu)
    resultado = _segmentar_core(placa_img)
    
    # Intento 2: Umbral fijo
    if len(resultado[0]) < 6:
        resultado2 = _segmentar_core(placa_img, umbral_fijo=True)
        if len(resultado2[0]) > len(resultado[0]):
            resultado = resultado2

    # Intento 3: Recortar un porcentaje de abajo
    if len(resultado[0]) < 6:
        resultado3 = _segmentar_core(placa_img, recortar_inferior=True)
        if len(resultado3[0]) > len(resultado[0]):
            resultado = resultado3

    return resultado




def _segmentar_core(placa_img, umbral_fijo=False, recortar_inferior=False,  erosion_fallback=False):
    """
    Función núcleo que realiza la segmentación.
    """
    if len(placa_img.shape) == 3:
        gray = cv2.cvtColor(placa_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = placa_img.copy()
    
    alto, ancho = gray.shape
    
    # Binarizar
    if umbral_fijo:
        _, binaria = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Erosión preliminar para separar caracteres de bordes blancos
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    erosionada = cv2.erode(binaria, kernel_erosion, iterations=2)
    
    # Invertir imagen
    invertida = cv2.bitwise_not(erosionada)
    
    # Buscar la región útil de la patente
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(invertida, connectivity=8)
    
    mejor_rect = None
    mejor_area = 0
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h if h > 0 else 0
        
        if 1.5 < aspect_ratio < 7.0 and area > mejor_area and area > (alto * ancho * 0.2):
            mejor_area = area
            mejor_rect = (x, y, w, h)
    
    if mejor_rect is None:
        mejor_rect = (0, 0, ancho, alto)
    
    x, y, w, h = mejor_rect
    
    # Recortar la región de la patente real 
    patente_recortada = gray[y:y+h, x:x+w]

    # Re-Binarizar el recorte limpio
    if umbral_fijo:
        _, binaria_limpia = cv2.threshold(patente_recortada, 127, 255, cv2.THRESH_BINARY)
    else:
        _, binaria_limpia = cv2.threshold(patente_recortada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Erosión para separar caracteres pegados
    kernel_separar = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    binaria_limpia = cv2.erode(binaria_limpia, kernel_separar, iterations=1)
    
    alto_r, ancho_r = binaria_limpia.shape

    if recortar_inferior:
        recorte = int(alto_r * 0.15)
        binaria_limpia[-recorte:, :] = 0  # Poner en negro el borde inferior


    # Tamaños esperados
    ancho_char_esperado = ancho_r / 8
    alto_char_esperado = alto_r * 0.6
    
    min_ancho = ancho_char_esperado * 0.05 
    max_ancho = ancho_char_esperado * 1.8
    min_alto = alto_char_esperado * 0.5
    max_alto = alto_char_esperado * 1.5
    
    # Extracción de componentes 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria_limpia, connectivity=8)
    
    caracteres = []
    for i in range(1, num_labels):
        cx, cy, cw, ch, area = stats[i]
        
        if min_ancho < cw < max_ancho and min_alto < ch < max_alto:
            caracteres.append({'x': cx, 'y': cy, 'w': cw, 'h': ch, 'area': area})
    
    caracteres.sort(key=lambda c: c['x'])
    
    if len(caracteres) > 6:
        caracteres = sorted(caracteres, key=lambda c: c['area'], reverse=True)[:6]
        caracteres.sort(key=lambda c: c['x'])
    
    imagenes = [binaria_limpia[c['y']:c['y']+c['h'], c['x']:c['x']+c['w']] for c in caracteres]
    
    img_resultado = cv2.cvtColor(binaria_limpia, cv2.COLOR_GRAY2BGR)
    for c in caracteres:
        cv2.rectangle(img_resultado, (c['x'], c['y']), (c['x']+c['w'], c['y']+c['h']), (0, 255, 0), 1)
    
    return imagenes, caracteres, binaria_limpia, img_resultado

# MAIN
if __name__ == "__main__":
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Patentes")
    VER = 1
    
    print(f"{'Imagen':<10} | {'Estado':<6} | {'Chars':<5}")
    print("-" * 30)

    for i in range(1, 13):
        path = os.path.join(carpeta, f"img{i:02d}.png")
        if not os.path.exists(path): continue
        
        img = cv2.imread(path)
        img_display = img.copy()
        
        # Pipeline
        pre = etapa1_preprocesado_realce(img)
        morfo = etapa2_definicion_objeto(pre)
        rect = detectar_placa(morfo, img_display, img)
        
        chars_count = 0
        if rect:
            x, y, w, h = rect
            # Mostramos el numero final
            chars_count = len(segmentar_caracteres(img[y:y+h, x:x+w])[0])
            
        estado = "OK" if chars_count >= 5 else "BAJO"
        print(f"img{i:02d}.png | {estado}     | {chars_count}")
        
        if i == VER and rect:
             x, y, w, h = rect
             _, _, _, resultado = segmentar_caracteres(img[y:y+h, x:x+w])
             cv2.imshow("Placa Detectada", img_display)
             cv2.imshow("Caracteres", resultado)
             cv2.waitKey(0)
             cv2.destroyAllWindows()
