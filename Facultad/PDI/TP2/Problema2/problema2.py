import cv2 
import numpy as np 
import os 

def mostrar(titulo, imagen):
    cv2.imshow(titulo, imagen) 
    print(f"Presiona una tecla en '{titulo}' para continuar...") 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

# PREPROCESADO 
def etapa1_preprocesado_realce(img_bgr):

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0) 
    
    sobelx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3) 
    abs_sobelx = cv2.convertScaleAbs(sobelx) 
    
    realce = cv2.normalize(abs_sobelx, None, 0, 255, cv2.NORM_MINMAX) 

    _, th_otsu = cv2.threshold(realce, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    
    return th_otsu # Devuelve la imagen binarizada (bordes de Sobel)

# DEFINICIÓN DE OBJETO
def etapa2_definicion_objeto(img_binaria):
    
    # EROSIÓN VERTICAL 
    kernel_corte = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    # Aplica erosion para desconectar de la parrilla/paragolpes
    img_cortada = cv2.erode(img_binaria, kernel_corte, iterations=1)

    #  CLAUSURA HORIZONTAL
    # Kernel 
    kernel_union = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 1))
    morfologia = cv2.morphologyEx(img_cortada, cv2.MORPH_CLOSE, kernel_union)

    return morfologia # Devuelve la imagen con el bloque sólido (candidato a patente)

# CLASIFICACIÓN
def etapa3_clasificacion(img_morfologia, img_original_para_dibujar):

    contours, _ = cv2.findContours(img_morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posibles_placas = []
    # Obtiene la altura total de la imagen para el filtro posicional
    alto_imagen = img_morfologia.shape[0] 
    # Crea una copia para dibujar los resultados de debug
    img_debug = img_original_para_dibujar.copy()

    for cnt in contours:
        # Obtiene el Bounding Box (coordenadas y dimensiones)
        x, y, w, h = cv2.boundingRect(cnt)
        #Calcula el Descriptor de Forma (Relación de Aspecto)
        aspect_ratio = float(w) / h
        #Calcula el Descriptor de Tamaño (Área del Bounding Box)
        area = w * h
        # Calcula la coordenada inferior (para filtro posicional)
        y_bottom = y + h
        
        # Dibuja un rectángulo AZUL para cada contorno analizado
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), (255, 0, 0), 1)

        # Filtros de Clasificación (Basados en geometría y posición)
        # Filtra por Proporción (entre 1.5 y 7.0 para aceptar inclinación)
        filtro_ratio = 1.5 < aspect_ratio < 7.0 
        # Filtra por Área (entre un mínimo de ruido y un máximo del coche)
        filtro_area = 500 < area < 20000 
        # Filtra por Posición (descartar el 5% inferior de la imagen - suelo)
        filtro_posicion = y_bottom < (0.95 * alto_imagen) 

        if filtro_ratio and filtro_area and filtro_posicion:
            # Si pasa todos los filtros, lo añade como candidato
            posibles_placas.append(cnt)
            cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("6 - Debug Clasificacion", img_debug) # Muestra la etapa de filtrado
    
    if len(posibles_placas) == 0: return None # Si no hay candidatos, devuelve None

    # Selección por Mayor Área (Criterio de Máxima Confianza)
    mejor_candidato = sorted(posibles_placas, key=cv2.contourArea, reverse=True)[0]
    # Devuelve las coordenadas del mejor candidato
    return cv2.boundingRect(mejor_candidato) 

# SEGMENTACIÓN DE CARACTERES  
def etapa4_segmentacion_caracteres(roi_placa, roi_color_para_dibujar):
    
    # Umbralización Adaptativa  - Aplicada al recorte de la placa
    th_chars = cv2.adaptiveThreshold(roi_placa, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 4)
    
    kernel_separacion = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8)) 
    # erosion(2 iteraciones) para intentar separar letras fusionadas
    th_chars = cv2.erode(th_chars, kernel_separacion, iterations=2) 
    
    # Búsqueda de Contornos 
    contours, _ = cv2.findContours(th_chars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chars_validos = []
    h_placa, w_placa = roi_placa.shape
    img_chars = roi_color_para_dibujar.copy()

    margen = 3 # Filtro de borde

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Descriptores
        h_rel = h / float(h_placa) # Altura relativa
        ratio = w / float(h) # Ratio
        area = w * h
        
        # Filtros Geométricos
        cond_geo = (0.25 < h_rel < 0.95) and (0.15 < ratio < 1.1) and (area > 30)
        # Filtro de Posición (Exclusión de bordes)
        cond_borde = (x > margen) and ((x + w) < (w_placa - margen))

        if cond_geo and cond_borde:
            chars_validos.append((x, y, w, h))
            cv2.rectangle(img_chars, (x, y), (x+w, y+h), (0, 0, 255), 1)
    
    # Ordenar y devolver la lista de coordenadas
    chars_validos = sorted(chars_validos, key=lambda c: c[0])
    
    print(f"Total caracteres detectados: {len(chars_validos)}")
    return img_chars, len(chars_validos)



if __name__ == "__main__":

    
    import os
    directorio = os.path.dirname(os.path.abspath(__file__))
    archivo = "img04.png" 
    ruta = os.path.join(directorio, "Patentes", archivo)
    
    img = cv2.imread(ruta)
    
    if img is not None:
        # SECUENCIA COMPLETA
        
        # A. DETECCIÓN DE PLACA
        th_otsu = etapa1_preprocesado_realce(img)
        img_morfologia = etapa2_definicion_objeto(th_otsu)
        rectangulo_placa = etapa3_clasificacion(img_morfologia, img)
        
        if rectangulo_placa is not None:
            x, y, w, h = rectangulo_placa
            
            # Recortes (ROI)
            roi_gris = th_otsu[y:y+h, x:x+w] # Usamos la binaria o la gris para procesar
            # Mejor usar la gris original para el threshold nuevo:
            gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi_para_procesar = gray_original[y:y+h, x:x+w]
            
            roi_color = img[y:y+h, x:x+w] 
            
            cv2.imshow("1. Placa Detectada", roi_color)
            
            # B. SEGMENTACIÓN DE CARACTERES
            img_resultado_chars, num_chars = etapa4_segmentacion_caracteres(roi_para_procesar, roi_color)
            
            cv2.imshow("2. Caracteres Segmentados", img_resultado_chars)
            print("Proceso terminado exitosamente.")
            
        else:
            print("Fallo en Etapa 3 (No se detectó placa).")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error de archivo.")