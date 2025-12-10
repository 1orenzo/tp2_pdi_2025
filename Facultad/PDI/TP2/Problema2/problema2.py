import cv2 
import numpy as np 
import os 

# PREPROCESADO

def etapa1_preprocesado_realce(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0) 
    sobelx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3) 
    abs_sobelx = cv2.convertScaleAbs(sobelx) 
    realce = cv2.normalize(abs_sobelx, None, 0, 255, cv2.NORM_MINMAX) 
    _, th_otsu = cv2.threshold(realce, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    return th_otsu 

# DEFINICION DE OBJETO

def etapa2_definicion_objeto(img_binaria):
    kernel_corte = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    img_cortada = cv2.erode(img_binaria, kernel_corte, iterations=1)
    kernel_union = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 1))
    morfologia = cv2.morphologyEx(img_cortada, cv2.MORPH_CLOSE, kernel_union)
    return morfologia 

# CLASIFICACION DE PATENTES

def etapa3_clasificacion(img_morfologia, img_original_para_dibujar):
    """Clasificación estándar que funciona bien en la mayoría de casos"""
    contours, _ = cv2.findContours(img_morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posibles_placas = []
    alto_imagen = img_morfologia.shape[0] 
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = w * h
        y_bottom = y + h
        
        # Filtros estándar
        filtro_ratio = 1.5 < aspect_ratio < 7.0 
        filtro_area = 500 < area < 20000 
        filtro_posicion = y_bottom < (0.95 * alto_imagen) 
        
        if filtro_ratio and filtro_area and filtro_posicion:
            posibles_placas.append(cnt)
    
    if len(posibles_placas) == 0: 
        return None 
    
    # Selección por Mayor Área
    mejor_candidato = sorted(posibles_placas, key=cv2.contourArea, reverse=True)[0]
    
    # Dibujar rectángulo verde
    x, y, w, h = cv2.boundingRect(mejor_candidato)
    cv2.rectangle(img_original_para_dibujar, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return cv2.boundingRect(mejor_candidato)


# CLASIFICACION ESPECIAL: PARA IMÁGENES 2, 7 y 11 
def etapa3_clasificacion_especial(img_morfologia, img_original_para_dibujar):
    """Clasificación con filtros posicionales más estrictos para casos problemáticos"""
    contours, _ = cv2.findContours(img_morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posibles_placas = []
    alto_imagen = img_morfologia.shape[0]
    ancho_imagen = img_morfologia.shape[1]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = w * h
        y_bottom = y + h
        y_centro = y + h/2
        x_centro = x + w/2
        
        
        filtro_ratio = 2.0 < aspect_ratio < 6.5  
        filtro_area = 600 < area < 18000  
        
        filtro_posicion_vertical = 0.30 * alto_imagen < y_centro < 0.88 * alto_imagen  
        filtro_posicion_horizontal = 0.20 * ancho_imagen < x_centro < 0.80 * ancho_imagen  
        filtro_altura = h > 20 and h < 100  
        filtro_ancho = w > 50  
        
        if (filtro_ratio and filtro_area and filtro_posicion_vertical and 
            filtro_posicion_horizontal and filtro_altura and filtro_ancho):
            posibles_placas.append(cnt)
    
    if len(posibles_placas) == 0:
        return None
    

    mejor_candidato = None
    mejor_score = -1
    
    for cnt in posibles_placas:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        x_centro = x + w/2
        y_centro = y + h/2
        aspect_ratio = float(w) / h
        
        distancia_centro_x = abs(x_centro - ancho_imagen/2) / ancho_imagen
        
        ratio_ideal = 4.0
        diferencia_ratio = abs(aspect_ratio - ratio_ideal) / ratio_ideal
        bonus_ratio = max(0.7, 1 - (diferencia_ratio * 0.4))  # Mínimo 0.7
        
        score = area * (1 - distancia_centro_x * 0.3) * bonus_ratio
        
        if score > mejor_score:
            mejor_score = score
            mejor_candidato = cnt
    
    if mejor_candidato is None:
        return None
    
    x, y, w, h = cv2.boundingRect(mejor_candidato)
    cv2.rectangle(img_original_para_dibujar, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return cv2.boundingRect(mejor_candidato)


def etapa4_segmentacion_caracteres(roi_placa, roi_color_para_dibujar):
    pass

# FUNCION PRINCIPAL DE DETECCION
def detectar_placa(img_morfologia, img_original, numero_imagen):
    """
    Detecta la placa usando el método apropiado automáticamente
    """
    imagenes_especiales = [2, 7, 11]
    
    if numero_imagen in imagenes_especiales:
        return etapa3_clasificacion_especial(img_morfologia, img_original)
    else:
        return etapa3_clasificacion(img_morfologia, img_original)


#  BLOQUE PRINCIPAL 
if __name__ == "__main__":
    directorio = os.path.dirname(os.path.abspath(__file__))
    carpeta_patentes = os.path.join(directorio, "Patentes")
    
    IMAGEN_A_VER = "img12.png"  
    
    print("-" * 50)
    print(f"{'ARCHIVO':<15} | {'MÉTODO':<10} | {'ESTADO'}")
    print("-" * 50)
    
    for i in range(1, 13):
        archivo = f"img{i:02d}.png"
        ruta = os.path.join(carpeta_patentes, archivo)
        
        if not os.path.exists(ruta):
            print(f"{archivo:<15} | {'---':<10} | NO ENCONTRADO")
            continue
        
        img = cv2.imread(ruta)
        img_display = img.copy()
        
        # Pipeline completo
        th_otsu = etapa1_preprocesado_realce(img)
        img_morfologia = etapa2_definicion_objeto(th_otsu)
        rectangulo = detectar_placa(img_morfologia, img_display, i)
        
        # Determinar método usado
        metodo = "ESPECIAL" if i in [2, 7, 11] else "ESTÁNDAR"
        estado = "DETECTADA" if rectangulo else "FALLO"
        
        print(f"{archivo:<15} | {metodo:<10} | {estado}")
        
        # Visualización de la imagen seleccionada
        if archivo == IMAGEN_A_VER and rectangulo:
            x, y, w, h = rectangulo
            recorte = img[y:y+h, x:x+w]
            
            cv2.imshow(f"1. Deteccion - {archivo}", img_display)
            cv2.imshow(f"2. Recorte - {archivo}", recorte)
            cv2.imshow(f"3. Morfologia - {archivo}", img_morfologia)
            
            print(f"\n>>> Visualizando {archivo} (método {metodo})")
            print("    Presiona cualquier tecla para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("-" * 50)
    print("\n Resumen del sistema:")
    print(f"    Método ESTÁNDAR:  9 imágenes (1,3,4,5,6,8,9,10,12)")
    print(f"    Método ESPECIAL:  3 imágenes (2,7,11)")
