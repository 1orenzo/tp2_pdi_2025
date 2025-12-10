import cv2
import numpy as np
import os

def mostrar(titulo, imagen):
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(titulo, 1000, 800)
    cv2.imshow(titulo, imagen)
    print(f"Presiona una tecla en '{titulo}' para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- PREPROCESADO ---

# Asegura que busque la imagen en la misma carpeta que el script
directorio_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(directorio_script, "monedas.jpg")
img_original = cv2.imread(ruta_imagen)

if img_original is None:
    print("Error: No se encuentra la imagen 'monedas.jpg' en el directorio.")
    exit()

# (ESCALADO)
scale_percent = 800 / img_original.shape[1]
width = int(img_original.shape[1] * scale_percent)
height = int(img_original.shape[0] * scale_percent)
img = cv2.resize(img_original, (width, height), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Corregir fondo no uniforme (Top-Hat / Sustracción de fondo)
kernel_fondo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
img_sin_fondo = cv2.morphologyEx(img_blur, cv2.MORPH_TOPHAT, kernel_fondo)

# mostrar("1. Preprocesado (Blur + Top-Hat)", img_sin_fondo)

# --- DETECCION DE BORDES ---
bordes = cv2.Canny(img_blur, 30, 80)

# Refinamiento: Dilatación + Clausura 
kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

# Dilatación 
bordes_procesados = cv2.dilate(bordes, kernel_morf, iterations=1)

# Clausura
bordes_procesados = cv2.morphologyEx(bordes_procesados, cv2.MORPH_CLOSE, kernel_morf, iterations=2)

# CORRECCIÓN DE FUSIÓN (Separar objetos pegados)
kernel_separacion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
bordes_separados = cv2.erode(bordes_procesados, kernel_separacion, iterations=3)

# mostrar("Corrección 1: Objetos Separados (Erosión)", bordes_separados)

contours, _ = cv2.findContours(bordes_separados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contornos encontrados: {len(contours)}")

# --- CLASIFICACION ---
img_resultado = img.copy()
monedas_cnt = []
dados_cnt = []

# Umbral para detectar los puntos dentro de los dados
thresh_puntos = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)

print(f"\n--- ANÁLISIS HÍBRIDO (Fp + Vértices) ---")

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    
    if perimetro == 0 or area < 800: continue 

    # Factor de forma (Circularidad)
    fp = area / (perimetro ** 2)
    
    # Aproximación de polígonos
    approx = cv2.approxPolyDP(cnt, 0.02 * perimetro, True)
    num_vertices = len(approx)

    es_moneda = False
    
    # Lógica de clasificación Híbrida
    if fp >= 0.068:
        es_moneda = True
    elif fp < 0.045:
        es_moneda = False
    else:
        # Zona gris: desempatar con vértices
        if num_vertices > 6:
            es_moneda = True
        else:
            es_moneda = False

    if not es_moneda:
        # --- ES UN DADO ---
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Margen de Seguridad para la ROI
        margen = 8 
        roi = thresh_puntos[y+margen : y+h-margen, x+margen : x+w-margen]
        
        subs, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        puntos = 0
        puntos_detectados = [] 
        
        for s in subs:
            area_punto = cv2.contourArea(s)
            
            # Filtro de tamaño para los puntos (pips)
            if 15 < area_punto < 400:
                
                # Circularidad del Punto 
                perim_punto = cv2.arcLength(s, True)
                if perim_punto == 0: continue
                
                circularidad_punto = (4 * np.pi * area_punto) / (perim_punto ** 2)
                
                # Solo contamos si parece un círculo 
                if circularidad_punto > 0.65:
                    puntos += 1
                    
                    # Guardamos la posición para dibujarlo visualmente
                    M_p = cv2.moments(s)
                    if M_p["m00"] != 0:
                        cx_p = int(M_p["m10"] / M_p["m00"]) + x + margen
                        cy_p = int(M_p["m01"] / M_p["m00"]) + y + margen
                        puntos_detectados.append((cx_p, cy_p))
        
        etiqueta_visual = f"D-{puntos}"
        dados_cnt.append(puntos)
        color = (0, 0, 255) # Rojo para dados
        
        # Dibujamos puntos amarillos sobre los pips detectados para verificar
        for pt in puntos_detectados:
            cv2.circle(img_resultado, pt, 3, (0, 255, 255), -1)
        
    else:
        # --- ES UNA MONEDA ---
        # Clasificación por tamaño (Area)
        if area < 3500:
            val = "10c"
        elif area > 4500:
            val = "50c"
        else:
            val = "1p"
            
        etiqueta_visual = val
        monedas_cnt.append(val)
        color = (0, 255, 0) # Verde para monedas

    # --- VISUALIZACIÓN ---
    cv2.drawContours(img_resultado, [cnt], -1, color, 2)
    
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        texto = etiqueta_visual
        # Texto con contorno negro para mejor contraste
        cv2.putText(img_resultado, texto, (cX - 25, cY + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4) 
        cv2.putText(img_resultado, texto, (cX - 25, cY + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

mostrar("Resultado Clasificado Hibrido", img_resultado)

# --- REPORTE ---
print("\n" + "=" * 30)
print("       REPORTE FINAL")
print("=" * 30)

print(f"MONEDAS DETECTADAS: {len(monedas_cnt)}")
print(f" -> 10c: {monedas_cnt.count('10c')}")
print(f" -> 1p:  {monedas_cnt.count('1p')}")
print(f" -> 50c: {monedas_cnt.count('50c')}")
print("-" * 30)
print(f"DADOS DETECTADOS: {len(dados_cnt)}")
for val in sorted(set(dados_cnt)):
    print(f" -> Valor {val}: {dados_cnt.count(val)}")
print("=" * 30)