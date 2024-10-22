# Asegúrate de instalar la librería ultralytics si aún no la tienes
# Puedes instalarla con: pip install ultralytics

from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO("C:/Users/Brandon/Documents/MCID/Inteligencia artificial/Practicas/Detector_yolo_autos_4/runs/detect/train/weights/best.pt")

# Leer la imagen
image_path = "C:/Users/Brandon/Documents/MCID/Inteligencia artificial/Practicas/Detector_yolo_autos_4/pruebas/prueba_20.jpg"
image = cv2.imread(image_path)

# Realizar la detección con YOLO
#results = model(image)
results = model.predict(image, conf = 0.6)

# Dibujar las cajas de detección en la imagen
annotated_image = results[0].plot()

# Mostrar la imagen con las detecciones
cv2.imshow('YOLO Detections', annotated_image)
cv2.waitKey(0)  # Espera a que presiones cualquier tecla para cerrar la ventana
cv2.destroyAllWindows()

# Si deseas guardar la imagen anotada, descomenta la siguiente línea:
cv2.imwrite('ruta_de_salida/imagen_con_detecciones.jpg', annotated_image)
