# Asegúrate de instalar la librería ultralytics si aún no la tienes
# Puedes instalarla con: pip install ultralytics

from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO("C:/Users/Brandon/Documents/MCID/Inteligencia artificial/Practicas/Detector_yolo_autos_4/runs/detect/train/weights/best.pt")

# Abrir un video o utilizar la webcam (cambiar 'ruta_al_video.mp4' si deseas usar un video)
cap = cv2.VideoCapture(0)  # Reemplaza '0' por la ruta al video si es un archivo de video
#cap = cv2.VideoCapture("C:/Users/Axel/Desktop/Tareas Inteligencia Artificial/Detector_yolo_autos/pruebas/prueba.jpeg")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la detección con YOLO
    #results = model(frame)
    results = model.predict(frame, conf = 0.60)

    # Dibujar las cajas de detección en la imagen
    annotated_frame = results[0].plot()

    # Mostrar el video con las detecciones
    cv2.imshow('YOLO Detections', annotated_frame)

    # Salir del bucle si presionas la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
