# Asegúrate de instalar la librería ultralytics si aún no la tienes
# Puedes instalarla con: pip install ultralytics

from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO("C:/Users/Brandon/Documents/MCID/Inteligencia artificial/Practicas/Detector_yolo_autos_4/runs/detect/train/weights/best.pt")

# Leer el video
video_path = "C:/Users/Brandon/Documents/MCID/Inteligencia artificial/Practicas/Detector_yolo_autos_4/pruebas/Video_3.mp4"
cap = cv2.VideoCapture(video_path)

# Obtener las propiedades del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Crear el objeto para guardar el video con las detecciones
output_path = "C:/Users/Brandon/Documents/MCID/Inteligencia artificial/Practicas/Detector_yolo_autos_4/pruebas/video_con_detecciones_3.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la detección con YOLO
    results = model.predict(frame, conf=0.6)
    
    # Dibujar las cajas de detección en el frame
    annotated_frame = results[0].plot()

    # Escribir el frame anotado en el video de salida
    out.write(annotated_frame)

    # Mostrar el frame con las detecciones (opcional)
    cv2.imshow('YOLO Detections', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los objetos de video
cap.release()
out.release()
cv2.destroyAllWindows()
