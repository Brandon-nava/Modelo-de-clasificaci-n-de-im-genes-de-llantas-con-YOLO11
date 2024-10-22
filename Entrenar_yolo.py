from ultralytics import YOLO

model = YOLO("yolo11n.pt")

if __name__ == '__main__':
    model.train(data="C:/Users/Brandon/Documents/MCID/Inteligencia artificial/Practicas/Detector_yolo_autos_4/data.yaml", epochs= 100, imgsz=640, workers= 4)