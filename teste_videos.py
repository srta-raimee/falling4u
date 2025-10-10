from ultralytics import YOLO

caminho_do_modelo = 'runs/detect/kfold_kfold_dataset_stratified_10_10folds_fold_2/weights/best.pt'
MODEL = YOLO(caminho_do_modelo)

# Defina o caminho do vídeo que você quer processar
caminho_do_video = 'arquivos_teste/IMG_6015.MOV'

# Faça a predição com save=True
# O YOLO vai cuidar de tudo e salvar o resultado
results = MODEL.predict(caminho_do_video, save=True)

print("\nProcesso concluído! O vídeo com as detecções foi salvo.")