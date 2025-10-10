from ultralytics import YOLO
from PIL import Image

# 1. Carregue o seu modelo campeão
caminho_do_modelo = 'runs/detect/kfold_kfold_dataset_stratified_10_10folds_fold_2/weights/best.pt'
modelo_campeao = YOLO(caminho_do_modelo)

# 2. Escolha uma imagem para testar
#    Pode ser qualquer imagem, até uma que não está no seu dataset.
caminho_da_imagem = 'caminho/para/uma/imagem_de_teste.jpg'

# 3. Faça a predição
results = modelo_campeao.predict(caminho_da_imagem)

# 4. Visualize o resultado
#    O YOLO vai desenhar as bounding boxes na imagem e mostrar para você.
for r in results:
    im_array = r.plot()  # plota as caixas na imagem
    im = Image.fromarray(im_array[..., ::-1])  # Converte para formato de imagem
    im.show() # Mostra a imagem
    im.save('resultado_predicao.jpg') # Salva a imagem com a predição