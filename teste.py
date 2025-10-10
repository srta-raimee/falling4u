import os
from ultralytics import YOLO
from PIL import Image

# kfold stratified 10 folds - fold 2 model
caminho_do_modelo = 'runs/detect/kfold_kfold_dataset_stratified_10_10folds_fold_2/weights/best.pt'
MODELO = YOLO(caminho_do_modelo)

pasta_de_entrada = 'imgs_teste'
pasta_de_saida = 'resultados_predicao'

os.makedirs(pasta_de_saida, exist_ok=True)

for nome_arquivo in os.listdir(pasta_de_entrada):
    if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        
        caminho_da_imagem = os.path.join(pasta_de_entrada, nome_arquivo)
        print(f"Processando a imagem: {caminho_da_imagem}...")

        results = MODELO.predict(caminho_da_imagem)

        for r in results:
            im_array = r.plot()  # plota as caixas na imagem
            im = Image.fromarray(im_array[..., ::-1])  # Converte para formato de imagem
            
            base, ext = os.path.splitext(nome_arquivo)
            caminho_salvo = os.path.join(pasta_de_saida, f"{base}_predito{ext}")
            
            #im.show() 
            im.save(caminho_salvo) # Salva a imagem com a predição
            print(f"--> Resultado salvo em: {caminho_salvo}")

print("\nProcesso concluído!")