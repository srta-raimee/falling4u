import os
import shutil
from sklearn.model_selection import train_test_split

def split(input_dir, output_dir, validation_size):
    INPUT_BASE_DIR = input_dir
    OUTPUT_BASE_DIR = output_dir
    
    SOURCE_IMAGES_DIR = os.path.join(INPUT_BASE_DIR, 'all_images')
    SOURCE_LABELS_DIR = os.path.join(INPUT_BASE_DIR, 'all_labels')

    DEST_IMG_TRAIN_DIR = os.path.join(OUTPUT_BASE_DIR, 'images', 'train')
    DEST_IMG_VAL_DIR = os.path.join(OUTPUT_BASE_DIR, 'images', 'val')
    DEST_LABEL_TRAIN_DIR = os.path.join(OUTPUT_BASE_DIR, 'labels', 'train')
    DEST_LABEL_VAL_DIR = os.path.join(OUTPUT_BASE_DIR, 'labels', 'val')

    VALIDATION_SIZE = validation_size
    RANDOM_STATE = 42
    
    print(f"Iniciando a separação do dataset de '{INPUT_BASE_DIR}' para '{OUTPUT_BASE_DIR}'...")

    for path in [DEST_IMG_TRAIN_DIR, DEST_IMG_VAL_DIR, DEST_LABEL_TRAIN_DIR, DEST_LABEL_VAL_DIR]:
        os.makedirs(path, exist_ok=True)

    try:
        image_files = [f for f in os.listdir(SOURCE_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    except FileNotFoundError:
        print(f"ERRO: A pasta de origem '{SOURCE_IMAGES_DIR}' não foi encontrada!")
        return

    if not image_files:
        print(f"ERRO: Nenhuma imagem encontrada em '{SOURCE_IMAGES_DIR}'.")
        return

    train_files, val_files = train_test_split(
        image_files,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print("\n--- Resumo da Divisão ---")
    print(f"Total de imagens encontradas: {len(image_files)}")
    print(f"Imagens de treino: {len(train_files)}")
    print(f"Imagens de validação: {len(val_files)}")
    print("--------------------------\n")

    def copy_files(file_list, dest_img_dir, dest_label_dir, set_name):
        print(f"Copiando arquivos para o conjunto de {set_name}...")
        copied_count = 0
        for filename in file_list:
            src_img_path = os.path.join(SOURCE_IMAGES_DIR, filename)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            src_label_path = os.path.join(SOURCE_LABELS_DIR, label_filename)

            dest_img_path = os.path.join(dest_img_dir, filename)
            dest_label_path = os.path.join(dest_label_dir, label_filename)

            shutil.copy(src_img_path, dest_img_path)
            
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)
            else:
                print(f"    AVISO: Label '{label_filename}' não encontrado para a imagem '{filename}'. Apenas a imagem será copiada.")
            
            copied_count += 1
        print(f"{copied_count} pares de arquivos copiados para {set_name}.")

    copy_files(train_files, DEST_IMG_TRAIN_DIR, DEST_LABEL_TRAIN_DIR, "TREINO")
    copy_files(val_files, DEST_IMG_VAL_DIR, DEST_LABEL_VAL_DIR, "VALIDAÇÃO")
    
    print(f"\nSeparação concluída com sucesso! Novo dataset criado em '{OUTPUT_BASE_DIR}'.")


if __name__ == '__main__':
    # holdout 70/30
    split(input_dir='fall_dataset', output_dir='fall_dataset_7030', validation_size=0.3)

    # holdout 60/40
    split(input_dir='fall_dataset', output_dir='fall_dataset_6040', validation_size=0.4)

    # holdout 80/20
    split(input_dir='fall_dataset', output_dir='fall_dataset_8020', validation_size=0.2)
