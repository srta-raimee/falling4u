import os
import shutil
import numpy as np
from sklearn.model_selection import KFold

def split_kfold(input_dir, output_dir, k_folds):
    print(f"Iniciando a preparação da estrutura K-Fold em '{output_dir}' com K={k_folds}...")
    
    SOURCE_IMG_DIR = os.path.join(input_dir, 'all_images')
    SOURCE_LABEL_DIR = os.path.join(input_dir, 'all_labels')
    
    if os.path.exists(output_dir):
        print(f"A pasta de destino '{output_dir}' já existe. Removendo para começar do zero.")
        shutil.rmtree(output_dir)

    all_image_files = sorted([f for f in os.listdir(SOURCE_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_image_files)):
        fold_num = fold_idx + 1
        print(f"\n--- Preparando Fold {fold_num}/{k_folds} ---")

        fold_base_dir = os.path.join(output_dir, f"fold_{fold_num}")
        train_img_dir = os.path.join(fold_base_dir, 'images', 'train')
        val_img_dir = os.path.join(fold_base_dir, 'images', 'val')
        train_label_dir = os.path.join(fold_base_dir, 'labels', 'train')
        val_label_dir = os.path.join(fold_base_dir, 'labels', 'val')

        for path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
            os.makedirs(path, exist_ok=True)
            
        print(f"Copiando {len(train_indices)} arquivos de treino para o Fold {fold_num}...")
        for idx in train_indices:
            filename = all_image_files[idx]
            shutil.copy(os.path.join(SOURCE_IMG_DIR, filename), train_img_dir)
            label_name = os.path.splitext(filename)[0] + '.txt'
            if os.path.exists(os.path.join(SOURCE_LABEL_DIR, label_name)):
                shutil.copy(os.path.join(SOURCE_LABEL_DIR, label_name), train_label_dir)

        print(f"Copiando {len(val_indices)} arquivos de validação para o Fold {fold_num}...")
        for idx in val_indices:
            filename = all_image_files[idx]
            shutil.copy(os.path.join(SOURCE_IMG_DIR, filename), val_img_dir)
            label_name = os.path.splitext(filename)[0] + '.txt'
            if os.path.exists(os.path.join(SOURCE_LABEL_DIR, label_name)):
                shutil.copy(os.path.join(SOURCE_LABEL_DIR, label_name), val_label_dir)
    
    print("\nEstrutura K-Fold criada com sucesso!")

if __name__ == '__main__':
    INPUT_DATA_DIR = 'fall_dataset'
    NUM_FOLDS = [5, 10, 15, 20]
    
    for fold in NUM_FOLDS:
        OUTPUT_KFOLD_DIR = f'kfold_dataset_{fold}'
        print(f"\nIniciando a divisão K-Fold com K={fold}...")
        split_kfold(
            input_dir=INPUT_DATA_DIR,
            output_dir=OUTPUT_KFOLD_DIR,
            k_folds=fold
        )