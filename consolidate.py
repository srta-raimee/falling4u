import os
import shutil

# mandar os dados todos juntos para uma so pasta de imagens e uma so de labels para reorganizar posteriormente com metodos de validacao
def consolidate_and_rename():
    SOURCE_IMAGE_FOLDERS = [
        'fall_dataset/images/train',
        'fall_dataset/images/val'
    ]

    DEST_IMG_DIR = 'fall_dataset/all_images'
    DEST_LABEL_DIR = 'fall_dataset/all_labels'
    FILENAME_PREFIX = 'fall'

    print("Iniciando a consolidação e renomeação dos arquivos...")

    os.makedirs(DEST_IMG_DIR, exist_ok=True)
    os.makedirs(DEST_LABEL_DIR, exist_ok=True)
    
    file_counter = 1

    for folder in SOURCE_IMAGE_FOLDERS:
        print(f"\nProcessando pasta: {folder}")
        if not os.path.isdir(folder):
            print(f"  AVISO: Pasta não encontrada. Pulando: {folder}")
            continue

        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("  Nenhuma imagem encontrada nesta pasta.")
            continue
            
        for old_filename in image_files:
            old_base_name, extension = os.path.splitext(old_filename)
            new_base_name = f"{FILENAME_PREFIX}_{file_counter:05d}"
            
            new_image_filename = new_base_name + extension
            new_label_filename = new_base_name + '.txt'
            
            old_image_path = os.path.join(folder, old_filename)
            
            label_folder = folder.replace('images', 'labels', 1)
            old_label_path = os.path.join(label_folder, old_base_name + '.txt')

            dest_image_path = os.path.join(DEST_IMG_DIR, new_image_filename)
            dest_label_path = os.path.join(DEST_LABEL_DIR, new_label_filename)

            shutil.copy(old_image_path, dest_image_path)
            
            if os.path.exists(old_label_path):
                shutil.copy(old_label_path, dest_label_path)
            else:
                print(f"  AVISO: Label para '{old_filename}' não encontrado. Apenas a imagem foi copiada.")

            file_counter += 1

    print(f"\nConsolidação concluída! {file_counter - 1} arquivos foram processados.")
    print(f"Seus dados consolidados estão em '{DEST_IMG_DIR}' e '{DEST_LABEL_DIR}'.")

if __name__ == '__main__':
    consolidate_and_rename()