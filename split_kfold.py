
import os
import shutil
import csv
import time
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import KFold

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s"

def run_kfold_experiment(num_folds, nome_experimento):
    SOURCE_IMG_DIR = 'fall_dataset/all_images'
    SOURCE_LABEL_DIR = 'fall_dataset/all_labels'
    TEMP_DATASET_DIR = 'kfold_temp_dataset'
    CSV_RESULTS_FILE = 'resultados_kfold_tcc.csv'
    
    K_FOLDS = 5
    EPOCHS = 100
    BATCH_SIZE = 8
    NOME_MODELO_INICIAL = "yolo11n.pt"
    EXPERIMENT_PREFIX = 'kfold_fall_detection'

    os.makedirs(TEMP_DATASET_DIR, exist_ok=True)
    
    all_image_files = sorted([f for f in os.listdir(SOURCE_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    all_fold_metrics = []
    total_start_time = time.perf_counter()

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_image_files)):
        fold_num = fold_idx + 1
        print("\n" + "="*60)
        print(f"INICIANDO FOLD {fold_num}/{K_FOLDS}")
        print("="*60)

        temp_train_img_dir = os.path.join(TEMP_DATASET_DIR, 'images', 'train')
        temp_val_img_dir = os.path.join(TEMP_DATASET_DIR, 'images', 'val')
        temp_train_label_dir = os.path.join(TEMP_DATASET_DIR, 'labels', 'train')
        temp_val_label_dir = os.path.join(TEMP_DATASET_DIR, 'labels', 'val')

        if os.path.exists(TEMP_DATASET_DIR):
            shutil.rmtree(TEMP_DATASET_DIR)

        for path in [temp_train_img_dir, temp_val_img_dir, temp_train_label_dir, temp_val_label_dir]:
            os.makedirs(path, exist_ok=True)

        for idx in train_indices:
            filename = all_image_files[idx]
            shutil.copy(os.path.join(SOURCE_IMG_DIR, filename), temp_train_img_dir)
            label_name = os.path.splitext(filename)[0] + '.txt'
            if os.path.exists(os.path.join(SOURCE_LABEL_DIR, label_name)):
                shutil.copy(os.path.join(SOURCE_LABEL_DIR, label_name), temp_train_label_dir)

        for idx in val_indices:
            filename = all_image_files[idx]
            shutil.copy(os.path.join(SOURCE_IMG_DIR, filename), temp_val_img_dir)
            label_name = os.path.splitext(filename)[0] + '.txt'
            if os.path.exists(os.path.join(SOURCE_LABEL_DIR, label_name)):
                shutil.copy(os.path.join(SOURCE_LABEL_DIR, label_name), temp_val_label_dir)

        yaml_path = os.path.join(TEMP_DATASET_DIR, 'kfold_data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"path: {os.path.abspath(TEMP_DATASET_DIR)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("nc: 2\n")
            f.write("names: ['fall', 'not fall']\n")
        
        model = YOLO(NOME_MODELO_INICIAL)
        
        fold_experiment_name = f"{EXPERIMENT_PREFIX}_fold_{fold_num}"
        model.train(data=yaml_path, epochs=EPOCHS, batch=BATCH_SIZE, device=0, name=fold_experiment_name)
        
        best_model = YOLO(f'runs/detect/{fold_experiment_name}/weights/best.pt')
        results = best_model.val()

        fold_metrics = {
            'mAP50_95': results.box.map, 'mAP50': results.box.map50,
            'precision': results.box.mp, 'recall': results.box.mr
        }
        all_fold_metrics.append(fold_metrics)
        print(f"Métricas do Fold {fold_num}: {fold_metrics}")
    
    total_end_time = time.perf_counter()
    total_duration_str = format_time(total_end_time - total_start_time)

    avg_metrics = {
        'avg_mAP50_95': np.mean([m['mAP50_95'] for m in all_fold_metrics]),
        'std_mAP50_95': np.std([m['mAP50_95'] for m in all_fold_metrics]),
        'avg_mAP50': np.mean([m['mAP50'] for m in all_fold_metrics]),
        'std_mAP50': np.std([m['mAP50'] for m in all_fold_metrics]),
        'avg_precision': np.mean([m['precision'] for m in all_fold_metrics]),
        'std_precision': np.std([m['precision'] for m in all_fold_metrics]),
        'avg_recall': np.mean([m['recall'] for m in all_fold_metrics]),
        'std_recall': np.std([m['recall'] for m in all_fold_metrics]),
    }

    print("\n" + "="*60)
    print("RESULTADO FINAL DO K-FOLD CROSS-VALIDATION")
    print("="*60)
    print(f"Tempo Total de Execução: {total_duration_str}")
    print(f"Métricas (Média ± Desvio Padrão) calculadas com {K_FOLDS} folds:")
    print(f"mAP50-95: {avg_metrics['avg_mAP50_95']:.4f} ± {avg_metrics['std_mAP50_95']:.4f}")
    print(f"mAP50:    {avg_metrics['avg_mAP50']:.4f} ± {avg_metrics['std_mAP50']:.4f}")
    print(f"Precisão: {avg_metrics['avg_precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
    print(f"Recall:   {avg_metrics['avg_recall']:.4f} ± {avg_metrics['std_recall']:.4f}")

    header = ['experimento_base', 'k_folds', 'tempo_total', 'avg_mAP50_95', 'std_mAP50_95', 'avg_mAP50', 'std_mAP50', 'avg_precision', 'std_precision', 'avg_recall', 'std_recall']
    data_row = [
        EXPERIMENT_PREFIX, K_FOLDS, total_duration_str,
        f"{avg_metrics['avg_mAP50_95']:.4f}", f"{avg_metrics['std_mAP50_95']:.4f}",
        f"{avg_metrics['avg_mAP50']:.4f}", f"{avg_metrics['std_mAP50']:.4f}",
        f"{avg_metrics['avg_precision']:.4f}", f"{avg_metrics['std_precision']:.4f}",
        f"{avg_metrics['avg_recall']:.4f}", f"{avg_metrics['std_recall']:.4f}"
    ]

    file_exists = os.path.isfile(CSV_RESULTS_FILE)
    with open(CSV_RESULTS_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)
    
    print(f"\nResultados médios salvos em '{CSV_RESULTS_FILE}'")
    shutil.rmtree(TEMP_DATASET_DIR)
    print(f"Pasta temporária '{TEMP_DATASET_DIR}' removida.")

if __name__ == '__main__':
    run_kfold_experiment()