import json
import shutil
from pathlib import Path

# Rutas (ajusta según tu sistema)
dataset_path = Path(r"C:\Users\Ffern\Downloads\tennis_court_det_dataset\data")
dest_path = Path(r"C:\Users\Ffern\OneDrive\Desktop\CourtKP\TennisCourtDetector\data")

# Crear carpetas de destino
(dest_path / "train" / "images").mkdir(parents=True, exist_ok=True)
(dest_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
(dest_path / "val" / "images").mkdir(parents=True, exist_ok=True)
(dest_path / "val" / "labels").mkdir(parents=True, exist_ok=True)

def process_json(json_file, output_img_dir, output_label_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for item in data:
        # Usar extensión .png (confirmado en tus archivos)
        img_name = f"{item['id']}.png"  # Cambiado a .png
        img_src = dataset_path / "images" / img_name
        img_dest = output_img_dir / img_name
        
        # Verificar si la imagen existe antes de copiar
        if not img_src.exists():
            print(f"⚠️ Advertencia: {img_name} no existe en la carpeta images/")
            continue
            
        shutil.copy(img_src, img_dest)
        
        # Guardar anotaciones (14 puntos)
        txt_path = output_label_dir / f"{item['id']}.txt"
        with open(txt_path, 'w') as f_txt:
            for x, y in item["kps"]:
                f_txt.write(f"{x} {y}\n")

# Procesar archivos
print("Procesando data_train.json...")
process_json(dataset_path / "data_train.json", dest_path / "train" / "images", dest_path / "train" / "labels")

print("\nProcesando data_val.json...")
process_json(dataset_path / "data_val.json", dest_path / "val" / "images", dest_path / "val" / "labels")

print("\n✅ Dataset organizado correctamente. Archivos faltantes:")
print("(Si hay advertencias arriba, esas imágenes no se copiaron)")