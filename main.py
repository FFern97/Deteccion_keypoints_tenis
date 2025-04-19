import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter
from tracknet import BallTrackerNet
import argparse
import cv2
import numpy as np

class CourtDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        # Rutas absolutas para evitar problemas
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(base_dir, 'data', mode)
        self.image_dir = os.path.join(self.data_path, 'images')
        self.label_dir = os.path.join(self.data_path, 'labels')
        
        # Verificar existencia de directorios
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Directorio de imágenes no encontrado: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Directorio de etiquetas no encontrado: {self.label_dir}")
        
        # Obtener lista de imágenes PNG
        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith('.png')]
        
        if not self.image_files:
            raise FileNotFoundError(f"No se encontraron imágenes PNG en {self.image_dir}")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Cargar imagen PNG y redimensionar
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 360))  # Ajustar al tamaño esperado por el modelo
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # Convertir a CxHxW
        
        # Cargar etiquetas y crear mapa de calor
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            points = [list(map(float, line.strip().split())) for line in f.readlines()]
        
        # Crear mapa de calor (15 canales para 15 puntos clave)
        heatmap = torch.zeros((15, 360, 640))  # (canales, altura, ancho)
        for i, (x, y) in enumerate(points):
            x_px = int(x * 640)
            y_px = int(y * 360)
            if 0 <= x_px < 640 and 0 <= y_px < 360:
                heatmap[i, y_px, x_px] = 1.0
                # Aplicar pequeño blur gaussiano (CORRECCIÓN APLICADA AQUÍ)
                heatmap[i] = F.conv2d(heatmap[i].unsqueeze(0).unsqueeze(0), 
                                    torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=torch.float32)/16,
                                    padding=1).squeeze()
        
        return image, heatmap

def train(model, loader, optimizer, criterion, device, epoch, steps_per_epoch):
    model.train()
    total_loss = 0
    
    for i, (images, heatmaps) in enumerate(loader):
        if i >= steps_per_epoch:
            break
            
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Asegurar que las dimensiones coincidan
        if outputs.shape[2:] != heatmaps.shape[2:]:
            outputs = F.interpolate(outputs, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)
        
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {i}, Loss: {loss.item():.4f}')
        
    return total_loss / steps_per_epoch

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, heatmaps in loader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            outputs = model(images)
            
            # Ajuste dimensional si es necesario
            if outputs.shape[2:] != heatmaps.shape[2:]:
                outputs = F.interpolate(outputs, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, heatmaps)
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
    return total_loss / total_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--exp_id', type=str, default='default')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--val_intervals', type=int, default=5)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    args = parser.parse_args()
    
    # Configuración inicial
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    try:
        # Datasets
        print("Cargando datasets...")
        train_dataset = CourtDataset('train')
        val_dataset = CourtDataset('val')
        
        # Verificación de dimensiones
        sample_img, sample_heatmap = train_dataset[0]
        print(f"\nVerificación de dimensiones:")
        print(f"Imagen de ejemplo: {sample_img.shape} (debe ser [3, 360, 640])")
        print(f"Heatmap de ejemplo: {sample_heatmap.shape} (debe ser [15, 360, 640])")
        
        print(f"\nCantidad de datos:")
        print(f"Entrenamiento: {len(train_dataset)} imágenes")
        print(f"Validación: {len(val_dataset)} imágenes")
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Modelo
        model = BallTrackerNet(out_channels=15).to(device)  # 15 canales para 15 puntos clave
        
        # Verificación de salida del modelo
        test_output = model(sample_img.unsqueeze(0).to(device))
        print(f"Salida del modelo: {test_output.shape} (debe ser [1, 15, H, W])\n")
        
        # Optimizador y pérdida
        optimizer = Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        
        # Configuración de logs
        exps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exps', args.exp_id)
        os.makedirs(exps_path, exist_ok=True)
        
        log_writer = SummaryWriter(os.path.join(exps_path, 'logs'))
        model_last_path = os.path.join(exps_path, 'model_last.pth')
        model_best_path = os.path.join(exps_path, 'model_best.pth')
        
        # Entrenamiento
        best_val_loss = float('inf')
        print("Comenzando entrenamiento...\n")
        
        for epoch in range(args.num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.steps_per_epoch)
            log_writer.add_scalar('Loss/train', train_loss, epoch)
            
            print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}')
            
            # Validación periódica
            if (epoch + 1) % args.val_intervals == 0:
                val_loss = validate(model, val_loader, criterion, device)
                log_writer.add_scalar('Loss/val', val_loss, epoch)
                
                print(f'Validation Loss: {val_loss:.4f}')
                
                # Guardar mejor modelo
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_best_path)
                    print("¡Nuevo mejor modelo guardado!")
                
                # Guardar último modelo
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, model_last_path)
        
        print("\n¡Entrenamiento completado con éxito!")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        raise