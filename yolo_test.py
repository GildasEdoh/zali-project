import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Télécharger le modèle une seule fois
model_path = hf_hub_download(
    repo_id="foduucom/plant-leaf-detection-and-classification",
    filename="best.pt"
)

model = YOLO(model_path)

def detecter_feuilles(image_path):
    
    # Charger image
    img = cv2.imread(image_path)
    
    # Détection
    results = model.predict(source=image_path, conf=0.35)

    for r in results:
        for box in r.boxes:
            
            # Coordonnées
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            classe_id = int(box.cls[0])
            nom_classe = r.names[classe_id]
            score = float(box.conf[0])
            print("plant name = ", nom_classe, "confidence= ", score)

            # 🔵 Dessiner rectangle
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),   # couleur verte
                2              # épaisseur
            )

            # 🟢 Ajouter texte
            label = f"{nom_classe} ({score:.2f})"
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Affichage final
    cv2.imshow("Feuilles detectees", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test
detecter_feuilles('/Users/v9/Documents/Documents personnels EDOH Yao Gildas/agriVision/image_test.jpg')