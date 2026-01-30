import cv2              #size change krte aani prt save krte
import os

input_folder = "dataset"
output_size = (224, 224)

for folder in ["cow", "buffalo"]:
    path = os.path.join(input_folder, folder)        #os.path.join is a function   
    for img_name in os.listdir(path):                
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)                    #img stored

        if img is not None:
            img = cv2.resize(img, output_size)
            cv2.imwrite(img_path, img)

print("✅ All images resized to 224x224")
