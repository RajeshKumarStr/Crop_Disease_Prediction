from PIL import Image
import os

def check_corrupt_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            img_path = os.path.join(root, file)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Check if the image can be opened
                    img.close()
                
                # Try reopening and converting to ensure readability
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # Ensure it is in a readable format
                    img.save(img_path)  # Overwrite with a valid format
            except (IOError, SyntaxError, OSError) as e:
                print(f"Corrupt image found and removed: {img_path}")
                os.remove(img_path)  # Remove the corrupt image

# Run validation on train and test datasets
check_corrupt_images("dataset/train")
check_corrupt_images("dataset/test")

# Corrupt image found and removed: dataset/train\Maize leaf beetle\leaf beetle90_.jpg
# Corrupt image found and removed: dataset/test\Tomato leaf curl\leaf curl43_.jpg
# Corrupt image found and removed: dataset/train\maize healthy\healthy189_.jpg
# Corrupt image found and removed: dataset/train\maize healthy\healthy87_.jpg
# Corrupt image found and removed: dataset/train\maize leaf blight\leaf blight379_.jpg
# Corrupt image found and removed: dataset/train\maize streak virus\streak virus118_.jpg
# Corrupt image found and removed: dataset/train\maize streak virus\streak virus421_.jpg
# Corrupt image found and removed: dataset/train\tomato leaf curl\leaf curl439_.jpg
# Corrupt image found and removed: dataset/test\maize healthy\healthy189_.jpg
# Corrupt image found and removed: dataset/test\maize streak virus\streak virus554_.jpg