import cv2
import os
import shutil


image_dir = "data/DogFaceNet/after_4_bis"

output_dir = "data/DogFaceNet/sorted"


key_to_folder = {
    "h": "happy",
    "u": "unhappy",
    "a": "angry",
    "r": "relaxed",
    "s": "surprised",
    "f": "fear",
    "c": "curious",
    "n": "neutral",
    "d": "rejected"
}

for folder in key_to_folder.values():
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

for root, dirs, files in os.walk(image_dir):
    for filename in files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(root, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read {filename}")
            continue
        
        cv2.imshow("Image Sorter", img)
        print(f"Displaying: {filename}")
        print("Press h = happy, u = unhappy, a = angry, r = relaxed, s = surprised, f = fear, c = curious, n = neutral, d = reject, q = quit")
    
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

        key_char = chr(key)
        if key_char in key_to_folder:
            dest_folder = os.path.join(output_dir, key_to_folder[key_char])
            shutil.move(image_path, dest_folder)
            print(f"Moved {filename} to {dest_folder}")
        else:
            print("Key not recognized, skipping image.")

cv2.destroyAllWindows()
