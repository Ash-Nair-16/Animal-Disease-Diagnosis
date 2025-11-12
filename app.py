from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# === Initialize Flask ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Load Models ===
cow_model = load_model('cow_disease_model.h5')
poultry_model = load_model('poultry_disease_model.keras')

# === Class Labels ===
cow_class_names = ['healthycows', 'lumpycows']
poultry_class_labels = ['Healthy', 'coryza', 'crd', 'Fowlpox', 'Bumblefoot']

# === Disease Info Dictionaries ===
cow_disease_info = {
    "healthycows": {
        "description": "The cow appears healthy with no visible disease symptoms.",
        "treatment": "Continue proper nutrition, hygiene, and vaccination schedule."
    },
    "lumpycows": {
        "description": "Lumpy Skin Disease — a viral infection spread by insects.",
        "treatment": "Isolate infected animals, provide antiviral medication, and maintain fly control. Consult a veterinarian immediately."
    }
}

poultry_disease_info = {
    "Healthy": {
        "description": "The bird appears healthy and shows no disease symptoms.",
        "treatment": "Continue providing a balanced diet, clean water, and proper coop hygiene."
    },
    "coryza": {
        "description": "Infectious Coryza — a bacterial disease causing swelling of the face and nasal discharge.",
        "treatment": "Treat with broad-spectrum antibiotics and maintain proper ventilation."
    },
    "crd": {
        "description": "Chronic Respiratory Disease — caused by Mycoplasma gallisepticum infection.",
        "treatment": "Administer Tylosin or Tiamulin as prescribed and disinfect housing areas."
    },
    "Fowlpox": {
        "description": "Fowlpox — a viral disease that causes lesions on the comb, wattles, and eyelids.",
        "treatment": "Vaccinate healthy birds and isolate infected ones. Keep coop clean and dry."
    },
    "Bumblefoot": {
        "description": "A bacterial infection on the footpad due to injury or poor sanitation.",
        "treatment": "Clean affected area, apply antiseptic, and consult a vet for antibiotics."
    }
}

# === Image Preprocessing ===
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# === Flask Routes ===
@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    confidence = None
    image_url = None
    description = None
    treatment = None
    animal_type = None

    if request.method == "POST":
        animal_type = request.form.get("animal_type")
        file = request.files["image"]

        if file and animal_type:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            image_url = filepath

            img_array = preprocess_image(filepath)

            if animal_type == "cow":
                preds = cow_model.predict(img_array)
                prediction = preds[0][0]
                class_index = int(prediction > 0.5)
                confidence = prediction if class_index == 1 else 1 - prediction
                predicted_label = cow_class_names[class_index]
                label = f"{predicted_label} ({confidence*100:.2f}%)"
                info = cow_disease_info.get(predicted_label, {})
                description = info.get("description", "No info available.")
                treatment = info.get("treatment", "No treatment information available.")

            elif animal_type == "poultry":
                preds = poultry_model.predict(img_array)
                pred_idx = np.argmax(preds[0])
                confidence = np.max(preds[0])
                predicted_label = poultry_class_labels[pred_idx]
                label = f"{predicted_label} ({confidence*100:.2f}%)"
                info = poultry_disease_info.get(predicted_label, {})
                description = info.get("description", "No info available.")
                treatment = info.get("treatment", "No treatment information available.")

    return render_template("index3.html",
                           label=label,
                           confidence=confidence,
                           image_url=image_url,
                           animal_type=animal_type,
                           description=description,
                           treatment=treatment)

if __name__ == "__main__":
      app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
