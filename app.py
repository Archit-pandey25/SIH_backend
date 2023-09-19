from flask import Flask, render_template, request, jsonify
from ml_model import predict_image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # return jsonify({'error': 'No selected file'})
        # Save the uploaded image to a temporary folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform image classification using your pre-trained model
        result = predict_image(file_path)  # Implement this function in ml_model.py

        # You can also remove the uploaded file at this point
        os.remove(file_path)

        class_labels_dict = {
            'Aloevera': "Known for its soothing properties, found in arid regions, and used for skin ailments and digestive health.",
            'Amla': "Rich in vitamin C, found in India, and used in Ayurvedic medicine for its antioxidant and immune-boosting properties.",
            'Amruthballi': "An Ayurvedic herb found in India, used for potential health benefits and immunity enhancement.",
            'Arali': "A tree native to Asia with medicinal bark, used in traditional medicine for various ailments.",
            'Ashoka': "A tree with medicinal bark found in India, used for women's health and natural remedies.",
            'Astma_weed': "Known for its potential benefits in managing asthma symptoms and found in various regions.",
            'Badipala': "Found in tropical regions, used in traditional medicine for its potential healing properties.",
            'Balloon_Vine': "A vine with balloon-like seed pods, found in tropical regions and used for potential medicinal properties.",
            'Bamboo': "A versatile plant with various medicinal uses, found in many parts of the world.",
            'Beans': "Edible legumes found globally, used in traditional medicine for potential health benefits.",
            'Betal': "Widely used in Asia for making betel quid, with cultural significance and potential medicinal properties.",
            'Bhrami': "An Ayurvedic herb known for cognitive-enhancing properties, found in India and other parts of Asia.",
            'Bringaraja': "Used in hair care products and Ayurveda, found in India, and known for hair health benefits.",
            'camphor': "A crystalline substance with aromatic properties, used in traditional medicine and various products.",
            'Caricature': "A plant with vibrant flowers, often used for ornamental purposes and found in gardens.",
            'Castor': "Known for its seeds, castor oil is derived from this plant and has various medicinal and industrial uses.",
            'Catharanthus': "Used in traditional medicine, this plant is known for its potential medicinal properties.",
            'Chakte': "A type of wood used in woodworking and crafting, commonly found in regions where it grows.",
            'Chilly': "Spicy pepper varieties used in culinary dishes worldwide, known for potential health benefits.",
            'Citron lime (herelikai)': "A citrus fruit used in cooking and beverages, found in regions with a suitable climate.",
            'Coffee': "A popular beverage made from coffee beans, known for its stimulating effect, and grown in coffee-producing regions worldwide.",
            'Common rue(nagadalli)': "An herb with various medicinal uses in traditional medicine, commonly found in Mediterranean regions and parts of Asia.",
            'Coriander': "An herb used as a spice and garnish in cooking, cultivated and found in many culinary cultures around the globe.",
            'Curry': "A blend of spices used to flavor various dishes, often used in Indian cuisine, and found in Indian kitchens and worldwide markets.",
            'Doddpathre': "A leafy vegetable commonly used in South Indian cuisine, found in the Indian subcontinent.",
            'Drumstick': "A tree known for its long, slender pods, used in cooking and often found in tropical and subtropical regions.",
            'Ekka': "A tree with various medicinal uses, especially in Ayurveda, found in India and Southeast Asia.",
            'Eucalyptus': "A tree known for its aromatic leaves and used in essential oils, commonly found in Australia and other regions with suitable climates.",
            'Ganigale': "A plant often used for its medicinal properties in traditional medicine, found in parts of India and other tropical regions.",
            'Ganike': "A type of wild cucumber found in tropical regions and used in traditional dishes and medicines.",
            'Gasagase': "Poppy seeds used as a spice and for their nutritional value, commonly found in various culinary traditions.",
            'Ginger': "A rhizome known for its strong flavor and medicinal properties, cultivated in tropical and subtropical regions worldwide.",
            'Globe Amarnath': "A type of amaranth with globe-shaped flower heads, often grown as an ornamental plant in gardens.",
            'Guava': "A tropical fruit known for its sweet and tangy flavor, grown in tropical and subtropical regions globally.",
            'Henna': "A plant used for dyeing hair, skin, and creating temporary tattoos, commonly found in North Africa, the Middle East, and South Asia.",
            'Hibiscus': "A flowering plant known for its vibrant and showy blossoms, hibiscus is found in tropical and subtropical regions worldwide.",
            'Honge': "A tree known for its medicinal properties and used in traditional medicine, found in India and Southeast Asia.",
            'Insulin': "A hormone produced in the pancreas, essential for regulating blood sugar levels, and used in diabetes management.",
            'Jackfruit': "A popular tropical fruit known for its large size and sweet flavor, commonly found in tropical regions.",
            'Jasmine': "A fragrant flower used in perfumes and for its ornamental beauty, often grown in gardens and used globally in perfumery.",
            'Kamakasturi': "A plant known for its aromatic seeds used in perfumes and traditional medicine, found in regions with a suitable climate.",
            'Kambajala': "A type of climbing vine found in tropical regions, known for its potential medicinal properties.",
            'Kasambruga': "A plant often used for its medicinal properties in traditional medicine, found in tropical and subtropical regions.",
            'Kepala': "A tropical fruit also known as breadfruit, used in cooking and found in tropical and subtropical regions.",
            'Kohlrabi': "A vegetable with a taste similar to broccoli stems, used in salads and dishes, cultivated in many parts of the world.",
            'Lantana': "A flowering plant known for its colorful clusters of flowers, often found in gardens and natural landscapes.",
            'Lemon': "A citrus fruit used in cooking, beverages, and for its refreshing scent, grown in various regions with suitable climates.",
            'Lemongrass': "An aromatic herb used in cooking and for its essential oils, commonly found in Asian and tropical regions.",
            'Malabar_Nut': "A medicinal plant used in traditional medicine for respiratory issues, found in tropical and subtropical areas.",
            'Malabar_Spinach': "A leafy green vegetable often used in Asian cuisine, grown in tropical and subtropical regions.",
            'Mango': "A popular tropical fruit known for its sweet and juicy flesh, widely cultivated in tropical regions.",
            'Marigold': "A flowering plant with vibrant, orange or yellow blossoms, commonly grown in gardens and used for ornamental purposes.",
            'Mint': "An aromatic herb used in cooking and for its refreshing flavor, cultivated and found worldwide.",
            'Neem': "A tree with medicinal properties, used in skincare and traditional medicine, commonly found in South Asia.",
            'Nelavembu': "A herb known for its potential health benefits in traditional medicine, found in parts of South India.",
            'Nerale': "A fruit known as jamun or black plum, used in culinary and traditional medicine, found in tropical regions.",
            'Nooni': "A fruit known for its sweet and tangy flavor, found in regions with a suitable climate.",
            'Onion': "A pungent vegetable widely used in cooking, cultivated and consumed globally.",
            'Padri': "A type of snake gourd used in culinary dishes, often found in South Asian cuisine.",
            'Palak(Spinach)': "A leafy green vegetable rich in nutrients, used in cooking and salads, cultivated and consumed worldwide.",
            'Papaya': "A tropical fruit known for its sweet and orange flesh, widely cultivated in tropical and subtropical regions.",
            'Parijatha': "A flowering plant with fragrant white blossoms, often used in traditional medicine and cultivated for its ornamental beauty in India and Southeast Asia.",
            'Pea': "Edible legumes available in various varieties and used in cooking worldwide, known for their protein content and nutritional value.",
            'Pepper': "A spice made from dried peppercorns, used to add flavor and heat to dishes, grown in tropical regions globally.",
            'Pomegranate': "A fruit known for its juicy seeds and potential health benefits, cultivated in Mediterranean and tropical regions.",
            'Pumpkin': "A versatile vegetable used in cooking, baking, and carving for Halloween, grown in many parts of the world.",
            'Raddish': "A root vegetable known for its crisp texture and spicy flavor, cultivated and consumed globally.",
            'Rose': "A flowering plant known for its beautiful and fragrant blossoms, cultivated in gardens worldwide and used in perfumery.",
            'Sampige': "A plant with fragrant white or yellow flowers used for ornamental purposes, commonly grown in tropical regions.",
            'Sapota': "A tropical fruit with sweet and grainy flesh, grown in tropical and subtropical regions.",
            'Seethaashoka': "A tree known for its significance in Hindu mythology and traditional medicine, found in India and Southeast Asia.",
            'Seethapala': "A fruit known for its sweet and creamy pulp, grown in tropical regions worldwide.",
            'Spinach1': "A type of spinach often used in salads and cooking, known for its nutritional value and cultivated globally.",
            'Tamarind': "A tropical fruit known for its sour taste used in various culinary dishes, grown in tropical regions worldwide.",
            'Taro': "A starchy root vegetable used in cooking and making snacks, cultivated and consumed in many parts of the world.",
            'Tecoma': "A flowering plant with bright orange or yellow trumpet-shaped blossoms, grown for its ornamental beauty in various regions.",
            'Thumbe': "A herb used in traditional medicine for its potential health benefits, found in parts of India and other tropical regions.",
            'Tomato': "A widely used fruit (often considered a vegetable) in cooking, cultivated and consumed worldwide.",
            'Tulsi': "A sacred herb with various medicinal uses, grown and revered in India and other parts of Asia.",
            'Turmeric': "A spice with a bright yellow color known for its anti-inflammatory properties, cultivated and used in South Asia and beyond."
        }

        return jsonify({'result': result,
                        'description': class_labels_dict[result]})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)
