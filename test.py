import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from effdet import create_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.tree import DecisionTreeRegressor

#estimates green light time based on vehicle count
class TimeEstimatorAI:
    def __init__(self):
        self.model = DecisionTreeRegressor()
        self.model.fit([[0], [9], [11], [15], [17], [20]], [0, 5, 10, 15, 20, 25])

    def predict_time(self, num_vehicles):
        estimated_time = self.model.predict(np.array([[num_vehicles]]))
        return estimated_time[0]

#define the model and loading function
def load_model(model_path):
    #load model
    model = create_model('tf_efficientdet_lite3', pretrained=True, num_classes=6)

    #load state of model
    state_dict = torch.load(model_path, map_location='cpu')

    #load into model
    model.load_state_dict(state_dict)

    model.eval()
    return model

#preprocess image
def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) #add bat dim

#draw bboxes(fail)
def draw_boxes(images, boxes_list, scores_list, labels_list, class_names, num_vehicles_list, total_vehicles, traffic_signal):
    rows = 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

    for i, (image, boxes, scores, labels) in enumerate(zip(images, boxes_list, scores_list, labels_list)):
        ax = axs[i // cols, i % cols]
        ax.imshow(image)
        for box, score, label in zip(boxes, scores, labels):
            box = [int(x) for x in box]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], f'{class_names[label]}: {score:.2f}',
                    bbox=dict(facecolor='yellow', alpha=0.5))
        
        ax.set_title(f'Signal {i + 1}\nVehicles Detected: {num_vehicles_list[i]}\n'
                     f'Red Light: {traffic_signal.red_durations[i]}\n'
                     f'Yellow Light: {traffic_signal.yellow_durations[i]}\n'
                     f'Green Light: {traffic_signal.green_durations[i]}')
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle(f'Total Vehicles: {total_vehicles}', y=1.02)
    plt.show()

#reshape, pading, trimming, parse predictions from model output
def process_predictions(predictions, threshold=1):
    global CLASS_NAMES
    boxes, scores, labels = [], [], []

    bbox_preds, class_preds = predictions
    
    for bbox_pred, class_pred in zip(bbox_preds, class_preds):
        if isinstance(bbox_pred, torch.Tensor) and isinstance(class_pred, torch.Tensor):
            bbox_pred = bbox_pred.view(-1, 4).cpu().numpy()
            class_pred = class_pred.view(-1, len(CLASS_NAMES)).cpu().numpy()
            
            #apply treashold
            for i in range(class_pred.shape[0]):
                score = np.max(class_pred[i])
                if score > threshold:
                    boxes.append(bbox_pred[i])
                    scores.append(score)
                    labels.append(np.argmax(class_pred[i]))
    
    #list to numpy array
    final_boxes = np.array(boxes) if boxes else np.array([])
    final_scores = np.array(scores) if scores else np.array([])
    final_labels = np.array(labels) if labels else np.array([])

    return final_boxes, final_scores, final_labels

class TrafficSignal:
    def __init__(self):
        self.green_durations = []
        self.yellow_durations = []
        self.red_durations = []

    def predict_green_light_duration(self, vehicle_count):
        return max(5, min(30, vehicle_count * 2))

    def calculate_signal_durations(self, vehicle_counts):
        for i in range(len(vehicle_counts)):
            if i == 0:
                T_g = self.predict_green_light_duration(vehicle_counts[i])
                T_y = 5
                T_r = 0
            else:
                T_r = self.yellow_durations[i - 1] + self.green_durations[i - 1] + (self.red_durations[i - 1] if i > 1 else 0)
                T_y = 5
                T_g = self.predict_green_light_duration(vehicle_counts[i])

            self.green_durations.append(T_g)
            self.yellow_durations.append(T_y)
            self.red_durations.append(T_r)

    def display_timings(self):
        for i in range(len(self.green_durations)):
            print(f"Signal {i + 1}:")
            print(f"  Red Light Duration: {self.red_durations[i]}")
            print(f"  Yellow Light Duration: {self.yellow_durations[i]}")
            print(f"  Green Light Duration: {self.green_durations[i]}")
            print()

def main():
    global CLASS_NAMES
    MODEL_SAVE_PATH = '/home/allan/project/sih/efficientdet.pth'
    IMAGE_PATHS = ['/home/allan/project/sih/tes.jpg', 
                   '/home/allan/project/sih/te2s.jpg', 
                   '/home/allan/project/sih/te3s.jpg', 
                   '/home/allan/project/sih/te4s.jpg']
    INPUT_SIZE = 512
    CLASS_NAMES = ['car', 'cart', 'truck', 'rickshaw', 'bike', 'ambulance']
    THRESHOLDS = [1.1182, 1.0, 1.315, 1.172]

    model = load_model(MODEL_SAVE_PATH)
    print("Model loaded")

    time_estimator = TimeEstimatorAI()
    images = []
    boxes_list, scores_list, labels_list, num_vehicles_list = [], [], [], []

    vehicle_counts = []

    for i, image_path in enumerate(IMAGE_PATHS):
        image_tensor = preprocess_image(image_path, INPUT_SIZE)
        
        with torch.no_grad():
            predictions = model(image_tensor)
            print(f"Inference done for {image_path}")

        boxes, scores, labels = process_predictions(predictions, threshold=THRESHOLDS[i])
        num_vehicles = len(scores)
        vehicle_counts.append(num_vehicles)
        print(f"No of vehicles in {image_path}: {num_vehicles}")

        # Print boxes, scores, and labels for debugging
        print(f"Boxes for {image_path}: {boxes}")
        print(f"Scores for {image_path}: {scores}")
        print(f"Labels for {image_path}: {labels}")

        images.append(Image.open(image_path))
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)
        num_vehicles_list.append(num_vehicles)


    traffic_signal = TrafficSignal()
    traffic_signal.calculate_signal_durations(vehicle_counts)
    traffic_signal.display_timings()

    draw_boxes(images, boxes_list, scores_list, labels_list, CLASS_NAMES, num_vehicles_list, sum(vehicle_counts), traffic_signal)

if __name__ == '__main__':
    main()
