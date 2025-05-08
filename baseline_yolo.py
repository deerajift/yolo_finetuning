from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

source = "datasets/b/*.png"
# Run batched inference on a list of images
results = model(source, stream=True)  # return a list of Results objects

result_save_folder = "datasets/b-results/"
# Process results list
for result in results:
    name = result.path
    #print("this is the name",name)
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    image_name = name.split('/')[-1]
    result_filename = result_save_folder+image_name
    result.save(filename=result_filename)  # save to disk