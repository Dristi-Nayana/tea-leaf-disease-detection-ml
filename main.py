# %% [markdown]
# ## Preprocessing

# %%
import cv2
import os
​
# Function to process images in a folder
def process_images_in_folder(folder_path):
    print("Processing folder:", folder_path)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            print("Processing image:", img_path)
            # Read image
            img = cv2.imread(img_path)
            
            # Increase brightness
            # Example for OpenCV
            brightness_factor = 1.2
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
            
            # Resize image
            # Example for OpenCV
            desired_size = (255, 255)
            img = cv2.resize(img, desired_size)
            
            # Remove noise
            # Example for OpenCV - Gaussian Blur
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Save processed image
            cv2.imwrite(img_path, img)
​
# Path to your dataset
dataset_path = "D:\\tea sickness dataset output 2222"
​
# Process images in each folder within the dataset directory
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        process_images_in_folder(folder_path)
​

# %% [markdown]
# ## preprocessing:otsu's thresholding

# %%
import cv2
import os
​
# Function to preprocess images using Otsu's thresholding
def preprocess_with_otsu(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Write the preprocessed image to the output path
    cv2.imwrite(output_path, thresholded)
​
# Path to the directory containing the dataset folders
dataset_dir = "D:\\tea sickness dataset output 2222"
​
# Path to the directory where preprocessed images will be saved
output_dir = "D:\\preprocessed"
​
# Iterate through each folder in the dataset directory
for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    
    # Create a corresponding folder in the output directory
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through each image in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        # Preprocess the image using Otsu's thresholding
        output_path = os.path.join(output_folder, image_name)
        preprocess_with_otsu(image_path, output_path)

# %% [markdown]
# ## tan triggs normalization

# %%
import cv2
import os
​
# Function to preprocess images using Tan-Triggs normalization
def preprocess_with_tan_triggs(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Tan-Triggs normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_image = clahe.apply(image)
    
    # Write the preprocessed image to the output path
    cv2.imwrite(output_path, normalized_image)
​
# Path to the directory containing the dataset folders
dataset_dir = "D:\\tea sickness dataset output 2222"
​
# Path to the directory where preprocessed images will be saved
output_dir = "D:\\normalization"
​
# Iterate through each folder in the dataset directory
for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    
    # Create a corresponding folder in the output directory
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through each image in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        # Preprocess the image using Tan-Triggs normalization
        output_path = os.path.join(output_folder, image_name)
        preprocess_with_tan_triggs(image_path, output_path)

# %% [markdown]
# ## resizeeee

# %%
port cv2
​
def resize_and_crop_dataset(input_dir, output_dir, target_width=128, target_height=96):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
​
    # Iterate through each folder in the input directory
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip if it's not a directory
        
        # Create corresponding output folder
        output_folder_path = os.path.join(output_dir, folder_name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
​
        # Iterate through each image file in the folder
        for filename in os.listdir(folder_path):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder_path, filename)
​
            # Read the image
            image = cv2.imread(input_path)
​
            # Check if the image was read successfully
            if image is not None:
                # Resize the image while maintaining aspect ratio
                aspect_ratio = image.shape[1] / image.shape[0]
                new_height = int(target_width / aspect_ratio)
                resized_image = cv2.resize(image, (target_width, new_height))
​
                # Calculate the cropping parameters
                crop_start_y = max(0, int((new_height - target_height) / 2))
                crop_end_y = min(new_height, crop_start_y + target_height)
​
                # Crop the resized image to remove excess background
                cropped_image = resized_image[crop_start_y:crop_end_y, :]
​
                # Save the cropped image
                cv2.imwrite(output_path, cropped_image)
                print(f"Processed {filename} successfully.")
            else:
                print(f"Error: Unable to read {filename} in folder {folder_name}.")
​
# Set the input directory containing the dataset
input_directory = "D:\\tea sickness dataset output 2222"
​
# Set the output directory where resized and cropped images will be saved
output_directory = "D:\\new dataset"
​
# Set the target dimensions for resizing
target_width = 128
target_height = 96
​
# Resize and crop the dataset
resize_and_crop_dataset(input_directory, output_directory, target_width, target_height)
​

# %% [markdown]
# ## color segment

# %%
import cv2
import numpy as np
import os
​
# Path to your dataset directory
dataset_dir = "D:\\tea sickness dataset output 2222"
​
# Function to perform color segmentation
def segment_leaf_color(image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read {image_path}")
        return
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for green color in HSV
    lower_green = np.array([40, 40, 40])  # Adjust these values as needed
    upper_green = np.array([70, 255, 255]) # Adjust these values as needed
    
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply bitwise AND to extract green regions
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    # Save the segmented image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, segmented)
    print(f"Image {filename} segmented and saved to {output_path}")
​
# Iterate through each folder in the dataset
for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    if os.path.isdir(folder_path):
        output_folder = os.path.join(dataset_dir, folder_name + "_segmented_color")
        os.makedirs(output_folder, exist_ok=True)
        # Iterate through each image in the folder and perform color segmentation
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            segment_leaf_color(image_path, output_folder)

# %%

import cv2
import numpy as np
import os
​
# Path to your dataset directory
dataset_dir = "D:\\tea sickness dataset output 2222"
​
# Function to perform color segmentation excluding specified colors
def segment_leaf_color(image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read {image_path}")
        return
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for red, blue, green, and yellow colors
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    blue_lower = np.array([100, 100, 100])
    blue_upper = np.array([140, 255, 255])
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([70, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    # Create masks for each color range
    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Combine the masks using logical OR operation
    combined_mask = mask_red | mask_blue | mask_green | mask_yellow
    
    # Invert the combined mask
    inverted_mask = cv2.bitwise_not(combined_mask)
    
    # Apply the inverted mask to the original image
    segmented = cv2.bitwise_and(image, image, mask=inverted_mask)
    
    # Save the segmented image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, segmented)
    print(f"Image {filename} segmented and saved to {output_path}")
​
# Iterate through each folder in the dataset
for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    if os.path.isdir(folder_path):
        output_folder = os.path.join(dataset_dir, folder_name + "_segmented_color")
        os.makedirs(output_folder, exist_ok=True)
        # Iterate through each image in the folder and perform color segmentation
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            segment_leaf_color(image_path, output_folder)
​

# %%

import cv2
import numpy as np
import random
​
# Load the image
image = cv2.imread("D:\\tea sickness dataset output 2222\\Anthracnose\\output_IMG_20220503_144322.jpg")
​
# Reduce image size
scale_percent = 20  # Adjust as needed
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
image = cv2.resize(image, (width, height))
​
# Define the target colors (red, green, blue) in RGB format
target_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
​
# Define the population size and number of generations
population_size = 50  # Reduced population size
num_generations = 30  # Reduced number of generations
​
# Define the mutation rate
mutation_rate = 0.01
​
# Define the fitness function
def fitness(individual):
    """Calculate the fitness of an individual."""
    total_difference = 0
    for target_color in target_colors:
        total_difference += np.sum(np.abs(np.subtract(individual, target_color)))
    return -total_difference  # Negative because we want to maximize fitness
​
# Initialize the population with random pixels
population = [np.random.randint(0, 256, (image.shape[0], image.shape[1], 3), dtype=np.uint8) for _ in range(population_size)]
​
# Main loop
for generation in range(num_generations):
    # Evaluate the fitness of each individual
    fitness_values = [fitness(individual) for individual in population]
​
    # Select the top individuals (elite) for reproduction
    elite_indices = np.argsort(fitness_values)[-population_size // 2:]
    elite_population = [population[i] for i in elite_indices]
​
    # Create the next generation through crossover and mutation
    next_generation = []
    while len(next_generation) < population_size:
        parent1 = random.choice(elite_population)
        parent2 = random.choice(elite_population)
        crossover_point = random.randint(0, image.shape[1])
        child = np.concatenate([parent1[:, :crossover_point], parent2[:, crossover_point:]], axis=1)
        
        # Mutation
        if random.random() < mutation_rate:
            mutation_x = random.randint(0, image.shape[0] - 1)
            mutation_y = random.randint(0, image.shape[1] - 1)
            child[mutation_x, mutation_y] = np.random.randint(0, 256, 3)
        
        next_generation.append(child)
​
    population = next_generation
​
# Select the best individual from the final population
best_individual = max(population, key=fitness)
​
# Display the segmented image
cv2.imshow("Segmented Image", best_individual)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% [markdown]
# ## texture feature like contrast,LBP etc calculation using GLCM

# %%

import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
​
# Function to calculate texture features
def calculate_texture_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    # Calculate LBP
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    
    # Calculate LBP histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize histogram
    
    return contrast, dissimilarity, homogeneity, energy, correlation, hist
​
# Function to process images in each folder
def process_images_in_dataset(dataset_path):
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            print("Processing folder:", folder_name)
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(folder_path, filename)
                    # Read image
                    img = cv2.imread(img_path)
                    
                    # Calculate texture features
                    contrast, dissimilarity, homogeneity, energy, correlation, lbp_hist = calculate_texture_features(img)
                    
                    # Print or store the texture features
                    print("Image:", filename)
                    print("GLCM Contrast:", contrast)
                    print("GLCM Dissimilarity:", dissimilarity)
                    print("GLCM Homogeneity:", homogeneity)
                    print("GLCM Energy:", energy)
                    print("GLCM Correlation:", correlation)
                    print("LBP Histogram:", lbp_hist)
                    print("------------------------")
​
# Path to your dataset
dataset_path = "D:\\tea sickness dataset output 2222"
​
# Process images in the dataset
process_images_in_dataset(dataset_path)
​

# %% [markdown]
# ## training

# %%
import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
​
# Define the function to extract features
def extract_features(image):
    # Extract color feature by calculating the histogram of the image
    color_feature = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]).flatten()
    
    # Extract texture features, shape features, and size feature
    # Similar to previous implementation
​
    # Concatenate all features into a single feature vector
    features = np.concatenate((color_feature, [shape_area, shape_perimeter, shape_aspect_ratio, size_area]))
​
    return features
​
# Directory containing images for each class
root_dir = "D:\\mini project\\tea sickness dataset output"
​
# Initialize lists to store features and labels
X = []
y = []
​
# Iterate over each class folder
for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)
    if os.path.isdir(class_dir):
        # Iterate over images in the class folder
        for filename in os.listdir(class_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Read the image
                image = cv2.imread(os.path.join(class_dir, filename))
​
                # Extract features
                features = extract_features(image)
​
                # Append features to the list of features
                X.append(features)
​
                # Assign label based on the folder name
                y.append(class_name)
​
                # Check if the number of features exceeds a threshold (e.g., 1000)
                if len(X) >= 1000:
                    # Convert lists to numpy arrays
                    X = np.array(X)  # Features
                    y = np.array(y)  # Labels
​
                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
                    # Initialize Random Forest classifier
                    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
​
                    # Train the classifier
                    rf_classifier.fit(X_train, y_train)
​
                    # Predict on the testing set
                    y_pred = rf_classifier.predict(X_test)
​
                    # Evaluate the model
                    accuracy = accuracy_score(y_test, y_pred)
                    print("Accuracy:", accuracy)
​
                    # Reset lists for the next batch
                    X = []
                    y = []
​
# Convert remaining features to numpy arrays and train the final model
X = np.array(X)  # Features
y = np.array(y)  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:", accuracy)

# %% [markdown]
# ## split dataset

# %%
import os
from sklearn.model_selection import train_test_split
​
# Assuming your dataset folders are named class_0, class_1, ..., class_7
data_dir = "D:\\mini project\\tea sickness dataset output"
classes = sorted(os.listdir(data_dir))
​
# Define variables to store training and testing data
X_train = []
X_test = []
y_train = []
y_test = []
​
# Loop through each class
for class_index, class_name in enumerate(classes):
    class_path = os.path.join(data_dir, class_name)
    files = os.listdir(class_path)
    # Split the data into training and testing sets for each class
    X_train_class, X_test_class, _, _ = train_test_split(
        [os.path.join(class_path, file) for file in files],
        [class_index] * len(files),
        test_size=0.2,  # You can adjust the test size as needed
        random_state=42  # Set random state for reproducibility
    )
    X_train.extend(X_train_class)
    X_test.extend(X_test_class)
    y_train.extend([class_index] * len(X_train_class))
    y_test.extend([class_index] * len(X_test_class))
​
# Now you have X_train, X_test, y_train, y_test containing your training and testing data
​
# Optionally, you can shuffle the data
# import random
# combined_data = list(zip(X_train, y_train))
# random.shuffle(combined_data)
# X_train, y_train = zip(*combined_data)

# %%
print("Number of training samples:", len(X_train))
print("Number of testing samples:", len(X_test))
print("Number of training labels:", len(y_train))
print("Number of testing labels:", len(y_test))

# %% [markdown]
# ## display

# %% [markdown]
# ## hog features

# %%

pip install opencv-python scikit-image matplotlib

# %%
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
​
def extract_hog_features(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to read the image.")
        return None, None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute HOG features
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    
    return features, hog_image
​
def visualize_hog(image_path):
    # Extract HOG features and visualize
    features, hog_image = extract_hog_features(image_path)
    
    if features is None or hog_image is None:
        return
    
    # Plot the original image and HOG image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ax1.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Features')
    ax2.axis('off')
    plt.show()
​
# Example usage:
image_path = "D:\\mini project\\tea sickness dataset\\brown blight\\UNADJUSTEDNONRAW_thumb_14b.jpg"
visualize_hog(image_path)
​

# %%
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
​
def extract_hog_features(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to read the image.")
        return None, None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute HOG features
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    
    return features, hog_image
​
def visualize_hog(image_path):
    # Extract HOG features and visualize
    features, hog_image = extract_hog_features(image_path)
    
    if features is None or hog_image is None:
        return
    
    # Plot the original image and HOG image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ax1.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Features')
    ax2.axis('off')
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
​
# Example usage:
image_path = "D:\\tea sickness dataset output 2222\\Anthracnose\\output_IMG_20220503_144322.jpg"
visualize_hog(image_path)
​

# %%
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
​
def extract_hog_features(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to read the image.")
        return None, None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute HOG features
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    
    return features, hog_image
​
def visualize_hog(image_path):
    # Extract HOG features and visualize
    features, hog_image = extract_hog_features(image_path)
    
    if features is None or hog_image is None:
        return
    
    # Plot the original image and HOG image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ax1.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Features')
    ax2.axis('off')
    plt.show()
​
# Example usage:
image_path = "D:\\new dataset\\brown blight\\output_UNADJUSTEDNONRAW_thumb_12a.jpg"
visualize_hog(image_path)
​

# %%

import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
​
def extract_hog_features(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to read the image.")
        return None, None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute HOG features
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    
    return features, hog_image
​
def visualize_hog(image_path):
    # Extract HOG features and visualize
    features, hog_image = extract_hog_features(image_path)
    
    if features is None or hog_image is None:
        return
    
    # Plot the original image and HOG image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ax1.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Features')
    ax2.axis('off')
    plt.show()
​
# Example usage:
image_path = "D:\\mini project\\tea sickness dataset\\algal leaf\\UNADJUSTEDNONRAW_thumb_3e.jpg"
visualize_hog(image_path)
​

# %% [markdown]
# ## LPB features

# %%

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
​
def extract_lbp_features(image_path, radius=1, n_points=8):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return None
    
    # Compute LBP features
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist
​
def visualize_lbp(image_path, radius=1, n_points=8):
    # Extract LBP features
    lbp_features = extract_lbp_features(image_path, radius, n_points)
    
    if lbp_features is None:
        return
    
    # Plot the LBP histogram
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(lbp_features)), lbp_features)
    plt.title('LBP Features')
    plt.xlabel('LBP Code')
    plt.ylabel('Frequency')
    plt.show()
​
# Path to your image
image_path = "D:\\image gray\\brown blight\\output_UNADJUSTEDNONRAW_thumb_13e.jpg"
​
# Visualize LBP features for the specified image
visualize_lbp(image_path)

# %% [markdown]
# ## pixelwise gray

# %%
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
​
def extract_lbp_features(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to read the image.")
        return None
    
    # Compute LBP features
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    return lbp
​
def visualize_lbp(image_path):
    # Extract LBP features
    lbp_image = extract_lbp_features(image_path)
    
    if lbp_image is None:
        return
    
    # Visualize LBP image
    plt.figure(figsize=(8, 6))
    plt.imshow(lbp_image, cmap='gray')
    plt.title('Local Binary Pattern (LBP) Image')
    plt.axis('off')
    plt.show()
​
# Example usage:
image_path = "D:\\image gray\\brown blight\\output_UNADJUSTEDNONRAW_thumb_13e.jpg"
visualize_lbp(image_path)
​

# %% [markdown]
# ## color histograms

# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
​
def extract_color_features(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to read the image.")
        return None
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Compute color histograms
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    
    return hist_r, hist_g, hist_b
​
def visualize_color_features(image_path):
    # Extract color features
    hist_r, hist_g, hist_b = extract_color_features(image_path)
    
    if hist_r is None or hist_g is None or hist_b is None:
        return
    
    # Visualize color histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(hist_r, color='r')
    plt.title('Red Channel Histogram')
    
    plt.subplot(1, 3, 2)
    plt.plot(hist_g, color='g')
    plt.title('Green Channel Histogram')
    
    plt.subplot(1, 3, 3)
    plt.plot(hist_b, color='b')
    plt.title('Blue Channel Histogram')
    
    plt.tight_layout()
    plt.show()
​
# Example usage:
image_path = "D:\\tea sickness dataset output 2222\\Anthracnose\\output_IMG_20220503_143717.jpg"
visualize_color_features(image_path)
​

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
​
def visualize_color_histogram(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to read the image.")
        return
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split the image into RGB channels
    r, g, b = cv2.split(image_rgb)
    
    # Plot histograms for each color channel
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.hist(r.flatten(), bins=256, color='red')
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 1, 2)
    plt.hist(g.flatten(), bins=256, color='green')
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 1, 3)
    plt.hist(b.flatten(), bins=256, color='blue')
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
​
# Example usage:
image_path = "D:\\tea sickness dataset output 2222\\Anthracnose\\output_IMG_20220503_143717.jpg"
visualize_color_histogram(image_path)

# %%



