import os 
import cv2
import shutil
import csv

data_path = 'siameseData/Extracted Faces'
output_path = 'siameseData/output_pth'

if not os.path.exists(output_path):
    print('Created {0}'.format(output_path))
    os.makedirs(output_path)
    for folders in os.listdir(data_path):
        folders_path = os.path.join(data_path,folders)
        for imgs in os.listdir(folders_path):
            imgs_path = os.path.join(folders_path,imgs)
            new_image_name = str(folders) + '_' + str(imgs)
            new_path = os.path.join(output_path,new_image_name)
            print('Doing for {0}'.format(imgs_path))
            shutil.copy2(imgs_path,new_path)

def generate_csv_from_images(folder_path, csv_path):
    """
    Generates a CSV file with filenames and their corresponding labels from a folder of images.
    
    Args:
        folder_path (str): Path to the folder containing the image files.
        csv_path (str): Path to save the generated CSV file.
        
    Returns:
        None
    """
    # Prepare the output data
    data = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Process only JPG files
            # Extract the label (the first number before the '_')
            label = filename.split("_")[0]
            # Append the filename and label to the data
            data.append([filename, label])

    # Write the data to the CSV file
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header
        writer.writerow(["filename", "label"])
        # Write the rows
        writer.writerows(data)
    
    print(f"CSV file has been generated and saved at: {csv_path}")

generate_csv_from_images(output_path,csv_path='siameseData/data.csv')


