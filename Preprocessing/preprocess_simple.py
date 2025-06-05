import pydicom
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import json

class ThyroidDicomProcessor:
    def __init__(self, dataset_path):
        """
        Initialize the DICOM processor
        
        Args:
            dataset_path (str): Path to the Dataset folder
        """
        self.dataset_path = Path(dataset_path)
        self.categories = {
            '1. goiter-diffuse goiter': 'goiter_diffuse',
            '2. MNG': 'mng',
            '3. diffuse toxic goiter': 'diffuse_toxic_goiter',
            '4. thyroiditis': 'thyroiditis',
            '5. cold nodule': 'cold_nodule',
            '6. hot nodule': 'hot_nodule',
            '7. warm nodule': 'warm_nodule',
            '8. normal': 'normal'
        }
        self.dicom_data = defaultdict(list)
        self.image_data = defaultdict(list)
        self.metadata = defaultdict(list)
    
    def read_dicom_files(self):
        """
        Read all DICOM files from each category folder
        """
        print("Reading DICOM files...")
        
        for category_folder, category_name in self.categories.items():
            category_path = self.dataset_path / category_folder
            
            if not category_path.exists():
                print(f"Warning: Category folder '{category_folder}' not found")
                continue
            
            # Find all .dcm files in the category folder
            dicom_files = list(category_path.glob("*.dcm"))
            print(f"Found {len(dicom_files)} DICOM files in '{category_folder}'")
            
            for dicom_file in dicom_files:
                try:
                    # Read DICOM file
                    dicom_data = pydicom.dcmread(str(dicom_file))
                    
                    # Store the complete DICOM object
                    self.dicom_data[category_name].append({
                        'file_path': str(dicom_file),
                        'filename': dicom_file.name,
                        'dicom_object': dicom_data
                    })
                    
                    # Extract pixel data
                    if hasattr(dicom_data, 'pixel_array'):
                        pixel_array = dicom_data.pixel_array
                        
                        # Store image data
                        self.image_data[category_name].append({
                            'filename': dicom_file.name,
                            'image': pixel_array,
                            'shape': pixel_array.shape,
                            'dtype': str(pixel_array.dtype)
                        })
                        
                        # Extract important metadata
                        metadata = self.extract_metadata(dicom_data, dicom_file.name)
                        self.metadata[category_name].append(metadata)
                    
                except Exception as e:
                    print(f"Error reading {dicom_file}: {str(e)}")
        
        self.print_summary()
    
    def extract_metadata(self, dicom_obj, filename):
        """
        Extract important metadata from DICOM object
        """
        metadata = {
            'filename': filename,
            'patient_id': getattr(dicom_obj, 'PatientID', 'Unknown'),
            'study_date': getattr(dicom_obj, 'StudyDate', 'Unknown'),
            'modality': getattr(dicom_obj, 'Modality', 'Unknown'),
            'image_type': getattr(dicom_obj, 'ImageType', 'Unknown'),
            'rows': getattr(dicom_obj, 'Rows', None),
            'columns': getattr(dicom_obj, 'Columns', None),
            'pixel_spacing': getattr(dicom_obj, 'PixelSpacing', None),
            'bits_allocated': getattr(dicom_obj, 'BitsAllocated', None),
            'bits_stored': getattr(dicom_obj, 'BitsStored', None),
            'window_center': getattr(dicom_obj, 'WindowCenter', None),
            'window_width': getattr(dicom_obj, 'WindowWidth', None)
        }
        return metadata
    
    def normalize_image(self, image, method='minmax'):
        """
        Normalize image pixel values
        
        Args:
            image: Input image array
            method: Normalization method ('minmax', 'zscore', 'window')
        """
        if method == 'minmax':
            # Min-max normalization to 0-255
            image_min, image_max = image.min(), image.max()
            if image_max > image_min:
                normalized = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image, dtype=np.uint8)
        
        elif method == 'zscore':
            # Z-score normalization then scale to 0-255
            mean, std = image.mean(), image.std()
            if std > 0:
                normalized = (image - mean) / std
                normalized = ((normalized - normalized.min()) / 
                            (normalized.max() - normalized.min()) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image, dtype=np.uint8)
        
        elif method == 'window':
            # Windowing (if window center/width available)
            # This would require metadata - simplified version here
            normalized = np.clip(image, 0, np.percentile(image, 99))
            normalized = ((normalized / normalized.max()) * 255).astype(np.uint8)
        
        return normalized
    
    def convert_to_png(self, output_dir, normalization_method='minmax', resize_shape=None):
        """
        Convert DICOM images to PNG format
        
        Args:
            output_dir (str): Output directory for PNG files
            normalization_method (str): Method for normalization
            resize_shape (tuple): Optional resize shape (width, height)
        """
        output_path = Path(output_dir)
        
        print(f"Converting to PNG format...")
        conversion_log = []
        
        for category_name, images in self.image_data.items():
            category_output_path = output_path / category_name
            category_output_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing {len(images)} images for category: {category_name}")
            
            for i, img_data in enumerate(images):
                try:
                    image = img_data['image']
                    filename = Path(img_data['filename']).stem  # Remove .dcm extension
                    
                    # Handle different image dimensions
                    if len(image.shape) == 3:
                        # Multi-channel image - take first channel or convert to grayscale
                        if image.shape[2] == 3:  # RGB
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        else:
                            image = image[:, :, 0]  # Take first channel
                    
                    # Normalize the image
                    normalized_image = self.normalize_image(image, normalization_method)
                    
                    # Resize if specified
                    if resize_shape:
                        normalized_image = cv2.resize(normalized_image, resize_shape)
                    
                    # Convert to PIL Image and save as PNG
                    pil_image = Image.fromarray(normalized_image)
                    png_filename = f"{filename}.png"
                    png_path = category_output_path / png_filename
                    pil_image.save(png_path)
                    
                    conversion_log.append({
                        'category': category_name,
                        'original_file': img_data['filename'],
                        'png_file': png_filename,
                        'original_shape': img_data['shape'],
                        'final_shape': normalized_image.shape
                    })
                    
                except Exception as e:
                    print(f"Error converting {img_data['filename']}: {str(e)}")
        
        # Save conversion log
        log_path = output_path / "conversion_log.json"
        with open(log_path, 'w') as f:
            json.dump(conversion_log, f, indent=2)
        
        print(f"PNG conversion completed. Files saved to: {output_path}")
        print(f"Conversion log saved to: {log_path}")
    
    def print_summary(self):
        """
        Print summary of loaded data
        """
        print("\n" + "="*50)
        print("DICOM LOADING SUMMARY")
        print("="*50)
        
        total_files = 0
        for category_name, images in self.image_data.items():
            count = len(images)
            total_files += count
            print(f"{category_name}: {count} files")
            
            if count > 0:
                # Show image statistics
                shapes = [img['shape'] for img in images]
                unique_shapes = set(shapes)
                print(f"  - Unique image shapes: {unique_shapes}")
        
        print(f"\nTotal files loaded: {total_files}")
    
    def get_category_data(self, category_name):
        """
        Get all data for a specific category
        
        Args:
            category_name (str): Name of the category
            
        Returns:
            dict: Dictionary containing images, dicom_objects, and metadata
        """
        return {
            'images': self.image_data.get(category_name, []),
            'dicom_objects': self.dicom_data.get(category_name, []),
            'metadata': self.metadata.get(category_name, [])
        }
    
    def visualize_samples(self, samples_per_category=2, figsize=(15, 10)):
        """
        Visualize sample images from each category
        """
        categories = list(self.image_data.keys())
        n_categories = len(categories)
        
        if n_categories == 0:
            print("No image data available for visualization")
            return
        
        fig, axes = plt.subplots(n_categories, samples_per_category, 
                               figsize=figsize, squeeze=False)
        
        for i, category in enumerate(categories):
            images = self.image_data[category]
            n_samples = min(samples_per_category, len(images))
            
            for j in range(samples_per_category):
                ax = axes[i, j]
                
                if j < n_samples:
                    img = images[j]['image']
                    # Handle multi-dimensional images
                    if len(img.shape) == 3:
                        img = img[:, :, 0]  # Take first channel
                    
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"{category}\n{images[j]['filename']}")
                else:
                    ax.set_title(f"{category}\n(No more samples)")
                
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig("sample_visualization")

print("Done!")
# Usage example
def main():
    # Initialize the processor
    dataset_path = "/home/kherad/AlirezaMottaghi/Thyroid/Dataset"  # Adjust path as needed
    processor = ThyroidDicomProcessor(dataset_path)
    
    # Read all DICOM files
    processor.read_dicom_files()
    
    # Convert to PNG
    output_dir = "Thyroid/PNG_Dataset"
    processor.convert_to_png(
        output_dir=output_dir,
        normalization_method='minmax',  # Options: 'minmax', 'zscore', 'window'
        resize_shape=(512, 512)  # Optional: resize all images to 512x512
    )
    
    # Visualize samples (optional)
    processor.visualize_samples(samples_per_category=3)
    
    # Access specific category data
    normal_data = processor.get_category_data('normal')
    print(f"Normal category has {len(normal_data['images'])} images")
    
    # Access the organized data
    print("\nAvailable categories:")
    for category in processor.image_data.keys():
        print(f"- {category}: {len(processor.image_data[category])} images")

if __name__ == "__main__":
    main()


# # Alternative simple function for quick loading
# def load_thyroid_dicom_simple(dataset_path):
#     """
#     Simple function to quickly load DICOM files into a dictionary
    
#     Returns:
#         dict: Dictionary with category names as keys and list of image arrays as values
#     """
#     categories = {
#         '1. goiter-diffuse goiter': 'goiter_diffuse',
#         '2. MNG': 'mng',
#         '3. diffuse toxic goiter': 'diffuse_toxic_goiter',
#         '4. thyroiditis': 'thyroiditis',
#         '5. cold nodule': 'cold_nodule',
#         '6. hot nodule': 'hot_nodule',
#         '7. warm nodule': 'warm_nodule',
#         '8. normal': 'normal'
#     }
    
#     data = {}
#     for folder, category in categories.items():
#         category_path = Path(dataset_path) / folder
#         if category_path.exists():
#             images = []
#             for dcm_file in category_path.glob("*.dcm"):
#                 try:
#                     dicom_data = pydicom.dcmread(str(dcm_file))
#                     if hasattr(dicom_data, 'pixel_array'):
#                         images.append(dicom_data.pixel_array)
#                 except Exception as e:
#                     print(f"Error reading {dcm_file}: {e}")
#             data[category] = images
#             print(f"Loaded {len(images)} images for {category}")
    
#     return data