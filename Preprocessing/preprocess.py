import os
import glob
import pydicom
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from skimage import exposure, filters, morphology
from scipy import ndimage
import seaborn as sns

class EnhancedThyroidProcessor:
    def __init__(self, dataset_path):
        """
        Enhanced DICOM processor for nuclear medicine thyroid images
        
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
        self.enhanced_data = defaultdict(list)
        self.metadata = defaultdict(list)
        
        # Enhancement parameters
        self.target_size = (256, 256)  # Optimal for nuclear medicine images
        self.clahe_clip_limit = 2.0
        self.clahe_tile_grid_size = (8, 8)
        self.gaussian_sigma = 0.8
    
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
                        
                        # Store original image data
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
    
    def nuclear_medicine_enhancement(self, image):
        """
        Apply specialized enhancement for nuclear medicine images
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image array
        """
        # Convert to float for processing
        if image.dtype != np.float64:
            img_float = image.astype(np.float64)
        else:
            img_float = image.copy()
        
        # Handle multi-dimensional images
        if len(img_float.shape) == 3:
            if img_float.shape[2] == 3:  # RGB
                img_float = cv2.cvtColor(img_float.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img_float = img_float[:, :, 0]  # Take first channel
        
        # Step 1: Normalize to 0-1 range
        img_min, img_max = img_float.min(), img_float.max()
        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min)
        else:
            img_normalized = np.zeros_like(img_float)
        
        # Step 2: Apply Gaussian filtering for noise reduction
        img_filtered = filters.gaussian(img_normalized, sigma=self.gaussian_sigma)
        
        # Step 3: Convert to uint8 for CLAHE
        img_uint8 = (img_filtered * 255).astype(np.uint8)
        
        # Step 4: Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, 
            tileGridSize=self.clahe_tile_grid_size
        )
        img_clahe = clahe.apply(img_uint8)
        
        # Step 5: Gamma correction for nuclear medicine images
        # Nuclear medicine images often benefit from gamma < 1 to enhance uptake areas
        gamma = 0.8
        img_gamma = exposure.adjust_gamma(img_clahe / 255.0, gamma)
        
        # Step 6: Histogram equalization (adaptive)
        img_eq = exposure.equalize_adapthist(img_gamma, clip_limit=0.03)
        
        # Step 7: Resize to target size
        img_resized = cv2.resize(img_eq, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Step 8: Final normalization to 0-255 range
        img_final = ((img_resized - img_resized.min()) / 
                    (img_resized.max() - img_resized.min()) * 255).astype(np.uint8)
        
        return img_final
    
    def enhance_all_images(self):
        """
        Apply enhancement to all loaded images
        """
        print("Applying nuclear medicine image enhancement...")
        
        for category_name, images in self.image_data.items():
            print(f"Enhancing {len(images)} images for category: {category_name}")
            
            for img_data in images:
                try:
                    original_image = img_data['image']
                    enhanced_image = self.nuclear_medicine_enhancement(original_image)
                    
                    self.enhanced_data[category_name].append({
                        'filename': img_data['filename'],
                        'original_image': original_image,
                        'enhanced_image': enhanced_image,
                        'original_shape': img_data['shape'],
                        'enhanced_shape': enhanced_image.shape
                    })
                    
                except Exception as e:
                    print(f"Error enhancing {img_data['filename']}: {str(e)}")
        
        print("Enhancement completed!")
    
    def save_enhanced_images(self, output_dir, save_originals=False):
        """
        Save enhanced images to PNG format
        
        Args:
            output_dir (str): Output directory for enhanced images
            save_originals (bool): Whether to save original images as well
        """
        output_path = Path(output_dir)
        enhanced_path = output_path / "enhanced"
        
        print(f"Saving enhanced images to {enhanced_path}")
        
        if save_originals:
            original_path = output_path / "original"
            original_path.mkdir(parents=True, exist_ok=True)
        
        save_log = []
        
        for category_name, images in self.enhanced_data.items():
            # Create category directories
            category_enhanced_path = enhanced_path / category_name
            category_enhanced_path.mkdir(parents=True, exist_ok=True)
            
            if save_originals:
                category_original_path = original_path / category_name
                category_original_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving {len(images)} enhanced images for category: {category_name}")
            
            for img_data in images:
                try:
                    filename_stem = Path(img_data['filename']).stem
                    
                    # Save enhanced image
                    enhanced_filename = f"{filename_stem}_enhanced.png"
                    enhanced_filepath = category_enhanced_path / enhanced_filename
                    enhanced_pil = Image.fromarray(img_data['enhanced_image'])
                    enhanced_pil.save(enhanced_filepath)
                    
                    # Save original image if requested
                    if save_originals:
                        original_image = img_data['original_image']
                        if len(original_image.shape) == 3:
                            original_image = original_image[:, :, 0]
                        
                        # Normalize original for saving
                        orig_min, orig_max = original_image.min(), original_image.max()
                        if orig_max > orig_min:
                            original_normalized = ((original_image - orig_min) / 
                                                 (orig_max - orig_min) * 255).astype(np.uint8)
                        else:
                            original_normalized = np.zeros_like(original_image, dtype=np.uint8)
                        
                        original_resized = cv2.resize(original_normalized, self.target_size)
                        original_filename = f"{filename_stem}_original.png"
                        original_filepath = category_original_path / original_filename
                        original_pil = Image.fromarray(original_resized)
                        original_pil.save(original_filepath)
                    
                    save_log.append({
                        'category': category_name,
                        'original_filename': img_data['filename'],
                        'enhanced_filename': enhanced_filename,
                        'original_shape': img_data['original_shape'],
                        'enhanced_shape': img_data['enhanced_shape']
                    })
                    
                except Exception as e:
                    print(f"Error saving {img_data['filename']}: {str(e)}")
        
        # Save processing log
        log_path = output_path / "enhancement_log.json"
        with open(log_path, 'w') as f:
            json.dump(save_log, f, indent=2)
        
        print(f"Enhanced images saved to: {enhanced_path}")
        print(f"Enhancement log saved to: {log_path}")
    
    def visualize_enhancement_comparison(self, samples_per_category=2, figsize=(20, 12)):
        """
        Visualize original vs enhanced images for comparison
        """
        categories = list(self.enhanced_data.keys())
        n_categories = len(categories)
        
        if n_categories == 0:
            print("No enhanced data available for visualization")
            return
        
        # Create subplots: categories x samples x 2 (original vs enhanced)
        fig, axes = plt.subplots(n_categories, samples_per_category * 2, 
                               figsize=figsize, squeeze=False)
        
        for i, category in enumerate(categories):
            images = self.enhanced_data[category]
            n_samples = min(samples_per_category, len(images))
            
            for j in range(samples_per_category):
                # Original image column
                ax_orig = axes[i, j * 2]
                # Enhanced image column  
                ax_enh = axes[i, j * 2 + 1]
                
                if j < n_samples:
                    # Original image
                    orig_img = images[j]['original_image']
                    if len(orig_img.shape) == 3:
                        orig_img = orig_img[:, :, 0]
                    
                    # Normalize original for display
                    orig_min, orig_max = orig_img.min(), orig_img.max()
                    if orig_max > orig_min:
                        orig_display = (orig_img - orig_min) / (orig_max - orig_min)
                    else:
                        orig_display = np.zeros_like(orig_img)
                    
                    ax_orig.imshow(orig_display, cmap='hot', aspect='equal')
                    ax_orig.set_title(f"{category}\nOriginal")
                    
                    # Enhanced image
                    enh_img = images[j]['enhanced_image']
                    ax_enh.imshow(enh_img, cmap='hot', aspect='equal')
                    ax_enh.set_title(f"{category}\nEnhanced")
                else:
                    ax_orig.set_title(f"{category}\nNo sample")
                    ax_enh.set_title(f"{category}\nNo sample")
                
                ax_orig.axis('off')
                ax_enh.axis('off')
        
        plt.tight_layout()
        plt.suptitle('Nuclear Medicine Image Enhancement Comparison\n(Original vs Enhanced)', 
                    fontsize=16, y=0.98)
        plt.savefig("enhanced_sample")
    
    def analyze_enhancement_effects(self):
        """
        Analyze the effects of enhancement on image properties
        """
        print("\n" + "="*60)
        print("ENHANCEMENT ANALYSIS")
        print("="*60)
        
        enhancement_stats = {}
        
        for category_name, images in self.enhanced_data.items():
            orig_contrasts = []
            enh_contrasts = []
            orig_means = []
            enh_means = []
            orig_stds = []
            enh_stds = []
            
            for img_data in images:
                # Original image stats
                orig_img = img_data['original_image']
                if len(orig_img.shape) == 3:
                    orig_img = orig_img[:, :, 0]
                
                orig_contrasts.append(orig_img.std())
                orig_means.append(orig_img.mean())
                orig_stds.append(orig_img.std())
                
                # Enhanced image stats
                enh_img = img_data['enhanced_image']
                enh_contrasts.append(enh_img.std())
                enh_means.append(enh_img.mean())
                enh_stds.append(enh_img.std())
            
            enhancement_stats[category_name] = {
                'original_contrast_mean': np.mean(orig_contrasts),
                'enhanced_contrast_mean': np.mean(enh_contrasts),
                'contrast_improvement': np.mean(enh_contrasts) / np.mean(orig_contrasts) if np.mean(orig_contrasts) > 0 else 0,
                'original_intensity_mean': np.mean(orig_means),
                'enhanced_intensity_mean': np.mean(enh_means),
                'count': len(images)
            }
            
            print(f"\n{category_name.upper()}:")
            print(f"  Images: {len(images)}")
            print(f"  Contrast improvement: {enhancement_stats[category_name]['contrast_improvement']:.2f}x")
            print(f"  Original contrast (std): {enhancement_stats[category_name]['original_contrast_mean']:.2f}")
            print(f"  Enhanced contrast (std): {enhancement_stats[category_name]['enhanced_contrast_mean']:.2f}")
        
        return enhancement_stats
    
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
        print(f"Target enhanced size: {self.target_size}")


# Usage example
def main():
    # Initialize the enhanced processor
    dataset_path = "/home/kherad/AlirezaMottaghi/Thyroid/Dataset"  # Path relative to preprocessing script
    processor = EnhancedThyroidProcessor(dataset_path)
    
    # Step 1: Read all DICOM files
    print("Step 1: Reading DICOM files...")
    processor.read_dicom_files()
    
    # Step 2: Apply enhancement to all images
    print("\nStep 2: Applying nuclear medicine enhancement...")
    processor.enhance_all_images()
    
    # Step 3: Analyze enhancement effects
    print("\nStep 3: Analyzing enhancement effects...")
    enhancement_stats = processor.analyze_enhancement_effects()
    
    # Step 4: Save enhanced images
    print("\nStep 4: Saving enhanced images...")
    output_dir = "/home/kherad/AlirezaMottaghi/Thyroid/ProcessedData"
    processor.save_enhanced_images(output_dir, save_originals=True)
    
    # Step 5: Visualize enhancement comparison
    print("\nStep 5: Visualizing enhancement results...")
    processor.visualize_enhancement_comparison(samples_per_category=2)
    
    print("\nEnhanced preprocessing completed successfully!")
    print(f"Enhanced images saved to: {output_dir}/enhanced/")
    print(f"Original images saved to: {output_dir}/original/")

if __name__ == "__main__":
    main()