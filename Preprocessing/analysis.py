import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import cv2
from PIL import Image
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import json
import warnings
warnings.filterwarnings('ignore')

class ThyroidDataAnalyzer:
    def __init__(self, enhanced_data_path):
        """
        Initialize the data analyzer for thyroid classification
        
        Args:
            enhanced_data_path (str): Path to enhanced images directory
        """
        self.enhanced_data_path = Path(enhanced_data_path)
        self.categories = [
            'goiter_diffuse', 'mng', 'diffuse_toxic_goiter', 'thyroiditis',
            'cold_nodule', 'hot_nodule', 'warm_nodule', 'normal'
        ]
        
        # Data storage
        self.class_distribution = {}
        self.image_paths = defaultdict(list)
        self.images = defaultdict(list)
        self.labels = []
        self.image_data = []
        
        # Balancing strategies
        self.balanced_datasets = {}
        
    def load_enhanced_images(self):
        """
        Load all enhanced images and create class distribution analysis
        """
        print("Loading enhanced images for analysis...")
        
        total_images = 0
        
        for category in self.categories:
            category_path = self.enhanced_data_path / category
            
            if not category_path.exists():
                print(f"Warning: Category '{category}' not found at {category_path}")
                self.class_distribution[category] = 0
                continue
            
            # Find all PNG files in category
            image_files = list(category_path.glob("*.png"))
            count = len(image_files)
            
            self.class_distribution[category] = count
            self.image_paths[category] = image_files
            total_images += count
            
            print(f"  {category}: {count} images")
            
            # Load images for balancing operations
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self.images[category].append(img)
                        self.image_data.append(img.flatten())  # Flatten for SMOTE
                        self.labels.append(category)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        print(f"\nTotal images loaded: {total_images}")
        
        # Convert to numpy arrays
        self.image_data = np.array(self.image_data)
        self.labels = np.array(self.labels)
        
        return self.class_distribution
    
    def analyze_class_distribution(self):
        """
        Comprehensive analysis of class distribution
        """
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Basic statistics
        total_samples = sum(self.class_distribution.values())
        mean_samples = total_samples / len(self.categories)
        
        print(f"Total samples: {total_samples}")
        print(f"Number of classes: {len(self.categories)}")
        print(f"Mean samples per class: {mean_samples:.1f}")
        
        # Class distribution details
        sorted_classes = sorted(self.class_distribution.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nClass distribution (sorted by count):")
        for class_name, count in sorted_classes:
            percentage = (count / total_samples) * 100
            print(f"  {class_name:20}: {count:4d} ({percentage:5.1f}%)")
        
        # Imbalance metrics
        max_count = max(self.class_distribution.values())
        min_count = min(self.class_distribution.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nImbalance Analysis:")
        print(f"  Largest class: {max_count} samples")
        print(f"  Smallest class: {min_count} samples")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Classification of imbalance severity
        if imbalance_ratio < 2:
            severity = "Mild"
        elif imbalance_ratio < 5:
            severity = "Moderate"
        elif imbalance_ratio < 10:
            severity = "Severe"
        else:
            severity = "Extreme"
        
        print(f"  Imbalance severity: {severity}")
        
        return {
            'total_samples': total_samples,
            'imbalance_ratio': imbalance_ratio,
            'severity': severity,
            'class_counts': dict(sorted_classes)
        }
    
    def visualize_class_distribution(self, figsize=(15, 10)):
        """
        Create comprehensive visualizations of class distribution
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Bar plot of class distribution
        classes = list(self.class_distribution.keys())
        counts = list(self.class_distribution.values())
        
        ax1 = axes[0, 0]
        bars = ax1.bar(classes, counts, color=plt.cm.Set3(np.linspace(0, 1, len(classes))))
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom')
        
        # 2. Pie chart
        ax2 = axes[0, 1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        # 3. Imbalance visualization
        ax3 = axes[1, 0]
        sorted_data = sorted(zip(classes, counts), key=lambda x: x[1])
        sorted_classes, sorted_counts = zip(*sorted_data)
        
        bars = ax3.barh(sorted_classes, sorted_counts, color=plt.cm.RdYlBu(np.linspace(0, 1, len(classes))))
        ax3.set_title('Class Imbalance (Sorted by Count)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Samples')
        
        # Add value labels
        for bar, count in zip(bars, sorted_counts):
            ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    str(count), ha='left', va='center')
        
        # 4. Imbalance ratio heatmap
        ax4 = axes[1, 1]
        
        # Create imbalance matrix
        n_classes = len(classes)
        imbalance_matrix = np.zeros((n_classes, n_classes))
        
        for i, count_i in enumerate(counts):
            for j, count_j in enumerate(counts):
                if count_j > 0:
                    imbalance_matrix[i, j] = count_i / count_j
        
        im = ax4.imshow(imbalance_matrix, cmap='RdYlBu_r', aspect='auto')
        ax4.set_title('Pairwise Imbalance Ratios', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(n_classes))
        ax4.set_yticks(range(n_classes))
        ax4.set_xticklabels([c[:10] for c in classes], rotation=45)
        ax4.set_yticklabels([c[:10] for c in classes])
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig("balacing_visualization")
    
    def create_balanced_datasets(self):
        """
        Create multiple balanced datasets using different strategies
        """
        print("\n" + "="*60)
        print("CREATING BALANCED DATASETS")
        print("="*60)
        
        # Strategy 1: Random Oversampling
        print("1. Creating Random Oversampling balanced dataset...")
        self.balanced_datasets['random_oversample'] = self._random_oversample()
        
        # Strategy 2: Random Undersampling
        print("2. Creating Random Undersampling balanced dataset...")
        self.balanced_datasets['random_undersample'] = self._random_undersample()
        
        # Strategy 3: SMOTE (Synthetic Minority Oversampling)
        print("3. Creating SMOTE balanced dataset...")
        self.balanced_datasets['smote'] = self._apply_smote()
        
        # Strategy 4: ADASYN (Adaptive Synthetic Sampling)
        print("4. Creating ADASYN balanced dataset...")
        self.balanced_datasets['adasyn'] = self._apply_adasyn()
        
        # Strategy 5: Combined SMOTE + Edited Nearest Neighbours
        print("5. Creating SMOTE + ENN balanced dataset...")
        self.balanced_datasets['smote_enn'] = self._apply_smote_enn()
        
        # Strategy 6: Class weights (for training)
        print("6. Calculating class weights for training...")
        self.balanced_datasets['class_weights'] = self._calculate_class_weights()
        
        # Compare all strategies
        self._compare_balancing_strategies()
    
    def _random_oversample(self):
        """Random oversampling to match the largest class"""
        max_count = max(self.class_distribution.values())
        balanced_images = defaultdict(list)
        balanced_paths = defaultdict(list)
        
        for category in self.categories:
            current_images = self.images[category]
            current_paths = self.image_paths[category]
            current_count = len(current_images)
            
            if current_count == 0:
                continue
            
            # Oversample to match max_count
            if current_count < max_count:
                # Randomly sample with replacement
                indices = np.random.choice(current_count, max_count, replace=True)
                balanced_images[category] = [current_images[i] for i in indices]
                balanced_paths[category] = [current_paths[i] for i in indices]
            else:
                balanced_images[category] = current_images
                balanced_paths[category] = current_paths
        
        return {
            'images': balanced_images,
            'paths': balanced_paths,
            'method': 'Random Oversampling'
        }
    
    def _random_undersample(self):
        """Random undersampling to match the smallest class"""
        min_count = min([count for count in self.class_distribution.values() if count > 0])
        balanced_images = defaultdict(list)
        balanced_paths = defaultdict(list)
        
        for category in self.categories:
            current_images = self.images[category]
            current_paths = self.image_paths[category]
            current_count = len(current_images)
            
            if current_count == 0:
                continue
            
            # Undersample to match min_count
            if current_count > min_count:
                indices = np.random.choice(current_count, min_count, replace=False)
                balanced_images[category] = [current_images[i] for i in indices]
                balanced_paths[category] = [current_paths[i] for i in indices]
            else:
                balanced_images[category] = current_images
                balanced_paths[category] = current_paths
        
        return {
            'images': balanced_images,
            'paths': balanced_paths,
            'method': 'Random Undersampling'
        }
    
    def _apply_smote(self):
        """Apply SMOTE for synthetic oversampling"""
        try:
            # Reduce dimensionality for SMOTE (it's computationally intensive)
            from sklearn.decomposition import PCA
            
            print("   Applying PCA for dimensionality reduction...")
            pca = PCA(n_components=50)  # Reduce to 50 components
            X_reduced = pca.fit_transform(self.image_data)
            
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X_reduced, self.labels)
            
            # Transform back to original space
            X_resampled_full = pca.inverse_transform(X_resampled)
            
            # Organize by categories
            balanced_data = defaultdict(list)
            for img_flat, label in zip(X_resampled_full, y_resampled):
                # Reshape back to image shape (256, 256)
                img = img_flat.reshape(256, 256).astype(np.uint8)
                balanced_data[label].append(img)
            
            return {
                'images': balanced_data,
                'method': 'SMOTE',
                'synthetic': True
            }
            
        except Exception as e:
            print(f"   SMOTE failed: {e}")
            return None
    
    def _apply_adasyn(self):
        """Apply ADASYN for adaptive synthetic oversampling"""
        try:
            from sklearn.decomposition import PCA
            
            print("   Applying PCA for dimensionality reduction...")
            pca = PCA(n_components=50)
            X_reduced = pca.fit_transform(self.image_data)
            
            adasyn = ADASYN(random_state=42, n_neighbors=3)
            X_resampled, y_resampled = adasyn.fit_resample(X_reduced, self.labels)
            
            # Transform back to original space
            X_resampled_full = pca.inverse_transform(X_resampled)
            
            # Organize by categories
            balanced_data = defaultdict(list)
            for img_flat, label in zip(X_resampled_full, y_resampled):
                img = img_flat.reshape(256, 256).astype(np.uint8)
                balanced_data[label].append(img)
            
            return {
                'images': balanced_data,
                'method': 'ADASYN',
                'synthetic': True
            }
            
        except Exception as e:
            print(f"   ADASYN failed: {e}")
            return None
    
    def _apply_smote_enn(self):
        """Apply SMOTE + Edited Nearest Neighbours"""
        try:
            from sklearn.decomposition import PCA
            
            print("   Applying PCA for dimensionality reduction...")
            pca = PCA(n_components=30)  # Smaller for combined method
            X_reduced = pca.fit_transform(self.image_data)
            
            smote_enn = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smote_enn.fit_resample(X_reduced, self.labels)
            
            # Transform back to original space
            X_resampled_full = pca.inverse_transform(X_resampled)
            
            # Organize by categories
            balanced_data = defaultdict(list)
            for img_flat, label in zip(X_resampled_full, y_resampled):
                img = img_flat.reshape(256, 256).astype(np.uint8)
                balanced_data[label].append(img)
            
            return {
                'images': balanced_data,
                'method': 'SMOTE + ENN',
                'synthetic': True
            }
            
        except Exception as e:
            print(f"   SMOTE + ENN failed: {e}")
            return None
    
    def _calculate_class_weights(self):
        """Calculate class weights for use in training"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Calculate class weights
        unique_classes = np.unique(self.labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=self.labels
        )
        
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        return {
            'weights': class_weight_dict,
            'method': 'Class Weights',
            'usage': 'Use in model training loss function'
        }
    
    def _compare_balancing_strategies(self):
        """Compare different balancing strategies"""
        print("\n" + "="*60)
        print("BALANCING STRATEGIES COMPARISON")
        print("="*60)
        
        comparison_data = []
        
        for strategy_name, strategy_data in self.balanced_datasets.items():
            if strategy_data is None:
                continue
                
            if strategy_name == 'class_weights':
                print(f"\n{strategy_data['method']}:")
                print("  - Weights for training loss function")
                for class_name, weight in strategy_data['weights'].items():
                    print(f"    {class_name}: {weight:.3f}")
                continue
            
            print(f"\n{strategy_data['method']}:")
            
            total_samples = 0
            min_samples = float('inf')
            max_samples = 0
            
            for category in self.categories:
                if category in strategy_data['images']:
                    count = len(strategy_data['images'][category])
                    total_samples += count
                    min_samples = min(min_samples, count)
                    max_samples = max(max_samples, count)
                    print(f"  {category}: {count} samples")
            
            balance_ratio = max_samples / min_samples if min_samples > 0 else 0
            print(f"  Total samples: {total_samples}")
            print(f"  Balance ratio: {balance_ratio:.2f}:1")
            
            comparison_data.append({
                'Strategy': strategy_data['method'],
                'Total Samples': total_samples,
                'Balance Ratio': balance_ratio,
                'Min Samples': min_samples,
                'Max Samples': max_samples
            })
        
        # Create comparison DataFrame
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print(f"\nStrategy Comparison Summary:")
            print(df_comparison.to_string(index=False))
        
        return comparison_data
    
    def save_balanced_dataset(self, strategy_name, output_dir):
        """
        Save a balanced dataset to disk
        
        Args:
            strategy_name (str): Name of balancing strategy
            output_dir (str): Output directory path
        """
        if strategy_name not in self.balanced_datasets:
            print(f"Strategy '{strategy_name}' not found!")
            return
        
        strategy_data = self.balanced_datasets[strategy_name]
        if strategy_data is None:
            print(f"Strategy '{strategy_name}' failed to generate data!")
            return
        
        output_path = Path(output_dir) / f"balanced_{strategy_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {strategy_name} balanced dataset to {output_path}")
        
        save_log = []
        
        for category, images in strategy_data['images'].items():
            category_path = output_path / category
            category_path.mkdir(exist_ok=True)
            
            for i, img in enumerate(images):
                filename = f"{category}_{i:04d}.png"
                filepath = category_path / filename
                
                # Save image
                cv2.imwrite(str(filepath), img)
                
                save_log.append({
                    'category': category,
                    'filename': filename,
                    'index': i
                })
        
        # Save metadata
        metadata = {
            'strategy': strategy_data['method'],
            'total_images': len(save_log),
            'categories': list(strategy_data['images'].keys()),
            'samples_per_category': {cat: len(imgs) for cat, imgs in strategy_data['images'].items()}
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(save_log)} images")
        print(f"Metadata saved to {output_path / 'metadata.json'}")
    
    def recommend_strategy(self):
        """
        Recommend the best balancing strategy based on dataset characteristics
        """
        analysis = self.analyze_class_distribution()
        
        print("\n" + "="*60)
        print("BALANCING STRATEGY RECOMMENDATION")
        print("="*60)
        
        imbalance_ratio = analysis['imbalance_ratio']
        total_samples = analysis['total_samples']
        severity = analysis['severity']
        
        print(f"Dataset characteristics:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Imbalance ratio: {imbalance_ratio:.2f}:1")
        print(f"  - Severity: {severity}")
        
        recommendations = []
        
        if severity == "Mild":
            recommendations.append("Class Weights - Sufficient for mild imbalance")
            recommendations.append("Random Oversampling - Simple and effective")
        
        elif severity == "Moderate":
            recommendations.append("SMOTE - Good for moderate imbalance")
            recommendations.append("Class Weights - Can handle moderate imbalance")
            recommendations.append("Random Oversampling - Backup option")
        
        elif severity == "Severe":
            recommendations.append("SMOTE - Preferred for severe imbalance")
            recommendations.append("ADASYN - Adaptive approach for severe cases")
            recommendations.append("SMOTE + ENN - Combined approach")
        
        else:  # Extreme
            recommendations.append("SMOTE + ENN - Best for extreme imbalance")
            recommendations.append("ADASYN - Adaptive approach")
            recommendations.append("Consider collecting more data for minority classes")
        
        print(f"\nRecommended strategies (in order of preference):")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return recommendations


# Usage example
def main():
    # Initialize analyzer
    enhanced_data_path = "/home/kherad/AlirezaMottaghi/Thyroid/ProcessedData/enhanced"
    analyzer = ThyroidDataAnalyzer(enhanced_data_path)
    
    # Step 1: Load and analyze data
    print("Step 1: Loading enhanced images...")
    analyzer.load_enhanced_images()
    
    # Step 2: Analyze class distribution
    print("\nStep 2: Analyzing class distribution...")
    analysis_results = analyzer.analyze_class_distribution()
    
    # Step 3: Visualize distribution
    print("\nStep 3: Creating visualizations...")
    analyzer.visualize_class_distribution()
    
    # Step 4: Create balanced datasets
    print("\nStep 4: Creating balanced datasets...")
    analyzer.create_balanced_datasets()
    
    # Step 5: Get recommendations
    print("\nStep 5: Getting strategy recommendations...")
    recommendations = analyzer.recommend_strategy()
    
    # Step 6: Save recommended balanced dataset
    print("\nStep 6: Saving recommended balanced dataset...")
    # Save the top recommended strategy
    if 'smote' in analyzer.balanced_datasets and analyzer.balanced_datasets['smote'] is not None:
        analyzer.save_balanced_dataset('smote', "/home/kherad/AlirezaMottaghi/Thyroid/ProcessedData/balanced")
    else:
        analyzer.save_balanced_dataset('random_oversample', "/home/kherad/AlirezaMottaghi/Thyroid/ProcessedData/balanced")
    
    print("\nClass balancing analysis completed!")

if __name__ == "__main__":
    main()