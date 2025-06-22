# scripts/analyze_datasets.py
"""
Comprehensive dataset analysis for selfies and glasses
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from typing import Dict, List, Any, Tuple
import logging
from tqdm import tqdm
import json

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.path_utils import ProjectPaths
from image_processing.utils.image_utils import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """Analyzes selfies and glasses datasets comprehensively"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.db_manager = DatabaseManager()
        self.paths = ProjectPaths()
        self.image_processor = ImageProcessor()
        
        logger.info("DatasetAnalyzer initialized")
    
    def analyze_selfies_dataset(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of selfies dataset
        
        Returns:
            Analysis results
        """
        logger.info("🔍 Analyzing selfies dataset...")
        
        # Get data from database
        selfies_df = self.db_manager.get_selfies_data()
        
        if selfies_df.empty:
            return {
                'status': 'empty',
                'message': 'No selfies found in database',
                'recommendation': 'Run download-selfies command first'
            }
        
        analysis = {
            'basic_stats': self._analyze_basic_selfie_stats(selfies_df),
            'image_quality': self._analyze_selfie_quality(selfies_df),
            'face_detection': self._analyze_face_detection_rate(selfies_df),
            'data_distribution': self._analyze_selfie_distribution(selfies_df),
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        analysis['recommendations'] = self._generate_selfie_recommendations(analysis)
        
        logger.info("✅ Selfies analysis completed")
        return analysis
    
    def analyze_glasses_dataset(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of glasses dataset
        
        Returns:
            Analysis results
        """
        logger.info("🔍 Analyzing glasses dataset...")
        
        # Get data from database
        glasses_df = self.db_manager.get_glasses_data()
        
        if glasses_df.empty:
            return {
                'status': 'empty',
                'message': 'No glasses found in database',
                'recommendation': 'Check database connection'
            }
        
        analysis = {
            'basic_stats': self._analyze_basic_glasses_stats(glasses_df),
            'brand_analysis': self._analyze_glasses_brands(glasses_df),
            'image_analysis': self._analyze_glasses_images(),
            'angle_analysis': self._analyze_glasses_angles(),
            'data_quality': self._analyze_glasses_quality(glasses_df),
            'recommendations': []
        }
        
        # Add recommendations
        analysis['recommendations'] = self._generate_glasses_recommendations(analysis)
        
        logger.info("✅ Glasses analysis completed")
        return analysis
    
    def _analyze_basic_selfie_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic selfie statistics"""
        stats = {
            'total_count': len(df),
            'file_size_stats': {
                'mean_kb': df['file_size_kb'].mean() if 'file_size_kb' in df.columns else 0,
                'median_kb': df['file_size_kb'].median() if 'file_size_kb' in df.columns else 0,
                'min_kb': df['file_size_kb'].min() if 'file_size_kb' in df.columns else 0,
                'max_kb': df['file_size_kb'].max() if 'file_size_kb' in df.columns else 0
            },
            'image_dimensions': {
                'mean_width': df['image_width'].mean() if 'image_width' in df.columns else 0,
                'mean_height': df['image_height'].mean() if 'image_height' in df.columns else 0,
                'width_range': [df['image_width'].min(), df['image_width'].max()] if 'image_width' in df.columns else [0, 0],
                'height_range': [df['image_height'].min(), df['image_height'].max()] if 'image_height' in df.columns else [0, 0]
            },
            'preprocessing_status': df['preprocessing_status'].value_counts().to_dict() if 'preprocessing_status' in df.columns else {}
        }
        return stats
    
    def _analyze_basic_glasses_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic glasses statistics"""
        stats = {
            'total_count': len(df),
            'unique_brands': df['title'].str.split().str[0].nunique() if 'title' in df.columns else 0,
            'published_status': df['published'].value_counts().to_dict() if 'published' in df.columns else {},
            'has_additional_images': (df['additional_images'].notna() & (df['additional_images'] != '')).sum() if 'additional_images' in df.columns else 0,
            'highlight_status': df['highlight'].value_counts().to_dict() if 'highlight' in df.columns else {}
        }
        return stats
    
    def _analyze_glasses_brands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze glasses brands and models"""
        if 'title' not in df.columns:
            return {'error': 'No title column found'}
        
        # Extract brand from title (first word)
        df['brand'] = df['title'].str.split().str[0]
        
        brand_analysis = {
            'top_brands': df['brand'].value_counts().head(10).to_dict(),
            'total_brands': df['brand'].nunique(),
            'brand_distribution': df['brand'].value_counts().describe().to_dict()
        }
        
        # Analyze model patterns
        model_patterns = {}
        for brand in df['brand'].value_counts().head(5).index:
            brand_titles = df[df['brand'] == brand]['title'].tolist()
            model_patterns[brand] = {
                'count': len(brand_titles),
                'sample_models': brand_titles[:3]
            }
        
        brand_analysis['model_patterns'] = model_patterns
        return brand_analysis
    
    def _analyze_glasses_images(self) -> Dict[str, Any]:
        """Analyze downloaded glasses images"""
        # Check raw glasses directories
        raw_dirs = [d for d in self.paths.raw_glasses_dir.iterdir() if d.is_dir()]
        processed_dirs = [d for d in self.paths.processed_glasses_dir.iterdir() if d.is_dir()]
        
        analysis = {
            'raw_directories': len(raw_dirs),
            'processed_directories': len(processed_dirs),
            'processing_rate': len(processed_dirs) / len(raw_dirs) if raw_dirs else 0,
            'sample_analysis': {}
        }
        
        # Analyze a few sample directories
        sample_dirs = raw_dirs[:3]
        for i, dir_path in enumerate(sample_dirs):
            images = list(dir_path.glob('*.*'))
            image_sizes = []
            
            for img_path in images[:5]:  # Sample first 5 images
                try:
                    img = self.image_processor.load_image(img_path)
                    if img is not None:
                        image_sizes.append(img.shape[:2])
                except:
                    continue
            
            analysis['sample_analysis'][f'sample_{i+1}'] = {
                'directory_id': dir_path.name,
                'image_count': len(images),
                'sample_sizes': image_sizes
            }
        
        return analysis
    
    def _analyze_glasses_angles(self) -> Dict[str, Any]:
        """Analyze glasses from different angles"""
        raw_dirs = [d for d in self.paths.raw_glasses_dir.iterdir() if d.is_dir()]
        
        angle_analysis = {
            'multi_angle_glasses': 0,
            'single_image_glasses': 0,
            'average_images_per_glasses': 0,
            'angle_distribution': {}
        }
        
        total_images = 0
        
        for dir_path in raw_dirs[:10]:  # Sample first 10 directories
            images = list(dir_path.glob('*.*'))
            total_images += len(images)
            
            if len(images) > 1:
                angle_analysis['multi_angle_glasses'] += 1
            else:
                angle_analysis['single_image_glasses'] += 1
        
        if raw_dirs:
            angle_analysis['average_images_per_glasses'] = total_images / min(len(raw_dirs), 10)
        
        # Common patterns in multi-angle glasses
        angle_analysis['patterns'] = {
            'typical_angles': ['front', 'side', 'three_quarter', 'detail'],
            'common_counts': [1, 2, 3, 4, 5],  # Common number of images per glasses
            'analysis_needed': 'Check if images show different angles or just different lighting'
        }
        
        return angle_analysis
    
    def _analyze_selfie_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze selfie image quality"""
        quality_analysis = {
            'face_detection_rate': 0,
            'size_distribution': {},
            'quality_scores': {}
        }
        
        if 'face_detected' in df.columns:
            quality_analysis['face_detection_rate'] = df['face_detected'].sum() / len(df)
        
        # Analyze file sizes as quality indicator
        if 'file_size_kb' in df.columns:
            quality_analysis['size_distribution'] = {
                'very_small': (df['file_size_kb'] < 50).sum(),  # Likely low quality
                'small': ((df['file_size_kb'] >= 50) & (df['file_size_kb'] < 200)).sum(),
                'medium': ((df['file_size_kb'] >= 200) & (df['file_size_kb'] < 500)).sum(),
                'large': (df['file_size_kb'] >= 500).sum()
            }
        
        return quality_analysis
    
    def _analyze_face_detection_rate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze face detection performance"""
        if 'face_detected' not in df.columns:
            return {'error': 'No face detection data available'}
        
        detection_stats = {
            'total_processed': len(df),
            'faces_detected': df['face_detected'].sum(),
            'detection_rate': df['face_detected'].mean(),
            'failed_detection': (~df['face_detected']).sum(),
            'by_preprocessing_status': {}
        }
        
        if 'preprocessing_status' in df.columns:
            for status in df['preprocessing_status'].unique():
                subset = df[df['preprocessing_status'] == status]
                detection_stats['by_preprocessing_status'][status] = {
                    'count': len(subset),
                    'detection_rate': subset['face_detected'].mean() if len(subset) > 0 else 0
                }
        
        return detection_stats
    
    def _analyze_selfie_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze selfie data distribution"""
        distribution = {}
        
        # Time distribution if available
        if 'upload_date' in df.columns:
            df['upload_date'] = pd.to_datetime(df['upload_date'])
            distribution['temporal'] = {
                'date_range': [df['upload_date'].min().isoformat(), df['upload_date'].max().isoformat()],
                'uploads_per_day': df.groupby(df['upload_date'].dt.date).size().describe().to_dict()
            }
        
        # Size distribution
        if 'image_width' in df.columns and 'image_height' in df.columns:
            df['aspect_ratio'] = df['image_width'] / df['image_height']
            distribution['dimensions'] = {
                'common_widths': df['image_width'].value_counts().head(5).to_dict(),
                'common_heights': df['image_height'].value_counts().head(5).to_dict(),
                'aspect_ratios': df['aspect_ratio'].describe().to_dict()
            }
        
        return distribution
    
    def _analyze_glasses_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze glasses dataset quality"""
        quality = {
            'completeness': {},
            'consistency': {},
            'usability': {}
        }
        
        # Completeness analysis
        quality['completeness'] = {
            'has_main_image': df['main_image'].notna().sum() if 'main_image' in df.columns else 0,
            'has_additional_images': (df['additional_images'].notna() & (df['additional_images'] != '')).sum() if 'additional_images' in df.columns else 0,
            'has_title': df['title'].notna().sum() if 'title' in df.columns else 0,
            'complete_records': len(df)  # Records with all essential fields
        }
        
        # URL validity (basic check)
        if 'main_image' in df.columns:
            valid_urls = df['main_image'].str.startswith('http').sum()
            quality['consistency']['valid_main_image_urls'] = valid_urls
        
        return quality
    
    def _generate_selfie_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for selfie dataset"""
        recommendations = []
        
        basic_stats = analysis.get('basic_stats', {})
        face_detection = analysis.get('face_detection', {})
        
        # Size recommendations
        if basic_stats.get('total_count', 0) < 100:
            recommendations.append("Dataset is small - consider downloading more selfies for better training")
        
        # Quality recommendations
        detection_rate = face_detection.get('detection_rate', 0)
        if detection_rate < 0.8:
            recommendations.append(f"Low face detection rate ({detection_rate:.1%}) - consider improving face detection or data quality")
        
        # Processing recommendations
        preprocessing = basic_stats.get('preprocessing_status', {})
        if 'pending' in preprocessing and preprocessing['pending'] > 0:
            recommendations.append("Some selfies need preprocessing - run selfie preprocessing pipeline")
        
        return recommendations
    
    def _generate_glasses_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for glasses dataset"""
        recommendations = []
        
        basic_stats = analysis.get('basic_stats', {})
        image_analysis = analysis.get('image_analysis', {})
        angle_analysis = analysis.get('angle_analysis', {})
        
        # Dataset size
        if basic_stats.get('total_count', 0) > 1000:
            recommendations.append("Large glasses dataset available - good for training diversity")
        
        # Processing rate
        processing_rate = image_analysis.get('processing_rate', 0)
        if processing_rate < 0.1:
            recommendations.append("Low glasses processing rate - run glasses preprocessing pipeline")
        
        # Multi-angle analysis
        multi_angle = angle_analysis.get('multi_angle_glasses', 0)
        single_angle = angle_analysis.get('single_image_glasses', 0)
        if multi_angle > single_angle:
            recommendations.append("Dataset has multi-angle glasses - consider using different angles for training")
        
        return recommendations
    
    def create_dataset_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Create comprehensive dataset analysis report
        
        Args:
            output_path: Path to save report
            
        Returns:
            Path to saved report
        """
        logger.info("📊 Creating comprehensive dataset report...")
        
        # Analyze both datasets
        selfies_analysis = self.analyze_selfies_dataset()
        glasses_analysis = self.analyze_glasses_dataset()
        
        # Create comprehensive report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'selfies_analysis': selfies_analysis,
            'glasses_analysis': glasses_analysis,
            'training_readiness': self._assess_training_readiness(selfies_analysis, glasses_analysis),
            'next_steps': self._generate_next_steps(selfies_analysis, glasses_analysis)
        }
        
        # Save report
        if output_path is None:
            output_path = self.paths.logs_dir / f"dataset_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Report saved to: {output_path}")
        
        # Print summary
        self._print_report_summary(report)
        
        return output_path
    
    def _assess_training_readiness(self, selfies_analysis: Dict, glasses_analysis: Dict) -> Dict[str, Any]:
        """Assess if datasets are ready for training"""
        readiness = {
            'selfies_ready': False,
            'glasses_ready': False,
            'overall_ready': False,
            'issues': [],
            'requirements_met': {}
        }
        
        # Check selfies readiness
        selfies_count = selfies_analysis.get('basic_stats', {}).get('total_count', 0)
        if selfies_count >= 100:
            readiness['selfies_ready'] = True
        else:
            readiness['issues'].append(f"Need more selfies (have {selfies_count}, need 100+)")
        
        # Check glasses readiness
        glasses_count = glasses_analysis.get('basic_stats', {}).get('total_count', 0)
        glasses_processed = glasses_analysis.get('image_analysis', {}).get('processed_directories', 0)
        
        if glasses_count >= 50 and glasses_processed >= 10:
            readiness['glasses_ready'] = True
        else:
            readiness['issues'].append(f"Need more processed glasses (have {glasses_processed}, need 10+)")
        
        # Overall readiness
        readiness['overall_ready'] = readiness['selfies_ready'] and readiness['glasses_ready']
        
        return readiness
    
    def _generate_next_steps(self, selfies_analysis: Dict, glasses_analysis: Dict) -> List[str]:
        """Generate next steps based on analysis"""
        steps = []
        
        # Selfies steps
        if selfies_analysis.get('status') == 'empty':
            steps.append("1. Download selfie dataset from Google Drive")
            steps.append("2. Run: python scripts/run_pipeline.py download-selfies --drive-url YOUR_URL")
        
        # Glasses steps
        processing_rate = glasses_analysis.get('image_analysis', {}).get('processing_rate', 0)
        if processing_rate < 0.5:
            steps.append("3. Process more glasses images")
            steps.append("4. Run: python scripts/run_pipeline.py preprocess-glasses --limit 50")
        
        # Training steps
        training_ready = self._assess_training_readiness(selfies_analysis, glasses_analysis)
        if training_ready['overall_ready']:
            steps.append("5. Start model training")
            steps.append("6. Run: python scripts/train_model.py --epochs 10 --batch-size 8")
        
        return steps
    
    def _print_report_summary(self, report: Dict[str, Any]):
        """Print report summary to console"""
        print("\n" + "="*60)
        print("📊 DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        # Selfies summary
        selfies = report['selfies_analysis']
        if selfies.get('status') != 'empty':
            selfies_count = selfies.get('basic_stats', {}).get('total_count', 0)
            detection_rate = selfies.get('face_detection', {}).get('detection_rate', 0)
            print(f"👤 SELFIES: {selfies_count} images, {detection_rate:.1%} face detection rate")
        else:
            print("👤 SELFIES: No selfies found - need to download dataset")
        
        # Glasses summary  
        glasses = report['glasses_analysis']
        if glasses.get('status') != 'empty':
            glasses_count = glasses.get('basic_stats', {}).get('total_count', 0)
            brands = glasses.get('brand_analysis', {}).get('total_brands', 0)
            processed = glasses.get('image_analysis', {}).get('processed_directories', 0)
            print(f"👓 GLASSES: {glasses_count} items, {brands} brands, {processed} processed")
        else:
            print("👓 GLASSES: No glasses found")
        
        # Training readiness
        readiness = report['training_readiness']
        status = "✅ READY" if readiness['overall_ready'] else "⚠️ NOT READY"
        print(f"🎯 TRAINING: {status}")
        
        # Issues
        if readiness['issues']:
            print("\n❌ ISSUES TO RESOLVE:")
            for issue in readiness['issues']:
                print(f"   • {issue}")
        
        # Next steps
        if report['next_steps']:
            print("\n📋 NEXT STEPS:")
            for step in report['next_steps']:
                print(f"   {step}")
        
        print("="*60)


def main():
    """Main analysis function"""
    analyzer = DatasetAnalyzer()
    
    # Create comprehensive report
    report_path = analyzer.create_dataset_report()
    
    print(f"\n📄 Full report saved to: {report_path}")
    print("🔍 Run this script anytime to check dataset status")

if __name__ == "__main__":
    main()