# image_processing/utils/glasses_dataset_analyzer.py
"""
Glasses dataset analyzer for understanding the glasses dataset structure
Analyzes glasses images from multiple angles and creates enhanced metadata
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.path_utils import ProjectPaths
from image_processing.utils.image_utils import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlassesDatasetAnalyzer:
    """Analyzes and enhances glasses dataset with angle and style information"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the analyzer
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.paths = ProjectPaths()
        self.image_processor = ImageProcessor()
        
        # Glass style categories
        self.style_keywords = {
            'round': ['round', 'circular', 'circle'],
            'square': ['square', 'rectangular', 'rectangle'],
            'aviator': ['aviator', 'pilot', 'teardrop'],
            'cat_eye': ['cat', 'eye', 'butterfly'],
            'wayfarer': ['wayfarer', 'clubmaster'],
            'oversized': ['oversized', 'large', 'big'],
            'sport': ['sport', 'athletic', 'wrap'],
            'vintage': ['vintage', 'retro', 'classic'],
            'rimless': ['rimless', 'borderless'],
            'half_rim': ['half', 'semi']
        }
        
        # Frame materials
        self.material_keywords = {
            'metal': ['metal', 'steel', 'titanium', 'aluminum'],
            'plastic': ['plastic', 'acetate', 'nylon'],
            'wood': ['wood', 'bamboo'],
            'carbon': ['carbon', 'fiber']
        }
        
        logger.info("GlassesDatasetAnalyzer initialized")
    
    def create_enhanced_glasses_table(self) -> bool:
        """
        Create enhanced glasses metadata table
        
        Returns:
            True if successful, False otherwise
        """
        from sqlalchemy import text
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.db_manager.schema}.glasses_enhanced (
            id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(255),
            brand VARCHAR(100),
            model VARCHAR(100),
            style_category VARCHAR(50),
            frame_material VARCHAR(50),
            lens_type VARCHAR(50),
            
            -- Image analysis
            total_images INTEGER DEFAULT 0,
            front_view_available BOOLEAN DEFAULT FALSE,
            side_view_available BOOLEAN DEFAULT FALSE,
            angle_45_available BOOLEAN DEFAULT FALSE,
            
            -- Technical specifications
            frame_width_mm INTEGER,
            lens_width_mm INTEGER,
            bridge_width_mm INTEGER,
            temple_length_mm INTEGER,
            
            -- Color analysis
            primary_color VARCHAR(50),
            secondary_color VARCHAR(50),
            color_hex VARCHAR(7),
            
            -- Quality metrics
            image_quality_score FLOAT,
            transparency_quality FLOAT,
            background_removed BOOLEAN DEFAULT FALSE,
            
            -- URLs and paths
            main_image_url TEXT,
            additional_image_urls TEXT[],
            best_front_view_url TEXT,
            best_side_view_url TEXT,
            
            -- Processing status
            analysis_completed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Foreign key to original frames table
            FOREIGN KEY (id) REFERENCES {self.db_manager.schema}.frames(id)
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_glasses_enhanced_style ON {self.db_manager.schema}.glasses_enhanced(style_category);
        CREATE INDEX IF NOT EXISTS idx_glasses_enhanced_brand ON {self.db_manager.schema}.glasses_enhanced(brand);
        CREATE INDEX IF NOT EXISTS idx_glasses_enhanced_material ON {self.db_manager.schema}.glasses_enhanced(frame_material);
        CREATE INDEX IF NOT EXISTS idx_glasses_enhanced_quality ON {self.db_manager.schema}.glasses_enhanced(image_quality_score);
        """
        
        try:
            with self.db_manager.engine.connect() as connection:
                for statement in create_table_sql.split(';'):
                    if statement.strip():
                        connection.execute(text(statement))
                connection.commit()
                
            logger.info("✅ Enhanced glasses table created successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create enhanced glasses table: {e}")
            return False
    
    def analyze_glasses_from_database(self, limit: Optional[int] = None) -> Dict:
        """
        Analyze glasses dataset from database
        
        Args:
            limit: Maximum number of glasses to analyze
            
        Returns:
            Analysis results
        """
        try:
            logger.info("Analyzing glasses dataset from database...")
            
            # Get glasses data
            glasses_df = self.db_manager.get_glasses_data(limit=limit)
            
            if glasses_df.empty:
                return {'error': 'No glasses data found in database'}
            
            # Create enhanced table
            if not self.create_enhanced_glasses_table():
                return {'error': 'Failed to create enhanced table'}
            
            results = {
                'total_glasses': len(glasses_df),
                'analyzed': 0,
                'failed': 0,
                'style_distribution': {},
                'brand_distribution': {},
                'material_distribution': {},
                'image_analysis': {
                    'total_images_found': 0,
                    'avg_images_per_glasses': 0,
                    'front_view_rate': 0,
                    'side_view_rate': 0
                }
            }
            
            logger.info(f"Analyzing {len(glasses_df)} glasses...")
            
            for idx, row in tqdm(glasses_df.iterrows(), total=len(glasses_df), desc="Analyzing glasses"):
                try:
                    analysis = self._analyze_single_glasses(row)
                    
                    if analysis['success']:
                        results['analyzed'] += 1
                        
                        # Update distributions
                        style = analysis['style_category']
                        if style:
                            results['style_distribution'][style] = results['style_distribution'].get(style, 0) + 1
                        
                        brand = analysis['brand']
                        if brand:
                            results['brand_distribution'][brand] = results['brand_distribution'].get(brand, 0) + 1
                        
                        material = analysis['frame_material']
                        if material:
                            results['material_distribution'][material] = results['material_distribution'].get(material, 0) + 1
                        
                        # Update image analysis
                        results['image_analysis']['total_images_found'] += analysis['total_images']
                        
                    else:
                        results['failed'] += 1
                        
                except Exception as e:
                    logger.debug(f"Failed to analyze glasses {row.get('id', 'unknown')}: {e}")
                    results['failed'] += 1
            
            # Calculate final statistics
            if results['analyzed'] > 0:
                results['image_analysis']['avg_images_per_glasses'] = results['image_analysis']['total_images_found'] / results['analyzed']
            
            logger.info(f"✅ Glasses analysis completed: {results['analyzed']}/{results['total_glasses']} successful")
            return results
            
        except Exception as e:
            logger.error(f"❌ Glasses analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_single_glasses(self, glasses_row) -> Dict:
        """
        Analyze a single glasses item
        
        Args:
            glasses_row: Database row for glasses
            
        Returns:
            Analysis results for single glasses
        """
        try:
            glasses_id = glasses_row.get('id', '')
            title = glasses_row.get('title', '')
            main_image_url = glasses_row.get('main_image', '')
            additional_images = glasses_row.get('additional_images', '')
            
            # Parse title for brand and model
            brand, model = self._parse_title(title)
            
            # Determine style category
            style_category = self._determine_style_category(title)
            
            # Determine frame material
            frame_material = self._determine_frame_material(title)
            
            # Analyze images
            image_urls = [main_image_url]
            if additional_images:
                additional_urls = [url.strip() for url in additional_images.strip('{}').split(',') if url.strip()]
                image_urls.extend(additional_urls)
            
            image_analysis = self._analyze_glasses_images(image_urls)
            
            # Extract color information
            color_info = self._extract_color_info(title)
            
            # Store enhanced data
            enhanced_data = {
                'id': glasses_id,
                'title': title,
                'brand': brand,
                'model': model,
                'style_category': style_category,
                'frame_material': frame_material,
                'total_images': len(image_urls),
                'front_view_available': image_analysis['has_front_view'],
                'side_view_available': image_analysis['has_side_view'],
                'angle_45_available': image_analysis['has_45_angle'],
                'primary_color': color_info['primary'],
                'secondary_color': color_info['secondary'],
                'image_quality_score': image_analysis['avg_quality'],
                'main_image_url': main_image_url,
                'additional_image_urls': image_urls[1:] if len(image_urls) > 1 else [],
                'best_front_view_url': image_analysis['best_front_view'],
                'best_side_view_url': image_analysis['best_side_view'],
                'analysis_completed': True
            }
            
            # Insert into enhanced table
            success = self._insert_enhanced_data(enhanced_data)
            
            return {
                'success': success,
                'brand': brand,
                'style_category': style_category,
                'frame_material': frame_material,
                'total_images': len(image_urls)
            }
            
        except Exception as e:
            logger.debug(f"Single glasses analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_title(self, title: str) -> Tuple[str, str]:
        """Parse brand and model from title"""
        try:
            parts = title.split()
            if len(parts) >= 2:
                brand = parts[0]
                model = ' '.join(parts[1:])
                return brand, model
            else:
                return title, ''
        except Exception:
            return '', ''
    
    def _determine_style_category(self, title: str) -> Optional[str]:
        """Determine style category from title"""
        title_lower = title.lower()
        
        for style, keywords in self.style_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return style
        
        return 'unknown'
    
    def _determine_frame_material(self, title: str) -> Optional[str]:
        """Determine frame material from title"""
        title_lower = title.lower()
        
        for material, keywords in self.material_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return material
        
        return 'unknown'
    
    def _extract_color_info(self, title: str) -> Dict[str, Optional[str]]:
        """Extract color information from title"""
        color_keywords = {
            'black': ['black', 'noir'],
            'brown': ['brown', 'marron', 'tortoise'],
            'gold': ['gold', 'golden', 'or'],
            'silver': ['silver', 'argent'],
            'blue': ['blue', 'bleu'],
            'red': ['red', 'rouge'],
            'green': ['green', 'vert'],
            'clear': ['clear', 'transparent'],
            'white': ['white', 'blanc']
        }
        
        title_lower = title.lower()
        colors_found = []
        
        for color, keywords in color_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    colors_found.append(color)
                    break
        
        return {
            'primary': colors_found[0] if colors_found else None,
            'secondary': colors_found[1] if len(colors_found) > 1 else None
        }
    
    def _analyze_glasses_images(self, image_urls: List[str]) -> Dict:
        """Analyze glasses images for angles and quality"""
        analysis = {
            'has_front_view': False,
            'has_side_view': False,
            'has_45_angle': False,
            'avg_quality': 0.0,
            'best_front_view': None,
            'best_side_view': None,
            'image_scores': []
        }
        
        if not image_urls:
            return analysis
        
        # For now, use heuristics based on image URLs and simple analysis
        # In a full implementation, you'd use computer vision to detect angles
        
        for i, url in enumerate(image_urls):
            # Simple heuristics for angle detection
            score = 0.5  # Default quality score
            
            # Assume first image is usually front view
            if i == 0:
                analysis['has_front_view'] = True
                analysis['best_front_view'] = url
            
            # Assume side views are often in later images
            if i > 0 and i < len(image_urls):
                if not analysis['has_side_view']:
                    analysis['has_side_view'] = True
                    analysis['best_side_view'] = url
            
            # If multiple images, assume some are different angles
            if len(image_urls) >= 3:
                analysis['has_45_angle'] = True
            
            analysis['image_scores'].append(score)
        
        # Calculate average quality
        if analysis['image_scores']:
            analysis['avg_quality'] = sum(analysis['image_scores']) / len(analysis['image_scores'])
        
        return analysis
    
    def _insert_enhanced_data(self, data: Dict) -> bool:
        """Insert enhanced data into database"""
        try:
            from sqlalchemy import text
            
            insert_sql = f"""
            INSERT INTO {self.db_manager.schema}.glasses_enhanced 
            (id, title, brand, model, style_category, frame_material, total_images,
             front_view_available, side_view_available, angle_45_available,
             primary_color, secondary_color, image_quality_score,
             main_image_url, additional_image_urls, best_front_view_url, best_side_view_url,
             analysis_completed)
            VALUES (:id, :title, :brand, :model, :style_category, :frame_material, :total_images,
                    :front_view_available, :side_view_available, :angle_45_available,
                    :primary_color, :secondary_color, :image_quality_score,
                    :main_image_url, :additional_image_urls, :best_front_view_url, :best_side_view_url,
                    :analysis_completed)
            ON CONFLICT (id) DO UPDATE SET
                brand = EXCLUDED.brand,
                model = EXCLUDED.model,
                style_category = EXCLUDED.style_category,
                frame_material = EXCLUDED.frame_material,
                total_images = EXCLUDED.total_images,
                front_view_available = EXCLUDED.front_view_available,
                side_view_available = EXCLUDED.side_view_available,
                angle_45_available = EXCLUDED.angle_45_available,
                primary_color = EXCLUDED.primary_color,
                secondary_color = EXCLUDED.secondary_color,
                image_quality_score = EXCLUDED.image_quality_score,
                main_image_url = EXCLUDED.main_image_url,
                additional_image_urls = EXCLUDED.additional_image_urls,
                best_front_view_url = EXCLUDED.best_front_view_url,
                best_side_view_url = EXCLUDED.best_side_view_url,
                analysis_completed = EXCLUDED.analysis_completed,
                updated_at = CURRENT_TIMESTAMP
            """
            
            with self.db_manager.engine.connect() as connection:
                connection.execute(text(insert_sql), data)
                connection.commit()
            
            return True
            
        except Exception as e:
            logger.debug(f"Failed to insert enhanced data: {e}")
            return False
    
    def get_enhanced_glasses_stats(self) -> Dict:
        """Get statistics about enhanced glasses dataset"""
        try:
            from sqlalchemy import text
            
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_analyzed,
                COUNT(CASE WHEN style_category != 'unknown' THEN 1 END) as categorized_styles,
                COUNT(CASE WHEN frame_material != 'unknown' THEN 1 END) as categorized_materials,
                COUNT(CASE WHEN front_view_available = true THEN 1 END) as front_views,
                COUNT(CASE WHEN side_view_available = true THEN 1 END) as side_views,
                COUNT(CASE WHEN angle_45_available = true THEN 1 END) as angle_45_views,
                AVG(total_images) as avg_images_per_glasses,
                AVG(image_quality_score) as avg_quality_score,
                COUNT(DISTINCT brand) as unique_brands,
                COUNT(DISTINCT style_category) as unique_styles
            FROM {self.db_manager.schema}.glasses_enhanced
            WHERE analysis_completed = true
            """
            
            with self.db_manager.engine.connect() as connection:
                result = connection.execute(text(stats_sql)).fetchone()
                
                if result and result[0] > 0:
                    return {
                        'total_analyzed': result[0],
                        'categorized_styles': result[1],
                        'categorized_materials': result[2],
                        'front_views': result[3],
                        'side_views': result[4],
                        'angle_45_views': result[5],
                        'avg_images_per_glasses': round(result[6], 1) if result[6] else 0,
                        'avg_quality_score': round(result[7], 2) if result[7] else 0,
                        'unique_brands': result[8],
                        'unique_styles': result[9],
                        'front_view_rate': result[3] / result[0] if result[0] > 0 else 0,
                        'side_view_rate': result[4] / result[0] if result[0] > 0 else 0
                    }
                else:
                    return {'message': 'No analyzed glasses found'}
        
        except Exception as e:
            logger.error(f"Failed to get enhanced glasses stats: {e}")
            return {'error': str(e)}
    
    def export_analysis_report(self, output_path: Optional[Path] = None) -> Path:
        """Export comprehensive analysis report"""
        try:
            if output_path is None:
                output_path = self.paths.logs_dir / f"glasses_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Get comprehensive statistics
            stats = self.get_enhanced_glasses_stats()
            
            # Get distribution data
            from sqlalchemy import text
            
            # Style distribution
            style_sql = f"SELECT style_category, COUNT(*) FROM {self.db_manager.schema}.glasses_enhanced GROUP BY style_category ORDER BY COUNT(*) DESC"
            brand_sql = f"SELECT brand, COUNT(*) FROM {self.db_manager.schema}.glasses_enhanced GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 20"
            
            with self.db_manager.engine.connect() as connection:
                style_results = connection.execute(text(style_sql)).fetchall()
                brand_results = connection.execute(text(brand_sql)).fetchall()
            
            report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'summary_stats': stats,
                'style_distribution': {row[0]: row[1] for row in style_results},
                'top_brands': {row[0]: row[1] for row in brand_results},
                'analysis_metadata': {
                    'analyzer_version': '1.0',
                    'total_categories': len(self.style_keywords),
                    'material_categories': len(self.material_keywords)
                }
            }
            
            # Save report
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ Analysis report exported: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Failed to export analysis report: {e}")
            return None


# Convenience functions
def analyze_glasses_dataset(limit: Optional[int] = None) -> Dict:
    """
    Convenience function to analyze glasses dataset
    
    Args:
        limit: Maximum number of glasses to analyze
        
    Returns:
        Analysis results
    """
    analyzer = GlassesDatasetAnalyzer()
    return analyzer.analyze_glasses_from_database(limit=limit)

# Example usage and testing
if __name__ == "__main__":
    print("Testing GlassesDatasetAnalyzer...")
    
    # Initialize analyzer
    analyzer = GlassesDatasetAnalyzer()
    
    # Test database connection
    if analyzer.db_manager.test_connection():
        print("✅ Database connection successful")
        
        # Test analysis with small sample
        print("Analyzing 5 sample glasses...")
        results = analyzer.analyze_glasses_from_database(limit=5)
        
        if 'error' not in results:
            print(f"✅ Analysis completed: {results['analyzed']}/{results['total_glasses']} glasses")
            print(f"   Style distribution: {results['style_distribution']}")
            print(f"   Brand distribution: {results['brand_distribution']}")
            
            # Get enhanced stats
            stats = analyzer.get_enhanced_glasses_stats()
            if 'error' not in stats:
                print(f"✅ Enhanced stats: {stats['total_analyzed']} analyzed")
                print(f"   Front view rate: {stats.get('front_view_rate', 0):.1%}")
                print(f"   Unique brands: {stats.get('unique_brands', 0)}")
        else:
            print(f"❌ Analysis failed: {results['error']}")
        
        print("✅ GlassesDatasetAnalyzer test completed")
    else:
        print("❌ Database connection failed")