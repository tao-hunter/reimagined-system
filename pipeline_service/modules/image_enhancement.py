"""
Image Enhancement Module for improving image quality before 3D generation.
Focuses on sharpness, color vibrancy, and detail preservation.
"""

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Optional
from logger_config import logger


class ImageEnhancer:
    """Enhances images for better 3D generation quality."""
    
    def __init__(self):
        pass
    
    def enhance_for_3d(
        self, 
        image: Image.Image,
        sharpen_factor: float = 1.3,
        color_factor: float = 1.15,
        contrast_factor: float = 1.1,
        apply_unsharp: bool = True
    ) -> Image.Image:
        """
        Enhance image quality for better 3D generation.
        
        Args:
            image: Input PIL Image
            sharpen_factor: Sharpness enhancement (1.0 = no change, >1.0 = sharper)
            color_factor: Color saturation (1.0 = no change, >1.0 = more vibrant)
            contrast_factor: Contrast enhancement (1.0 = no change, >1.0 = more contrast)
            apply_unsharp: Apply unsharp mask for edge enhancement
            
        Returns:
            Enhanced PIL Image
        """
        try:
            enhanced = image.convert('RGB')
            
            # 1. Sharpen for better edge definition (critical for shape)
            if sharpen_factor != 1.0:
                sharpener = ImageEnhance.Sharpness(enhanced)
                enhanced = sharpener.enhance(sharpen_factor)
            
            # 2. Enhance color saturation (critical for color/texture)
            if color_factor != 1.0:
                color_enhancer = ImageEnhance.Color(enhanced)
                enhanced = color_enhancer.enhance(color_factor)
            
            # 3. Adjust contrast for better detail visibility
            if contrast_factor != 1.0:
                contrast_enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = contrast_enhancer.enhance(contrast_factor)
            
            # 4. Apply unsharp mask for edge enhancement (improves shape quality)
            if apply_unsharp:
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            logger.info(f"Image enhanced: sharpen={sharpen_factor}, color={color_factor}, contrast={contrast_factor}")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}, returning original")
            return image
    
    def enhance_colors_for_matching(self, image: Image.Image) -> Image.Image:
        """
        Specifically enhance colors to improve matching with original image.
        More aggressive color enhancement for better texture quality.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Color-enhanced PIL Image
        """
        try:
            enhanced = image.convert('RGB')
            
            # Convert to numpy for advanced processing
            img_array = np.array(enhanced, dtype=np.float32) / 255.0
            
            # Enhance color vibrancy using HSV
            from PIL import ImageColor
            img_hsv = enhanced.convert('HSV')
            h, s, v = img_hsv.split()
            
            # Increase saturation
            s_array = np.array(s, dtype=np.float32)
            s_array = np.clip(s_array * 1.25, 0, 255).astype(np.uint8)
            s = Image.fromarray(s_array, mode='L')
            
            # Slightly boost value for brightness
            v_array = np.array(v, dtype=np.float32)
            v_array = np.clip(v_array * 1.05, 0, 255).astype(np.uint8)
            v = Image.fromarray(v_array, mode='L')
            
            enhanced = Image.merge('HSV', (h, s, v)).convert('RGB')
            
            logger.info("Colors enhanced for better matching")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Color enhancement failed: {e}, returning original")
            return image
    
    def prepare_for_trellis(self, image: Image.Image) -> Image.Image:
        """
        Final preparation of image before sending to Trellis.
        Optimizes for maximum 3D quality.
        
        Args:
            image: Input PIL Image (already background-removed)
            
        Returns:
            Optimized PIL Image ready for Trellis
        """
        # Apply conservative enhancements that won't distort geometry
        return self.enhance_for_3d(
            image,
            sharpen_factor=1.2,  # Moderate sharpening
            color_factor=1.1,     # Slight color boost
            contrast_factor=1.05, # Minimal contrast adjustment
            apply_unsharp=True    # Edge enhancement
        )

