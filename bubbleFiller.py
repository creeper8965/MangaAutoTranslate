from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

class SpeechBubbleFiller:
    def __init__(self, 
                 background_color=(255, 255, 255), 
                 font_path='mangat.ttf', 
                 font_size=30, 
                 threshold=120, 
                 minfont=5):
        """
        Initialize the SpeechBubbleFiller with configuration options.
        
        Args:
            background_color: Color of the speech bubble background (default: white).
            font_path: Path to the font file.
            font_size: Size of the font.
            threshold: Threshold for contour detection.
            minfont: Minimum font size to use.
        """
        self.background_color = background_color
        self.font_path = font_path
        self.font_size = font_size
        self.threshold = threshold
        self.minfont = minfont

    def wrap_text(self, text, font, max_width):
        """Wrap text to fit within the specified width."""
        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            width, _ = self.textsize(test_line, font=font)

            if width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def calculate_text_height(self, draw, lines, font):
        """Calculate the total height of the text lines."""
        return sum(self.textsize(line, font=font)[1] for line in lines)

    def fill_speech_bubble(self, image, bbox, text=''):
        """Fill the speech bubble area inside the bounding box and add text."""
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2].copy()

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_roi, self.threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(roi)
        cv2.fillPoly(mask, contours, (1, 1, 1))
        background = np.full(roi.shape, self.background_color, dtype=np.uint8)
        roi_filled = np.where(mask == 1, background, roi)

        image[y1:y2, x1:x2] = roi_filled
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(self.font_path, self.font_size)

        max_text_width = x2 - x1 - 10
        wrapped_text = self.wrap_text(text, font, max_text_width)
        total_text_height = self.calculate_text_height(draw, wrapped_text, font)

        bubble_height = y2 - y1

        while total_text_height > bubble_height and self.font_size > self.minfont:
            self.font_size -= 1
            font = ImageFont.truetype(self.font_path, self.font_size)
            wrapped_text = self.wrap_text(text, font, max_text_width)
            total_text_height = self.calculate_text_height(draw, wrapped_text, font)

        text_y = y1 + (bubble_height - total_text_height) / 2

        for line in wrapped_text:
            text_width, text_height = self.textsize(line, font=font)
            text_x = x1 + (x2 - x1 - text_width) / 2
            draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font)
            text_y += text_height

        return np.array(pil_image)

    def textsize(self, text, font):
        """Get the size of the text when rendered with the given font."""
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height

    def process_image_with_bubbles(self, image, bboxes, texts):
        """Process the image to fill speech bubbles and add texts."""
        for bbox, text in zip(bboxes, texts):
            image = self.fill_speech_bubble(image, bbox, text=text)
        return image
