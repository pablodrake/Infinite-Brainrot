import logging
import os
from dataclasses import dataclass
from typing import List, Optional
from seleniumbase import SB
from selenium.webdriver.common.keys import Keys

logger = logging.getLogger(__name__)

@dataclass
class TikTokVideo:
    """Data class to hold video information"""
    file_path: str
    description: str
    tags: List[str]

    @property
    def title(self) -> str:
        """Extract title from filename without extension"""
        return os.path.splitext(os.path.basename(self.file_path))[0]

class TikTokUploader:
    """Class to handle TikTok upload functionality"""
    def __init__(self, cookies_path: str, headless: bool = False):
        self.cookies_path = cookies_path
        self.headless = headless
        
    def initialize_sb(self) -> SB:
        """Initialize and return a SeleniumBase driver instance"""
        return SB(uc=True, headless=self.headless)

    def authenticate_with_cookies_file(self, driver, cookies_file_path: str):
        """
        Authenticates a browser session using a cookies.txt file in Netscape format
        
        Args:
            driver: Selenium WebDriver instance
            cookies_file_path (str): Path to the cookies.txt file
        
        Returns:
            WebDriver: The authenticated driver instance
        """
        # Load cookies from file
        with open(cookies_file_path, "r", encoding="utf-8") as file:
            lines = file.read().split("\n")
        
        # Parse cookies
        cookies = []
        for line in lines:
            split = line.split('\t')
            if len(split) < 6:
                continue

            split = [x.strip() for x in split]
            
            try:
                split[4] = int(split[4])
            except ValueError:
                split[4] = None

            cookie = {
                'name': split[5],
                'value': split[6],
                'domain': split[0],
                'path': split[2],
            }
            
            if split[4]:
                cookie['expiry'] = split[4]
                
            cookies.append(cookie)
        
        # Navigate to TikTok main page
        driver.open('https://www.tiktok.com')
        
        # Add cookies to browser session
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception as e:
                logger.error('Failed to add cookie %s: %s', cookie, str(e))
        
        # Refresh page to apply cookies
        driver.refresh()
        
        return driver

    def upload_to_tiktok(self, videos: List[TikTokVideo], sb: Optional[SB] = None):
        """
        Uploads videos to TikTok using browser automation
        
        Args:
            videos (List[TikTokVideo]): List of TikTokVideo objects containing video information
            sb (Optional[SB]): Optional SeleniumBase instance
        """
        try:
            if sb is None:
                sb = self.initialize_sb()
                
            self.authenticate_with_cookies_file(sb, self.cookies_path)
            
            for video in videos:
                # Navigate to upload page
                sb.open('https://www.tiktok.com/upload')
                
                # Wait for the upload container to be present
                sb.wait_for_element_present("css selector", "div.upload-card.before-upload-new-stage")
                
                # Show hidden file choosers
                sb.show_file_choosers()
                
                # Find and use the file input
                file_input = 'input[type="file"]'
                sb.choose_file(file_input, video.file_path)
                
                # Wait for upload to complete
                sb.wait_for_element_not_present("css selector", ".upload-loading", timeout=60)
                
                # Add description and tags
                caption_input = "div[contenteditable='true']"
                sb.wait_for_element_present("css selector", caption_input)
                
                # Add the title and description
                full_text = f"{video.title}\n{video.description}"
                sb.send_keys(caption_input, full_text)
                
                # Add tags
                for tag in video.tags:
                    sb.send_keys(caption_input, f" #{tag}")
                    # Wait for hashtag suggestion container and first suggestion
                    sb.wait_for_element_present("css selector", ".hashtag-suggestion-item.jsx-937316377", timeout=120)
                    sb.send_keys(caption_input, Keys.ENTER)
                
                # Find and click the Post button
                post_button = "button.TUXButton.TUXButton--default.TUXButton--large.TUXButton--primary"
                sb.wait_for_element_clickable(post_button, timeout=120)
                sb.click(post_button)
                
                # Wait for upload confirmation modal
                sb.wait_for_element_present("css selector", "div.jsx-1540291114.common-modal-header", timeout=120)
                
        finally:
            if sb and not isinstance(sb, SB):
                sb.__exit__(None, None, None)  # Properly close the browser if we created it

# Example usage
if __name__ == "__main__":
    # Example video data
    videos = [
        TikTokVideo(
            file_path="test.mp4",
            description="Follow for more AITA stories!",
            tags=["aita", "AITAH", "reddit", "redditstories", "storytime", 
                 "minecraft", "relationship", "reddit_tiktok", "minecraftmemes"]
        )
    ]
    
    # Initialize uploader
    uploader = TikTokUploader(cookies_path='cookies.txt', headless=False)
    
    # Upload videos
    with uploader.initialize_sb() as sb:
        uploader.upload_to_tiktok(videos, sb=sb)