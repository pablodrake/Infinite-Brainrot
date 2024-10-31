import time
import logging
from pathlib import Path
import datetime
from seleniumbase import SB
import torch
from TTS.api import TTS
from main import (
    initialize_sb, 
    get_reddit_post_text, 
    process_video, 
    initialize_whisper_model,
    SUBREDDIT_DEFAULT_URL,
    CHROME_USER_DATA_DIR
)
from pytok import TikTokUploader, TikTokVideo
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./monitor.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RedditMonitor:
    def __init__(self):
        self.processed_posts_file = Path("processed_posts.json")
        self.processed_posts = self.load_processed_posts()
        self.tts = None
        self.uploader = TikTokUploader(cookies_path='cookies.txt', headless=False)
        
    def load_processed_posts(self) -> set:
        """Load processed posts from JSON file"""
        try:
            if self.processed_posts_file.exists():
                with open(self.processed_posts_file, 'r') as f:
                    return set(json.load(f))
            return set()
        except Exception as e:
            logger.error(f"Error loading processed posts: {str(e)}")
            return set()
    
    def save_processed_posts(self):
        """Save processed posts to JSON file"""
        try:
            with open(self.processed_posts_file, 'w') as f:
                json.dump(list(self.processed_posts), f)
        except Exception as e:
            logger.error(f"Error saving processed posts: {str(e)}")

    def initialize_tts(self):
        if self.tts is None:
            logger.info("Loading TTS model...")
            self.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2"
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return self.tts

    def process_new_posts(self):
        # Set up directories based on current date
        date_today = datetime.date.today().strftime("%Y-%m-%d")
        output_dir = Path(f"./output/{date_today}")
        final_dir = output_dir / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        try:
            with initialize_sb() as sb:
                # Get latest posts
                posts = get_reddit_post_text(SUBREDDIT_DEFAULT_URL, sb, posts_number=5)
                
                new_posts = []
                for post in posts:
                    if post['title'] not in self.processed_posts:
                        new_posts.append(post)
                        self.processed_posts.add(post['title'])
                        # Save immediately after adding each post
                        self.save_processed_posts()

                if not new_posts:
                    logger.info("No new posts found")
                    return

                logger.info(f"Found {len(new_posts)} new posts")
                
                # Process each new post
                final_videos = []
                for i, post in enumerate(new_posts, 1):
                    try:
                        # Generate audio
                        output_file_name = f"{i}.wav"
                        audio_path = output_dir / f"aitah_audio_{output_file_name}"
                        
                        # Generate voice
                        self.initialize_tts()
                        self.tts.tts_to_file(
                            text=post['text'],
                            speed=2,
                            file_path=str(audio_path),
                            speaker_wav=["voices/xd.wav"],
                            language="en"
                        )

                        # Store the title
                        title_file = output_dir / f"title_{i}.txt"
                        with open(title_file, "w") as f:
                            f.write(post['title'])

                        # Process video
                        final_video = process_video(str(output_dir), str(audio_path), i)
                        if final_video:
                            final_videos.append(final_video)

                    except Exception as e:
                        logger.error(f"Error processing post {i}: {str(e)}")
                        continue

                # Upload videos if any were generated
                if final_videos:
                    videos = [
                        TikTokVideo(
                            file_path=video_path,
                            description="Follow for more AITA stories!",
                            tags=["aita", "AITAH", "reddit", "redditstories", 
                                 "storytime", "minecraft", "relationship", 
                                 "reddit_tiktok", "minecraftmemes"]
                        ) for video_path in final_videos
                    ]
                    
                    with self.uploader.initialize_sb() as sb:
                        self.uploader.upload_to_tiktok(videos, sb=sb)

        except Exception as e:
            logger.error(f"Error in process_new_posts: {str(e)}")

    def run(self, check_interval=300):  # 5 minutes default
        logger.info("Starting Reddit monitor...")
        initialize_whisper_model()

        while True:
            try:
                self.process_new_posts()
                logger.info(f"Sleeping for {check_interval} seconds...")
                time.sleep(check_interval)
            except KeyboardInterrupt:
                logger.info("Monitor stopped by user")
                self.save_processed_posts()  # Save before exiting
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    monitor = RedditMonitor()
    monitor.run()