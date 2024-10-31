from altair import Key
from seleniumbase import SB
import html
from pathlib import Path
from TTS.api import TTS
import re
import torch
import datetime
import os
import subprocess
import random
from faster_whisper import WhisperModel
import logging
from selenium.webdriver.common.keys import Keys
from pytok import upload_to_tiktok

# Constants
NUM_VIDEOS_TO_GENERATE = 10
VIDEO_SOURCE_PATH = "./videos/minecraft.mp4"
CHROME_USER_DATA_DIR = "/home/pdaloxd/.config/google-chrome/Default"
SUBREDDIT_DEFAULT_URL = "https://www.reddit.com/r/AITAH/hot/"
VAAPI_DEVICE = "/dev/dri/renderD128"
WHISPER_MODEL = None
final_video_paths = []

# Video encoding settings
VIDEO_ENCODING = {
    'QP': '18',
    'BITRATE': '8M',
    'MAXRATE': '10M',
    'BUFSIZE': '16M'
}

# Subtitle styling
SUBTITLE_STYLE = (
    'Fontsize=24,Alignment=10,MarginV=50,'
    'PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,'
    'BorderStyle=1,Outline=2'
)

# Reddit acronym mappings
REDDIT_ACRONYMS = {
    "AITAH": "Am I The Ahole",
    "AITA": "Am I The Ahole",
    "WIBTA": "Would I Be The Ahole",
    "YTA": "You're The Ahole",
    "NTA": "Not The Ahole",
    "NAH": "No Aholes Here",
    "ESH": "Everyone Sucks Here",
    "TLDR": "Too Long Didn't Read",
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./video_generation.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up directories based on current date
date_today = datetime.date.today().strftime("%Y-%m-%d")
output_dir = Path(f"./output/{date_today}")
final_dir = output_dir / "final"
# Create the directories if they don't exist
output_dir.mkdir(parents=True, exist_ok=True)
final_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Created output directories: {output_dir}, {final_dir}")


# Add this initialization function
def initialize_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        logger.info("Loading Whisper model...")
        WHISPER_MODEL = WhisperModel("base", device="auto", compute_type="auto")
    return WHISPER_MODEL


def overlay_screenshot(input_video, screenshot_path, output_video):
    logger.info("Overlaying screenshot on video...")
    temp_output = output_video.replace('.mp4', '_temp.mp4')

    # Step 1: Overlay screenshot using software processing
    overlay_command = [
        "ffmpeg",
        "-i", input_video,
        "-i", screenshot_path,
        "-filter_complex",
        # Force scale to 1920 width, calculate height to maintain aspect ratio
        "[1:v]scale=960:-2,format=rgb24[img];"
        # Maintain placement above the center of the bottom half
        "[0:v][img]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2+main_h/4[v]",
        "-map", "[v]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-c:a", "copy",
        temp_output,
        "-y"
    ]

    run_command(overlay_command)

    # Step 2: Encode video using VAAPI
    encode_command = [
        "ffmpeg",
        "-hwaccel", "vaapi",
        "-hwaccel_device", "/dev/dri/renderD128",
        "-i", temp_output,
        "-vf", "format=nv12,hwupload",
        "-c:v", "h264_vaapi",
        "-qp", "18",
        "-b:v", "8M",
        "-maxrate", "10M",
        "-bufsize", "16M",
        "-c:a", "copy",
        output_video,
        "-y"
    ]
    run_command(encode_command)

    # Clean up temporary file
    os.remove(temp_output)

    logger.info(f"Video with screenshot overlay created: {output_video}")


def run_command(command):
    logger.info(f"Executing command: {' '.join(command) if isinstance(command, list) else command}")
    try:
        if isinstance(command, list):
            process = subprocess.run(
                command, check=True, capture_output=True, text=True)
        else:
            process = subprocess.run(
                command, check=True, shell=True, capture_output=True, text=True)
        logger.info("Command executed successfully")
        return process.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        raise


def get_duration(file_path):
    logger.info(f"Getting duration for file: {file_path}")
    command = f"ffprobe -i \"{file_path}\" -show_entries format=duration -v quiet -of csv=\"p=0\""
    duration = float(run_command(command))
    logger.info(f"Duration: {duration} seconds")
    return int(duration)


def transcribe_audio(audio_path, working_directory):
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    subtitle_path = os.path.join(working_directory, f"{audio_basename}.srt")
    logger.info(f"Transcribing audio to {subtitle_path}...")

    try:
        model = initialize_whisper_model()
        logger.info("Transcribing audio...")
        segments, info = model.transcribe(audio_path, word_timestamps=True)

        logger.info("Generating SRT with one word per line...")
        srt_content = generate_word_level_srt(segments)

        logger.info("Writing SRT file...")
        with open(subtitle_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        logger.info(f"Transcription completed. Subtitle file saved at {subtitle_path}")
        return subtitle_path

    except Exception as e:
        logger.error(f"An error occurred during transcription: {str(e)}")
        raise


def generate_word_level_srt(segments):
    srt_content = ""
    index = 1
    for segment in segments:
        for word in segment.words:
            start_time = format_timestamp(word.start)
            end_time = format_timestamp(word.end)
            text = word.word.strip()
            if text:  # Only add non-empty words
                srt_content += f"{index}\n{start_time} --> {end_time}\n{text}\n\n"
                index += 1
    return srt_content


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"


def calculate_start_time(audio_duration, video_duration):
    logger.info("Calculating random start time...")
    if video_duration <= 0:
        raise ValueError("Video duration is zero or negative")
    max_start = video_duration - audio_duration
    if max_start < 0:
        raise ValueError("Video is shorter than the audio.")
    start_time = random.randint(0, max_start)
    logger.info(f"Random start time: {start_time} seconds")
    return start_time


def extract_clip_and_add_audio(video_path, audio_path, start_time, audio_duration, working_directory):
    logger.info("Extracting clip and adding audio...")
    temp_video = os.path.join(working_directory, "temp_video.mp4")
    command = [
        "ffmpeg",
        "-vaapi_device", "/dev/dri/renderD128",
        "-ss", str(start_time),
        "-i", video_path,
        "-i", audio_path,
        "-t", str(audio_duration),
        "-filter_complex", "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30,format=nv12,hwupload[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "h264_vaapi",
        "-qp", "18",
        "-b:v", "8M",
        "-maxrate", "10M",
        "-bufsize", "16M",
        "-c:a", "aac",
        "-ac", "2",
        "-ar", "44100",
        "-b:a", "96k",
        temp_video,
        "-y"
    ]

    run_command(command)
    logger.info(f"Temporary video created: {temp_video}")
    return temp_video


def burn_subtitles(temp_video, subtitle_path, output_video):
    logger.info("Burning subtitles into video...")
    command = [
        "ffmpeg",
        "-vaapi_device", "/dev/dri/renderD128",
        "-i", temp_video,
        "-vf", f"subtitles={subtitle_path}:force_style='Fontsize=24,Alignment=10,MarginV=50,PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2':fontsdir=/path/to/fonts,format=nv12,hwupload",
        "-c:v", "h264_vaapi",
        "-qp", "18",
        "-b:v", "8M",
        "-maxrate", "10M",
        "-bufsize", "16M",
        "-c:a", "copy",
        output_video
    ]
    run_command(command)
    logger.info(f"Final video created: {output_video}")


def process_video(working_directory, audio_path, counter=1):
    try:
        if not os.path.isfile(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None  # Exit the function early if the file is missing

        title_file = Path(working_directory) / f"title_{counter}.txt"
        # Check if the title file exists before proceeding
        if not title_file.exists():
            logger.error(f"Title file not found: {title_file}")
            return None

        with open(title_file, "r") as f:
            safe_title = f.read().strip()

        video_path = VIDEO_SOURCE_PATH
        logger.info("Starting video processing...")

        # Use generic names for intermediate files
        subtitle_path = os.path.join(working_directory, f"temp_subtitles_{counter}.srt")
        temp_video = os.path.join(working_directory, f"temp_video_{counter}.mp4")
        output_video_with_subs = os.path.join(working_directory, f"temp_with_subs_{counter}.mp4")
        screenshot_path = os.path.join(working_directory, f"reddit_post_screenshot_{counter}.png")
        
        # Only use the title for the final output
        final_output_video = os.path.join(final_dir, f"{safe_title}.mp4")

        subtitle_path = transcribe_audio(audio_path, working_directory)
        audio_duration = get_duration(audio_path)
        video_duration = get_duration(video_path)
        start_time = calculate_start_time(audio_duration, video_duration)

        temp_video = extract_clip_and_add_audio(
            video_path, audio_path, start_time, audio_duration, working_directory)

        burn_subtitles(temp_video, subtitle_path, output_video_with_subs)
        overlay_screenshot(output_video_with_subs, screenshot_path, final_output_video)

        logger.info("Cleaning up temporary files...")
        os.remove(temp_video)
        os.remove(output_video_with_subs)
        os.remove(title_file)  # Clean up the title file
        logger.info(f"Process completed. Final output file: {final_output_video}")

        # Add the final path to our list
        final_video_paths.append(final_output_video)
        return final_output_video

    except Exception as e:
        logger.error(f"An error occurred during video processing: {str(e)}")
        return None


# Track files generated per day for AITAH posts
aitah_counter = 1


def initialize_sb():
    return SB(uc=True, user_data_dir=CHROME_USER_DATA_DIR, headless=True)


def redditpost_to_text(url, sb):
    global aitah_counter
    global output_dir

    logger.info(f"Opening URL: {url}")
    sb.open(url)

    post_title = sb.get_text("h1")
    post_text = sb.get_text("div[class='md text-14']")

    # Create a safe filename from the title
    safe_title = "".join(c for c in post_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title[:100]  # Limit length to avoid too long filenames

    # Save text to file
    filename = output_dir / f"{safe_title}_{aitah_counter}.txt"
    with open(filename, "w") as f:
        f.write(post_title + "\n===\n" + post_text + "\n")
    logger.info(f"Saved post text to {filename}")
    aitah_counter += 1

    # Return title and combined text for TTS
    return {
        'title': safe_title,
        'text': substitute_acronyms(post_title + " " + post_text)
    }


def select_reddit_post(subreddit_url, sb, posts_number=1):
    logger.info(f"Opening subreddit: {subreddit_url}")
    sb.open(subreddit_url)
    unique_links = []  # Using a list to maintain order
    last_length = 0
    screenshot_counter = 1

    # Continue until we collect the required number of unique posts
    while len(unique_links) < posts_number:
        post_links_elements = sb.find_elements(
            "article", limit=posts_number)  # Using a simple CSS selector
        for elem in post_links_elements:
            link_element = elem.find_element(
                by="css selector", value="a[href*='/r/AITAH/comments/']:first-child")
            href = link_element.get_attribute('href')
            if href not in unique_links:  # Check to avoid duplicates
                unique_links.append(href)
                # Take a screenshot of the article element which is the post preview
                screenshot_path = output_dir / \
                    f"reddit_post_screenshot_{screenshot_counter}.png"
                elem.screenshot(str(screenshot_path))
                logger.info(f"Screenshot saved: {screenshot_path}")
                screenshot_counter += 1

        # Check if new links have been added
        if last_length == len(unique_links):
            sb.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            sb.sleep(2)  # Wait for more posts to load
        last_length = len(unique_links)

        if sb.get_current_url().endswith('#'):
            logger.info("Reached the end of the page or no new posts are loading.")
            break

    logger.info(f"Found {len(unique_links[:posts_number])} unique post links")
    return unique_links[:posts_number]


def get_reddit_post_text(subreddit_url, sb, posts_number=1):
    post_links = select_reddit_post(subreddit_url, sb, posts_number)
    posts = []
    for post_link in post_links:
        post_data = redditpost_to_text(post_link, sb)
        posts.append(post_data)
    return posts


def gen_voice(text, language, output_file_name="example.wav"):
    text = html.unescape(text)
    # Use consistent naming pattern: aitah_audio_{number}.wav
    output_file_path = output_dir / f"aitah_audio_{output_file_name}"
    logger.info(f"Generating speech for output file: {output_file_path}")
    tts.tts_to_file(
        text=text,
        speed=2,
        file_path=output_file_path,
        speaker_wav=["voices/xd.wav"],
        language=language
    )
    logger.info(f"Generated speech saved to {output_file_path}")


def generate_aitah_audio(sb, subreddit_url=SUBREDDIT_DEFAULT_URL, language="en"):
    logger.info("Starting AITAH audio generation...")
    posts = get_reddit_post_text(subreddit_url, sb, NUM_VIDEOS_TO_GENERATE)
    for i, post in enumerate(posts, 1):
        # Use consistent naming pattern
        output_file_name = f"{i}.wav"  # This will become aitah_audio_1.wav, etc.
        gen_voice(post['text'], language, output_file_name)
        # Store the title for later use
        title_file = output_dir / f"title_{i}.txt"
        with open(title_file, "w") as f:
            f.write(post['title'])


def substitute_acronyms(text):
    acronyms = {
        "AITAH": "Am I The Ahole",
        "AITA": "Am I The Ahole",
        "WIBTA": "Would I Be The Ahole",
        "YTA": "You're The Ahole",
        "NTA": "Not The Ahole",
        "NAH": "No Aholes Here",
        "ESH": "Everyone Sucks Here",
        "TLDR": "Too Long Didn't Read",
    }

    def replace(match):
        word = match.group(0)
        return acronyms.get(word, word)

    # Replace acronyms
    pattern = r'\b(' + '|'.join(re.escape(key)
                                for key in acronyms.keys()) + r')\b'
    expanded_text = re.sub(pattern, replace, text)

    # Split into lines
    lines = expanded_text.split('\n')

    # Process each line
    processed_lines = []
    for line in lines:
        words = line.split()
        chunks = []
        current_chunk = []
        for word in words:
            if len(' '.join(current_chunk + [word])) > 250:
                chunks.append(' '.join(current_chunk) + '.')
                current_chunk = [word]
            else:
                current_chunk.append(word)
        if current_chunk:
            chunks.append(' '.join(current_chunk) + '.')
        processed_lines.extend(chunks)

    return '\n'.join(line.strip() for line in processed_lines)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        logger.info("Loading TTS model...")
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        initialize_whisper_model()  # Initialize the model when the script starts

        logger.info("Starting AITAH audio generation process...")
        with initialize_sb() as sb:
            generate_aitah_audio(sb)
            
            for i in range(1, NUM_VIDEOS_TO_GENERATE + 1):
                audio_path = output_dir / f"aitah_audio_{i}.wav"
                process_video(str(output_dir), str(audio_path), i)

        if final_dir.exists():
            final_video_paths = [str(p) for p in final_dir.glob("*.mp4")]
            logger.info(f"Found {len(final_video_paths)} videos in {final_dir}")
            with SB(uc=True, user_data_dir=CHROME_USER_DATA_DIR, headless=False) as sb1:
                upload_to_tiktok(final_video_paths, 'cookies.txt', sb1)
    elif len(sys.argv) > 1 and sys.argv[1] == "upload":
        final_video_paths = [str(p) for p in final_dir.glob("*.mp4")]
        logger.info(f"Found {len(final_video_paths)} videos in {final_dir}")
        with SB(uc=True, user_data_dir=CHROME_USER_DATA_DIR, headless=False) as sb1:
            upload_to_tiktok(final_video_paths, 'cookies.txt', sb1)
    else:
        logger.error("Invalid command. Use 'generate' to generate videos or 'upload' to upload videos.")