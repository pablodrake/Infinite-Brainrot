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

# Add this near the top of the script, after the imports
# You can change this value to generate a different number of videos
NUM_VIDEOS_TO_GENERATE = 5

# Set up directory based on current date
date_today = datetime.date.today().strftime("%Y-%m-%d")
output_dir = Path(f"./output/{date_today}")
# Create the directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)


def overlay_screenshot(input_video, screenshot_path, output_video):
    print("\nOverlaying screenshot on video...")
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
        "-crf", "23",
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
        "-qp", "23",
        "-b:v", "5M",
        "-maxrate", "5M",
        "-bufsize", "10M",
        "-c:a", "copy",
        output_video,
        "-y"
    ]
    run_command(encode_command)

    # Clean up temporary file
    os.remove(temp_output)

    print(f"Video with screenshot overlay created: {output_video}")


def run_command(command):
    print(f"\nExecuting command: {' '.join(command) if isinstance(command, list) else command}")
    try:
        if isinstance(command, list):
            process = subprocess.run(
                command, check=True, capture_output=True, text=True)
        else:
            process = subprocess.run(
                command, check=True, shell=True, capture_output=True, text=True)
        print("Command executed successfully")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("Command output:", e.stdout)
        print("Command error:", e.stderr)
        raise


def get_duration(file_path):
    print(f"\nGetting duration for file: {file_path}")
    command = f"ffprobe -i \"{file_path}\" -show_entries format=duration -v quiet -of csv=\"p=0\""
    duration = float(run_command(command))
    print(f"Duration: {duration} seconds")
    return int(duration)


def transcribe_audio(audio_path, working_directory):
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    subtitle_path = os.path.join(working_directory, f"{audio_basename}.srt")
    print(f"\nTranscribing audio to {subtitle_path}...")

    try:
        print("Loading Whisper model...")
        model = WhisperModel("base", device="auto", compute_type="auto")

        print("Transcribing audio...")
        segments, info = model.transcribe(audio_path, word_timestamps=True)

        print("Generating SRT with one word per line...")
        srt_content = generate_word_level_srt(segments)

        print("Writing SRT file...")
        with open(subtitle_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        print(f"Transcription completed. Subtitle file saved at {subtitle_path}")
        return subtitle_path

    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
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
    print("\nCalculating random start time...")
    if video_duration <= 0:
        raise ValueError("Video duration is zero or negative")
    max_start = video_duration - audio_duration
    if max_start < 0:
        raise ValueError("Video is shorter than the audio.")
    start_time = random.randint(0, max_start)
    print(f"Random start time: {start_time} seconds")
    return start_time


def extract_clip_and_add_audio(video_path, audio_path, start_time, audio_duration, working_directory):
    print("\nExtracting clip and adding audio...")
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
        "-qp", "23",
        "-b:v", "5M",
        "-maxrate", "5M",
        "-bufsize", "10M",
        "-c:a", "aac",
        "-ac", "2",
        "-ar", "44100",
        "-b:a", "96k",
        temp_video,
        "-y"
    ]

    run_command(command)
    print(f"Temporary video created: {temp_video}")
    return temp_video


def burn_subtitles(temp_video, subtitle_path, output_video):
    print("\nBurning subtitles into video...")
    command = [
        "ffmpeg",
        "-vaapi_device", "/dev/dri/renderD128",
        "-i", temp_video,
        "-vf", f"subtitles={subtitle_path}:force_style='Fontsize=24,Alignment=10,MarginV=50,PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2':fontsdir=/path/to/fonts,format=nv12,hwupload",
        "-c:v", "h264_vaapi",
        "-qp", "20",
        "-b:v", "5M",
        "-maxrate", "5M",
        "-bufsize", "10M",
        "-c:a", "copy",
        output_video
    ]
    run_command(command)
    print(f"Final video created: {output_video}")


def process_video(working_directory, audio_path, counter=1):
    try:
        video_path = "/videos/output.mp4"
        print("\n--- Starting video processing ---")

        subtitle_path = transcribe_audio(audio_path, working_directory)

        audio_duration = get_duration(audio_path)
        video_duration = get_duration(video_path)

        start_time = calculate_start_time(audio_duration, video_duration)

        temp_video = extract_clip_and_add_audio(
            video_path, audio_path, start_time, audio_duration, working_directory)

        output_video_with_subs = os.path.join(
            working_directory, f"output_video_with_subs_{counter}.mp4")
        burn_subtitles(temp_video, subtitle_path, output_video_with_subs)

        # Add screenshot overlay
        screenshot_path = os.path.join(
            working_directory, f"reddit_post_screenshot_{counter}.png")
        final_output_video = os.path.join(
            working_directory, f"final_output_video_{counter}.mp4")
        overlay_screenshot(output_video_with_subs,
                           screenshot_path, final_output_video)

        print("\nCleaning up temporary files...")
        os.remove(temp_video)
        os.remove(output_video_with_subs)
        print(f"Process completed. Final output file: {final_output_video}")

        return final_output_video

    except Exception as e:
        print(f"\nAn error occurred during video processing: {str(e)}")
        return None

    except Exception as e:
        print(f"\nAn error occurred during video processing: {str(e)}")
        return None


# Track files generated per day for AITAH posts
aitah_counter = 1


def redditpost_to_text(url):
    global aitah_counter
    global output_dir  # Ensure that this is defined somewhere as a Path object
    # Make sure this is set to an appropriate path
    output_dir = Path(output_dir)

    with SB(uc=True, user_data_dir="/home/pdaloxd/.config/google-chrome/Default", headless=True) as sb:
        sb.open(url)

        # Assuming h1 contains the post title
        post_title = sb.get_text("h1")
        # Assuming this class correctly identifies the main text of the post
        post_text = sb.get_text("div[class='md text-14']")

        # Save text to file
        filename = output_dir / f"reddit_post_aitah_{aitah_counter}.txt"
        with open(filename, "w") as f:
            f.write(post_title + "\n===\n" + post_text + "\n")
        aitah_counter += 1  # Increment the file counter for next use

        # Return combined title and text for TTS
        return substitute_acronyms(post_title + " " + post_text)


def select_reddit_post(subreddit_url, posts_number=1):
    with SB(uc=True, user_data_dir="/home/pdaloxd/.config/google-chrome/Default", headless=True) as sb:
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
                    print(f"Screenshot saved: {screenshot_path}")
                    screenshot_counter += 1

            # Check if new links have been added
            if last_length == len(unique_links):
                sb.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);")
                sb.sleep(2)  # Wait for more posts to load
            last_length = len(unique_links)

            if sb.get_current_url().endswith('#'):
                print("Reached the end of the page or no new posts are loading.")
                break

        print("Unique Post Links:", unique_links[:posts_number])
        return unique_links[:posts_number]


def get_reddit_post_text(subreddit_url, posts_number=1):
    post_links = select_reddit_post(subreddit_url, posts_number)
    texts = []
    for post_link in post_links:
        text = redditpost_to_text(post_link)
        texts.append(text)
    return texts


def gen_voice(text, language, output_file_name="example.wav"):
    text = html.unescape(text)
    output_file_path = output_dir / output_file_name
    tts.tts_to_file(
        text=text,
        speed=2,
        file_path=output_file_path,
        speaker_wav=["voices/xd.wav"],
        language=language
    )
    print(f"Generated speech saved to {output_file_path}")


def generate_aitah_audio(subreddit_url="https://www.reddit.com/r/AITAH/top/?t=hour", language="en"):
    post_texts = get_reddit_post_text(subreddit_url, NUM_VIDEOS_TO_GENERATE)
    for i, post_text in enumerate(post_texts):
        output_file_name = f"aitah_audio_{i + 1}.wav"
        gen_voice(post_text, language, output_file_name)


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


# Load the TTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(
    torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Example usage
generate_aitah_audio()  # Automatically fetches top posts from AITAH and generates audio

# Replace the commented-out for loop with this:
for i in range(1, NUM_VIDEOS_TO_GENERATE + 1):
    audio_path = output_dir / f"aitah_audio_{i}.wav"
    process_video(str(output_dir), str(audio_path), i)
