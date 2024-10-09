# Infinite-Brainrot

Infinite-Brainrot is an automated TikTok-style video creation tool that transforms Reddit posts into engaging short-form content. This project streamlines the process of generating viral-worthy videos by combining web scraping, text-to-speech, and video editing techniques.

## Features

- Scrapes top posts from the r/AITAH (Am I The A-hole) subreddit
- Converts post text to speech using advanced TTS models
- Automatically generates subtitles for accessibility
- Overlays text on background video clips
- Adds screenshots of Reddit posts for visual context
- Utilizes hardware acceleration for efficient video processing

## Prerequisites

- Python 3.7+
- CUDA/ROCM-compatible GPU (recommended for faster processing)
- FFmpeg
- Google Chrome

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Infinite-Brainrot.git
   cd Infinite-Brainrot
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the necessary directories:
   ```
   mkdir -p output voices
   ```

4. Place your background video in the `/videos` directory and name it `output.mp4`.

5. Add your voice sample to the `voices` directory and name it `xd.wav`.

## Usage

Run the main script to generate videos:

```
python main.py
```

By default, the script will generate 5 videos. You can modify the `NUM_VIDEOS_TO_GENERATE` variable in the script to change this number.

## Configuration

- Adjust the `subreddit_url` in the `generate_aitah_audio` function to scrape from different subreddits or time periods.
- Modify the TTS model or voice settings in the `gen_voice` function.
- Customize video overlay and editing parameters in the `process_video` function.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This tool is for educational purposes only. Ensure you comply with Reddit's terms of service and respect content creators' rights when using this tool.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
