**Media Download Bot**
=====================

A Telegram bot that downloads videos from provided links, converts them to audio, and uploads both files to a Telegram group.

**Features**
------------

*   Downloads videos from provided links
*   Converts downloaded videos to high-quality audio (MP3)
*   Uploads both video and audio files to a Telegram group
*   Splits large files into smaller chunks for uploading
*   Validates video chunks before uploading
*   Cleans up downloaded and temporary files on every startup

**Usage**
-----

1.  Start the bot by sending the `/start` command.
2.  Send a video link to the bot, or use the `/pull` command followed by the video URL.
3.  The bot will analyze the video, download it, convert it to audio, and upload both files to the Telegram group.

**Requirements**
------------

*   Python 3.8+
*   `python-telegram-bot` library
*   `yt-dlp` library
*   `pydub` library
*   `ffmpeg` installed on the system

**Installation**
------------

1.  Clone the repository: `git clone https://github.com/ShadowPlague21/YT-DL-Telegram-bot.git`
2.  Install the required libraries: `pip install -r requirements.txt`
3.  Create a `TOKEN` environment variable with your bot's token
4.  Run the bot: `python main.py`

**Configuration**
-------------

*   `TOKEN`: Your bot's token
*   `DOWNLOAD_PATH`: The directory where downloaded files will be stored
*   `CPU_CORES`: The number of CPU cores available for processing
*   `THREAD_POOL_SIZE`: The number of threads in the thread pool
*   `CONCURRENT_FRAGMENTS`: The number of concurrent fragments for downloading
*   `BUFFER_SIZE`: The buffer size for downloading and uploading files
*   `MAX_FILE_SIZE`: The maximum file size for uploading
*   `CHUNK_SIZE`: The chunk size for splitting large files
*   `MIN_CHUNK_SIZE`: The minimum chunk size for splitting large files
*   `CONNECT_TIMEOUT`, `READ_TIMEOUT`, `WRITE_TIMEOUT`, `POOL_TIMEOUT`: Timeouts for HTTP requests

**Troubleshooting**
--------------

*   Check the bot's logs for errors
*   Ensure that the `ffmpeg` executable is installed and available in the system's PATH
*   Verify that the bot's token is correct and has the necessary permissions

**License**
-------

This project is licensed under the MPL 2.0 License.
