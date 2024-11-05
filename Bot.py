import os
import asyncio
from typing import Optional, Tuple, Dict, List
import logging
from datetime import datetime
from urllib.parse import urlparse
import concurrent.futures
import shutil
import multiprocessing
import math
import aiofiles
import aiofiles.os as async_os
import tempfile

from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest
from yt_dlp import YoutubeDL
from pydub import AudioSegment

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
TOKEN = 'YOUR-BOT-TOKEN'
DOWNLOAD_PATH = 'downloads/'

# System resources configuration
CPU_CORES = multiprocessing.cpu_count()
THREAD_POOL_SIZE = CPU_CORES * 2
CONCURRENT_FRAGMENTS = 8
BUFFER_SIZE = 8 * 1024 * 1024  # 8MB buffer

# File size limits (in bytes)
MAX_FILE_SIZE = 45 * 1024 * 1024  # 45MB max file size for Telegram
CHUNK_SIZE = 44 * 1024 * 1024     # Slightly smaller to account for overhead
MIN_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB minimum chunk size

# Timeouts (in seconds)
CONNECT_TIMEOUT = 60
READ_TIMEOUT = 300
WRITE_TIMEOUT = 300
POOL_TIMEOUT = 300

# Create thread pools
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=CPU_CORES)

# Video processing constants
MIN_CHUNK_DURATION = 1  # Minimum chunk duration in seconds
MAX_RETRIES = 3        # Maximum number of retries for failed operations
FFMPEG_TIMEOUT = 3600  # Timeout for ffmpeg operations (1 hour)

def format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def convert_audio(input_path: str, output_path: str, cpu_cores: int) -> str:
    """Standalone function for audio conversion."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(
            output_path,
            format='mp3',
            parameters=[
                "-q:a", "0",
                "-b:a", "192k",
                "-threads", str(cpu_cores)
            ]
        )
        return output_path
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return ""

class MediaBot:
    def __init__(self):
        self.setup_download_directory()

    def setup_download_directory(self):
        """Setup and clean download directory."""
        if os.path.exists(DOWNLOAD_PATH):
            try:
                shutil.rmtree(DOWNLOAD_PATH)
            except Exception as e:
                logger.error(f"Error cleaning download directory: {e}")
        os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_message = (
            "👋 Hi! I'm a Media Download Bot.\n\n"
            "Send me a video link, and I'll:\n"
            "1. Download the video in best quality\n"
            "2. Convert it to high-quality audio\n"
            "3. Split and upload files if they're larger than 45MB\n\n"
            "Just send me any video URL after /pull to get started!"
        )
        await update.message.reply_text(welcome_message)

    async def split_file_async(self, file_path: str, chunk_size: int = CHUNK_SIZE) -> List[Tuple[str, int]]:
        """Split a file into chunks asynchronously with proper video splitting."""
        file_size = os.path.getsize(file_path)
        if file_size <= chunk_size:
            return [(file_path, file_size)]
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp(dir=DOWNLOAD_PATH)
        chunks = []
        
        try:
            # Get video duration using ffprobe
            duration_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{file_path}"'
            process = await asyncio.create_subprocess_shell(
                duration_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stderr:
                logger.error(f"FFprobe error: {stderr.decode()}")
                return [(file_path, file_size)]
                
            duration = float(stdout.decode().strip())
            
            # Calculate chunks
            num_chunks = math.ceil(file_size / chunk_size)
            chunk_duration = duration / num_chunks
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                chunk_path = os.path.join(temp_dir, f'chunk_{i+1}.mp4')
                
                # Use ffmpeg to split video properly
                cmd = (
                    f'ffmpeg -i "{file_path}" -ss {start_time} -t {chunk_duration} '
                    f'-c copy -avoid_negative_ts 1 -y "{chunk_path}"'
                )
                
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if os.path.exists(chunk_path):
                    chunk_size = os.path.getsize(chunk_path)
                    if chunk_size > 0 and chunk_size <= MAX_FILE_SIZE:
                        chunks.append((chunk_path, chunk_size))
                    else:
                        # Remove invalid chunk
                        await async_os.remove(chunk_path)
            
            # Remove original file to save space
            await async_os.remove(file_path)
            
            # If no valid chunks were created, return original file
            if not chunks:
                return [(file_path, file_size)]
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting file: {e}")
            # Clean up temp directory on error
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temp directory: {cleanup_error}")
            return [(file_path, file_size)]

        # Add this helper function to validate video chunks
    async def validate_video_chunk(self, chunk_path: str) -> bool:
        """Validate if the video chunk is properly formatted."""
        try:
            cmd = f'ffprobe -v error "{chunk_path}"'
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            return not stderr  # If stderr is empty, the video is valid
        except Exception as e:
            logger.error(f"Error validating video chunk: {e}")
            return False        

        # Also modify the send_file method to include chunk validation
    async def send_file(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, 
                    file_path: str, caption: str, reply_to_message_id: int, 
                    is_video: bool = True) -> bool:
        """Send a single file with retries and validation."""
        max_retries = 3
        current_try = 0
        
        while current_try < max_retries:
            try:
                # Check file size before sending
                file_size = os.path.getsize(file_path)
                if file_size > MAX_FILE_SIZE:
                    logger.error(f"File too large: {file_size} bytes")
                    return False

                # For video files, validate the chunk before sending
                if is_video and not await self.validate_video_chunk(file_path):
                    logger.error(f"Invalid video chunk: {file_path}")
                    return False

                with open(file_path, 'rb', buffering=BUFFER_SIZE) as file:
                    if is_video:
                        await context.bot.send_video(
                            chat_id=chat_id,
                            video=file,
                            caption=caption,
                            reply_to_message_id=reply_to_message_id,
                            read_timeout=READ_TIMEOUT,
                            write_timeout=WRITE_TIMEOUT,
                            connect_timeout=CONNECT_TIMEOUT,
                            supports_streaming=True
                        )
                    else:
                        await context.bot.send_audio(
                            chat_id=chat_id,
                            audio=file,
                            caption=caption,
                            reply_to_message_id=reply_to_message_id,
                            read_timeout=READ_TIMEOUT,
                            write_timeout=WRITE_TIMEOUT,
                            connect_timeout=CONNECT_TIMEOUT
                        )
                return True
                    
            except Exception as e:
                current_try += 1
                logger.warning(f"Attempt {current_try}/{max_retries} failed: {str(e)}")
                if current_try >= max_retries:
                    logger.error(f"Failed to send file after {max_retries} attempts: {str(e)}")
                    return False
                await asyncio.sleep(2 ** current_try)  # Exponential backoff
        
        return False

    async def send_file_in_chunks(self, context: ContextTypes.DEFAULT_TYPE, 
                                chat_id: int, file_path: str, caption: str, 
                                reply_to_message_id: int, is_video: bool = True,
                                status_message = None) -> bool:
        """Enhanced chunked file sending with proper cleanup."""
        try:
            file_size = os.path.getsize(file_path)
            
            # If file is within limit, send normally
            if file_size <= CHUNK_SIZE:
                return await self.send_file(
                    context, chat_id, file_path, 
                    f"{caption}\nSize: {format_size(file_size)}", 
                    reply_to_message_id, is_video
                )
            
            # Split file into chunks
            chunks = await self.split_file_async(file_path)
            if not chunks:
                if status_message:
                    await status_message.edit_text("❌ Failed to split file into chunks.")
                return False
                
            total_parts = len(chunks)
            
            for i, (chunk_path, chunk_size) in enumerate(chunks, 1):
                chunk_caption = (
                    f"{caption}\n"
                    f"Part {i}/{total_parts}\n"
                    f"Size: {format_size(chunk_size)}"
                )
                
                if status_message:
                    try:
                        await status_message.edit_text(
                            f"📤 Uploading {caption.split(':')[0]}\n"
                            f"Part {i}/{total_parts} ({format_size(chunk_size)})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update status message: {e}")
                
                # Validate chunk size before sending
                if chunk_size > MAX_FILE_SIZE:
                    logger.error(f"Chunk {i} too large: {chunk_size} bytes")
                    await status_message.edit_text(
                        f"❌ Chunk {i} exceeds maximum size limit. Please try a smaller video."
                    )
                    return False
                
                # Send the chunk
                success = await self.send_file(
                    context, chat_id, chunk_path,
                    chunk_caption, reply_to_message_id, is_video
                )
                
                # Clean up the chunk file
                try:
                    await async_os.remove(chunk_path)
                except Exception as e:
                    logger.warning(f"Failed to remove chunk file: {e}")
                
                if not success:
                    await status_message.edit_text(
                        f"❌ Failed to send part {i}/{total_parts}"
                    )
                    return False
                
                # Add delay between chunks to prevent rate limiting
                await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in chunked upload: {e}")
            if status_message:
                await status_message.edit_text(
                    f"❌ An error occurred during upload: {str(e)}"
                )
            return False
        
    async def cleanup_files(self, *file_paths):
        """Asynchronously clean up downloaded and temporary files."""
        for file_path in file_paths:
            try:
                if file_path and await async_os.path.exists(file_path):
                    await async_os.remove(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")    

    async def get_video_info(self, url: str) -> Optional[Dict]:
        """Get video information with improved size estimation."""
        ydl_opts = {
            'format': 'best',
            'quiet': True,
            'no_warnings': True,
            'force_generic_extractor': False
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    lambda: ydl.extract_info(url, download=False)
                )
                
                if info:
                    # Try multiple ways to get filesize
                    filesize = 0
                    formats = info.get('formats', [])
                    
                    # First try the best format
                    best_format = None
                    max_quality = -1
                    
                    for f in formats:
                        quality = f.get('quality', -1)
                        if quality > max_quality and f.get('filesize'):
                            max_quality = quality
                            best_format = f
                    
                    if best_format:
                        filesize = best_format.get('filesize', 0)
                    
                    # If no filesize, estimate from bitrate and duration
                    if not filesize and info.get('duration'):
                        # Get the highest bitrate available
                        max_bitrate = max(
                            (f.get('tbr', 0) for f in formats if f.get('tbr')),
                            default=0
                        )
                        if max_bitrate:
                            # Estimate size: bitrate (bits/s) * duration (s) / 8 = bytes
                            filesize = int((max_bitrate * 1024 * info['duration']) / 8)
                    
                    info['filesize'] = filesize
                    
                return info
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None

    async def download_video(self, url: str, chat_id: int) -> Optional[str]:
        """Download video in best quality."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(DOWNLOAD_PATH, f'video_{chat_id}_{timestamp}.mp4')

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'concurrent_fragments': CONCURRENT_FRAGMENTS,
            'buffersize': BUFFER_SIZE,
            'http_chunk_size': BUFFER_SIZE,
            'postprocessor_args': [
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-threads', str(CPU_CORES)
            ]
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    lambda: ydl.download([url])
                )
            return output_path if os.path.exists(output_path) else None
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None

    async def convert_to_audio(self, video_path: str) -> Optional[str]:
        """Convert video to audio using process pool."""
        try:
            output_path = os.path.join(
                DOWNLOAD_PATH,
                f"{os.path.splitext(os.path.basename(video_path))[0]}.mp3"
            )

            await asyncio.get_event_loop().run_in_executor(
                process_pool,
                convert_audio,
                video_path,
                output_path,
                CPU_CORES
            )

            return output_path if os.path.exists(output_path) else None
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return None

    async def handle_video_link(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle direct video link messages."""
        url = update.message.text.strip()
        await self.process_video_request(update, context, url)

    async def pull_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /pull command for video downloads."""
        # Check if a URL was provided with the command
        if not context.args:
            await update.message.reply_text(
                "❌ Please provide a video URL after the /pull command.\n"
                "Example: /pull https://www.example.com/video",
                reply_to_message_id=update.message.message_id
            )
            return

        # Extract the URL from the command arguments
        url = context.args[0].strip()
        
        # Process the video using existing functionality
        await self.process_video_request(update, context, url)

    async def process_video_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
        """Process video download request from either command or direct link."""
        chat_id = update.effective_chat.id
        message_id = update.message.message_id

        if not self.is_valid_url(url):
            await update.message.reply_text(
                "❌ Please provide a valid URL.",
                reply_to_message_id=message_id
            )
            return

        status_message = await update.message.reply_text(
            "🔍 Analyzing video...",
            reply_to_message_id=message_id
        )

        try:
            # Get video information with timeout
            video_info = await asyncio.wait_for(
                self.get_video_info(url),
                timeout=30
            )
            
            if not video_info:
                await status_message.edit_text("❌ Failed to get video information.")
                return

            title = video_info.get('title', 'Video')
            duration = video_info.get('duration', 0)
            filesize = video_info.get('filesize', 0)

            size_message = (f"Estimated size: {format_size(filesize)}" 
                          if filesize > 0 
                          else "Size: Will be determined during download")

            await status_message.edit_text(
                f"📥 Downloading: {title}\n"
                f"Duration: {duration} seconds\n"
                f"{size_message}"
            )

            video_path = await asyncio.wait_for(
                self.download_video(url, chat_id),
                timeout=3600
            )
            
            if not video_path:
                await status_message.edit_text("❌ Download failed.")
                return

            video_task = asyncio.create_task(self.process_video(
                context, chat_id, video_path, title, message_id, status_message
            ))
            
            audio_task = asyncio.create_task(self.process_audio(
                context, chat_id, video_path, title, message_id, status_message
            ))

            await asyncio.gather(video_task, audio_task)

        except asyncio.TimeoutError:
            await status_message.edit_text(
                "❌ Operation timed out. Please try again with a shorter video."
            )
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            await status_message.edit_text(
                "❌ An error occurred while processing your request. Please try again."
            )

    async def process_video(self, context, chat_id, video_path, title, message_id, status_message):
        """Process and send video file."""
        try:
            video_size = os.path.getsize(video_path)
            await status_message.edit_text(
                f"📤 Uploading video ({format_size(video_size)})..."
            )

            video_success = await self.send_file_in_chunks(
                context, chat_id, video_path,
                f"🎥 Video: {title}",
                message_id, True, status_message
            )

            if not video_success:
                await status_message.edit_text("❌ Failed to send video.")
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            await status_message.edit_text("❌ Error processing video.")

    async def process_audio(self, context, chat_id, video_path, title, message_id, status_message):
        """Process and send audio file."""
        try:
            audio_path = await self.convert_to_audio(video_path)
            if audio_path and os.path.exists(audio_path):
                audio_size = os.path.getsize(audio_path)
                await status_message.edit_text(
                    f"📤 Uploading audio ({format_size(audio_size)})..."
                )

                audio_success = await self.send_file_in_chunks(
                    context, chat_id, audio_path,
                    f"🎵 Audio: {title}",
                    message_id, False, status_message
                )

                if audio_success:
                    await status_message.delete()
                else:
                    await status_message.edit_text("❌ Failed to send audio.")
            else:
                await status_message.edit_text("❌ Audio conversion failed.")
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            await status_message.edit_text("❌ Error processing audio.")

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def cleanup_files(self, *file_paths):
        """Clean up downloaded and temporary files."""
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")

async def shutdown():
    """Cleanup function to be called on shutdown."""
    try:
        # Clean up thread and process pools
        thread_pool.shutdown(wait=True)
        process_pool.shutdown(wait=True)
        
        # Clean up download directory
        if os.path.exists(DOWNLOAD_PATH):
            shutil.rmtree(DOWNLOAD_PATH)
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def main():
    """Start the bot."""
    request = HTTPXRequest(
        connect_timeout=CONNECT_TIMEOUT,
        read_timeout=READ_TIMEOUT,
        write_timeout=WRITE_TIMEOUT,
        pool_timeout=POOL_TIMEOUT
    )

    application = (
        Application.builder()
        .token(TOKEN)
        .request(request)
        .build()
    )
    
    # Initialize bot
    bot = MediaBot()
    
    # Add handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("pull", bot.pull_command))  # Add pull command handler
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        bot.handle_video_link
    ))

    # Register shutdown handler
    application.post_shutdown = shutdown

    # Start the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES, timeout=None)

if __name__ == '__main__':
    main()
