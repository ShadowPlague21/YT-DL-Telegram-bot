import os
import asyncio
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import aiohttp
import aiofiles
from collections import defaultdict
from datetime import datetime, timedelta
import concurrent.futures
import shutil
import multiprocessing
import math
import tempfile

from pathlib import Path
from telegram import Update
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    ContextTypes,
    filters
)
from telegram.request import HTTPXRequest
from yt_dlp import YoutubeDL
from pydub import AudioSegment

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Config:
    # Bot configuration
    TOKEN = 'Oompa-Loompa-Boompa-Token'
    DOWNLOAD_PATH = 'downloads/'
    
    # System resources
    CPU_CORES = multiprocessing.cpu_count()
    THREAD_POOL_SIZE = CPU_CORES * 2
    CONCURRENT_FRAGMENTS = 8
    BUFFER_SIZE = 8 * 1024 * 1024  # 8MB
    
    # File size limits
    MAX_FILE_SIZE = 45 * 1024 * 1024  # 45MB (Telegram limit)
    CHUNK_SIZE = 44 * 1024 * 1024     # Slightly smaller for overhead
    MIN_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB minimum chunk
    
    # Timeouts
    CONNECT_TIMEOUT = 60
    READ_TIMEOUT = 300
    WRITE_TIMEOUT = 300
    POOL_TIMEOUT = 300
    
    # Rate limiting
    RATE_LIMIT_SECONDS = 1
    
    # Video processing
    MIN_CHUNK_DURATION = 1
    MAX_RETRIES = 3
    FFMPEG_TIMEOUT = 3600

def convert_audio(video_path: str, output_path: str) -> Optional[str]:
    """Standalone function for audio conversion."""
    try:
        logger.info(f"Starting audio conversion: {video_path} -> {output_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Input video file does not exist: {video_path}")
            return None
            
        audio = AudioSegment.from_file(video_path)
        logger.info(f"Successfully loaded audio from video file")
        
        audio.export(
            output_path,
            format='mp3',
            parameters=[
                "-q:a", "0",
                "-b:a", "192k",
                "-threads", str(Config.CPU_CORES)
            ]
        )
        logger.info(f"Successfully exported audio file")
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Audio conversion complete. File size: {file_size / (1024*1024):.2f} MB")
            return output_path
        else:
            logger.error(f"Output file was not created: {output_path}")
            return None
            
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}", exc_info=True)
        return None

class MediaBot:
    def __init__(self):
        """Initialize the MediaBot with required components."""
        self.setup_directories()
        self.setup_thread_pools()
        self.user_last_request = defaultdict(datetime.now)
        
    def setup_directories(self):
        """Setup and clean download directories."""
        if os.path.exists(Config.DOWNLOAD_PATH):
            try:
                shutil.rmtree(Config.DOWNLOAD_PATH)
            except Exception as e:
                logger.error(f"Error cleaning download directory: {e}")
        os.makedirs(Config.DOWNLOAD_PATH, exist_ok=True)
        
    def setup_thread_pools(self):
        """Initialize thread and process pools."""
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=Config.THREAD_POOL_SIZE
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=Config.CPU_CORES
        )

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"

    async def check_rate_limit(self, user_id: int) -> bool:
        """Check if user is within rate limits."""
        now = datetime.now()
        if now - self.user_last_request[user_id] < timedelta(seconds=Config.RATE_LIMIT_SECONDS):
            return False
        self.user_last_request[user_id] = now
        return True
    
    async def run_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /run command."""
        # Only respond if command is specifically for this bot
        if update.message.text.lower() == '/run' or update.message.text.lower() == f'/run@{context.bot.username.lower()}':
            welcome_message = (
                "👋 Hi! I'm Your Media Download Bot.\n\n"
                "Send me a video link, and I'll:\n"
                "1. Download the video in best quality\n"
                "2. Convert it to high-quality audio\n"
                "3. Split and upload files if they're larger than 45MB\n\n"
                "Just send me any video URL after /fetch to get started!"
            )
            await update.message.reply_text(welcome_message)

    async def is_valid_url(self, url: str) -> bool:
        """Validate URL and check if it's supported."""
        # List of valid YouTube domains
        youtube_domains = [
            'youtube.com',
            'www.youtube.com',
            'youtu.be',
            'm.youtube.com',
            'music.youtube.com',
            'www.music.youtube.com'
        ]
        
        try:
            # First check if it's a YouTube URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            is_youtube = any(domain == yd for yd in youtube_domains)
            
            # If it's YouTube, we know it's supported
            if is_youtube:
                return True
                
            # For non-YouTube URLs, check with yt-dlp
            with YoutubeDL() as ydl:
                extractors = ydl.extract_info(url, download=False, process=False)
                return bool(extractors)
        except:
            return False

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
                    self.thread_pool,
                    lambda: ydl.extract_info(url, download=False)
                )
                
                if info:
                    # Try to get filesize
                    filesize = 0
                    formats = info.get('formats', [])
                    
                    # Get best format
                    best_format = None
                    max_quality = -1
                    
                    for f in formats:
                        quality = f.get('quality', -1)
                        if quality > max_quality and f.get('filesize'):
                            max_quality = quality
                            best_format = f
                    
                    if best_format:
                        filesize = best_format.get('filesize', 0)
                    
                    # Estimate from bitrate if no filesize
                    if not filesize and info.get('duration'):
                        max_bitrate = max(
                            (f.get('tbr', 0) for f in formats if f.get('tbr')),
                            default=0
                        )
                        if max_bitrate:
                            filesize = int((max_bitrate * 1024 * info['duration']) / 8)
                    
                    info['filesize'] = filesize
                    
                return info
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None

    async def download_video(self, url: str, chat_id: int) -> Optional[str]:
        """Download video in best quality."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(Config.DOWNLOAD_PATH, f'video_{chat_id}_{timestamp}.mp4')

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'concurrent_fragments': Config.CONCURRENT_FRAGMENTS,
            'buffersize': Config.BUFFER_SIZE,
            'http_chunk_size': Config.BUFFER_SIZE,
            'postprocessor_args': [
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-threads', str(Config.CPU_CORES)
            ]
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: ydl.download([url])
                )
            return output_path if os.path.exists(output_path) else None
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None

    async def convert_to_audio(self, video_path: str) -> Optional[str]:
        """Convert video to high-quality audio."""
        try:
            output_path = os.path.join(
                Config.DOWNLOAD_PATH,
                f"{os.path.splitext(os.path.basename(video_path))[0]}.mp3"
            )
            
            result = await asyncio.get_event_loop().run_in_executor(
                self.process_pool,
                convert_audio,
                video_path,
                output_path
            )
            
            return result
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return None
        
    async def split_file_async(self, file_path: str, chunk_size: int = Config.CHUNK_SIZE) -> List[Tuple[str, int]]:
        """Split a file into chunks asynchronously with proper video splitting."""
        file_size = os.path.getsize(file_path)
        if file_size <= chunk_size:
            return [(file_path, file_size)]
        
        temp_dir = tempfile.mkdtemp(dir=Config.DOWNLOAD_PATH)
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
                await process.communicate()
                
                if os.path.exists(chunk_path):
                    chunk_size = os.path.getsize(chunk_path)
                    if chunk_size > 0 and chunk_size <= Config.MAX_FILE_SIZE:
                        chunks.append((chunk_path, chunk_size))
                    else:
                        os.remove(chunk_path)
            
            return chunks if chunks else [(file_path, file_size)]
                
        except Exception as e:
            logger.error(f"Error splitting file: {e}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temp directory: {cleanup_error}")
            return [(file_path, file_size)]

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
            return not stderr
        except Exception as e:
            logger.error(f"Error validating video chunk: {e}")
            return False

    async def send_file(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, 
                    file_path: str, caption: str, reply_to_message_id: int, 
                    is_video: bool = True) -> bool:
        """Send a single file with retries and validation."""
        current_try = 0
        max_retries = Config.MAX_RETRIES
        
        while current_try < max_retries:
            try:
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    return False

                file_size = os.path.getsize(file_path)
                if file_size > Config.MAX_FILE_SIZE:
                    logger.error(f"File too large: {file_size / (1024*1024):.2f} MB")
                    return False

                if is_video and not await self.validate_video_chunk(file_path):
                    logger.error(f"Invalid video chunk: {file_path}")
                    return False

                async with aiofiles.open(file_path, 'rb') as file:
                    file_content = await file.read()
                    
                    if is_video:
                        await context.bot.send_video(
                            chat_id=chat_id,
                            video=file_content,
                            caption=caption,
                            reply_to_message_id=reply_to_message_id,
                            read_timeout=Config.READ_TIMEOUT,
                            write_timeout=Config.WRITE_TIMEOUT,
                            connect_timeout=Config.CONNECT_TIMEOUT,
                            supports_streaming=True
                        )
                    else:
                        await context.bot.send_audio(
                            chat_id=chat_id,
                            audio=file_content,
                            caption=caption,
                            reply_to_message_id=reply_to_message_id,
                            read_timeout=Config.READ_TIMEOUT,
                            write_timeout=Config.WRITE_TIMEOUT,
                            connect_timeout=Config.CONNECT_TIMEOUT
                        )
                    return True
                        
            except Exception as e:
                current_try += 1
                logger.warning(f"Attempt {current_try}/{max_retries} failed: {str(e)}")
                if current_try >= max_retries:
                    logger.error(f"Failed to send file after {max_retries} attempts: {str(e)}")
                    return False
                await asyncio.sleep(2 ** current_try)
            
        return False
    
    async def send_file_in_chunks(self, context: ContextTypes.DEFAULT_TYPE, 
                                chat_id: int, file_path: str, caption: str, 
                                reply_to_message_id: int, is_video: bool = True,
                                status_message = None) -> bool:
        """Enhanced chunked file sending with proper cleanup."""
        try:
            logger.info(f"Starting file upload: {file_path} (is_video={is_video})")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                if status_message:
                    await status_message.edit_text("❌ File not found for upload.")
                return False

            # Check if file is non-empty
            file_size = os.path.getsize(file_path)
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            if file_size == 0:
                logger.error(f"File is empty: {file_path}")
                if status_message:
                    await status_message.edit_text("❌ File is empty.")
                return False

            # For small files, send directly
            if file_size <= Config.CHUNK_SIZE:
                success = await self.send_file(
                    context, chat_id, file_path, 
                    f"{caption}\nSize: {self.format_size(file_size)}", 
                    reply_to_message_id, is_video
                )
                logger.info(f"Single file upload {'successful' if success else 'failed'}")
                if not success and status_message:
                    await status_message.edit_text("❌ Failed to send file.")
                return success

            # For larger files, split and send in chunks
            chunks = await self.split_file_async(file_path)
            if not chunks:
                if status_message:
                    await status_message.edit_text("❌ Failed to split file into chunks.")
                return False

            total_parts = len(chunks)
            sent_chunks = []

            for i, (chunk_path, chunk_size) in enumerate(chunks, 1):
                chunk_caption = (
                    f"{caption}\n"
                    f"Part {i}/{total_parts}\n"
                    f"Size: {self.format_size(chunk_size)}"
                )

                if status_message:
                    try:
                        await status_message.edit_text(
                            f"📤 Uploading {caption.split(':')[0]}\n"
                            f"Part {i}/{total_parts} ({self.format_size(chunk_size)})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update status message: {e}")

                if chunk_size > Config.MAX_FILE_SIZE:
                    logger.error(f"Chunk {i} exceeds size limit: {chunk_size / (1024*1024):.2f} MB")
                    if status_message:
                        await status_message.edit_text(
                            f"❌ Chunk {i} exceeds maximum size limit. Please try a smaller file."
                        )
                    # Clean up sent chunks
                    for sent_chunk in sent_chunks:
                        try:
                            if os.path.exists(sent_chunk):
                                os.remove(sent_chunk)
                        except Exception as e:
                            logger.warning(f"Failed to remove sent chunk: {e}")
                    return False

                success = await self.send_file(
                    context, chat_id, chunk_path,
                    chunk_caption, reply_to_message_id, is_video
                )

                if success:
                    sent_chunks.append(chunk_path)
                    try:
                        os.remove(chunk_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove chunk file: {e}")
                else:
                    logger.error(f"Failed to send chunk {i}/{total_parts}")
                    if status_message:
                        await status_message.edit_text(
                            f"❌ Failed to send part {i}/{total_parts}"
                        )
                    # Clean up remaining chunks
                    for chunk, _ in chunks[i:]:
                        try:
                            if os.path.exists(chunk):
                                os.remove(chunk)
                        except Exception as e:
                            logger.warning(f"Failed to remove chunk file: {e}")
                    return False

                await asyncio.sleep(2)  # Prevent rate limiting

            return True

        except Exception as e:
            logger.error(f"Error in chunked upload: {e}", exc_info=True)
            if status_message:
                await status_message.edit_text(
                    f"❌ An error occurred during upload: {str(e)}"
                )
            return False

    async def fetch_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /fetch command."""
        if not context.args:
            await update.message.reply_text(
                "❌ Please provide a video URL after the /fetch command.\n"
                "Example: /fetch https://www.example.com/video",
                reply_to_message_id=update.message.message_id
            )
            return

        url = context.args[0].strip()
        await self.process_video_request(update, context, url)

    async def handle_video_link(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle direct video link messages."""
        url = update.message.text.strip()
        await self.process_video_request(update, context, url)

    async def process_video_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
        """Process video download request."""
        chat_id = update.effective_chat.id
        message_id = update.message.message_id
        user_id = update.effective_user.id
        video_path = None
        audio_path = None
        status_message = None

        try:
            # Check rate limit
            if not await self.check_rate_limit(user_id):
                await update.message.reply_text(
                    "⏳ Please wait before making another request.",
                    reply_to_message_id=message_id
                )
                return

            # Check URL validity
            if not await self.is_valid_url(url):
                await update.message.reply_text(
                    "❌ Please provide a valid video URL.",
                    reply_to_message_id=message_id
                )
                return

            status_message = await update.message.reply_text(
                "🔍 Analyzing video...",
                reply_to_message_id=message_id
            )

            # Get video information
            video_info = await self.get_video_info(url)
            if not video_info:
                await status_message.edit_text("❌ Failed to get video information.")
                return

            title = video_info.get('title', 'Video')
            duration = video_info.get('duration', 0)
            filesize = video_info.get('filesize', 0)

            # Format duration
            duration_str = ""
            if duration:
                hours = duration // 3600
                minutes = (duration % 3600) // 60
                seconds = duration % 60
                if hours > 0:
                    duration_str = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    duration_str = f"{minutes}m {seconds}s"
                else:
                    duration_str = f"{seconds}s"

            # Warn if video is very long
            if duration > 1800:  # 30 minutes
                await status_message.edit_text(
                    f"⚠️ Warning: This video is {duration_str} long.\n"
                    "Processing may take a while and might fail.\n"
                    "Consider using a shorter video."
                )
                await asyncio.sleep(3)

            # Parse domain for YouTube check
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            is_youtube = any(domain == yd for yd in [
                'youtube.com', 'www.youtube.com', 'youtu.be',
                'm.youtube.com', 'music.youtube.com', 'www.music.youtube.com'
            ])

            download_message = (
                f"📥 Downloading: {title}\n"
                f"Duration: {duration_str}\n"
                f"Estimated size: {self.format_size(filesize)}\n"
            )
            
            if not is_youtube:
                download_message += "\n⚠️ Note: Non-YouTube links may take longer to process."
            
            await status_message.edit_text(download_message)

            # Download video
            video_path = await self.download_video(url, chat_id)
            if not video_path:
                await status_message.edit_text(
                    "❌ Download failed. This could be due to:\n"
                    "• Video is private or age-restricted\n"
                    "• Video is not available in your region\n"
                    "• Server issues\n\n"
                    "Please try again or use a different video."
                )
                return

            # Make a copy for audio processing
            audio_source_path = None
            if is_youtube or url.lower().endswith('audio'):
                try:
                    audio_source_path = f"{video_path}_audio_source.mp4"
                    shutil.copy2(video_path, audio_source_path)
                except Exception as e:
                    logger.error(f"Failed to copy video for audio processing: {e}")

            # Process video
            video_success = await self.process_video(
                context, chat_id, video_path, title, message_id, status_message
            )

            if not video_success:
                logger.error("Video processing failed")
                await status_message.edit_text("❌ Failed to process video.")
                return

            # Process audio if needed
            if audio_source_path and (is_youtube or url.lower().endswith('audio')):
                audio_path = await self.convert_to_audio(audio_source_path)
                if audio_path:
                    await self.process_audio(
                        context, chat_id, audio_path, title, message_id, status_message
                    )
                else:
                    await status_message.edit_text("❌ Audio conversion failed.")

        except Exception as e:
            logger.error(f"Error in process_video_request: {e}", exc_info=True)
            if status_message:
                await status_message.edit_text(
                    "❌ An error occurred while processing your request.\n"
                    "Please try again or use a different video."
                )
        finally:
            # Cleanup
            try:
                for path in [video_path, audio_source_path, audio_path]:
                    if path and os.path.exists(path):
                        os.remove(path)
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def process_video(self, context, chat_id, video_path, title, message_id, status_message):
        """Process and send video file."""
        try:
            if not os.path.exists(video_path):
                logger.error("Video file not found")
                await status_message.edit_text("❌ Video file not found.")
                return False

            video_size = os.path.getsize(video_path)
            if video_size == 0:
                logger.error("Video file is empty")
                await status_message.edit_text("❌ Video file is empty.")
                return False

            logger.info(f"Processing video of size: {video_size / (1024*1024):.2f} MB")
            await status_message.edit_text(
                f"📤 Uploading video ({self.format_size(video_size)})..."
            )

            video_success = await self.send_file_in_chunks(
                context, chat_id, video_path,
                f"🎥 Video: {title}",
                message_id, True, status_message
            )

            return video_success

        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            await status_message.edit_text("❌ Error processing video.")
            return False

    async def process_audio(self, context, chat_id, audio_path, title, message_id, status_message):
        """Process and send audio file."""
        try:
            if not os.path.exists(audio_path):
                logger.error("Audio file not found")
                await status_message.edit_text("❌ Audio file not found.")
                return False

            audio_size = os.path.getsize(audio_path)
            await status_message.edit_text(
                f"📤 Uploading audio ({self.format_size(audio_size)})..."
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
                
            return audio_success

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            await status_message.edit_text("❌ Error processing audio.")
            return False

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors in the telegram bot."""
    logger.error(f"Update {update} caused error {context.error}")
    
    error_message = "❌ An error occurred while processing your request."
    
    if isinstance(context.error, TimeoutError):
        error_message = "⏳ Request timed out. Please try again."
    
    if update.message:
        await update.message.reply_text(error_message)

def main():
    """Start the bot."""
    # Initialize request parameters
    request = HTTPXRequest(
        connect_timeout=Config.CONNECT_TIMEOUT,
        read_timeout=Config.READ_TIMEOUT,
        write_timeout=Config.WRITE_TIMEOUT,
        pool_timeout=Config.POOL_TIMEOUT
    )

    # Create application
    application = (
        Application.builder()
        .token(Config.TOKEN)
        .request(request)
        .build()
    )
    
    # Initialize bot
    bot = MediaBot()
    
    # Add handlers with specific filters for the bot
    application.add_handler(CommandHandler(
        "run", 
        bot.run_command,
        filters.ChatType.PRIVATE | filters.ChatType.GROUPS
    ))
    
    application.add_handler(CommandHandler(
        "fetch", 
        bot.fetch_command,
        filters.ChatType.PRIVATE | filters.ChatType.GROUPS
    ))
    
    # URL handler with regex filter
    url_filter = (
        filters.TEXT & 
        filters.Regex(r'https?://\S+') & 
        ~filters.COMMAND & 
        filters.ChatType.GROUPS
    )
    
    application.add_handler(MessageHandler(url_filter, bot.handle_video_link))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Start the bot
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
