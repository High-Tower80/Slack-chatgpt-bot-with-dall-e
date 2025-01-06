import os
import random
import sys
import requests
import gspread
import openai
import tempfile
import io
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.request import urlopen
from flask import Response
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from logging.handlers import RotatingFileHandler
import traceback
import time
import threading
from functools import wraps
from PyPDF2 import PdfReader
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account

# Move helper functions to the top, right after imports
def valid_input(value: Optional[str]) -> bool:
	return value is not None and value.strip() != ''

def get_env(key: str, default: Optional[str] = None) -> str:
	value = os.getenv(key)
	if not valid_input(value):
		if default is None:
			raise ValueError(f"Environment variable {key} is required but not set")
		return default
	return value

def log_function_call(func):
	"""Decorator to log function calls with timing"""
	@wraps(func)
	def wrapper(*args, **kwargs):
		logger.info(f"🔵 Starting {func.__name__}")
		start_time = time.time()
		try:
			result = func(*args, **kwargs)
			duration = time.time() - start_time
			logger.info(f"✅ Completed {func.__name__} in {duration:.2f}s")
			return result
		except Exception as e:
			duration = time.time() - start_time
			logger.error(f"❌ Error in {func.__name__} after {duration:.2f}s: {str(e)}")
			logger.error(traceback.format_exc())
			raise
	return wrapper

# Force reload of environment variables
os.environ.clear()
load_dotenv(override=True)

# Update logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('slack_bot')

# File handlers with proper formatting
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Main log file
file_handler = logging.FileHandler('bot.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Error log file - capture actual errors
error_handler = logging.FileHandler('bot.error.log')
error_handler.setFormatter(formatter)
error_handler.setLevel(logging.ERROR)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(error_handler)

# Remove default handlers to avoid duplication
logger.propagate = False

dm_channel_ids = {}
pdf_contexts = {}
pdf_processing_lock = {}
last_success_messages = {}  # Track last success message per channel

# New global variables
last_request_datetime = {}  # Track the last request time for each channel
history_expires_seconds = int(get_env('HISTORY_EXPIRES_IN', '900'))  # Use the existing environment variable

CODE_VERSION = "1.0.1"  # Increment this when you make changes

# Global set to track processed file IDs
processed_files = set()

# Define the correct credentials path as a constant at the top of the file
GOOGLE_CREDS_PATH = 'PATHTOYOURLOGCREDS/google_sheets_creds.json'

# Set configuration constants
PDF_CONTEXT_EXPIRES_IN = int(get_env('PDF_CONTEXT_EXPIRES_IN', '3600'))  # 1 hour for PDF contexts
HISTORY_EXPIRES_IN = int(get_env('HISTORY_EXPIRES_IN', '900'))  # 15 minutes for regular chat
HISTORY_SIZE = int(get_env('HISTORY_SIZE', '3'))

# Integration tokens and keys
SLACK_BOT_TOKEN = get_env('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = get_env('SLACK_APP_TOKEN')
OPENAI_API_KEY = get_env('OPENAI_API_KEY')

# Initialize OpenAI configuration
openai.api_key = OPENAI_API_KEY

# Initialize the Slack app and client
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(token=SLACK_BOT_TOKEN)
slack_client = client

# Verify the client works on startup
try:
	auth_test = slack_client.auth_test()
	bot_user_id = auth_test["user_id"]
	logger.info(f"Bot user ID: {bot_user_id}")
except SlackApiError as e:
	logger.error(f"Error authenticating with Slack: {e}")
	raise

# ChatGPT configuration
model = get_env('GPT_MODEL', 'gpt-4')
system_desc = get_env('GPT_SYSTEM_DESC')
image_size = get_env('GPT_IMAGE_SIZE', '1024x1024')

def log_interaction_to_sheet(user_id, interaction, gpt_reply, channel_id=None, event_type=None, error_info=None, response_time=None):
	"""Log interactions to Google Sheet with proper user identification"""
	try:
		# Use the defined constant path
		if not os.path.exists(GOOGLE_CREDS_PATH):
			logger.error(f"Google Sheets credentials not found at {GOOGLE_CREDS_PATH}")
			return

		scope = ["https://spreadsheets.google.com/feeds",
				"https://www.googleapis.com/auth/spreadsheets",
				"https://www.googleapis.com/auth/drive.file",
				"https://www.googleapis.com/auth/drive"]

		creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_PATH, scope)
		sheets_client = gspread.authorize(creds)
		
		# Get user info from Slack
		try:
			user_info = slack_client.users_info(user=user_id)
			username = user_info['user']['name']
		except Exception as e:
			logger.error(f"Could not get user info: {e}")
			username = user_id

		worksheet = sheets_client.open("Slackbot ChatGPT Logs").sheet1

		current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		row_data = [
			current_time,
			username,  # Use actual username instead of just ID
			str(interaction)[:1000] if interaction else '',
			str(gpt_reply)[:1000] if gpt_reply else '',
			channel_id or '',
			event_type or '',
			str(error_info)[:500] if error_info else '',
			response_time or ''
		]

		worksheet.append_row(row_data)
		logger.info(f"Successfully logged interaction for user {username} to Google Sheet")

	except Exception as e:
		logger.error(f"Error logging to Google Sheet: {str(e)}")
		logger.error(traceback.format_exc())

# Rest of your code...

@log_function_call
def handle_pdf(file_info, channel_id, thread_ts=None):
	"""Process PDF Files."""
	try:
		
		user_id = file_info.get('shared_by_user')
		logger.info(f"Processing PDF: {file_info['name']} in channel {channel_id} from user {user_id}")

		slack_client.chat_postMessage(
			channel=channel_id,
			thread_ts=thread_ts,
			text="📄 Processing your PDF, please wait..."
		)

		# Get file content
		headers = {'Authorization': f'Bearer {SLACK_BOT_TOKEN}'}
		response = requests.get(file_info['url_private'], headers=headers, timeout=10)
		response.raise_for_status()

		# Process PDF content
		with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
			tmp_file.write(response.content)
			tmp_file.flush()
			pdf_text = extract_text_from_pdf(tmp_file.name)
			os.unlink(tmp_file.name)

		if not pdf_text:
			raise ValueError("No text could be extracted from PDF")

		# Store context
		pdf_contexts[channel_id] = {
			'text': pdf_text,
			'timestamp': datetime.now(timezone.utc),
			'processing': False,
			'file_name': file_info.get('name', 'Unknown'),
			'shared_by_user': user_id,
			'expires_at': datetime.now(timezone.utc) + timedelta(seconds=PDF_CONTEXT_EXPIRES_IN)
		}

		slack_client.chat_postMessage(
			channel=channel_id,
			thread_ts=thread_ts,
			text=f"✅ I've successfully read '{file_info.get('name')}'. You can now ask me questions about this PDF!"
		)

	except Exception as e:
		logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
		if channel_id in pdf_contexts:
			pdf_contexts.pop(channel_id)
		slack_client.chat_postMessage(
			channel=channel_id,
			thread_ts=thread_ts,
			text="❌ Sorry, I couldn't process the PDF. Please try again."
		)

def extract_text_from_pdf(pdf_path):
	"""Extract text from a PDF file."""
	try:
		reader = PdfReader(pdf_path)
		text = ""
		for page in reader.pages:
			text += page.extract_text() or ""
		return text.strip()
	except Exception as e:
		logger.error(f"Error extracting PDF text: {str(e)}", exc_info=True)
		return None

@app.event("app_mention")
def handle_app_mention(body, say):
	"""Handle mentions of the bot."""
	try:
		event = body['event']
		text = event.get('text', '').strip()
		channel = event['channel']
		thread_ts = event.get('thread_ts')
		user = event['user']

		# Remove bot mention from text
		text = text.replace(f'<@{bot_user_id}>', '').strip()

		if text.lower().startswith('image:'):
			prompt = text[6:].strip()
			handle_image_request(prompt, channel, thread_ts)
		else:
			response = handle_prompt(text, user, channel, thread_ts)
			if response:
				say(text=response, thread_ts=thread_ts)

	except Exception as e:
		logger.error(f"Error in app_mention handler: {str(e)}", exc_info=True)
		say("Sorry, I encountered an error processing your request.")

@app.event("message")
def handle_message(body, say):
	"""Handle direct messages and channel messages."""
	try:
		event = body['event']
		
		# Ignore bot messages
		if event.get('subtype') == 'bot_message' or event.get('bot_id'):
			return

		# Handle file shares separately
		if event.get('files'):
			return

		text = event.get('text', '').strip()
		channel = event['channel']
		thread_ts = event.get('thread_ts')
		user = event.get('user')

		if not text:
			return

		# Handle image requests
		if text.lower().startswith('image:'):
			prompt = text[6:].strip()
			if not prompt:
				say("Please provide a description of the image you'd like me to generate.")
				return
			handle_image_generation(prompt, channel, thread_ts, say, user)
			return

		response = handle_prompt(text, user, channel, thread_ts)
		if response:
			say(text=response, thread_ts=thread_ts)

	except Exception as e:
		logger.error(f"Error in message handler: {str(e)}", exc_info=True)
		say("Sorry, I encountered an error processing your request.")

def handle_prompt(prompt, user, channel, thread_ts=None, direct_message=False, in_thread=False):
	"""Handle all types of prompts."""
	logger.info(f'Channel {channel} received message: {prompt}')

	# Check if we need to send a loading message
	if channel not in last_request_datetime:
		last_request_datetime[channel] = datetime.fromtimestamp(0)

	if last_request_datetime[channel] + timedelta(seconds=history_expires_seconds) < datetime.now():
		client.chat_postMessage(
			channel=channel,
			thread_ts=thread_ts,
			text=random.choice([
				'Generating... :gear:',
				'Beep beep :robot_face:',
				'hm :thinking_face:',
				'On it :saluting_face:'
			])
		)

	try:
		start_time = time.time()
		
		messages = [{"role": "system", "content": system_desc}]
		
		if channel in pdf_contexts:
			pdf_text = pdf_contexts[channel]['text']
			messages.append({"role": "system", "content": f"Document content: {pdf_text}"})
		
		messages.append({"role": "user", "content": prompt})

		response = openai.chat.completions.create(
			model=model,
			messages=messages,
			temperature=0.7,
			max_tokens=1000
		)

		reply = response.choices[0].message.content.strip()
		# Add markdown conversion here
		reply = convert_to_slack_markdown(reply)
		
		# Log the interaction with timing
		response_time = f"{(time.time() - start_time):.2f}s"
		log_interaction_to_sheet(
			user_id=user,
			interaction=prompt,
			gpt_reply=reply,
			channel_id=channel,
			event_type='chat',
			response_time=response_time
		)

		return reply

	except Exception as e:
		logger.error(f"Error in handle_prompt: {str(e)}")
		# Log the error
		log_interaction_to_sheet(
			user_id=user,
			interaction=prompt,
			gpt_reply=None,
			channel_id=channel,
			event_type='error',
			error_info=str(e)
		)
		return "Sorry, I encountered an error processing your request."

@log_function_call
def handle_image_request(prompt, channel, thread_ts=None):
	"""Handle image generation requests"""
	filename = f"{prompt[:30].strip().replace(' ', ' ').lower()}.png"
	client.chat_postMessage(
		channel=channel,
		thread_ts=thread_ts,
		text=":art: Creating your image with DALL-E 3... (this may take up to 15 seconds)"
	)

	try:
		response = openai.images.generate(
			model="dall-e-3",
			prompt=prompt,
			n=1,
			size="1024x1024",
			quality="hd"
		)

		image_url = response.data[0].url
		image_response = requests.get(image_url)

		with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
			tmp_file.write(image_response.content)
			tmp_file.flush()

			client.files_upload_v2(
				channel=channel,
				thread_ts=thread_ts,
				file=tmp_file.name,
				filename=filename,
				initial_comment=f"Here's your DALL-E 3 image of: {prompt}"
			)

		os.unlink(tmp_file.name)

	except Exception as e:
		client.chat_postMessage(
			channel=channel,
			thread_ts=thread_ts,
			text=f"❌ Sorry, I couldn't generate the image: {str(e)}"
		)

def handle_image_generation(prompt, channel, thread_ts, say, user):
	"""Handle image generation separately"""
	try:
		start_time = time.time()
		
		filename = f"{prompt[:30].strip().replace(' ', ' ').lower()}.png"
		say(":noto_paint: Creating your image with DALL-E 3... (this may take up to 15 seconds)")

		response = openai.images.generate(
			model="dall-e-3",
			prompt=prompt,
			n=1,
			size="1024x1024",
			quality="hd"
		)

		image_url = response.data[0].url
		image_response = requests.get(image_url)

		with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
			tmp_file.write(image_response.content)
			tmp_file.flush()

			client.files_upload_v2(
				channel=channel,
				thread_ts=thread_ts,
				file=tmp_file.name,
				filename=filename,
				initial_comment=f"Here's your DALL-E 3 image of: {prompt}"
			)

		os.unlink(tmp_file.name)

		# Log successful image generation
		response_time = f"{(time.time() - start_time):.2f}s"
		log_interaction_to_sheet(
			user_id=user,
			interaction=f"Image generation: {prompt}",
			gpt_reply="Image generated successfully",
			channel_id=channel,
			event_type='image_generation',
			response_time=response_time
		)

	except Exception as e:
		error_msg = str(e)
		say(f"❌ Sorry, I couldn't generate the image: {error_msg}")
		
		# Log failed image generation
		log_interaction_to_sheet(
			user_id=None,  # We'll need to pass user ID from the message handler
			interaction=f"Image generation: {prompt}",
			gpt_reply=None,
			channel_id=channel,
			event_type='image_generation_error',
			error_info=error_msg
		)

def handle_image(image_prompt, channel, thread_ts, say):
	# Add a loading message first
	loading_message = client.chat_postMessage(
				channel=channel,
				thread_ts=thread_ts,
				text="🎨 Generating your image... (this may take up to 30 seconds)"
			)

	image_path = None
	try:
		# DALL-E call
		response = openai.images.generate(
			model="dall-e-3",
			prompt=image_prompt,
			n=1,
			size="1024x1024"
		)
		image_url = shorten_url(response.data[0].url)

		# Use context manager for file operations
		image_name = f"{image_prompt[:30].replace(' ', '_')}.png"
		image_path = os.path.join('./tmp', image_name)

		with urlopen(image_url) as response, open(image_path, 'wb') as image_file:
			image_file.write(response.read())

		upload_response = client.files_upload_v2(
			channel=channel,
			thread_ts=thread_ts,
			initial_comment="Here is your image:",
			file=image_path
		)
	except Exception as e:
		logger.error(f'ChatGPT image error: {e}')
		say(text=str(e), thread_ts=thread_ts)
	finally:
		if image_path and os.path.exists(image_path):
			os.remove(image_path)

@app.event("file_shared")
def handle_file_shared(body, say):
	try:
		event = body['event']
		file_id = event['file_id']
		channel_id = event['channel_id']
		
		logger.info(f"File shared event received - ID: {file_id}, Channel: {channel_id}")
		
		# Get file info
		try:
			file_info = slack_client.files_info(file=file_id).get('file')
			if file_info['filetype'].lower() == 'pdf':
				handle_pdf(file_info, channel_id)
		except SlackApiError as e:
			logger.error(f"Error getting file info: {e.response['error']}")
			
	except Exception as e:
		logger.error(f"Error in file_shared handler: {str(e)}", exc_info=True)

def shorten_url(url: str) -> str:
	"""Shorten a URL using TinyURL service"""
	try:
		response = requests.get(f'https://tinyurl.com/api-create.php?url={url}')
		return response.text if response.status_code == 200 else url
	except requests.RequestException:
		return url

def convert_to_slack_markdown(text):
	"""Convert standard markdown to Slack's markdown format"""
	text = text.replace('**', '*')
	text = text.replace('__', '_')
	text = text.replace('```', '```')
	text = text.replace('`', '`')
	text = text.replace('###', '*')
	text = text.replace('##', '*')
	text = text.replace('#', '*')

	lines = text.split("\n")
	formatted_lines = []
	for line in lines:
		if line.startswith("- "):
			formatted_lines.append("• " + line[2:])
		elif line.startswith("    - "):
			formatted_lines.append("    • " + line[6:])
		elif line.startswith("  - "):
			formatted_lines.append("  • " + line[4:])
		else:
			formatted_lines.append(line)
	text = "\n".join(formatted_lines)

	return text

# Add at the bottom of the file
if __name__ == "__main__":
	try:
		handler = SocketModeHandler(app, SLACK_APP_TOKEN)
		handler.start()
	except Exception as e:
		logger.error(f"Fatal error: {str(e)}")
		logger.error(traceback.format_exc())
