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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

from dotenv import load_dotenv
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

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(error_handler)

# Remove default handlers to avoid duplication
logger.propagate = False

dm_channel_ids = {}
pdf_contexts = {}
pdf_processing_lock = {}
last_success_messages = {}  # Track last success message per channel

CODE_VERSION = "1.0.1"  # Increment this when you make changes

# Global set to track processed file IDs
processed_files = set()

def valid_input(value: Optional[str]) -> bool:
	return value is not None and value.strip() != ''

def get_env(key: str, default: Optional[str] = None) -> str:
	value = os.getenv(key)
	if not valid_input(value):
		if default is None:
			raise ValueError(f"Environment variable {key} is required but not set")
		return default
	return value

# Set configuration constants
PDF_CONTEXT_EXPIRES_IN = int(get_env('PDF_CONTEXT_EXPIRES_IN', '3600'))  # 1 hour for PDF contexts
HISTORY_EXPIRES_IN = int(get_env('HISTORY_EXPIRES_IN', '900'))  # 15 minutes for regular chat

# Integration tokens and keys
SLACK_BOT_TOKEN = get_env('SLACK_BOT_TOKEN', None)
SLACK_APP_TOKEN = get_env('SLACK_APP_TOKEN', None)
OPENAI_API_KEY = get_env('OPENAI_API_KEY', None)

# Event API, Web API and OpenAI API
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)

try:
	bot_info = client.auth_test()
	bot_user_id = bot_info["user_id"]
	logger.info(f"Bot user ID: {bot_user_id}")
except Exception as e:
	logger.error(f"Could not get bot user ID: {e}")
	raise

openai.api_key = OPENAI_API_KEY

# ChatGPT configuration
model = get_env('GPT_MODEL', 'gpt-4o')
system_desc = """You are a helpful assistant specializing in document analysis and information extraction. 
When asked about documents:
1. Provide clear, organized summaries
2. Extract key information accurately
3. Answer questions about the content directly
4. Format responses for easy reading and copying
5. Maintain professional tone

You can also help with general questions and generate images when requested."""
image_size = get_env('GPT_IMAGE_SIZE', '1024x1024')

def log_interaction_to_sheet(user_id, interaction, gpt_reply, channel_id=None, event_type=None, error_info=None, response_time=None):
	"""Log interactions to Google Sheet if credentials exist"""
	try:
		# Use actual filename
		creds_path = '/Users/tate/gptfixes/google_sheets_creds.json'
		logger.info(f"Attempting to use credentials from: {creds_path}")  # Debug log
		
		if not os.path.exists(creds_path):
			logger.error(f"Google Sheets credentials not found at {creds_path}")
			return

		scope = ["https://spreadsheets.google.com/feeds",
				"https://www.googleapis.com/auth/spreadsheets",
				"https://www.googleapis.com/auth/drive.file",
				"https://www.googleapis.com/auth/drive"]

		creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
		client = gspread.authorize(creds)
		sheet = client.open("Slackbot ChatGPT Logs").sheet1
		
		current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		row_data = [
			current_time,
			user_id,
			str(interaction)[:1000] if interaction else '',
			str(gpt_reply)[:1000] if gpt_reply else '',
			channel_id or '',
			event_type or '',
			str(error_info)[:500] if error_info else '',
			response_time or ''
		]
		
		sheet.append_row(row_data)
		logger.info(f"Successfully logged interaction to Google Sheet at {current_time}")
		
	except Exception as e:
		logger.error(f"Error logging to Google Sheet: {str(e)}")
		logger.error(traceback.format_exc())

def convert_to_slack_markdown(text):
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

def extract_text_from_pdf(pdf_path):
	"""Extract text from a PDF file."""
	try:
		logger.info(f"=== EXTRACTING TEXT FROM PDF: {os.path.basename(pdf_path)} ===")
		
		reader = PdfReader(pdf_path)
		text = ""
		total_pages = len(reader.pages)
		
		logger.info(f"PDF has {total_pages} pages")
		
		for i, page in enumerate(reader.pages, 1):
			logger.info(f"Processing page {i}/{total_pages}")
			page_text = page.extract_text() or ""
			text += page_text
			logger.info(f"Page {i} extracted: {len(page_text)} characters")
			
		return text.strip()
		
	except Exception as e:
		logger.error(f"Error extracting PDF text: {str(e)}", exc_info=True)
		return None

def log_function_call(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		logger.info(f"🔵 Starting {func.__name__}")
		try:
			result = func(*args, **kwargs)
			logger.info(f"✅ Completed {func.__name__}")
			return result
		except Exception as e:
			logger.error(f"❌ Error in {func.__name__}: {str(e)}")
			logger.error(traceback.format_exc())
			raise
	return wrapper

@log_function_call
def handle_pdf(file_info, channel_id, thread_ts=None):
	"""Process PDF Files."""
	try:
		logger.info(f"Processing PDF: {file_info['name']} in channel {channel_id}")
		
		client.chat_postMessage(
			channel=channel_id,
			thread_ts=thread_ts,
			text="📄 Processing your PDF, please wait..."
		)
		
		# Clear existing context
		if channel_id in pdf_contexts:
			logger.info(f"Clearing existing PDF context for channel {channel_id}")
			pdf_contexts.pop(channel_id)
		
		# Process the PDF...
		headers = {'Authorization': f'Bearer {SLACK_BOT_TOKEN}'}
		response = requests.get(file_info['url_private'], headers=headers, timeout=10)
		response.raise_for_status()
		
		# Extract text from PDF
		pdf_text = ""
		with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
			tmp_file.write(response.content)
			tmp_file.flush()
			pdf_text = extract_text_from_pdf(tmp_file.name)
			os.unlink(tmp_file.name)
			
		if not pdf_text:
			raise ValueError("No text could be extracted from PDF")
		
		# Update context
		pdf_contexts[channel_id] = {
			'text': pdf_text,
			'timestamp': datetime.now(timezone.utc),
			'processing': False,
			'file_name': file_info.get('name', 'Unknown'),
			'expires_at': datetime.now(timezone.utc) + timedelta(seconds=PDF_CONTEXT_EXPIRES_IN)
		}
		
		# Success message with clear instructions
		client.chat_postMessage(
			channel=channel_id,
			thread_ts=thread_ts,
			text=f"✅ I've successfully read '{file_info.get('name')}'. You can now ask me questions about this PDF!"
		)
		
	except Exception as e:
		logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
		if channel_id in pdf_contexts:
			pdf_contexts.pop(channel_id)
		client.chat_postMessage(
			channel=channel_id,
			thread_ts=thread_ts,
			text="❌ Sorry, I couldn't process the PDF. Please try again."
		)

@log_function_call
def generate_response(prompt, channel_id=None):
	"""Generate response with timeout"""
	logger.info(f"Generating for channel: {channel_id}")
	
	try:
		# Check if we have stored context
		if channel_id not in pdf_contexts:
			logger.info(f"No PDF context found for channel {channel_id}")
			return "Sorry, I don't have any PDF loaded to answer questions about."
			
		# Get the stored text - don't try to extract again
		stored_text = pdf_contexts[channel_id].get('text')
		logger.info(f"Found stored text of length: {len(stored_text) if stored_text else 0}")
		
		if not stored_text:
			logger.warning(f"No text found in stored context for channel {channel_id}")
			return "Sorry, I couldn't find the PDF content. Please try uploading again."
			
		messages = [
			{"role": "system", "content": "You are analyzing a PDF document. Provide direct, clear answers based on its content."},
			{"role": "system", "content": f"Document content: {stored_text}"},
			{"role": "user", "content": prompt}
		]
		
		logger.info("Sending request to OpenAI")
		response = openai.chat.completions.create(
			model=model,
			messages=messages,
			temperature=0.3,
			max_tokens=1000
		)
		
		return response.choices[0].message.content.strip()
		
	except Exception as e:
		logger.error(f"Error generating response: {e}")
		logger.error(traceback.format_exc())
		return "Sorry, I encountered an error processing your request."

chat_history = {
	'general': []
}
history_expires_seconds = int(get_env('HISTORY_EXPIRES_IN', '900'))
history_size = int(get_env('HISTORY_SIZE', '3'))

last_request_datetime = {}

# Remove file system dependencies
if not os.path.exists('./tmp'):
	try:
		os.makedirs('./tmp')
	except Exception as e:
		logger.warning(f"Could not create tmp directory: {e}")

def clear_all_caches():
	"""Clear all caches and states"""
	global processed_files, pdf_contexts
	processed_files = set()
	pdf_contexts = {}
	logger.info("Cleared all caches and states")

# Add this right after your imports and before app initialization
clear_all_caches()

@app.event("app_mention")
def handle_app_mention_events(body, say):
	event = body['event']
	text = event.get('text', '').strip()
	channel = event['channel']
	thread_ts = event.get('thread_ts')
	
	# Remove bot mention
	actual_message = text.replace(f'<@{bot_user_id}>', '').strip()
	
	# If it starts with 'image:', ONLY do image generation
	if actual_message.lower().startswith('image:'):
		prompt = actual_message[6:].strip()
		handle_image_request(prompt, channel, thread_ts)
		return  # Exit here, do nothing else
		
	# Only reach here if it's NOT an image request
	response = handle_prompt(actual_message, event['user'], channel, thread_ts, direct_message=False)
	if response:
		say(text=response, thread_ts=thread_ts)

@app.event("message")
def handle_message(body, say):
	try:
		# Ignore bot messages
		if body['event'].get('subtype') == 'bot_message' or body['event'].get('bot_id'):
			return
			
		# Ignore messages with files (including PDFs)
		if body['event'].get('files'):
			logger.info("Ignoring message with file attachment")
			return
			
		# Ignore messages that are part of a file share
		if body['event'].get('subtype') == 'file_share':
			logger.info("Ignoring file share message")
			return
			
		text = body['event'].get('text', '').strip()
		channel = body['event']['channel']
		thread_ts = body['event'].get('thread_ts')
		user = body['event'].get('user')

		# Skip empty messages
		if not text:
			return
			
		# Check if PDF is being processed
		if channel in pdf_contexts and pdf_contexts[channel].get('processing'):
			logger.info("PDF processing in progress - ignoring message")
			client.chat_postMessage(
				channel=channel,
				thread_ts=thread_ts,
				text="⏳ I'm still processing the PDF. Please wait until it's complete before asking questions."
			)
			return
			
		# Handle different message types
		if text.lower().startswith('image:'):
			prompt = text[6:].strip()
			if not prompt:
				say("Please provide a description of the image you'd like me to generate.")
				return
			handle_image_generation(prompt, channel, thread_ts, say)
			return
		
		# Regular message handling
		if channel in pdf_contexts and not pdf_contexts[channel].get('processing'):
			handle_pdf_question(text, channel, thread_ts, say)
		else:
			handle_regular_chat(text, channel, thread_ts, say)
			
	except Exception as e:
		logger.error(f"Error in message handler: {str(e)}", exc_info=True)
		say("Sorry, I encountered an error processing your request.")

def handle_image_generation(prompt, channel, thread_ts, say):
	"""Handle image generation separately"""
	filename = f"{prompt[:30].strip().replace(' ', ' ').lower()}.png"
	say(":noto_paint: Creating your image with DALL-E 3... (this may take up to 15 seconds)")
	
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
		say(f"❌ Sorry, I couldn't generate the image: {str(e)}")

def handle_pdf_question(text, channel, thread_ts, say):
	"""Handle PDF-related questions"""
	try:
		response = generate_response(text, channel)
		if response:
			say(text=response, thread_ts=thread_ts)
	except Exception as e:
		logger.error(f"Error handling PDF question: {e}")
		say("Sorry, I had trouble processing your question about the PDF.")

def handle_regular_chat(text, channel, thread_ts, say):
	"""Handle regular chat messages"""
	try:
		response = handle_prompt(text, None, channel, thread_ts)
		if response:
			say(text=response, thread_ts=thread_ts)
	except Exception as e:
		logger.error(f"Error in regular chat: {e}")
		say("Sorry, I had trouble processing your message.")

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

def shorten_url(url: str) -> str:
	try:
		response = requests.get(f'https://tinyurl.com/api-create.php?url={url}')
		return response.text if response.status_code == 200 else url
	except requests.RequestException:
		return url

@log_function_call
def handle_prompt(prompt, user, channel, thread_ts=None, direct_message=False):
	"""Handle all types of prompts"""
	logger.info(f"Prompt: {prompt[:50]}...")
	
	start_time = time.time()
	
	try:
		logger.info(f'Channel {channel} received message: {prompt}')
		
		# Send single thinking message
		
		# Check if we have PDF context first
		if channel in pdf_contexts:
			logger.info(f"PDF context for channel {channel} already exists.")
			return
		
		# If no PDF context, proceed with regular chat
		messages = [{"role": "system", "content": system_desc}]
		
		# Get chat history
		history = chat_history.get(channel, [])
		if thread_ts:
			history = [msg for msg in history if msg.get('thread_ts') == thread_ts]
		
		logger.info(f'Using {len(history)} messages from chat history')
		
		for msg in history[-4:]:
			messages.append({
				"role": "user" if msg.get('role') == "user" else "assistant",
				"content": msg.get('content', '')
			})
		
		messages.append({"role": "user", "content": prompt})
		
		logger.info('Sending request to OpenAI')
		response = openai.chat.completions.create(
			model=model,
			messages=messages,
			temperature=0.7,
			max_tokens=1000
		)
		
		answer = response.choices[0].message.content.strip()
		# Convert markdown to Slack format
		answer = convert_to_slack_markdown(answer)
		logger.info(f'ChatGPT response: {answer}')
		
		# Log interaction
		response_time = round(time.time() - start_time, 2)
		log_interaction_to_sheet(
			user_id=user,
			interaction=prompt,
			gpt_reply=answer,
			channel_id=channel,
			event_type='message',
			response_time=f"{response_time}s"
		)
		
		# Update chat history
		chat_history.setdefault(channel, []).extend([
			{
				'role': 'user',
				'content': prompt,
				'thread_ts': thread_ts,
				'timestamp': datetime.now().isoformat()
			},
			{
				'role': 'assistant',
				'content': answer,
				'thread_ts': thread_ts,
				'timestamp': datetime.now().isoformat()
			}
		])
		
		return answer
		
	except Exception as e:
		logger.error(f"Error in handle_prompt: {str(e)}")
		logger.error(traceback.format_exc())
		return "I apologize, but I encountered an error. Could you please try again?"

def handle_image_request(prompt, channel, thread_ts=None):
	"""Handle image generation requests"""
	logger.info(f"Starting image generation for prompt: '{prompt}'")
	try:
		# Send processing message
		client.chat_postMessage(
			channel=channel,
			thread_ts=thread_ts,
			text="🎨 Generating your image... (this may take up to 15 seconds)"
		)
		
		# Generate image using DALL-E
		logger.info("Calling OpenAI image generation API")
		response = openai.images.generate(
			model="dall-e-2",
			prompt=prompt,
			n=1,
			size="1024x1024"
		)
		
		image_url = response.data[0].url
		logger.info("Image generated successfully")
		
		# Send the image
		client.files_upload_v2(
			channel=channel,
			thread_ts=thread_ts,
			initial_comment="Here's your generated image:",
			file=image_url
		)
		
	except Exception as e:
		logger.error(f"Image generation error: {str(e)}")
		logger.error(f"Full error: {traceback.format_exc()}")
		client.chat_postMessage(
			channel=channel,
			thread_ts=thread_ts,
			text=f"❌ Sorry, I couldn't generate the image: {str(e)}"
		)

def cleanup_chat_history(channel, thread_ts):
	"""Clean up old messages from chat history"""
	try:
		now = datetime.now()
		if channel in chat_history:
				chat_history[channel] = [
					msg for msg in chat_history[channel]
					if msg.get('thread_ts') == thread_ts
				][-10:]  # Keep only last 10 messages per thread
	except Exception as e:
		logger.error(f"Error in cleanup_chat_history: {str(e)}")

def manage_history_size(channel, thread_ts):
	"""Maintain history size limit"""
	thread_messages = [msg for msg in chat_history[channel] if msg['thread_ts'] == thread_ts]
	if len(thread_messages) >= (history_size + 1) * 2:
		# Remove oldest pair of messages
		to_remove = thread_messages[:2]
		chat_history[channel] = [msg for msg in chat_history[channel] if msg not in to_remove]

def cleanup_expired_contexts():
	"""Clean up expired PDF contexts"""
	now = datetime.now()
	expired_channels = [
		channel for channel, context in pdf_contexts.items()
		if (now - context['timestamp']).total_seconds() > PDF_CONTEXT_EXPIRES_IN
	]
	for channel in expired_channels:
		del pdf_contexts[channel]

@app.event("file_shared")
def handle_file_shared(body, say):
	try:
		event = body['event']
		file_id = event['file_id']
		channel_id = event['channel_id']
		
		logger.info(f"File shared - ID: {file_id}, Channel: {channel_id}")
		
		if file_id in processed_files:
			return
			
		processed_files.add(file_id)
		file_info = client.files_info(file=file_id)['file']
		
		if file_info['filetype'].lower() == 'pdf':
			# Process PDF without any initial text
			handle_pdf(file_info, channel_id)
			
	except Exception as e:
		logger.error(f"Error in file_shared handler: {str(e)}", exc_info=True)

def handle_pdf_dm(event, channel_id):
	"""Handle PDF in direct messages"""
	logger.info(f'[Version {CODE_VERSION}] === START PDF PROCESSING ===')
	
	try:
		# Get file info first
		file_info = client.files_info(file=event['file_id'])['file']
		
		if file_info['filetype'].lower() != 'pdf':
			client.chat_postMessage(
				channel=channel_id,
				text="Sorry, I can only process PDF files."
			)
			return
		
		# Process PDF and store context
		if handle_pdf(file_info, channel_id):
			# Remove auto-summary - wait for user to ask questions instead
			pass
	except Exception as e:
		logger.error(f"Error in handle_pdf_dm: {e}")
		client.chat_postMessage(
				channel=channel_id,
				text="Sorry, I encountered an error processing your PDF."
			)

def handle_pdf_channel(event, channel_id):
	"""Handle PDF in channels - more structured"""
	# Only process when explicitly mentioned
	# Wait for thread to be created
	pass  # We'll implement this if needed

def handle_pdf_query(prompt, channel_id, thread_ts=None):
	"""Handle queries about PDF content"""
	logger.info(f'[Version {CODE_VERSION}] Processing PDF query in channel {channel_id}')
	
	try:
		# First check if we have the context
		if channel_id not in pdf_contexts:
			logger.info(f"No PDF context found for channel {channel_id}")
			return "Sorry, I don't have any PDF loaded to answer questions about."
			
		# Use the already extracted text from context
		context = pdf_contexts[channel_id]
		pdf_text = context.get('text')
		
		if not pdf_text:
			logger.warning(f"No text found in stored context for channel {channel_id}")
			return "Sorry, I couldn't find the PDF content. Please try uploading again."
		
		logger.info(f"Found stored text of length: {len(pdf_text)}")
		
		# Generate response using the stored text
		messages = [
			{"role": "system", "content": "You are analyzing a PDF document. Provide direct, clear answers based on its content."},
			{"role": "system", "content": f"Document content: {pdf_text}"},
			{"role": "user", "content": prompt}
		]
		
		response = openai.chat.completions.create(
			model=model,
			messages=messages,
			temperature=0.3,
			max_tokens=1000
		)
		
		return response.choices[0].message.content.strip()
		
	except Exception as e:
		logger.error(f"Error in handle_pdf_query: {e}")
		logger.error(traceback.format_exc())
		return "Sorry, I encountered an error processing your request."

def create_thread(channel, prompt):
	"""Create a new thread for channel conversations"""
	try:
		# Create initial thread message
		response = client.chat_postMessage(
			channel=channel,
			text=f"Starting new conversation about: {prompt[:50]}..."
		)
		return response['ts']  # Return thread timestamp
	except Exception as e:
		logger.error(f"Error creating thread: {e}")
		return None

def handle_channel_query(prompt, channel, thread_ts):
	"""Handle queries in public/private channels"""
	logger.info(f'[Version {CODE_VERSION}] Processing channel query in {channel}')
	
	try:
		# Check for PDF context first
		if channel in pdf_contexts:
			return handle_pdf_query(prompt, channel, thread_ts)
			
		# Otherwise handle as regular chat
		messages = [
			{"role": "system", "content": system_desc},
			{"role": "user", "content": prompt}
		]
		
		response = openai.chat.completions.create(
			model=model,
			messages=messages,
			temperature=0.7,
			max_tokens=1000
		)
		
		answer = response.choices[0].message.content.strip()
		
		client.chat_postMessage(
			channel=channel,
			thread_ts=thread_ts,
			text=answer
		)
		
		return None  # Already sent message
		
	except Exception as e:
		logger.error(f"Error in channel query: {e}")
		return "Sorry, I encountered an error processing your request."

def cli_test():
	"""Test mode for CLI debugging"""
	print("Starting CLI test mode...")
	
	while True:
		try:
			command = input("\nEnter command (or 'quit' to exit): ").strip()
			
			if command.lower() == 'quit':
				print("Exiting test mode...")
				break
				
			# Simulate message event
			event = {
				'type': 'message',
				'text': command,
				'channel': 'CLI_TEST',
				'user': 'CLI_USER'
			}
			
			response = handle_prompt(
				prompt=command,
				user='CLI_USER',
				channel='CLI_TEST',
				direct_message=True
			)
			
			if response:
				print(f"\nBot response:\n{response}")
				
		except KeyboardInterrupt:
			print("\nExiting test mode...")
			break
		except Exception as e:
			print(f"Error in test mode: {e}")
			logger.error(traceback.format_exc())

if __name__ == '__main__':
	try:
		clear_all_caches()  # Clear caches on startup
		handler = SocketModeHandler(app, SLACK_APP_TOKEN)
		handler.start()
	except Exception as e:
		logger.error(f"Fatal error: {str(e)}")
		logger.error(traceback.format_exc())
		
