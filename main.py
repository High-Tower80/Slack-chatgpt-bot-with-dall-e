import os
import random
import sys
import requests
from datetime import datetime, timedelta
from typing import Optional
from urllib.request import urlopen
import openai
from flask import Response

from dotenv import load_dotenv
from openai import InvalidRequestError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging

from _info import __version__
from home import build_app_home_blocks

dm_channel_ids = {}

def generate_image(prompt):
	# Call the DALL-E API to generate the image (replace with actual API call)
	response = openai.Image.create(prompt=prompt, n=1)
	# Extract the image URL from the response (replace with actual logic to get the URL)
	image_url = response['data'][0]['url']
	return image_url

def process_message(message):
	# Process the message and generate a response
	# Replace this with your desired message processing logic
	response = f"Got it :salute:"
	return response

def valid_input(value: Optional[str]) -> bool:
	return value is not None and value.strip() != ''


def get_env(key: str, default: Optional[str]) -> str:
	value = os.getenv(key, default)
	if not valid_input(value):
		value = default
	return value


def log(content: str, error: bool = False):
	now = datetime.now()
	print(f'[{now.isoformat()}] {content}', flush=True, file=sys.stderr if error else sys.stdout)


# Load environment variables
load_dotenv()

# Integration tokens and keys
SLACK_BOT_TOKEN = get_env('SLACK_BOT_TOKEN', None)
SLACK_APP_TOKEN = get_env('SLACK_APP_TOKEN', None)
OPENAI_API_KEY = get_env('OPENAI_API_KEY', None)

# Event API, Web API and OpenAI API
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)
bot_user_id = client.auth_test()["user_id"]
openai.api_key = OPENAI_API_KEY

# ChatGPT configuration: change to 'gpt-3.5-turbo' if needed
# Image size can be changed here, Must be one of 256x256, 512x512, or 1024x1024.
model = get_env('GPT_MODEL', 'gpt-4')
system_desc = get_env('GPT_SYSTEM_DESC', 'You are a helpful assistant.')
image_size = get_env('GPT_IMAGE_SIZE', '512x512')

# Slash Command function haven't got this work if any one can figure out, thx
def handle_slash_command(ack, command, respond):
    text = command['text']
    
    response_text = generate_response(text)
    
    response = {
        'text': response_text,
        'response_type': 'in_channel',  # Add this line to make the response visible to everyone in the channel
    }
    
    respond(response)
    ack()

app.command('/chatgpt')(handle_slash_command)



# Function to generate a response using GPT-4
def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": system_desc}, {"role": "user", "content": prompt}],
        max_tokens=150,
        n=1,
        temperature=0.9,
    )
    return response.choices[0].message['content'].strip()

# Keep chat history to provide context for future prompts
chat_history = {
	'general': []
}
history_expires_seconds = int(get_env('HISTORY_EXPIRES_IN', '900'))  # 15 minutes
history_size = int(get_env('HISTORY_SIZE', '3'))

# Keep timestamps of last requests per channel
last_request_datetime = {}

if not os.path.exists('./tmp'):
	os.makedirs('./tmp')

@app.event("app_home_opened")
def handle_app_home(body, logger):
	user = body["event"]["user"]
	logger.info(f"App Home opened by user {user}")
	
	# Open a direct message channel with the user
	dm_channel = client.conversations_open(users=user)
	
	# Get the channel ID from the response
	channel_id = dm_channel["channel"]["id"]
	dm_channel_ids[user] = channel_id

	# Fetch the conversation history using the channel ID
	result = client.conversations_history(channel=channel_id, limit=1)
	
	if not result["messages"]:
		# Send a welcome message if no previous message exists
		welcome_message = "Welcome:robot_face:! To start interacting with me, send a message right here! To utilize DALL-E (a powerful image generation AI), start your input statement with image:\nGPT MODEL: gpt-3.5-turbo and 4"
		client.chat_postMessage(channel=user, text=welcome_message)
			
	# Update the App Home tab
	update_app_home(body, logger)
	
def update_app_home(body, logger):
	user_id = body["event"]["user"]
	# Get the block kit structure from home.py
	blocks = build_app_home_blocks()
	try:
		client.views_publish(
			user_id=user_id,
			view={
				"type": "home",
				"blocks": blocks
			}
		)
	except SlackApiError as e:
		logger.error(f"Error publishing App Home: {e}")
		
def update_home_tab(client, event, logger):
	try:
		client.views_publish(
			user_id=event["user"],
			view={
				"type": "home",
				"blocks": build_app_home_blocks(),
			},
		)
	except Exception as e:
		logger.error(f"Error publishing home tab: {e}")
		

@app.action("go_to_messages")
def handle_go_to_messages(ack, body, logger, client):
	user_id = body["user"]["id"]
	ack()

	prompt = "puppy at sunset"  # Modify this as needed
	image_url = generate_image(prompt)

	client.chat_postEphemeral(
		channel=body["container"]["channel_id"],
		user=user_id,
		text=f"Here is an image of a {prompt}: {image_url}\nTo generate more images, please send a message like `image: {prompt}` in the *Messages* tab of the app home."
	)


# Buttons to link out to a URL of policies and resources of your choice in Slack App Home. Build a App home with Slack's block kit builder or turn off this feature from slack app setup
@app.action("policies-button-action")
def handle_some_action(ack, body, logger):
		ack()
		logger.info(body)

@app.shortcut("chatgpt")
def handle_shortcuts(ack, body, logger):
		ack()
		logger.info(body)

@app.action("button-action")
def handle_some_action(ack, body, logger):
		ack()
		logger.info(body)
		

# This NOT working, but also does not hinder DM messaging, its main use. Error msg in README.
# Activated when the bot is tagged in a channel. 
@app.event("app_mention")
def handle_message_events(body, say):
    if body['event'].get('subtype') == 'bot_message' or body['event'].get('bot_id'):
        return
    text = body['event'].get('text', '')
    if text.startswith('<@' + str(bot_user_id) + '>'):
        # Extract the actual message text
        actual_message = text.split('>', 1)[1].strip()
        if len(actual_message) > 0:
            # Process the message and generate a response
            response = process_message(actual_message)
            # Respond in the channel
            say(response)
        else:
            # If the bot is mentioned without any message
            say("Hi there! I'm here to help. Please provide a message after mentioning me.")

# Activated when the bot receives a direct message
@app.event('message')
def handle_message_events(body, logger):
	bot_id = body['event'].get('bot_id')
	if body['event'].get('subtype') == 'bot_message' or (bot_id is not None and '@' + bot_id in body['event']['text']):
		return  # Ignore bot messages and messages that tag the bot (handled by app_mention)

	prompt = str(body['event']['text']).strip()
	channel = body['event']['channel']
	user = body['event']['user']
	is_im_channel = client.conversations_info(channel=channel)['channel']['is_im']

	if is_im_channel:
		thread_ts = body['event']['thread_ts'] if 'thread_ts' in body['event'] else None
		handle_prompt(prompt, channel, user, thread_ts, direct_message=True)

#url shortener tinyurl for images
def shorten_url(url: str) -> str:
	try:
		response = requests.get(f'https://tinyurl.com/api-create.php?url={url}')
		if response.status_code == 200:
			return response.text
		else:
			return url
	except requests.RequestException:
		return url


def handle_prompt(prompt, user, channel, thread_ts=None, direct_message=False, in_thread=False):
	# Log requested prompt
	log(f'Channel {channel} received message: {prompt}')

	# Initialize the last request datetime for this channel
	if channel not in last_request_datetime:
		last_request_datetime[channel] = datetime.fromtimestamp(0)

	# Let the user know that we are busy with the request if enough time has passed since last message
	if last_request_datetime[channel] + timedelta(seconds=history_expires_seconds) < datetime.now():
		client.chat_postMessage(channel=channel,
								thread_ts=thread_ts,
								text=random.choice([
									'Generating... :gear:',
									'Beep beep :robot_face:',
									'hm :thinking_face:',
									'On it :saluting_face:'
								]))

	# Set current timestamp
	last_request_datetime[channel] = datetime.now()

	# Read parent message content if called inside thread conversation
	parent_message_text = None
	if thread_ts and not direct_message:
		conversation = client.conversations_replies(channel=channel, ts=thread_ts)
		if len(conversation['messages']) > 0 and valid_input(conversation['messages'][0]['text']):
			parent_message_text = conversation['messages'][0]['text']

	# Handle empty prompt
	if len(prompt.strip()) == 0 and parent_message_text is None:
		log('Empty prompt received')
		return

	if prompt.lower().startswith('image:'):
		# Generate DALL-E image command based on the prompt
		base_image_prompt = prompt[6:].strip()
		image_prompt = base_image_prompt

		# Append parent message text as prefix if exists
		if parent_message_text:
			image_prompt = f'{parent_message_text}. {image_prompt}'
			log('Using parent message inside thread')

		if len(image_prompt) == 0:
			text = 'Please check your input. To generate image use this format -> image: robot walking a dog'
		else:
			# Generate image based on prompt text
			try:
				response = openai.Image.create(prompt=image_prompt, n=1, size=image_size)
			except InvalidRequestError as e:
				log(f'ChatGPT image error: {e}', error=True)
				# Reply with error message
				client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=str(e))
				return

			image_long_image_url = response.data[0].url
			image_url = shorten_url(image_long_image_url)

			image_path = None
			try:
			   # Read image from URL
			   image_content = urlopen(image_url).read()
		   
			   # Prepare image name and path. 
			   short_prompt = base_image_prompt if valid_input(base_image_prompt) else image_prompt[:30].strip()
			   image_name = f"{short_prompt.replace(' ', '_')}.png"
			   image_path = f'./tmp/{image_name}'
		   
			   # Write file in temp directory
			   image_file = open(image_path, 'wb')
			   image_file.write(image_content)
			   image_file.close()
		   
			   # Upload image to Slack and send message with image to channel so you have it neatly named with your prompt and saved permanently
			   upload_response = client.files_upload_v2(
				   channel=dm_channel_ids.get(user, user),
				   thread_ts=thread_ts,
				   title=short_prompt,
				   filename=image_name,
				   file=image_path
			   )
		   
			   # Set text variable for logging purposes only
			   text = upload_response['file']['url_private']
			except SlackApiError as e:
			   text = None
			   log(f'Slack API error: {e}', error=True)
		   
		   # Remove temp image
			if image_path and os.path.exists(image_path):
			   os.remove(image_path)
	else:
		# Generate chat response
		now = datetime.now()

		# Add history messages if not expired
		history_messages = []
		if channel in chat_history:
			for channel_message in chat_history[channel]:
				if channel_message['created_at'] + timedelta(seconds=history_expires_seconds) < now or \
						channel_message['thread_ts'] != thread_ts or parent_message_text == channel_message['content']:
					continue
				history_messages.append({'role': channel_message['role'], 'content': channel_message['content']})
		else:
			chat_history[channel] = []

		# Log used history messages count
		log(f'Using {len(history_messages)} messages from chat history')

		# Append parent text message from current thread
		if parent_message_text:
			history_messages.append({'role': 'user', 'content': parent_message_text})
			log(f'Adding parent message from thread with timestamp: {thread_ts}')

		# Combine messages from history, current prompt and system if not disabled
		messages = [
			*history_messages,
			{'role': 'user', 'content': prompt}
		]
		if system_desc.lower() != 'none':
			messages.insert(0, {'role': 'system', 'content': system_desc})

		# Send request to ChatGPT
		try:
			response = openai.ChatCompletion.create(model=model, messages=messages)
		except InvalidRequestError as e:
			log(f'ChatGPT response error: {e}', error=True)
			# Reply with error message
			client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=str(e))
			return

		# Prepare response text
		text = response.choices[0].message.content.strip('\n')

		# Add messages to history
		chat_history[channel].append({'role': 'user', 'content': prompt, 'created_at': now, 'thread_ts': thread_ts})
		chat_history[channel].append(
			{'role': 'assistant', 'content': text, 'created_at': datetime.now(), 'thread_ts': thread_ts})

		# Remove the oldest 2 history message if the channel history size is exceeded for the current thread
		if len(list(filter(lambda x: x['thread_ts'] == thread_ts, chat_history[channel]))) >= (history_size + 1) * 2:
			# Create iterator for chat history list
			chat_history_list = (msg for msg in chat_history[channel] if msg['thread_ts'] == thread_ts)
			first_occurance = next(chat_history_list, None)
			second_occurance = next(chat_history_list, None)

			# Remove first occurance
			if first_occurance:
				chat_history[channel].remove(first_occurance)

			# Remove second occurance
			if second_occurance:
				chat_history[channel].remove(second_occurance)

		# Reply answer to thread
		if direct_message:
			target_channel = dm_channel_ids.get(user, user)
		else:
    			target_channel = channel

		client.chat_postMessage(channel=target_channel, thread_ts=thread_ts, text=text, reply_broadcast=in_thread)


	# Log response text
	log(f'ChatGPT response: {text}')


if __name__ == '__main__':
	try:
		print(f'ChatGPT Slackbot version {__version__}')
		SocketModeHandler(app, SLACK_APP_TOKEN).start()
	except KeyboardInterrupt:
		log('Stopping server')
