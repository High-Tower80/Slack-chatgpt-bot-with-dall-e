# Slack-chatgpt-bot-with-dall-e
# OpenAI ChatGPT Slackbot

A powerful Slackbot app built with OpenAI's GPT-4 (switch to 3.5-turbo in main.py if you dont have beta access) language model and DALL-E image generation API. This app allows users to interact with GPT-4 directly from Slack. It also supports generating images with DALL-E by giving a text prompt and saves it to the slackbot's DM.

![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/SCR-20230521-pezn.jpeg)

![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/SCR-20230521-pelt.png)
## Features

1. Directly chat with GPT-4 from your Slack workspace.
2. Generate DALL-E images with a text prompt.
3. Automatically update the App Home tab with a block kit structure.
4. Handle bot mentions and direct messages.
5. Maintain chat history to provide context for future prompts.
6. URL shortener for image URLs.

## Getting Started

Before you can use the OpenAI ChatGPT Slackbot, you need to set up and configure your environment. 

### Prerequisites

1. Python 3.6 and above
2. OpenAI Python Client (openai) 0.27.0 or later
3. Slack Bolt for Python (slack_bolt) 1.9.1 or later
4. Python-dotenv (python-dotenv) 0.19.1 or later

### Slack bot and App scopes
        "scopes": {
            "bot": [
                "app_mentions:read",
                "channels:history",
                "channels:join",
                "chat:write",
                "commands",
                "files:write",
                "groups:history",
                "im:history",
                "im:read",
                "im:write",
                "incoming-webhook",
                "mpim:history",
                "users:read",
                "workflow.steps:execute",
                "files:read"

            "bot_events": [
                "app_home_opened",
                "app_mention",
                "message.channels",
                "message.im",
                "workflow_step_execute"

### Installation

1. Clone the repository

```
git clone https://github.com/your_username_/slackbot-gpt-4.git
```

2. Change into the project directory

```
cd slackbot-gpt-4
```

3. Install the required packages

```
pip install -r requirements.txt
```

### Configuration

1. Set up the Slack bot following the instructions [here](https://api.slack.com/start).
2. Obtain your API keys from Slack and OpenAI. You need the following:
   - Slack Bot Token
   - Slack App Token
   - OpenAI API Key
3. Set up your environment variables in a `.env` file. The variables you need to set are:
   - `SLACK_BOT_TOKEN`
   - `SLACK_APP_TOKEN`
   - `OPENAI_API_KEY`
   - `GPT_MODEL` (e.g. 'gpt-4')
   - `GPT_SYSTEM_DESC` (default is 'You are a helpful assistant.')
   - `GPT_IMAGE_SIZE` (default is '512x512')
   - `HISTORY_EXPIRES_IN` (default is '900' which is 15 minutes)
   - `HISTORY_SIZE` (default is '3')

### Running the Slackbot

1. Run the Slackbot with the following command:

```
python main.py
```

## Usage
![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/GPT%20problemsolving%20in%20slack.png)



### Interacting with GPT-4 or 3.5-turbo

KNOWN ISSUE: To interact with GPT-4 in a channel, mention the bot with your message
Cannot seem to sort out the channel convos. When a member tags @bot with a request in a channel its expected to reply within that thread. but it fails with this error:

```
Error publishing App Home: The request to the Slack API failed. (url: https://www.slack.com/api/views.publish)
The server responded with: {'ok': False, 'error': 'invalid_arguments', 'response_metadata': {'messages': ['[ERROR] failed to match all allowed schemas [json-pointer:/view]', '[ERROR] missing required field: image_url [json-pointer:/view/blocks/3/accessory]', '[ERROR] missing required field: alt_text [json-pointer:/view/blocks/3/accessory]']}}
```

Please if anyone can figure out why.
DMing the bot is its main use and all features work in the App's Messages window
	
```

### Generating Images with DALL-E

To generate an image with DALL-E, send a message in the following format:

```
image: Your image description here
![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/SCR-20230521-pequ.jpeg)
```

## Notes

- This project is for demonstration purposes only and is not officially associated with OpenAI.
- This project does not include error handling for situations where the API keys are not set or incorrect. Please ensure that your API keys are correct before running the bot.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


