# Slackbot-with-chatgpt-dalle3-pdf-analyzer
# OpenAI ChatGPT Slackbot
** NEWLY ADDED PDF UPload and chat function ***

A powerful Slackbot app built with OpenAI's GPT-4o (switch to any model in .env) language model and DALL-E3 image generation API. This app allows users to interact with GPT-4o directly from Slack. It also supports generating images with DALL-E by giving a text prompt and saves it to the slackbot's DM.


![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/slackgpt%20image2.jpeg)

![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/Slackgpt%20summary.png)

## Features

1. Directly chat with GPT-4o from your Slack workspace.
2. Generate DALL-E images with a text prompt.
4. Upload PDFS direct to slack and ask it questions ala chatgpt plus
5. Extract Text and insightsnfrom PDFs
6. Maintain chat history to provide context for future prompts.

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
   - `GPT_MODEL=gpt-4o
   - GPT_SYSTEM_DESC="You are an AI assistant integrated into a bot application. Your primary goals are 	to: - Be helpful, knowledgeable, concise, and clear. - Provide accurate and effective responses 	tailored to the user's needs. - Adapt your tone and style to suit the context, remaining 		professional yet approachable. - Prioritize speed and clarity in delivering solutions. - Assist 	with tasks ranging from answering questions and summarizing documents to creative problem-solving 	and technical support. - Handle multimodal inputs like text, images, and audio, where applicable, 	with precision and efficiency. Always ensure your responses are easy to understand and aligned 		with the user’s intent."

   - GPT_IMAGE_SIZE=1024x1024

   - HISTORY_EXPIRES_IN=900
   - HISTORY_SIZE=3
   - PDF_CONTEXT_EXPIRES_IN=3600
   - THREAD_HISTORY_EXPIRES_IN=1800

### Running the Slackbot

1. Run the Slackbot with the following command:

```
python main.py
```

## Usage
![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/slackgpt%20sheets.png)



### Interacting with GPT-4o


```
	
```

### Generating Images with DALL-E

To generate an image with DALL-E, send a message in the following format:

```
image: Your image description here

![Example Image](https://github.com/High-Tower80/Slack-chatgpt-bot-with-dall-e/blob/main/slackgpt%20image1.jpeg)
```

## Notes

- This project is for demonstration purposes only and is not officially associated with OpenAI.
- This project does not include error handling for situations where the API keys are not set or incorrect. Please ensure that your API keys are correct before running the bot.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


