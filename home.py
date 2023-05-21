def build_app_home_blocks():
	return [
		{
			"type": "header",
			"text": {
				"type": "plain_text",
				"text": "My App Home",
				"emoji": True
			}
		},
		{
			"type": "divider"
		},
		{
			"type": "header",
			"text": {
				"type": "plain_text",
				"text": "ChatGPT 4!",
				"emoji": True
			}
		},
		{
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": ":wave: Hey there! I’m an AI-powered chatbot designed to have conversations with people like you.  You can ask me anything you want, and I’ll do my best to provide you with a helpful and informative answer"
			},
			"accessory": {
				"type": "image",
				"https://google.com"
				"alt_text": "Version"
			}
		},
		{
			"type": "divider"
		},
		{
			"type": "header",
			"text": {
				"type": "plain_text",
				"text": "DALL-E function included",
				"emoji": True
			}
		},
		{
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": "If you want to create a unique image, you can use DALL-E, Open AI's powerful image generation AI. Just start your input statement with\n \"*image:*\" and describe the object or scene you want to create \n```image: smiling robot sunset``` Try it out in the *Messages* Section:robot_face::sunrise:"
			},
			"accessory": {
				"type": "image",
				"image_url": "google.com",
				"alt_text": "gpt data"
			}
		},
		{
			"type": "divider"
		},
		{
			"type": "actions",
			"elements": [
				{
					"type": "button",
					"text": {
						"type": "plain_text",
						"text": "More Resources",
						"emoji": True
					},
					"style": "primary",
					"url": "https://www.example.com",
					"action_id": "button-action"
				},
				{
					"type": "button",
					"text": {
						"type": "plain_text",
						"text": "Policies",
						"emoji": True
					},
					"url": "www.example.com",
					"action_id": "policies-button-action"
				}
			]
		},
		{
			"type": "context",
			"elements": [
				{
					"type": "mrkdwn",
							"text": "Check the About section for more info."
						}
					]
				}
	]
