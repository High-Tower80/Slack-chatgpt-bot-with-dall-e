import sys
import logging
import os
from flask import Flask, request, jsonify
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Force stdout logging
logging.basicConfig(
	level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
	format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
	stream=sys.stdout,
	force=True
)

# Add a stream handler to also output to stderr
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)
logging.getLogger().addHandler(stderr_handler)

logger = logging.getLogger(__name__)
logger.info("=== Application Starting ===")

# Print some diagnostic information
print("Python version:", sys.version)
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())

# Initialize Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(flask_app)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    logger.info("Received slack event")
    return handler.handle(request)

# For Slack's URL verification
@flask_app.route("/slack/events", methods=["POST"])
def endpoint():
    if request.json and request.json.get("type") == "url_verification":
        logger.info("Handling URL verification challenge")
        return jsonify({"challenge": request.json["challenge"]})
    return handler.handle(request)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    flask_app.run(host="0.0.0.0", port=port)
		
