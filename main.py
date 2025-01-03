import sys
import logging
import os
from flask import Flask, request, jsonify
from slack_bolt.adapter.flask import SlackRequestHandler

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
handler = SlackRequestHandler(app)

@flask_app.route("/", methods=["GET"])
def hello():
    return jsonify({"message": "Choo Choo! Welcome to your Flask app 🚅"})

@flask_app.route("/slack/events", methods=["POST", "GET"])
def slack_events():
    # Handle URL verification
    if request.json and request.json.get("type") == "url_verification":
        logger.info("Handling URL verification challenge")
        return jsonify({"challenge": request.json.get("challenge")})
    
    # Handle other events
    logger.info(f"Received slack event: {request.json}")
    return handler.handle(request)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    flask_app.run(host="0.0.0.0", port=port)
		
