import logging
import os
import requests
import subprocess

from flask import Flask

# Set up logging
fmt = logging.Formatter()
fh = logging.StreamHandler()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

# Define Flask app
app = Flask(__name__)

# Construct blueprints

# Register blueprints

# Test endpoint
@app.route('/')
def index():
    return "This API is alive"

@app.route('/scrape-subreddit')
def scrape_subreddit(subreddit='wallstreetbets', n_results=5000, text='H', file_type='json'):
    """ Scrapes subreddit. Defaults to r/wallsreetbets. """
    os.chdir("urs")
    call = f"python ./Urs.py -r {subreddit} {text} {n_results} --{file_type} -y"
    subprocess.call(call.split(" "))
    os.chdir("..")
    return f"r/{subreddit} successfully scraped!"


# Run the flask app
if __name__ == '__main__':
    app.run(port=9656, host='0.0.0.0')
