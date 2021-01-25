import logging
import os
import subprocess

from flask import Flask, request

from models.SentimentModel import SentimentModel

# Set up logging
fmt = logging.Formatter()
fh = logging.StreamHandler()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

# Define Flask app
app = Flask(__name__)


##### ENDPOINTS #####

@app.route('/')
def index():
    return "This API is alive"


@app.route('/scrape_subreddit')
def scrape_subreddit(subreddit='wallstreetbets', n_results=5000, text='H', file_type='json'):
    """ Scrapes subreddit. Defaults to r/wallsreetbets. """
    os.chdir("urs")
    call = f"python ./Urs.py -r {subreddit} {text} {n_results} --{file_type} -y"
    subprocess.call(call.split(" "))
    os.chdir("..")
    return f"r/{subreddit} successfully scraped!"


@app.route('/SentimentModel/get_weights')
def get_weights(n=2, thresh=10, long_only=True):
    model = SentimentModel()
    n = float(request.args.get('n', n))
    thresh = float(request.args.get('thresh', thresh))
    long_only = bool(request.args.get('long_only', long_only))
    return model.get_weights(n=n, thresh=thresh, long_only=long_only).to_json()


@app.route('/SentimentModel/get_target_holdings')
def get_target_holdings(n=2, thresh=10, long_only=True):
    model = SentimentModel()
    n = float(request.args.get('n', n))
    thresh = float(request.args.get('thresh', thresh))
    long_only = bool(request.args.get('long_only', long_only))
    return str(model.get_target_holdings(n=n, thresh=thresh, long_only=long_only))


@app.route('/SentimentModel/rebalance')
def rebalance(capital=150000, order_type='limit', n=2, thresh=10, long_only=True, equity_mode=True):
    model = SentimentModel()
    capital = float(request.args.get('capital', capital))
    # Overwrite capital with just equity
    if equity_mode:
        capital = float(model.account.get_account().equity)
    order_type = request.args.get('order_type', order_type)
    n = float(request.args.get('n', n))
    thresh = float(request.args.get('thresh', thresh))
    long_only = bool(request.args.get('long_only', long_only))
    return str(model.rebalance(capital=capital, order_type=order_type, n=n, thresh=thresh, long_only=long_only))


# Run the flask app
if __name__ == '__main__':
    app.run(port=9656, host='0.0.0.0')
