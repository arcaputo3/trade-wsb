FROM python:3.7

WORKDIR /code

COPY requirements.txt .

RUN apt-get update && \
    pip install -r requirements.txt && \
    python -m nltk.downloader -d /usr/local/share/nltk_data all

EXPOSE 9656

COPY . .

# Command to run on container start
CMD [ "python", "app.py" ]
