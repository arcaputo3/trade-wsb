#!/usr/bin/python
"""
Created on Tue Jun 2 20:14:43 2020

Universal Reddit Scraper 3.1.0.

@author: Joseph Lai
@contact: urs_project@protonmail.com
"""
import os
import praw

from utils.Logger import LogMain
from utils.Tools import Run


class Main():
    """
    Run URS.
    """

    @staticmethod
    @LogMain.master_timer
    def main():
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD")
        )

        Run(reddit).run_urs()


if __name__ == "__main__":
    Main.main()
