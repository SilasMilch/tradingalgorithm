import pandas as pd
import datetime as dt
import time
import logging

from optibook.synchronous_client import Exchange
from optibook.common_types import SocialMediaFeed

from transformers import pipeline

class LLM:

    messages = []
    exchange = Exchange()
    n = 1  # Number of messages to consider for market direction
    c = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

    cscoNames = ["Cisco"]
    nvdaNames = ["Nvidia", "NVIDIA"]
    ingNames = ["ING", "Insurance Group", "InnoGlobal Group", "InG"]
    sanNames = ["Santander"]
    pfeNames = ["Pfizer"]

    testWordsSentiment = [
        "good", "bad"]
    testWordSentimentWights = [1, -1]

    def __init__(self):
        self.exchange.connect()

    def getMarketSentiment(self):
        """
        This function retrieves the current market direction as +/- 1.
        """
        # get last n messages
        lastMessages = self.messages#getLastMessages(self.n)
        dir = [0, 0, 0, 0, 0]
        if not lastMessages:
            return dir
        for message in lastMessages:
            if any(name in message for name in self.nvdaNames):
                dir[0] = 1
            if any(name in message for name in self.ingNames):
                dir[1] = 1
            if any(name in message for name in self.sanNames):
                dir[2] = 1
            if any(name in message for name in self.pfeNames):
                dir[3] = 1
            if any(name in message for name in self.cscoNames):
                dir[4] = 1
        #if all values are 0 return zero
        if (dir[0] == dir[1] and dir[2] == dir[1] and dir[2] == dir[3] and dir[3] == dir[4] and dir[0] == 0):
            return dir
        #compute all sentiment values for test words
        sentiment = 0
        for i, message in enumerate(lastMessages):
            factor = self.getMessageWeight(i, len(lastMessages))
            sentiments = self.c(message, self.testWordsSentiment, multi_label=False)
            for j, sentimentValue in enumerate(sentiments['scores']):
                entiment += factor * sentimentValue * self.testWordSentimentWights[j]
        return [sentiment * d for d in dir]

    def getMessageWeight(self, i, n):
        #get exponential function fo n-i
        return 1 / (2 ** (n - i))

    def getRandomTrainingData(self, n):
        """
        This function retrieves random training data from the market.
        It returns a list of messages.
        """
        data = pd.read_csv('training.csv')
        return data.sample(n)

    def getLastMessages(self, n):
        """
        This function retrieves the last n messages from the market.
        """
        lastFeed = self.exchange.poll_new_social_media_feeds()
        if not lastFeed:
            return self.messages[-n:]
        if len(self.messages) >= n:
            return lastFeed[-n:]
        messages = lastFeed + self.messages[-(n - len(self.messages)):]
        return messages

if __name__ == '__main__':
    llm = LLM()
    llm.messages = ["a", "Cisco", "bad"]
    direction = llm.getMarketDirection()