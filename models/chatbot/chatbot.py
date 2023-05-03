import os
import openai
import requests
from bs4 import BeautifulSoup
from utils.data_feeder import DataFeeder
from ..utils.JsonDBEngine import JsonDbEngine


class ChatBot:
    def __init__(self, use_openai=True, openai_api_key=None):
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        self.db_engine = JsonDbEngine("database")
        self.data_feeder = DataFeeder()
        
    def search_google(query):
        response = requests.get("https://www.google.com/search", params={"q": query})
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for g in soup.find_all('div', class_='r'):
            anchors = g.find_all('a')
            if anchors:
                title = g.find('h3').text
                link = anchors[0]['href']
                description = g.find(class_='st').text
                results.append({
                    "title": title,
                    "link": link,
                    "description": description
                })
        return results

    def search_duckduckgo(query):
        response = requests.get("https://duckduckgo.com/html", params={"q": query})
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for result in soup.select(".results .result"):
            link = result.find("a", recursive=False)
            if link:
                title = link.text.strip()
                url = link["href"].strip()
                description = result.find(".result__snippet").text.strip()
                results.append({
                    "title": title,
                    "link": url,
                    "description": description
                })
        return results
    
    def search(query):
        results = search_google(query)
        if not results:
            results = search_duckduckgo(query)
        return results

    def handle_message(self, message):
        # Check if the message is a command
        if message.startswith("/"):
            command_parts = message.split(" ")
            command = command_parts[0][1:].lower()
            if command == "learn":
                intent = command_parts[1].lower()
                response = " ".join(command_parts[2:])
                self.data_feeder.add_data(intent, response)
                return f"Thanks, I learned that {intent} is {response}"
            elif command == "search":
                query = " ".join(command_parts[1:])
                results = self.search(query)
                if len(results) > 0:
                    return f"I found {len(results)} results for '{query}':\n\n" + "\n\n".join([f"{i+1}. {result['title']}\n{result['link']}\n{result['description']}" for i, result in enumerate(results)])
                else:
                    return f"Sorry, I could not find any results for '{query}'"
            else:
                return f"Sorry, I do not recognize the command '/{command}'"

        # Use the data_feeder to retrieve a response
        response = self.data_feeder.get_response(message)
        if response:
            return response

        # If data_feeder has no response, use OpenAI API if available
        if self.use_openai and self.openai_api_key:
            response = self.openai_api_wrapper.get_response(message)
            if response:
                return response

        # If all else fails, return a default message
        return self.default_response

    def run(self):
        print("Hello! I'm a chatbot. What can I help you with?")
        while True:
            user_input = input("> ")
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            response = self.handle_message(user_input)
            print(response)

    def main():
        chatbot = ChatBot()
        chatbot.run()

    if __name__ == "__main__":
        main()

