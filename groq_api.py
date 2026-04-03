# !pip install -q groq python-dotenv requests beautifulsoup4
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv() 

def stream_chat_completion(client):
    streams = client.chat.completions.create(
        messages=[
            # commands to system
            {
                'role': 'system', 
                'content': 'You are a helpful asssistant. Answer as John Snow'
            },
            # user inputs
            {
                'role': 'user', 
                'content': 'Explain the importance of low latency LLMs'
            }
        ],
        model='llama-3.1-8b-instant',
        temperature=0.5, # controls the randomness of the output. Code → 0.1 – 0.3, content → 0.7 – 1.0
        max_tokens=1024, # maximum number of tokens to generate; too low may result in incomplete answers, too high may lead to unnecessary verbosity and increased latency, consume more tokens
        top_p=1, # nucleus sampling (in relevance to temperature), 0.1 means only the tokens comprising the top 10% probability mass are considered, 1 means all tokens are considered
        stop=None, # stop sequence, generation will stop if any one of the specified sequences is generated. Can be a string or an array of strings. If not specified, the model will stop when it reaches the max_tokens limit.
        stream=True # If True, partial message details will be sent as they become available. If False, the full message details will be sent once the message is complete.
    )

    for chunk in streams:
        print(chunk.choices[0].delta.content, end='')

def non_stream_chat_completion(client):
    chat_completion = client.chat.completions.create(
        messages=[
            # commands to system
            {
                'role': 'system', 
                'content': 'You are a helpful asssistant. Answer as John Snow'
            },
            # user inputs
            {
                'role': 'user', 
                'content': 'Explain the importance of low latency LLMs'
            }
        ],
        model='llama-3.1-8b-instant',
        temperature=0.5, # controls the randomness of the output. Code → 0.1 – 0.3, content → 0.7 – 1.0
        max_tokens=1024, # maximum number of tokens to generate; too low may result in incomplete answers, too high may lead to unnecessary verbosity and increased latency, consume more tokens
        top_p=1, # nucleus sampling (in relevance to temperature), 0.1 means only the tokens comprising the top 10% probability mass are considered, 1 means all tokens are considered
        stop=None, # stop sequence, generation will stop if any one of the specified sequences is generated. Can be a string or an array of strings. If not specified, the model will stop when it reaches the max_tokens limit.
        stream=False # If True, partial message details will be sent as they become available. If False, the full message details will be sent once the message is complete.
    )
    print(chat_completion.choices[0].message.content)

def stream_chat_completionwith_web_scraping(client): 
    from bs4 import BeautifulSoup
    import requests

    url = "https://paulgraham.com/greatwork.html"

    response = requests.get(url)
    html = response.text

    # 2. Parse HTML
    soup = BeautifulSoup(html, "html.parser")

    # 3. get text of page
    text = soup.get_text(separator="\n", strip=True)

    print(f"Text length: {len(text)} chars, ~{len(text)//4} tokens")

    max_chars = 20000  # ~5000 tokens, safe under 12k TPM limit

    response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant. Your job is to summarize the following text in 10 points'},
            {'role': 'user', 'content': text[:max_chars]}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True
    )

    for stream_chunk in response:
        print(stream_chunk.choices[0].delta.content or '', end="")


def main():
    api_key = os.getenv('GROQ_API_KEY')
    client = Groq(api_key=api_key)

    # non_stream_chat_completion(client)
    # stream_chat_completion(client)
    stream_chat_completionwith_web_scraping(client)


if __name__ == "__main__":
    main()

