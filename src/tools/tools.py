import os
import re
import requests
import trafilatura

from dotenv import load_dotenv
from tavily import TavilyClient
from bs4 import BeautifulSoup
from readability import Document

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information.
    Returns titles, URLs, and snippets.
    """
    results = tavily.search(query=query, max_results=5)

    output = []

    for result in results["results"]:
        output.append(
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Snippet: {result['content'][:300]}\n"
        )

    return "\n--------------------\n".join(output)


def scrape_url(url: str) -> str:
    """
    Scrape and extract clean readable content from a URL.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        html = response.text

        # Strategy 1: trafilatura
        extracted = trafilatura.extract(html)

        if extracted:
            return re.sub(r"\s+", " ", extracted)[:5000]

        # Strategy 2: readability
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if text:
            return re.sub(r"\s+", " ", text)[:5000]

        return "No readable content found."

    except Exception as e:
        return f"Error: {e}"