# Save this as app.py and run using: streamlit run app.py

import streamlit as st
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import anthropic


client = anthropic.Anthropic(api_key="enter ur api key")


def search_news_links(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            if 'href' in r:
                results.append(r['href'])
    return results

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        for s in soup(["script", "style"]): s.extract()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        return '\n'.join(line for line in lines if line)[:4000]
    except Exception as e:
        return f"Error: {e}"

def classify_news_with_claude(news_text, original_claim):
    prompt = f"""
The user has provided a news claim:

\"\"\"{original_claim}\"\"\"

Here is a related article or excerpt from the web:

\"\"\"{news_text}\"\"\"

Based on this content, is the original news claim likely to be REAL, FAKE, or UNSURE? Provide a one-word classification and explain briefly why.
"""
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

st.title("🧠 Fake News Classifier using Anthropic Claude")
news_claim = st.text_input("Enter a News Headline or Claim:")

if news_claim:
    st.write(f"🔍 Searching for articles related to: **{news_claim}**")
    with st.spinner("Searching web and analyzing..."):
        links = search_news_links(news_claim)
        if not links:
            st.error("No related news sources found.")
        else:
            for i, link in enumerate(links):
                st.markdown(f"### 🔗 Source {i+1}")
                st.markdown(f"[{link}]({link})")
                content = extract_text_from_url(link)
                if content.startswith("Error:"):
                    st.error(content)
                    continue
                st.markdown("**📄 Extracted Content (first few lines):**")
                st.text_area(f"Content {i+1}", content[:10000], height=200)
                prediction = classify_news_with_claude(content, news_claim)
                st.markdown(f"**🤖 Claude's Verdict:** `{prediction}`")
