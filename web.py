import requests
import feedparser
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import time
from dataclasses import dataclass

@dataclass
class TechUpdate:
    title: str
    content: str
    url: str
    source: str
    timestamp: datetime
    summary: str

class TechNewsRetriever:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        self.sources = {
            'hackernews': {
                'url': 'https://hnrss.org/frontpage?count=20',
                'type': 'rss'
            },
            'techcrunch': {
                'url': 'https://techcrunch.com/feed/',
                'type': 'rss'
            },
            'github_trending': {
                'url': 'https://api.github.com/search/repositories',
                'type': 'api'
            },
            'reddit_programming': {
                'url': 'https://www.reddit.com/r/programming/hot.json?limit=20',
                'type': 'api'
            }
        }

    def fetch_from_source(self, source: str) -> List[TechUpdate]:
        '''Fetching tech updates from the source'''
        if source not in self.sources:
            raise ValueError(f"Unknown source: {source}")

        source_config = self.sources[source]

        try:
            if source_config['type'] == 'rss':
                return self._fetch_from_rss(source, source_config['url'])
            elif source_config['type'] == 'api':
                return self._fetch_from_api(source, source_config['url'])
        except Exception as e:
            print(f"Error fetching from {source}: {e}")
            return []

    def _fetch_from_api(self, source: str, url: str) -> List[TechUpdate]:
        """Dispatcher to correct API fetcher"""
        if source == "github_trending":
            return self._fetch_github_trending(url)
        elif source == "reddit_programming":
            return self._fetch_reddit_programming(url)
        else:
            print(f"No handler implemented for API source: {source}")
            return []

    def _fetch_from_rss(self, source: str, url: str) -> List[TechUpdate]:
        """Fetch from RSS feed"""
        try:
            feed = feedparser.parse(url)
            updates = []

            for entry in feed.entries[:15]:
                content = entry.get('summary', entry.get('description', ''))
                content = self._clean_html(content)
                summary = self._create_summary(content)

                update = TechUpdate(
                    title=entry.title,
                    content=content,
                    url=entry.link,
                    source=source,
                    timestamp=datetime.now(),
                    summary=summary
                )
                updates.append(update)

            return updates
        except Exception as e:
            print(f"RSS fetch error for {source}: {e}")
            return []

    def _fetch_github_trending(self, url: str) -> List[TechUpdate]:
        """Fetch trending GitHub repositories"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        params = {
            'q': f'created:>{yesterday}',
            'sort': 'stars',
            'order': 'desc',
            'per_page': 10
        }

        response = self.session.get(url, params=params)
        if response.status_code != 200:
            return []

        data = response.json()
        updates = []

        for repo in data.get('items', []):
            content = f"{repo['stargazers_count']} stars | {repo['language'] or 'N/A'} | {repo['description'] or 'No description'}"
            description = repo.get('description', '') or 'No description'

            update = TechUpdate(
                title=repo['full_name'],
                content=content,
                url=repo['html_url'],
                source='github_trending',
                timestamp=datetime.now(),
                summary=f"Trending {repo['language'] or 'repository'}: {description[:100]}..."
            )
            updates.append(update)

        return updates

    def _fetch_reddit_programming(self, url: str) -> List[TechUpdate]:
        """Fetch hot posts from r/programming"""
        response = self.session.get(url)
        if response.status_code != 200:
            return []

        data = response.json()
        updates = []

        for post in data['data']['children'][:10]:
            post_data = post['data']
            if post_data.get('pinned', False):
                continue

            content = f"{post_data['score']} upvotes | {post_data['num_comments']} comments\n\n{post_data.get('selftext', '')[:300]}..."
            update = TechUpdate(
                title=post_data['title'],
                content=content,
                url=f"https://reddit.com{post_data['permalink']}",
                source='reddit_programming',
                timestamp=datetime.fromtimestamp(post_data['created_utc']),
                summary=f"Reddit discussion: {post_data['title'][:80]}..."
            )
            updates.append(update)

        return updates

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content"""
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > 500:
            text = text[:500] + "..."

        return text

    def _create_summary(self, content: str) -> str:
        """Create a brief summary of the content"""
        if not content:
            return "No content available"

        sentences = content.split('.')
        if sentences and len(sentences[0]) > 20:
            return sentences[0] + "."
        return content[:100] + "..." if len(content) > 100 else content

    def fetch_all_sources(self) -> List[TechUpdate]:
        """Fetch from all configured sources"""
        all_updates = []

        for source in self.sources.keys():
            try:
                updates = self.fetch_from_source(source)
                all_updates.extend(updates)
                time.sleep(0.5)  # To respect API limits
            except Exception as e:
                print(f"Error fetching from {source}: {e}")

        all_updates.sort(key=lambda x: x.timestamp, reverse=True)
        return all_updates

    def get_source_stats(self) -> Dict[str, int]:
        """Get statistics about available sources"""
        stats = {}
        for source in self.sources.keys():
            try:
                updates = self.fetch_from_source(source)
                stats[source] = len(updates)
            except:
                stats[source] = 0
        return stats

# ✅ Example Usage
if __name__ == "__main__":
    retriever = TechNewsRetriever()

    print("Fetching from HackerNews...")
    hn_updates = retriever.fetch_from_source('hackernews')

    for update in hn_updates[:3]:
        print(f"\nTitle: {update.title}")
        print(f"Source: {update.source}")
        print(f"Summary: {update.summary}")
        print(f"URL: {update.url}")
        print("-" * 50)
