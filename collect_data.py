import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

def fetch_reddit_rss(url):
    """
    Fetches raw RSS feed data using staggered timing gaps
    to systematically prevent HTTP 429 rate-limiting events.
    """
    print(f"Requesting RSS Feed: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 429:
            print("-> Hit rate limit (429). Initiating cooling-off buffer...")
            time.sleep(15)  # Back off for 15 seconds if flagged
            response = requests.get(url, headers=headers, verify=False)
            if response.status_code == 200:
                return response.text
        
        print(f"Error fetching data: HTTP Error {response.status_code}")
        return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def parse_rss_posts(xml_content):
    """
    Parses XML data streams into clean data rows.
    """
    extracted_posts = []
    if not xml_content:
        return extracted_posts

    soup = BeautifulSoup(xml_content, 'xml')
    entries = soup.find_all('entry')

    for entry in entries:
        title = entry.find('title').text.strip() if entry.find('title') else ""
        id_str = entry.find('id').text.strip().split('_')[-1] if entry.find('id') else ""
        
        content_tag = entry.find('content')
        selftext = ""
        
        if content_tag:
            content_soup = BeautifulSoup(content_tag.text, 'html.parser')
            paragraphs = content_soup.find_all('p')
            text_blocks = [p.text.strip() for p in paragraphs if "[link]" not in p.text and "[comments]" not in p.text]
            selftext = " ".join(text_blocks).strip()

        if title or selftext:
            extracted_posts.append({
                'id': id_str,
                'title': title,
                'text': selftext,
                'reddit_flair': 'None',
                'ai_prelabeled': False,
                'label': ''
            })

    return extracted_posts

def main():
    print("Starting secure RSS-driven data collection from r/TaylorSwift...")
    all_collected_posts = []
    
    # Target 1: Pull from /new
    new_url = "https://www.reddit.com/r/TaylorSwift/new/.rss?limit=100"
    raw_new_xml = fetch_reddit_rss(new_url)
    all_collected_posts += parse_rss_posts(raw_new_xml)
    print(f"Current pool size: {len(all_collected_posts)} total rows collected.")
    
    # Stagger request execution to allow the IP window to reset
    print("Waiting 10 seconds to maintain endpoint safety...")
    time.sleep(10)
    
    # Target 2: Pull from /hot (highly engaged discussion posts)
    hot_url = "https://www.reddit.com/r/TaylorSwift/hot/.rss?limit=100"
    raw_hot_xml = fetch_reddit_rss(hot_url)
    all_collected_posts += parse_rss_posts(raw_hot_xml)
    print(f"Current pool size: {len(all_collected_posts)} total rows collected.")
    
    print("Waiting 10 seconds to maintain endpoint safety...")
    time.sleep(10)
    
    # Target 3: Pull from /top of the month (rich pool for game brackets and polls)
    top_url = "https://www.reddit.com/r/TaylorSwift/top/.rss?sort=top&t=month&limit=100"
    raw_top_xml = fetch_reddit_rss(top_url)
    all_collected_posts += parse_rss_posts(raw_top_xml)
    
    # Combine data structures into an integrated DataFrame
    df = pd.DataFrame(all_collected_posts)
    
    if not df.empty:
        # Deduplicate using unique Reddit post IDs
        df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        print(f"\nTotal unique posts collected after deduplication: {len(df)}")
        
        output_filename = "taylor_swift_dataset.csv"
        df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"Success! Saved dataset to: {output_filename}")
    else:
        print("\nNo rows could be written. Verify connection parameters.")

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()