import sqlite3
import logging
import feedparser
from bs4 import BeautifulSoup
import requests
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DatabaseManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_data (
                    news_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    title TEXT,
                    source TEXT,
                    ticker TEXT
                )
            ''')
            conn.commit()

    def save_news(self, ticker: str, news_data: list):
        """news_data 테이블에 데이터를 저장"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                for news in news_data:
                    cursor.execute('''
                        INSERT OR IGNORE INTO news_data (title, source, timestamp, ticker)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        news['title'],
                        news['source'],
                        news['date'],
                        ticker
                    ))
                conn.commit()
                logging.info(f"{len(news_data)}건의 뉴스 데이터 저장 완료 - ticker={ticker}")
        except Exception as e:
            logging.error(f"news_data 데이터 저장 실패: error={e}")

    def get_existing_titles(self) -> list:
        """이미 데이터베이스에 저장된 뉴스 제목 리스트 반환"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT title FROM news_data')
                existing_titles = [row[0] for row in cursor.fetchall()]
                return existing_titles
        except Exception as e:
            logging.error(f"기존 뉴스 제목 조회 실패: error={e}")
            return []


class NewsCollector:
    @staticmethod
    def get_news(keyword='비트코인', db_manager=None) -> list:
        """CoinTelegraph RSS 피드에서 비트코인 뉴스 수집 (최대 3건)"""
        try:
            url = 'https://kr.cointelegraph.com/rss/tag/bitcoin '
            feed = feedparser.parse(url)

            if feed.bozo:
                raise Exception(f"RSS 파싱 오류: {feed.bozo_exception}")

            existing_titles = db_manager.get_existing_titles() if db_manager else []

            news_data = []
            for entry in feed.entries:
                if len(news_data) >= 3:
                    break

                title = entry.get('title', '')
                if not title or title in existing_titles:
                    continue

                # 날짜 파싱 (존재하지 않으면 현재 시간으로 대체)
                try:
                    if 'published_parsed' in entry:
                        dt = datetime(*entry.published_parsed[:6])
                        date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    logging.warning(f"날짜 파싱 실패: {e}")
                    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                news_data.append({
                    'title': title,
                    'source': 'CoinTelegraph',
                    'date': date_str
                })
                logging.info(f"수집된 뉴스: {title}")

            logging.info(f"CoinTelegraph 뉴스 {len(news_data)}건 수집 완료")
            return news_data

        except Exception as e:
            logging.error(f"뉴스 수집 실패: {e}")
            return []