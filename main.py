import os
import json
import time
import pandas as pd
import pyupbit
from dotenv import load_dotenv
from openai import OpenAI
import warnings
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import sys
from news_collector import DatabaseManager, NewsCollector
from trading_reflection import TradingReflectionManager

print(TradingReflectionManager)



# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading.log', encoding='utf-8')
    ]
)

warnings.filterwarnings('ignore', category=FutureWarning)


# 거래 설정값을 담은 클래스
@dataclass
class TradeConfig:
    MIN_ORDER_AMOUNT: int = 5000
    TRADING_FEE: float = 0.0005
    MAX_LOSS_PERCENT: float = 0.05
    PROFIT_TARGET: float = 0.15
    API_DELAY: float = 0.1
    TRADING_INTERVAL: int = 600


# 데이터베이스 관리 클래스
class DatabaseManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                            CREATE TABLE IF NOT EXISTS trading_base (
                                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                ticker TEXT,
                                decision TEXT NOT NULL,
                                price REAL,
                                amount REAL,
                                balance REAL,
                                reason TEXT
                                        )
                                    ''')
            cursor.execute('''
                            CREATE TABLE IF NOT EXISTS trading_result (
                                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                trade_id INTEGER,
                                trade_result TEXT,
                                FOREIGN KEY(trade_id) REFERENCES trading_base(trade_id)
                            )
                        ''')
            cursor.execute('''
                                        CREATE TABLE IF NOT EXISTS coin_info(
                                            coin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                            ticker TEXT UNIQUE,
                                            name TEXT,
                                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
                                    ''')
            cursor.execute('''
                            CREATE TABLE IF NOT EXISTS ai_analysis (
                                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                ticker TEXT,
                                analysis_result TEXT NOT NULL,
                                confidence_score REAL,
                                market_data_id INTEGER,
                                reason TEXT,
                                FOREIGN KEY(ticker) REFERENCES coin_info(ticker),
                                FOREIGN KEY(market_data_id) REFERENCES market_data(data_id)
)
                        ''')
            cursor.execute('''
                                        CREATE TABLE IF NOT EXISTS technical_indicators(
                                            indicator_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                            ticker TEXT,
                                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                            rsi REAL,
                                            macd REAL,
                                            signal REAL,
                                            ma5 REAL, 
                                            ma20 REAL,
                                            ma60 REAL,
                                            upper_band REAL,          -- 추가
                                            lower_band REAL,          -- 추가
                                            atr REAL,                 -- 추가
                                            fibonacci_0236 REAL,      -- 추가
                                            fibonacci_0382 REAL,      -- 추가
                                            fibonacci_05 REAL,        -- 추가
                                            fibonacci_0618 REAL,      -- 추가
                                            fibonacci_0786 REAL,      -- 추가
                                            FOREIGN KEY(ticker) REFERENCES coin_info(ticker)

            )
                                    ''')
            cursor.execute('''
                                    CREATE TABLE IF NOT EXISTS market_data (
                                        data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        ticker TEXT,
                                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                        current_price REAL,
                                        volume REAL,
                                        price_change REAL,
                                        indicator_id INTEGER,
                                        FOREIGN KEY(ticker) REFERENCES coin_info(ticker),
                                        FOREIGN KEY(indicator_id) REFERENCES technical_indicators(indicator_id)
                                    )
                                ''')
            cursor.execute('''
                        CREATE TABLE IF NOT EXISTS news_data (
                            news_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            title TEXT,
                            source TEXT,
                            ticker TEXT,
                            FOREIGN KEY(ticker) REFERENCES coin_info(ticker)
                        )
                    ''')
            cursor.execute('''
                        CREATE TABLE IF NOT EXISTS trading_reflection (
                             reflection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                          trade_id INTEGER,
                          market_condition TEXT,
                          decision_accuracy TEXT,
                          improvement_points TEXT,
                          next_strategy TEXT,
                          FOREIGN KEY(trade_id) REFERENCES trading_base(trade_id)

                        )
                    ''')
        pass

    def get_trade_statistics(self, lookback=50):
        """최근 lookback건의 거래 통계(평균 승률, 수익률 등) 반환"""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT decision, price FROM trading_base ORDER BY trade_id DESC LIMIT ?", (lookback,)
            )
            rows = cursor.fetchall()
            # 예시: BUY → SELL 시 차익 계산
            profit_list = []
            last_buy = None
            for dec, prc in rows[::-1]:  # 오래된 순서
                if dec == "BUY":
                    last_buy = prc
                elif dec == "SELL" and last_buy:
                    profit = (prc - last_buy) / last_buy
                    profit_list.append(profit)
                    last_buy = None
            win_count = sum(1 for pf in profit_list if pf > 0)
            trade_count = len(profit_list)
            win_rate = win_count / trade_count * 100 if trade_count else 0
            avg_profit = sum(profit_list) / trade_count if trade_count else 0
            return {
                "승률(%)": round(win_rate, 2),
                "평균수익률(%)": round(avg_profit * 100, 2)
            }


    def save_news(self, ticker: str, news_data: List[Dict]):
        """news_data 테이블에 데이터를 저장"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                for news in news_data:  # news_data 리스트 순회
                    cursor.execute('''
                        INSERT INTO news_data 
                        (title, source, timestamp, ticker)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        news['title'],  # 뉴스 제목
                        news['source'],  # 뉴스 출처
                        news['date'],  # 뉴스 날짜
                        ticker  # 해당 티커
                    ))
                conn.commit()
                logging.info(f"{len(news_data)}건의 뉴스 데이터 저장 완료 - ticker={ticker}")
        except Exception as e:
            logging.error(f"news_data 데이터 저장 실패: error={e}")

    def save_coin_info(self, ticker: str, name: str):
        """coin_info 테이블에 데이터를 저장"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO coin_info (ticker, name)
                    VALUES (?, ?)
                ''', (ticker, name))
                conn.commit()
                logging.info(f"coin_info 데이터 저장 완료: ticker={ticker}, name={name}")
        except Exception as e:
            logging.error(f"coin_info 데이터 저장 실패: ticker={ticker}, name={name}, error={e}")

    def save_trading_reflection(self, trade_id: int, market_condition: str, decision_accuracy: str,
                                improvement_points: str,
                                next_strategy: str):
        """
        트레이딩 반성 내용을 trading_reflection 테이블에 저장
        """
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_reflection 
                    (trade_id, market_condition, decision_accuracy, improvement_points, next_strategy)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    trade_id,
                    market_condition,
                    decision_accuracy,
                    improvement_points,
                    next_strategy
                ))
                conn.commit()
                logging.info(f"트레이딩 반성 기록 완료: trade_id={trade_id}")
        except Exception as e:
            logging.error(f"트레이딩 반성 기록 저장 실패: {e}")

    def save_trade(self, trade_data: Dict):
        logging.info(f"저장할 거래 데이터: {trade_data}")
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                # 거래 데이터에서 'reason' 키 확인 및 기본값 설정
                reason = trade_data.get('reason', "N/A")
                if not isinstance(reason, str):
                    logging.warning(f"'reason' 필드의 값이 잘못되었습니다. 기본값으로 설정합니다.")
                    reason = "N/A"
                # trading_base 테이블에 기본 거래 정보 저장
                cursor.execute("""
                            INSERT INTO trading_base (ticker, decision, price, amount, balance, reason, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                    trade_data['ticker'],
                    trade_data['decision'],
                    trade_data['price'],
                    trade_data['amount'],
                    trade_data['balance'],
                    trade_data['reason']

                ))
                trade_id = cursor.lastrowid
                logging.info(f"거래 기록 저장 완료: {trade_data['ticker']}")
                return trade_id
        except KeyError as e:
            logging.error(f"거래 데이터 필수 키가 누락되었습니다: {e}")
        except sqlite3.IntegrityError as e:
            logging.error(f"데이터베이스 무결성 오류: {e}")
        except Exception as e:
            logging.error(f"거래 기록 저장 실패: {e}")
            return None

    # save_market_data 내 로깅 추가
    def save_market_data(self, market_data: Dict) -> Optional[int]:
        try:
            # Validate market data
            if not self._validate_market_data(market_data):
                logging.error(f"유효하지 않은 시장 데이터: {market_data}")
                return None
            # Log processed data
            processed_data = self._process_market_data(market_data)
            logging.info(f"처리된 시장 데이터: {processed_data}")
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                # Insert technical indicators first
                cursor.execute('''
                    INSERT INTO technical_indicators
                    (ticker, timestamp, rsi, macd, signal, ma5, ma20, ma60, upper_band, lower_band, atr, 
                    fibonacci_0236, fibonacci_0382, fibonacci_05, fibonacci_0618, fibonacci_0786)
                    VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    processed_data['ticker'],
                    processed_data['rsi'],
                    processed_data['macd'],
                    processed_data['signal'],
                    processed_data['ma5'],
                    processed_data['ma20'],
                    processed_data['ma60'],
                    processed_data['upper_band'],
                    processed_data['lower_band'],
                    processed_data['atr'],
                    processed_data['fibonacci']['0.236'],  # 피보나치
                    processed_data['fibonacci']['0.382'],  # 피보나치
                    processed_data['fibonacci']['0.5'],  # 피보나치
                    processed_data['fibonacci']['0.618'],  # 피보나치
                    processed_data['fibonacci']['0.786']  # 피보나치
                ))
                # Save to market_data table
                indicator_id = cursor.lastrowid
                cursor.execute('''
                    INSERT INTO market_data
                    (ticker, timestamp, current_price, volume, price_change, indicator_id)
                    VALUES (?, datetime('now'), ?, ?, ?, ?)
                ''', (
                    processed_data['ticker'],
                    processed_data['current_price'],
                    processed_data['volume'],
                    processed_data['price_change'],
                    indicator_id
                ))
                conn.commit()
                logging.info(f"market_data 저장 성공: {processed_data['ticker']}, ID={cursor.lastrowid}")
                return cursor.lastrowid
        except Exception as e:
            logging.error(f"시장 데이터 저장 실패: {e}")
            return None

    def _process_market_data(self, market_data: Dict) -> Dict:
        """
            시장 데이터를 변환하거나 추가 처리합니다.
            현재는 데이터를 그대로 반환하지만, 필요에 따라 확장 가능합니다.
            """
        try:
            # 데이터 전처리 단계 추가 (필요 시 활용 가능)
            processed_data = {**market_data}
            # 예: 데이터의 특정 값을 변경하거나 제거
            # processed_data['new_field'] = processed_data['existing_field'] * 2
            return processed_data
        except Exception as e:
            logging.error(f"시장 데이터 처리 중 오류 발생: {e}")
            raise

    def save_ai_analysis(self, analysis_data: Dict):
        """AI 분석 결과를 데이터베이스에 저장"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ai_analysis 
                (ticker, analysis_result, confidence_score, market_data_id, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                ''', (
                    analysis_data.get('ticker'),
                    analysis_data.get('decision'),
                    analysis_data.get('confidence_score', 0.0),  # 확신도 저장
                    analysis_data.get('market_data_id'),
                    analysis_data.get('reason')
                ))
                conn.commit()
                logging.info(f"AI 분석 결과 저장 완료: {analysis_data.get('ticker')}")
        except Exception as e:
            logging.error(f"AI 분석 결과 저장 실패: {e}")

    def _validate_market_data(self, market_data: Dict) -> bool:
        """시장 데이터 유효성 검사"""
        try:
            required_fields = [
                'ticker',
                'current_price',
                'volume',
                'price_change',
                'rsi',
                'macd',
                'signal',
                'ma5',
                'ma20',
                'ma60'
            ]
            # 필수 필드 존재 확인
            if not all(field in market_data for field in required_fields):
                logging.error(f"필수 필드 누락: {[field for field in required_fields if field not in market_data]}")
                return False
            # 데이터 타입 검사
            for field in required_fields:
                if field == 'ticker':
                    if not isinstance(market_data[field], str):
                        logging.error(f"잘못된 ticker 타입: {type(market_data[field])}")
                        return False
                else:
                    if not isinstance(market_data[field], (int, float)):
                        logging.error(f"잘못된 {field} 타입: {type(market_data[field])}")
                        return False
            return True
        except Exception as e:
            logging.error(f"시장 데이터 검증 중 오류: {e}")
            return False


# 시장 데이터 수집기 클래스
# MarketDataCollector 클래스의 collect_market_data 메서드 수정
# 시장 데이터 수집기 클래스
class MarketDataCollector:
    REQUIRED_INDICATORS = ['rsi', 'macd', 'signal', 'ma5', 'ma20', 'ma60', 'upper_band', 'lower_band', 'atr']

    @staticmethod
    def collect_market_data(ticker: str) -> Optional[Dict]:
        RETRY_LIMIT = 3
        for attempt in range(RETRY_LIMIT):
            try:
                logging.info(f"{ticker}: 시장 데이터 수집 시도 {attempt + 1}...")
                df = pyupbit.get_ohlcv(ticker, interval="minute60", count=100)
                if df is None or df.empty:
                    logging.error(f"{ticker}: OHLCV 데이터가 유효하지 않습니다. 재시도 중...")
                    time.sleep(1)
                    continue
                if len(df) < 60:
                    logging.error(f"{ticker}: 데이터가 부족합니다. (최소: 60, 현재: {len(df)})")
                    return None
                # 필수 지표 계산
                df = MarketDataCollector.calculate_indicators(df)
                df = calculate_bollinger_bands(df)  # 볼린저 밴드 추가
                df = calculate_atr(df)  # ATR 추가
                fibonacci = calculate_fibonacci_retracement(df)  # 피보나치 추가
                # 시장 데이터 병합
                market_data = {
                    "ticker": ticker,
                    "current_price": pyupbit.get_current_price(ticker),
                    "volume": float(df["volume"].iloc[-1]),
                    "price_change": float(
                        ((df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]) * 100),
                    "rsi": df["rsi"].iloc[-1],
                    "macd": df["macd"].iloc[-1],
                    "signal": df["signal"].iloc[-1],
                    "ma5": df["ma5"].iloc[-1],
                    "ma20": df["ma20"].iloc[-1],
                    "ma60": df["ma60"].iloc[-1],
                    "upper_band": df["upper_band"].iloc[-1],  # 볼린저 밴드
                    "lower_band": df["lower_band"].iloc[-1],  # 볼린저 밴드
                    "atr": df["atr"].iloc[-1],  # ATR
                    "fibonacci": fibonacci  # 피보나치 값들
                }
                return market_data
            except Exception as e:
                logging.error(f"{ticker}: 시장 데이터 수집 중 오류 발생: {e}")
                return None

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        # (기존 지표 계산 로직)
        ...
        try:
            if df is None or len(df) < 60:  # 충분한 데이터가 있는지 확인
                logging.error("충분한 데이터가 없습니다")
                return None
            # RSI 계산
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)  # NaN 값 처리
            loss = (-delta.where(delta < 0, 0)).fillna(0)  # NaN 값 처리
            gain = gain.rolling(window=14).mean()
            loss = loss.rolling(window=14).mean()
            rs = gain / loss.replace(0, float('inf'))  # 0으로 나누기 방지
            df['rsi'] = 100 - (100 / (1 + rs))
            # MACD 계산
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            # 이동평균 계산 (MA)
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            return df
        except Exception as e:
            logging.error(f"기술적 지표 계산 중 오류 발생: {str(e)}")
            return None

    @staticmethod
    def validate_indicators(df: pd.DataFrame) -> bool:
        """
        데이터프레임에 필수 지표들이 존재하는지 확인합니다.
        """
        return all(indicator in df.columns for indicator in MarketDataCollector.REQUIRED_INDICATORS)


# 볼린저 밴드 계산 함수
def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df is None or len(df) < 20:
            logging.error("볼린저 밴드 계산을 위한 충분한 데이터가 없습니다")
            return None
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['stddev'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['ma20'] + (df['stddev'] * 2)
        df['lower_band'] = df['ma20'] - (df['stddev'] * 2)
        return df
    except Exception as e:
        logging.error(f"볼린저 밴드 계산 중 상세 오류: {str(e)}")
        return None


# ATR 계산 함수
def calculate_atr(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['high-low'] = df['high'] - df['low']
        df['high-close'] = abs(df['high'] - df['close'].shift())
        df['low-close'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high-low', 'high-close', 'low-close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        return df
    except Exception as e:
        logging.error(f"ATR 계산 중 오류 발생: {e}")
        return df


# 피보나치 되돌림 계산 함수
def calculate_fibonacci_retracement(df: pd.DataFrame) -> Dict[str, float]:
    try:
        max_price = df['high'].max()
        min_price = df['low'].min()
        retracements = {
            '0.236': max_price - 0.236 * (max_price - min_price),
            '0.382': max_price - 0.382 * (max_price - min_price),
            '0.5': max_price - 0.5 * (max_price - min_price),
            '0.618': max_price - 0.618 * (max_price - min_price),
            '0.786': max_price - 0.786 * (max_price - min_price)
        }
        return retracements
    except Exception as e:
        logging.error(f"피보나치 되돌림 계산 중 오류 발생: {e}")
        return {}

# AI 트레이더 메인 클래스
        DEFAULT_TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-XLM", "KRW-DOGE"]

class AITrader:
    def __init__(self):
        load_dotenv()
        self.config = TradeConfig()
        self.db = DatabaseManager('trading_history.db')
        self.is_running = True
        self.STOP_LOSS = {}
        self._initialize_apis()
        self.initialize_tickers()
        try:
            self.reflection_manager = TradingReflectionManager("trading_history.db")
            logging.info("TradingReflectionManager 초기화 성공")
        except Exception as e:
            logging.error(f"TradingReflectionManager 초기화 실패: {e}")

        pass

    def _process_trading_cycle(self):
        """
        거래 사이클을 처리하는 메서드
        """
        try:
            # 각 코인에 대해 시장 데이터를 수집하고 분석을 수행
            for ticker in self.tickers:
                logging.info(f"{ticker}에 대한 거래 사이클 처리 중...")

                # 시장 데이터 수집
                market_data = MarketDataCollector.collect_market_data(ticker)
                if not market_data:
                    logging.warning(f"{ticker}의 시장 데이터를 수집할 수 없습니다.")
                    continue

                # AI를 활용한 분석
                decision = self.analyze_with_ai(market_data)

                # 거래 실행
                if decision in ["BUY", "SELL"]:
                    reason = "AI 분석에 의해 결정된 거래"  # 이유를 상황에 따라 지정
                    self.execute_trade(ticker, decision, reason)

                # 손절 및 익절 체크
                self.check_stop_loss(ticker)

        except Exception as e:
            logging.error(f"거래 사이클 처리 중 오류 발생: {str(e)}")


    def _initialize_apis(self):
        try:
            self.upbit_access = os.getenv('UPBIT_ACCESS_KEY')
            self.upbit_secret = os.getenv('UPBIT_SECRET_KEY')
            if not self.upbit_access or not self.upbit_secret:
                raise ValueError("Upbit API 키가 설정되지 않았습니다.")
            self.upbit = pyupbit.Upbit(self.upbit_access, self.upbit_secret)
            openai_api_key = os.getenv('OPENROUTER_API_KEY')
            if not openai_api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            # OpenAI 클라이언트 초기화 수정
            self.ai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv('OPENROUTER_API_KEY'),
                default_headers={
                    "HTTP-Referer": "https://github.com/Parksehyun-3/-"
                }
            )
            logging.info("API 초기화 완료")
        except Exception as e:
            logging.error(f"API 초기화 실패: {e}")
            raise

    def reflect_on_trade(self, trade_id: int, ticker: str, decision: str, result: str, reason: str):
        """
        거래 결과를 바탕으로 트레이딩 반성을 수행
        """
        try:
            # 트레이딩 반성 데이터 생성
            reflection_data = self.reflection_manager.generate_reflection_data(
                trade_id=trade_id,
                ticker=ticker,
                decision=decision,
                result=result,
                reason=reason
            )

            # 데이터 유효성 검증
            if not reflection_data or 'trade_id' not in reflection_data:
                logging.error(f"반성 데이터 생성 실패. 무효한 데이터: {reflection_data}")
                return

            # 로깅
            logging.info(f"Reflection Data 생성됨: {reflection_data}")

            # 트레이딩 반성 데이터 저장
            self.reflection_manager.save_trading_reflection(
                trade_id=reflection_data["trade_id"],
                market_condition=reflection_data["market_condition"],
                decision_accuracy=reflection_data["decision_accuracy"],
                improvement_points=reflection_data["improvement_points"],
                next_strategy=reflection_data["next_strategy"]
            )

        except Exception as e:
            logging.error(f"트레이딩 반성 과정 중 오류 발생: {e}")

    def initialize_tickers(self):
        """거래할 4개 코인의 티커 초기화"""
        try:
            self.tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-XLM", "KRW-DOGE"]
            logging.info(f"거래 대상 티커: {self.tickers}")
        except Exception as e:
            logging.error(f"티커 초기화 실패: {e}")
            self.tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-XLM", "KRW-DOGE"]
            logging.info(f"기본 티커로 설정됨: {self.tickers}")

    def analyze_with_ai(self, market_data: Dict) -> str:
        try:
            # 입력 데이터 검증
            required_fields = ['ticker', 'current_price', 'rsi', 'macd', 'signal', 'ma5', 'ma20', 'ma60']
            for field in required_fields:
                if field not in market_data:
                    logging.error(f"필수 필드 누락: {field}")
                    return "HOLD"

            stats = self.db.get_trade_statistics(lookback=30)
            stats_text = f"최근 거래 승률: {stats['승률(%)']}%, 평균 수익률: {stats['평균수익률(%)']}%"

            prompt = f"""
    아래의 {market_data['ticker']} 시장 데이터를 분석하여 **최상의 투자 결정과 확신도**를 알려주세요.
    분석에는 단기 트렌드, 기술적 지표, 리스크 관리, 그리고 시장 심리를 바탕으로 결정을 내리세요.
    반드시 다음 형식으로만 답변해주세요:

    결정: [BUY / SELL / HOLD]  
    이유: [100자 이내의 이유]  
    확신도: [숫자 %로 표현, 예: 85%]

    시장 데이터:
    - 현재가: {market_data['current_price']} 원
    - RSI: {market_data['rsi']}
    - MACD: {market_data['macd']}
    - Signal: {market_data['signal']}
    - MA5: {market_data['ma5']}
    - MA20: {market_data['ma20']}
    - MA60: {market_data['ma60']}
    - 볼린저 밴드 상단: {market_data['upper_band']}
    - 볼린저 밴드 하단: {market_data['lower_band']}
    - ATR(변동성): {market_data['atr']}
    - 피보나치 되돌림 0.236: {market_data['fibonacci']['0.236']}
    - 피보나치 되돌림 0.382: {market_data['fibonacci']['0.382']}
    - 피보나치 되돌림 0.5: {market_data['fibonacci']['0.5']}
    - 피보나치 되돌림 0.618: {market_data['fibonacci']['0.618']}
    - 피보나치 되돌림 0.786: {market_data['fibonacci']['0.786']}

    ### 추가 고려 사항:
    1. **시장 심리 및 뉴스 영향**:
       - 최근 비트코인과 연관된 주요 뉴스 트렌드 및 경제적 사건이 가격에 미친 영향을 고려하세요.
       - 변동성이 높은 경우에는 ATR과 볼린저 밴드를 활용해 리스크 관리 전략을 추가하세요.

    2. **매수 / 매도의 강도 판단**:
       - RSI의 극단적 수치(30 이하, 70 이상)일 경우, 점진적 진입 / 청산 전략을 사용하세요.
       - 피보나치 되돌림 레벨이 강력한 지지선이나 저항선을 제공하는지 판단하세요.

    3. **손익 비율 기반 추천**:
       - 손실 리스크 대비 목표 수익(예: 1:2 이상)이 적절한지 여부를 판단하세요.
       - ATR 값을 활용해 단기 변동성에 따라 투자 금액을 조정하거나 조기 철수 전략을 추천하세요.

    4. **트렌드와 지표의 조합 고려**:
       - MA5, MA20, MA60의 정렬 상태(예: MA5 > MA20 > MA60일 때 강한 상승장)를 이용해 추세를 확신하세요.
       - MACD와 Signal의 교차 시점을 통해 추가적인 매수 / 매도 타이밍을 확인하세요.

    결정을 내리실 때 위 정보를 적극 활용해 답변을 알려주세요.  
    **결정, 이유, 확신도(숫자 %로 표시)를 반드시 포함하세요.**
            """
            response = self.ai_client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{
                    "role": "system",
                    "content": "한국어로만 응답하세요. 결정과 이유를 지정된 형식으로만 제공하세요."
                },
                    {
                        "role": "user",
                        "content": prompt
                    }],
                max_tokens=100,
            )
            response_text = response.choices[0].message.content.strip()
            print(response_text)
            # 응답 파싱
            try:
                decision_line = [line for line in response_text.split('\n') if line.startswith('결정:')][0]
                reason_line = [line for line in response_text.split('\n') if line.startswith('이유:')][0]
                confidence_line = [line for line in response_text.split('\n') if line.startswith('확신도:')][0]
                decision = decision_line.split(':')[1].strip().upper()
                reason = reason_line.split(':')[1].strip()
                confidence = float(confidence_line.split(':')[1].replace('%', '').strip()) / 100.0  # 퍼센트를 숫자로 변환
                valid_decisions = {"BUY", "SELL", "HOLD"}
                if decision not in valid_decisions:
                    logging.warning(f"유효하지 않은 AI 결정값: {decision}")
                    return "HOLD"
                # 결정과 이유 모두 로깅
                logging.info(f"AI 분석 결과: 결론={decision}, 확신도={confidence}")
                logging.info(f"분석 이유: {reason}")
                # AI 분석 결과 저장
                self.db.save_ai_analysis({
                    'ticker': market_data['ticker'],
                    'decision': decision,
                    'reason': reason,
                    'confidence_score': confidence,  # 확신도 추가
                    'market_data_id': self.db.save_market_data(market_data)  # market_data_id 저장
                })
                return decision
            except (IndexError, KeyError) as e:
                logging.error(f"AI 응답 파싱 실패: {e}")
                return "HOLD"
        except Exception as e:
            logging.error(f"AI 분석 중 오류 발생: {e}")
            return "HOLD"

    def execute_trade(self, ticker: str, decision: str, reason: str) -> Optional[int]:
        try:
            # 현재가 조회
            current_price = pyupbit.get_current_price(ticker)
            if current_price is None:
                logging.error(f"Failed to retrieve current price for {ticker}")
                return None

            # 잔액 조회
            balance = self.upbit.get_balance("KRW")
            if balance is None:
                logging.error(f"Failed to retrieve balance.")
                return None

            # 거래 기록
            try:
                trade_id = self.db.save_trade({
                    'ticker': ticker,
                    'decision': decision,
                    'price': current_price,
                    'amount': balance,
                    'balance': balance,
                    'reason': reason
                })
            except Exception as e:
                logging.error(f"Database operation failed: {str(e)}")
                return None

            if trade_id is None:
                logging.error("Failed to record trade in database.")
                return None

            # 매수 로직
            if decision == "BUY":
                if balance >= self.config.MIN_ORDER_AMOUNT:
                    order_amount = min(balance * 0.1, balance)  # 10% 비중 매수
                    try:
                        self.upbit.buy_market_order(ticker, order_amount)
                        logging.info(f"Buy order successful: ticker={ticker}, amount={order_amount}")
                    except Exception as e:
                        logging.error(f"Buy order failed: {e}")
                        return None
                else:
                    logging.warning("Insufficient balance for buy order.")
                    return None

            # 매도 로직
            elif decision == "SELL":
                ticker_balance = self.upbit.get_balance(ticker)
                if ticker_balance is None or ticker_balance <= 0:
                    logging.error(f"Insufficient balance to sell: ticker={ticker}")
                    return None
                try:
                    self.upbit.sell_market_order(ticker, ticker_balance)
                    logging.info(f"Sell order successful: ticker={ticker}, amount={ticker_balance}")
                except Exception as e:
                    logging.error(f"Sell order failed: {e}")
                    return None

            return trade_id

        except Exception as e:
            logging.error(f"Error during trade execution: {str(e)}")
            return None

    """
        """

    def stop(self):
        """트레이딩 봇 종료 처리"""
        try:
            self.is_running = False
            logging.info("트레이딩 봇 정상 종료 처리 완료")
        except Exception as e:
            logging.error(f"트레이딩 봇 종료 중 오류 발생: {e}")


    def check_stop_loss(self, ticker: str):
        """손절 및 익절 체크 메서드"""
        try:
            current_price = pyupbit.get_current_price(ticker)
            if current_price is None:
                return

            stop_loss_info = self.STOP_LOSS.get(ticker)
            if stop_loss_info:
                if current_price <= stop_loss_info['stop_loss']:
                    logging.info(f"{ticker} 손절가 도달")
                    self.execute_trade(ticker, "SELL")
                elif current_price >= stop_loss_info['take_profit']:
                    logging.info(f"{ticker} 목표가 도달")
                    self.execute_trade(ticker, "SELL")
        except Exception as e:
            logging.error(f"{ticker} 손절/익절 체크 중 오류: {e}")

    def run(self):
        try:
            logging.info("트레이딩 봇 시작...")
            while self.is_running:
                try:
                    logging.info("새로운 거래 사이클 시작...")
                    self._process_trading_cycle()
                    # 5분 대기
                    time.sleep(300)
                except Exception as e:
                    logging.error(f"거래 사이클 중 오류 발생: {e}")
                    time.sleep(5)
                    continue
        except KeyboardInterrupt:
            logging.info("트레이딩 봇 종료 요청 받음...")
        finally:
            self.stop()
            logging.info("트레이딩 봇 종료됨")


import logging

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 데이터베이스 매니저 초기화
        db_manager = DatabaseManager('trading_history.db')

        # 뉴스 수집 및 저장 실행
        ticker_keyword = '비트코인'
        collected_news = NewsCollector.get_news(keyword=ticker_keyword)

        trader = AITrader()
        trader.run()

        # 뉴스 데이터 유효성 확인 후 저장
        if collected_news and isinstance(collected_news, list):
            db_manager.save_news(ticker=ticker_keyword, news_data=collected_news)
        else:
            logging.warning("수집된 뉴스 데이터가 없거나 유효하지 않습니다.")
    except Exception as e:
        logging.error(f"프로그램 실행 중 치명적 오류 발생: {e}")
    finally:
        logging.info("프로그램 종료")