import sqlite3
import logging
from typing import Dict


class TradingReflectionManager:
    def __init__(self, db_name: str):
        self.db_name = db_name


    def save_trading_reflection(self, trade_id: int, market_condition: str, decision_accuracy: str,
                                improvement_points: str, next_strategy: str):
        """
        트레이딩 반성 테이블에 데이터를 저장합니다.
        """
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                        INSERT INTO trading_reflection 
                        (trade_id, market_condition, decision_accuracy, improvement_points, next_strategy)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        trade_id,
                        market_condition,
                        decision_accuracy,
                        improvement_points,
                        next_strategy
                    )
                )
                conn.commit()
                logging.info(f"트레이딩 반성 데이터 저장 성공. trade_id={trade_id}")
        except Exception as e:
            logging.error(f"트레이딩 반성 데이터 저장 실패: {e}")

    def retrieve_reflections(self):
        """
        모든 트레이딩 반성 데이터를 조회합니다.
        """
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM trading_reflection')
                reflections = cursor.fetchall()
                return reflections
        except Exception as e:
            logging.error(f"트레이딩 반성 데이터 조회 실패: {e}")
            return []

    def generate_reflection_data(self, trade_id: int, ticker: str, decision: str, result: str, reason: str) -> Dict:

        try:

            reflection_data = self.reflection_manager.generate_reflection_data(
                trade_id=trade_id,
                ticker=ticker,
                decision=decision,
                result=result,
                reason=reason
            )
            print("Reflection Data:", reflection_data)  # 디버깅용
        except Exception as e:
            logging.error(f"generate_reflection_data 호출 실패: {e}")

        if reflection_data:
            try:
                self.reflection_manager.save_trading_reflection(
                    trade_id=reflection_data["trade_id"],
                    market_condition=reflection_data["market_condition"],
                    decision_accuracy=reflection_data["decision_accuracy"],
                    improvement_points=reflection_data["improvement_points"],
                    next_strategy=reflection_data["next_strategy"]
                )
            except Exception as e:
                logging.error(f"save_trading_reflection 호출 실패: {e}")

        """
        AI의 응답을 기반으로 트레이딩 반성 데이터를 생성합니다.
        """
        try:
            # 트레이딩 반성 데이터 생성 로직
            prompt = f"""
                [트레이드 반성 분석 - {ticker}]
                **거래 정보**:
                - 거래 ID: {trade_id}
                - 티커: {ticker}
                - 결정: {decision}
                - 결과: {result}
                - 이유: {reason}
                **질문**:
                1. 거래 결과에 따른 AI 분석 판단의 정확성을 평가하세요.
                2. 주요 성공/실패 원인을 정리하세요.
                3. 향후 개선할 점과 새로운 전략을 제안하세요.
            """
            response = self.ai_client.chat.completions.create(
                model="deepseek/deepseek-r1:free",
                messages=[{"role": "system", "content": "간단 명확하게 대답하세요.영어나 한글로만 대답해주세요."},
                          {"role": "user", "content": prompt}],
                max_tokens=200
            )
            content = response.choices[0].message["content"]
            lines = content.splitlines()
            response = "AI Dummy Response"  # 예제용 응답
            # 가공된 AI 응답을 파싱합니다.

            return {
                "trade_id": trade_id,
                "market_condition": "시장 좋음",
                "decision_accuracy": "75%",
                "improvement_points": "분석 기간 부족",
                "next_strategy": "단기적 관점 강화",
            }
        except Exception as e:
            logging.error(f"트레이딩 반성 데이터 생성 실패: {e}")
            return {}
