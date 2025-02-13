import re
from dataclasses import dataclass
from enum import Enum
from random import choice
from typing import Optional


class QueryType(Enum):
    GREETING = "greeting"
    SYSTEM_QUERY = "system_query"
    TAX_QUERY = "tax_query"
    IRRELEVANT = "irrelevant"


@dataclass
class QueryAnalysisResult:
    query_type: QueryType
    confidence: float
    details: Optional[dict] = None


class QueryAnalyzer:
    def __init__(self):
        self.greeting_patterns = {
            r"^(добр[ий|ого]\s*)(ранк[у|ок]|ден[ь|я]|вечі[р|ора])": 0.9,
            r"^віта[ю|ння]": 0.9,
            r"^прив[і|e]т": 0.9,
            r"^здрастуй(те)?": 0.9,
            r"^хай$": 0.8,
            r"^hi$": 0.8,
            r"^hello$": 0.8,
        }

        self.system_query_keywords = {
            "асистент": 0.6,
            "помічник": 0.6,
            "бот": 0.6,
            "модель": 0.6,
            "працюєш": 0.7,
            "функціонуєш": 0.7,
            "влаштований": 0.7,
            "можливості": 0.6,
        }

        self.tax_keywords = {
            "податок": 0.8,
            "фоп": 0.9,
            "тов": 0.9,
            "єсв": 0.9,
            "пдфо": 0.9,
            "звітність": 0.8,
            "декларація": 0.8,
            "платник": 0.7,
            "реєстрація": 0.7,
            "підприємець": 0.8,
            "зарплата": 0.7,
            "податки": 0.8,
            "резидент": 0.8,
        }

    def analyze_query(self, text: str) -> QueryAnalysisResult:
        text = text.lower().strip()

        for pattern, confidence in self.greeting_patterns.items():
            if re.search(pattern, text):
                return QueryAnalysisResult(
                    QueryType.GREETING, confidence, {"pattern": pattern}
                )

        system_confidence = 0
        system_matches = []
        for keyword, weight in self.system_query_keywords.items():
            if keyword in text:
                system_confidence += weight
                system_matches.append(keyword)

        tax_confidence = 0
        tax_matches = []
        for keyword, weight in self.tax_keywords.items():
            if keyword in text:
                tax_confidence += weight
                tax_matches.append(keyword)

        system_confidence = min(system_confidence, 1.0)
        tax_confidence = min(tax_confidence, 1.0)

        if system_confidence > 0.6:
            return QueryAnalysisResult(
                QueryType.SYSTEM_QUERY,
                system_confidence,
                {"matched_keywords": system_matches},
            )
        elif tax_confidence > 0.6:
            return QueryAnalysisResult(
                QueryType.TAX_QUERY, tax_confidence, {"matched_keywords": tax_matches}
            )

        return QueryAnalysisResult(
            QueryType.IRRELEVANT, 0.7, {"reason": "No relevant keywords found"}
        )


class QueryHandler:
    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.greeting_responses = [
            "Вітаю! Чим можу допомогти з питань оподаткування?",
            "Доброго дня! Готовий допомогти вам з податковими питаннями.",
            "Вітаю! Я ваш податковий асистент. Яке у вас питання?",
        ]
        self.system_query_response = """
        Я - спеціалізований асистент з податкового законодавства України. 
        Моя база знань включає:
        - Податковий кодекс України
        - Актуальні ставки податків
        - Правила оподаткування для різних груп платників
        - Терміни подання звітності

        Я можу допомогти вам:
        - Розрахувати податки
        - Пояснити правила оподаткування
        - Надати інформацію про терміни та звітність
        - Відповісти на питання щодо ФОП та найманих працівників

        Чим можу бути корисним?
        """
        self.irrelevant_query_response = """
        Вибачте, але це питання виходить за межі моєї спеціалізації. 
        Я - податковий асистент і можу допомогти з питаннями щодо:
        - Оподаткування та податкової звітності
        - Реєстрації та ведення діяльності ФОП
        - Розрахунку податків та зборів
        - Термінів подання звітності

        Будь ласка, задайте питання, пов'язане з цими темами.
        """

    def handle_query(self, query: str, model_response_func=None) -> str:
        analysis = self.analyzer.analyze_query(query)

        if analysis.query_type == QueryType.GREETING:
            return choice(self.greeting_responses)

        elif analysis.query_type == QueryType.SYSTEM_QUERY:
            return self.system_query_response.strip()

        elif analysis.query_type == QueryType.TAX_QUERY:
            if model_response_func:
                return model_response_func(query)
            return ""

        else:
            return self.irrelevant_query_response.strip()
