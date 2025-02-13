import os
import re
import warnings
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings

from model.query_handler import QueryHandler, QueryType

warnings.filterwarnings("ignore", category=FutureWarning)


class TaxCodeAssistant:
    def __init__(
        self,
        persist_directory: str = "faiss/db",
        embeddings_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: str = "cpu",
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_retries: int = 5,
    ):
        load_dotenv()
        self.max_retries = max_retries
        self.persist_directory = persist_directory

        self.query_handler = QueryHandler()

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
            k=5,
        )

        self.llm = HuggingFaceHub(
            repo_id=model_name,
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            model_kwargs={
                "temperature": 0.5,
                "max_new_tokens": 512,
                "top_p": 0.95,
                "do_sample": True,
                "num_beams": 3,
                "return_full_text": False,
                "context_length": 8192,
                "early_stopping": True,
            },
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model, model_kwargs={"device": device}
        )

        if os.path.exists(os.path.join(persist_directory, "index.faiss")):
            self.vectorstore = FAISS.load_local(
                persist_directory, self.embeddings, allow_dangerous_deserialization=True
            )
        else:

            self.vectorstore = None

        self.prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""[INST] Ти - асистент з податкового законодавства України.
                ВАЖЛИВО: Відповідай українською мовою за замовчуванням.
                ВАЖЛИВО: Надавай тільки чітку, структуровану відповідь. Уникайте повторень та незрозумілих послідовностей.

                Історія чату:
                {chat_history}

                Використовуй наданий контекст для відповіді на питання.
                Відповідай лише на основі контексту. Якщо не можеш знайти відповідь в контексті,
                скажи що не знаєш. 

                Якщо в питанні є конкретні цифри, обов'язково зроби розрахунок та 
                покажи його покроково. Використовуй актуальні ставки податків.

                1. Якщо в запиті не вистачає інформації:
                    - Вкажи, яка саме інформація потрібна
                    - Задай конкретні уточнюючі питання
                    - Надай грунтовну відповідь на питання       

                2. При розрахунку податків:
                    - Показуй розрахунок покроково
                    - Вказуй формули розрахунку
                    - Пояснюй кожен крок тільки якщо про це напишуть

                3. Завжди вказуй:
                    - Посилання на статті Податкового Кодексу України
                    - Терміни сплати податків
                    - Терміни подання звітності

                4. Формат відповіді:                  
                    - Якщо у запиті попросять відповідати юридичною (професійною) мовою то відповідай як професіональний юрист
                    - Виділяй важливі цифри та дати

                5. При відповіді на питання про ФОП:
                    1. Обов'язково уточни групу ФОП
                    2. Перевір ліміти доходу для групи
                    3. Нагадай про ЄСВ
                    4. Вкажи обмеження щодо видів діяльності
                    5. Поясни різницю між групами якщо доречно           

                8. При відповіді на питання про найманих працівників:
                    1. Враховуй базову ставку ПДФО 
                    2. Не забудь про військовий збір 
                    3. Перевір право на податкову соціальну пільгу
                    4. Вкажи обов'язки роботодавця
                    5. Нагадай про терміни виплати зарплати

                    Контекст: {context}

                    Питання: {question} [/INST]""",
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def format_sources(self, context: List[Dict[str, Any]]) -> str:
        articles_dict = {}
        sorted_context = sorted(context, key=lambda x: x["score"])

        for doc in sorted_context:
            metadata = doc["metadata"]
            content = doc.get("content", "")
            score = doc["score"]

            article_matches = re.finditer(r"[Сс]таття (\d+(?:\.\d+)*)", content)
            for match in article_matches:
                article = match.group(1)
                if article not in articles_dict:
                    articles_dict[article] = (score, set())

            point_matches = re.finditer(r"(?:пункт[іу]?\s+)?(\d+(?:\.\d+)*)", content)
            for match in point_matches:
                point = match.group(1)

                if "." in point:
                    article = point.split(".")[0]
                    if article in articles_dict:
                        articles_dict[article][1].add(point)

        sorted_articles = sorted(articles_dict.items(), key=lambda x: x[1][0])

        sources = []
        for article, (_, points) in sorted_articles[:5]:
            source = f"Податковий кодекс України, Стаття {article}"
            if points:
                sorted_points = sorted(
                    points, key=lambda x: [float(n) for n in x.split(".")]
                )
                points_str = ", ".join(sorted_points[:3])
                if points_str:
                    source += f", пункти {points_str}"
            sources.append(source)

        if not sources:
            return "Податковий кодекс України"

        return "\n".join(sources)

    def get_context(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        context = []
        for doc, score in results:
            content = doc.page_content[:8192]
            context.append(
                {"content": content, "metadata": doc.metadata, "score": score}
            )
        return context

    def validate_response(self, response: str) -> dict:
        errors = []

        if not response:
            return {"is_valid": False, "errors": [""]}

        words = response.lower().split()

        unique_words = set(word for word in words if len(word) > 2)
        if len(unique_words) < 5:
            errors.append("")
            return {"is_valid": False, "errors": errors}

        consecutive_repeats = 0
        previous_word = None
        for word in words:
            if word == previous_word and len(word) > 2:
                consecutive_repeats += 1
                if consecutive_repeats > 5:
                    errors.append("")
                    break
            else:
                consecutive_repeats = 0
            previous_word = word

        word_counts = {}
        for word in words:
            if len(word) > 0:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] > len(words) * 0.15:
                    errors.append(f" ")
                    break

        legal_reference = r"\d+\.\d+(\.\d+)*"
        cleaned_response = re.sub(legal_reference, "LEGAL_REF", response)

        repetitive_pattern = r"(\d+\.){7,}"
        if re.search(repetitive_pattern, cleaned_response):
            errors.append("")

        numbers = len(re.findall(r"\d", response))
        text_length = len(response)
        if numbers > text_length * 0.5:
            errors.append("")

        invalid_pattern = r"^[I\d.]+$"
        if re.match(invalid_pattern, response.strip()):
            errors.append("")

        ukrainian_pattern = r"[а-яА-ЯіІїЇєЄґҐ]"
        if not re.search(ukrainian_pattern, response):
            errors.append("")

        nonsense_patterns = [
            r"(\b\w+\b)(\s+\1){10,}",
            r"[A-Za-z\s]{30,}",
        ]

        for pattern in nonsense_patterns:
            if re.search(pattern, response):
                errors.append("")
                break

        return {"is_valid": len(errors) == 0, "errors": errors}

    def get_response(self, query: str) -> str:

        for attempt in range(self.max_retries):
            try:
                context = self.get_context(query)
                if not context:
                    return "Не знайдено релевантної інформації для відповіді на це питання."

                context_text = "\n".join([doc["content"] for doc in context])
                if len(context_text) > 8192:
                    context_text = context_text[:8192] + "..."

                chat_history = self.memory.load_memory_variables({})["chat_history"]

                response = self.chain.invoke(
                    {
                        "question": query,
                        "context": context_text,
                        "chat_history": chat_history,
                    }
                )

                if isinstance(response, dict) and "text" in response:
                    response = response["text"]
                else:
                    response = str(response)

                sources = self.format_sources(context)
                response += f"\n\nДжерела:\n{sources}"

                validation_result = self.validate_response(response)
                if validation_result["is_valid"]:
                    return response
                else:
                    if attempt == self.max_retries - 1:
                        error_details = "\n".join(validation_result["errors"])
                        return (
                            f"Вибачте, але я не можу надати коректну відповідь на ваше запитання. "
                            f"Причини:\n{error_details}\n"
                            f"Будь ласка, спробуйте переформулювати запитання."
                        )
                    continue

            except Exception as e:
                print(f"Error in get_response: {str(e)}")
                if attempt == self.max_retries - 1:
                    return f"Виникла помилка при генерації відповіді: {str(e)}"
                continue

    def process_query(self, query: str) -> str:

        return self.query_handler.handle_query(
            query, model_response_func=self.get_response
        )
