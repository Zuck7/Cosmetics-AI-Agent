try:
    from groq import Groq
    _HAVE_GROQ = True
except ImportError:
    Groq = None
    _HAVE_GROQ = False

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass

import os


class SummarizationAgent:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        # Load .env file
        load_dotenv()

        self.model_name = model_name
        self.client = None

        if _HAVE_GROQ:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.client = Groq(api_key=api_key)
            else:
                print("⚠️  GROQ_API_KEY not found. Using simple demo summarization.")
        else:
            print("⚠️  groq package not installed. Using simple demo summarization.")

    def summarize(self, text, max_len=200):
        # If Groq is available and configured, use it
        if self.client is not None:
            prompt = f"Summarize the following text in less than {max_len} words:\n{text}"
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        else:
            # Enhanced fallback: extract sentences most relevant to the question
            import re

            # Try to split into question and context
            if text.lower().startswith("question:") and "context:" in text.lower():
                parts = text.split("Context:", 1) if "Context:" in text else text.split("context:", 1)
                if len(parts) == 2:
                    question_part = parts[0].replace("Question:", "").replace("question:", "").strip()
                    context_part = parts[1].strip()
                else:
                    question_part = ""
                    context_part = text
            else:
                question_part = ""
                context_part = text

            sentences = re.split(r'(?<=[.!?])\s+', context_part)

            if question_part:
                # Score sentences by overlap with query words
                query_words = set(question_part.lower().split())
                scored = []
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    # Score based on word overlap and sentence length
                    s_words = set(s.lower().split())
                    overlap = len(query_words.intersection(s_words))
                    # Prefer sentences with good overlap and reasonable length
                    score = overlap * 10 + min(len(s.split()), 20)  # Bonus for reasonable length
                    if overlap > 0:
                        scored.append((score, s))

                # Sort by score, take top sentences up to max_len words
                summary = []
                word_count = 0
                for _, s in sorted(scored, reverse=True):
                    if word_count + len(s.split()) > max_len:
                        break
                    summary.append(s)
                    word_count += len(s.split())

                if summary:
                    return ' '.join(summary)

            # Fallback: if no question or no good matches, take first sentences
            summary = []
            word_count = 0
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if word_count + len(s.split()) > max_len:
                    break
                summary.append(s)
                word_count += len(s.split())

            return ' '.join(summary)
