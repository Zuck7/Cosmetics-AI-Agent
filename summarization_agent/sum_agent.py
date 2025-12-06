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
            # Fallback: extract sentences most relevant to the question
            import re
            # Try to split into question and context
            if text.lower().startswith("question:") and "context:" in text:
                parts = text.split("Context:", 1)
                query = parts[0].replace("Question:", "").strip()
                context = parts[1].strip()
            else:
                query = ""
                context = text
            sentences = re.split(r'(?<=[.!?])\s+', context)
            query_words = set(query.lower().split())
            # Score sentences by overlap with query words
            scored = []
            for s in sentences:
                score = sum(1 for w in query_words if w in s.lower())
                if score > 0:
                    scored.append((score, s))
            # Sort by score, take top sentences up to max_len words
            summary = []
            word_count = 0
            for _, s in sorted(scored, reverse=True):
                if word_count + len(s.split()) > max_len:
                    break
                summary.append(s)
                word_count += len(s.split())
            # Fallback: if nothing matched, take first sentences
            if not summary:
                for s in sentences:
                    if word_count + len(s.split()) > max_len:
                        break
                    summary.append(s)
                    word_count += len(s.split())
            return ' '.join(summary)
