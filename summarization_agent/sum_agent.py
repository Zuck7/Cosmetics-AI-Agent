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
            # Fallback: simple extractive summarization
            sentences = text.split('. ')
            words_per_sentence = max_len // len(sentences) if sentences else max_len
            summary = '. '.join(
                s for s in sentences[:max(2, len(sentences) // 2)]
            )
            if not summary.endswith('.'):
                summary += '.'
            return summary[:max_len]
