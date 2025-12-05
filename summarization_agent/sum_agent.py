from groq import Groq
import os

class SummarizationAgent:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def summarize(self, text, max_len=200):
        prompt = (
            f"Summarize the following text in less than {max_len} words. "
            "Keep it concise and factual.\n\n"
            f"TEXT:\n{text}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content
