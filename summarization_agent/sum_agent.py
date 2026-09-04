from groq_config import get_groq_client, get_groq_api_key


class SummarizationAgent:
    def __init__(self, model_name="openai/gpt-oss-20b"):
        self.model_name = model_name
        self.client = get_groq_client()

        if self.client is None:
            if get_groq_api_key() is None:
                print("GROQ_API_KEY not found. Using simple demo summarization.")
            else:
                print("Groq client unavailable. Using simple demo summarization.")

    def summarize(self, text, max_len=200):
        # If Groq is available and configured, use it; otherwise fallback locally.
        if self.client is not None:
            prompt = f"Summarize the following text in less than {max_len} words:\n{text}"
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Groq summarization failed ({e}). Falling back to local summarization.")

        return self._fallback_summarize(text, max_len=max_len)

    def _fallback_summarize(self, text, max_len=200):
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
