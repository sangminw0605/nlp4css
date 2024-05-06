from openai import OpenAI
import openai
from os import environ

client = OpenAI(api_key='sk-proj-7yEkgWUG98DXhaQqcMO0T3BlbkFJbFKPcOEUKYSKph7j6lTg')


stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Summarize the implied meanings in the following comment briefly. Do not write a complete sentence, just synthesize the message: Bill Kristol and Ben Shaprio, two turds in the same toilet bowl."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")


