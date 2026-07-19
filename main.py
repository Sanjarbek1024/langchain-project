import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()


# Categories
categories = [
    "Math",
    "Music",
    "Programming",
]


# Structured Output Schema
class Classification(BaseModel):
    category_name: str = Field(
        description="Must be one of: Math, Music, Programming."
    )
    score: float = Field(
        description="Confidence score between 0 and 1."
    )


# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a text classification assistant.

Classify the user's message into ONLY ONE category.

Allowed categories:
{categories}

Return the response using the provided structured output schema.

Rules:
- category_name must be one of the allowed categories.
- score must be between 0 and 1.
""",
        ),
        (
            "human",
            "{message}",
        ),
    ]
)


# Structured Output
structured_llm = llm.with_structured_output(Classification)


# Chain
chain = prompt | structured_llm


def classify(message: str):
    return chain.invoke(
        {
            "message": message,
            "categories": ", ".join(categories),
        }
    )


def stream_response(message: str):
    print("\nStreaming Response:\n")

    stream_chain = prompt | llm

    for chunk in stream_chain.stream(
        {
            "message": message,
            "categories": ", ".join(categories),
        }
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":

    message = input("Enter message: ")

    result = classify(message)

    print("\nClassification Result")
    print("---------------------")
    print(f"Category : {result.category_name}")
    print(f"Score    : {result.score:.2f}")

    print("\n" + "=" * 50)

    stream_response(message)