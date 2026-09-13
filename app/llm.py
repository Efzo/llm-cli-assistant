import logging

import openai
from openai import OpenAI

from app.config import (
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MODEL,
    OPENAI_API_KEY,
    SYSTEM_INSTRUCTIONS,
    TIMEOUT_SECONDS,
)


logger = logging.getLogger(__name__)


client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=TIMEOUT_SECONDS,
    max_retries=MAX_RETRIES,
)


def ask_llm(question: str) -> str:
    logger.info(
        "Sending LLM request | question_length=%d",
        len(question),
    )

    try:
        stream = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=question,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        full_answer = ""
        completed_response = None

        print("\nAI: ", end="", flush=True)

        for event in stream:

            if event.type == "response.output_text.delta":
                print(
                    event.delta,
                    end="",
                    flush=True,
                )

                full_answer += event.delta

            elif event.type == "response.completed":
                completed_response = event.response

        print()

        if completed_response and completed_response.usage:
            usage = completed_response.usage

            print("\n--- Usage ---")
            print(
                f"Input tokens: {usage.input_tokens}"
            )
            print(
                f"Output tokens: {usage.output_tokens}"
            )
            print(
                f"Total tokens: {usage.total_tokens}"
            )
            print("-------------")

            logger.info(
                "LLM response completed | "
                "input_tokens=%d "
                "output_tokens=%d "
                "total_tokens=%d",
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
            )

        return full_answer

    except openai.APITimeoutError:
        logger.error("LLM request timed out")

        print(
            "\nError: The request timed out. "
            "Please try again."
        )

        return ""

    except openai.RateLimitError:
        logger.warning(
            "OpenAI rate limit reached"
        )

        print(
            "\nError: The API rate limit was reached."
        )

        return ""

    except openai.APIConnectionError:
        logger.error(
            "Could not connect to OpenAI API"
        )

        print(
            "\nError: Could not connect "
            "to the OpenAI API."
        )

        return ""

    except openai.APIStatusError as error:
        logger.error(
            "OpenAI API error | status_code=%d",
            error.status_code,
        )

        print(
            f"\nOpenAI API error: "
            f"{error.status_code}"
        )

        return ""

    except Exception:
        logger.exception(
            "Unexpected application error"
        )

        print(
            "\nAn unexpected error occurred."
        )

        return ""