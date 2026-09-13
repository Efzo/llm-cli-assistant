import logging

from app.llm import ask_llm


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


logger = logging.getLogger(__name__)


def main():
    logger.info("AI assistant started")

    print("AI Assistant")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {
            "exit",
            "quit",
        }:
            logger.info(
                "AI assistant stopped"
            )

            print("Goodbye.")

            break

        if not question:
            continue

        ask_llm(question)


if __name__ == "__main__":
    main()