# webapp/app.py
import gradio as gr
from .service import QAGenerationService
from .config import MODEL_PATH


class QAGeneratorApp:
    def __init__(self):
        self.service = QAGenerationService(MODEL_PATH)

    def generate_qa(self, paragraph: str) -> str:
        """Generate QA pairs from a paragraph"""
        if not paragraph or not paragraph.strip():
            return "Please enter a paragraph."

        try:
            # Get all generated QA pairs
            qa_pairs = self.service.process_paragraph(paragraph)

            if not qa_pairs:
                return "No questions could be generated."

            # Format output
            output_parts = ["Generated Question-Answer Pairs:\n"]

            for i, qa in enumerate(qa_pairs, 1):
                pair_text = [
                    f"\nPair {i}:",
                    f"Question: {qa['question']}",
                    f"Answer: {qa['answer']}",
                    f"Style: {qa['style']}"
                ]

                if qa.get('clue'):
                    pair_text.append(f"Clue: {qa['clue']}")

                pair_text.append("-" * 50)
                output_parts.append("\n".join(pair_text))

            return "\n".join(output_parts)

        except Exception as e:
            import traceback
            return f"Error generating questions:\n{str(e)}\n\n{traceback.format_exc()}"

    def launch(self):
        """Launch the Gradio interface"""
        interface = gr.Interface(
            fn=self.generate_qa,
            inputs=gr.Textbox(
                lines=5,
                placeholder="Enter your paragraph here...",
                label="Input Paragraph"
            ),
            outputs=gr.Textbox(
                label="Generated Q&A Pairs",
                lines=15
            ),
            title="Vietnamese Question-Answer Generator",
            description="Enter a paragraph to generate multiple question-answer pairs.",
            examples=[
                ["Hà Nội là thủ đô của Việt Nam, một thành phố với hơn 1000 năm lịch sử."]
            ]
        )

        interface.launch(share=True)


def main():
    app = QAGeneratorApp()
    app.launch()


if __name__ == "__main__":
    main()
