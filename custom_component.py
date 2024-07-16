from typing import List
from haystack import component, Pipeline


@component
class WelcomeTextGenerator:
    """
    A component generating personal welcome message and making it upper case
    """

    @component.output_types(welcome_text=str, note=str)
    def run(self, name: str):
        return {"welcome_text": ('Hello {name}, welcome to Haystack!'.format(name=name)).upper(),
                "note": "welcome message is ready"}


@component
class WhitespaceSplitter:
    """
    A component for splitting the text by whitespace
    """

    @component.output_types(splitted_text=List[str])
    def run(self, text: str):
        return {"splitted_text": text.split()}


text_pipeline = Pipeline()
text_pipeline.add_component(name="welcome_text_generator", instance=WelcomeTextGenerator())
text_pipeline.add_component(name="splitter", instance=WhitespaceSplitter())

text_pipeline.connect(sender="welcome_text_generator.welcome_text", receiver="splitter.text")

result = text_pipeline.run({"welcome_text_generator": {"name": "Bilge"}})

print(result["splitter"]["splitted_text"])