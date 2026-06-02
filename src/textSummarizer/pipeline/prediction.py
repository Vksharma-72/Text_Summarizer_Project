from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline 

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs =  {"length_penalty": 0.8, "num_beams":8 , "max_length": 128}

        pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\n Model Summary")
        print(output)

        return print(output)

# if __name__ == "__main__":
#     obj = PredictionPipeline()
#     obj.predict("""Hannah: Hey, do you have Betty's number?
# Amanda: Lemme check
# Hannah: <file_gif>
# Amanda: Sorry, can't find it.
# Amanda: Ask Larry
# Amanda: He called her last time we were at the park together
# Hannah: I don't know him well
# H """)