import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "The quarterback threw the football 87 yards for a touchdown"
ids = tokenizer.encode(text, allowed_special=set())
print(f"Text: '{text}'")
print(f"Encoded: {len(ids)} tokens - {ids}")
