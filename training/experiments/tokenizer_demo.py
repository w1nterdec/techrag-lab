from training.tokenizer.tokenizer import TechRAGTokenizer


tokenizer = TechRAGTokenizer()


text = "docker ps permission denied"


tokens = tokenizer.encode(text)


print("Text:")
print(text)

print("\nToken ids:")
print(tokens)


print("\nDecode:")
print(tokenizer.decode(tokens))
