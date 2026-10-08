print("Script started")

from datasets import load_dataset

print("Loading Common Voice...")

dataset = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "en",
    split="train",
    streaming=True
)

print("Common Voice loaded successfully!")

sample = next(iter(dataset))

print("Sample received!")
print(sample)