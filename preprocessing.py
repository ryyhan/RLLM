import json
import re
import glob

chat_files = glob.glob("chats/*.txt")
output_file = "dataset.jsonl"
data = []
current_sender = None
current_text = ""

# Your name in the chat
user_name = "Rehan"

def flush_message():
    global current_sender, current_text, data
    if current_sender:
        data.append({"sender": current_sender, "text": current_text.strip()})
        current_sender = None
        current_text = ""

# Aggregate consecutive messages
for file_path in chat_files:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in lines:
        # regex to handle Unicode spaces and lowercase am/pm
        match = re.match(
            r"(\d{2}/\d{2}/\d{4}),\s*(\d{1,2}:\d{2}\s*[APMapm]{2})\s*-\s*(.*?):\s*(.*)",
            line,
            re.IGNORECASE
        )
        if match:
            _, _, sender, message = match.groups()
            if sender == current_sender:
                current_text += " " + message
            else:
                flush_message()
                current_sender = sender
                current_text = message
    flush_message()

# Pair messages using name
pairs = []
for i in range(len(data) - 1):
    current = data[i]
    next_msg = data[i + 1]
    if (
        current["sender"].lower() != user_name.lower()
        and next_msg["sender"].lower() == user_name.lower()
    ):
        pairs.append({
            "prompt": f"{current['sender']}: {current['text']}",
            "response": next_msg["text"]
        })

# Save as JSONL
with open(output_file, "w", encoding="utf-8") as file:
    for entry in pairs:
        file.write(json.dumps(entry) + "\n")

print(f"Dataset saved as {output_file}, total samples: {len(pairs)}")