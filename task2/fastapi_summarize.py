import os
import json
import google.generativeai as genai

# ---------- Setup ----------
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY not found in environment variables!")
genai.configure(api_key=api_key)  # replace with your actual API key
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

DATA_DIR = r"C:\Users\shruti shreya\Downloads\assessment-infraintelai\task1\mycleaned_texts"
OUTPUT_FILE = "summaries.json"

# ---------- Prompt Template ----------
def make_prompt(note_text: str) -> str:
    return f"""Extract the following fields from the clinical note and return valid JSON with keys: Patient, Diagnosis, Treatment, Followup. Also, if there's some common typing mistake, correct it in the output.


Note:
{note_text}

JSON:
"""

# ---------- Batch Processing ----------
def summarize_notes():
    results = {}

    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                text = f.read().strip()

            print(f"📄 Processing {file}...")

            try:
                # Ask Gemini
                response = gemini_model.generate_content(make_prompt(text))
                output_text = response.text.strip()

                # Try to parse as JSON
                try:
                    summary = json.loads(output_text)
                except json.JSONDecodeError:
                    print(f"⚠️ JSON decode failed for {file}, saving raw text")
                    summary = {"raw_output": output_text}

                results[file] = summary

            except Exception as e:
                print(f"❌ Error with {file}: {e}")
                results[file] = {"error": str(e)}

    # Save all summaries
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(results, out, indent=4, ensure_ascii=False)

    print(f"\n✅ Summaries saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    summarize_notes()
