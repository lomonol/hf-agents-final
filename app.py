import os
import gradio as gr
import requests
import inspect
import pandas as pd
from openai import OpenAI
from smolagents import OpenAIServerModel, CodeAgent, InferenceClientModel, DuckDuckGoSearchTool, VisitWebpageTool, tool

GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Wikipedia/Wikimedia return 403 to the default python-requests User-Agent.
# Give every requests call (incl. smolagents' visit_webpage) a descriptive one.
AGENT_USER_AGENT = "GAIA-course-agent/1.0 (educational agent; python-requests)"
requests.utils.default_user_agent = lambda *args, **kwargs: AGENT_USER_AGENT

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Local run settings (no HF Space / OAuth login) ---
HF_USERNAME = os.getenv("HF_USERNAME", "lomonol")
AGENT_CODE_URL = os.getenv("AGENT_CODE_URL", "https://github.com/lomonol/hf-agents-final")

# --- Custom File Download Tool (Fixes Sandbox `open()` Restriction) ---
@tool
def download_task_file(task_id: str) -> str:
    """
    Downloads any file associated with a specific GAIA evaluation task ID. 
    For text-based files (.py, .txt, .csv, .json), it reads and returns the file content directly.
    For structured or binary files (.xlsx), it saves it locally and returns the path for pandas.
    
    Args:
        task_id: The unique task identifier string provided in the question context.
    """
    file_url = f"{DEFAULT_API_URL}/files/{task_id}"
    try:
        response = requests.get(file_url, timeout=30)
        if response.status_code == 200 and response.content:
            filename = "task_file.dat"
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition and "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[-1].strip('"\'')
            
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"Successfully downloaded file: {filename} ({len(response.content)} bytes)")
            
            # If it's a text/code file, return its text content directly to bypass sandbox open() blocks
            if filename.endswith((".py", ".txt", ".csv", ".json", ".md")):
                try:
                    with open(filename, "r", encoding="utf-8") as text_f:
                        content = text_f.read()
                    return f"File '{filename}' content:\n{content}"
                except Exception as read_err:
                    return f"Downloaded file saved as {filename}, but failed to read text: {read_err}"
            
            return f"File successfully downloaded and saved locally as: {filename}"
        else:
            # The course API currently returns 404 for every attachment; fall back to the
            # official GAIA dataset (gated: needs HF_TOKEN and accepting its terms once).
            return download_from_gaia_dataset(task_id)
    except Exception as e:
        return f"Error downloading file: {e}"


GAIA_REPO = "gaia-benchmark/GAIA"
GAIA_DIR = "2023/validation"
_gaia_names = None  # cached list of attachment file names in the dataset folder

def download_from_gaia_dataset(task_id: str) -> str:
    """Fetch a GAIA validation attachment whose name starts with task_id from the gaia-benchmark/GAIA dataset."""
    global _gaia_names
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN")
    if not token:
        return "Cannot download attachment: HF_TOKEN is not set in this terminal."
    try:
        if _gaia_names is None:
            files = HfApi().list_repo_tree(GAIA_REPO, path_in_repo=GAIA_DIR, repo_type="dataset", token=token)
            _gaia_names = [f.path.rsplit("/", 1)[-1] for f in files]
        matches = [n for n in _gaia_names if n.startswith(task_id + ".")]
        if not matches:
            return f"No attachment for task {task_id} in the GAIA dataset either."
        name = matches[0]
        url = f"https://huggingface.co/datasets/{GAIA_REPO}/resolve/main/{GAIA_DIR}/{name}"
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
        if resp.status_code != 200:
            return f"Attachment download failed (GAIA dataset): HTTP {resp.status_code}: {resp.text[:200]}"
        os.makedirs("gaia_files", exist_ok=True)
        path = os.path.join("gaia_files", name)
        with open(path, "wb") as out:
            out.write(resp.content)
    except Exception as e:
        return f"Attachment download failed (GAIA dataset): {type(e).__name__}: {str(e)[:300]}"
    print(f"Downloaded {name} from the GAIA dataset")
    if name.endswith((".py", ".txt", ".csv", ".json", ".md")):
        with open(path, "r", encoding="utf-8") as text_f:
            return f"File '{name}' content:\n{text_f.read()}"
    return f"File successfully downloaded and saved locally as: {path}"

# --- Audio Transcription Tool (Groq Whisper, free tier) ---
@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribes a downloaded audio file (e.g. .mp3) to text using Whisper.
    Use it after download_task_file has saved the audio file locally.

    Args:
        file_path: Path of the audio file saved by download_task_file (e.g. 'abc123.mp3').
    """
    try:
        client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_API_BASE)
        with open(file_path, "rb") as f:
            result = client.audio.transcriptions.create(model="whisper-large-v3-turbo", file=f)
        return f"Transcript of '{file_path}':\n{result.text}"
    except Exception as e:
        return f"Error transcribing audio: {e}"

# --- Smolagents Agent Definition ---
class SmolAgentsGAIAAgent:
    def __init__(self):
        print("Initializing Smolagents CodeAgent with Web Search, Web Page Reader, and File Downloader...")
        
        self.model = OpenAIServerModel(
            model_id="qwen/qwen3.8-27b",
            api_base=GROQ_API_BASE,
            api_key=os.environ["GROQ_API_KEY"],
        )
        self.tools = [DuckDuckGoSearchTool(), VisitWebpageTool(), download_task_file, transcribe_audio]
        
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            additional_authorized_imports=["requests", "bs4", "json", "math", "pandas", "pypdf", "openpyxl"],
            max_steps=10,
            verbosity_level=1
        )

    def __call__(self, question: str) -> str:
        print(f"Agent received prompt: {question}")
        
        prompt = (
            f"Answer the following question accurately. "
            f"If the question references an attached file, use your download_task_file tool with the given task_id to inspect it. "
            f"If the attached file is audio (.mp3), download it first, then pass the saved file path to transcribe_audio. "
            f"If you need deeper information from a URL, use visit_webpage. "
            f"Your response must contain ONLY the final direct answer (e.g., a number, a proper noun, or a short phrase). "
            f"Do NOT write full sentences, do NOT include markdown formatting, and do NOT include the text 'FINAL ANSWER'. "
            f"If your final result is a list of items, format them as a comma-separated string (e.g., item1, item2).\n\n"
            f"{question}"
        )
        
        try:
            raw_answer = self.agent.run(prompt)
            
            if isinstance(raw_answer, list):
                cleaned_answer = ", ".join([str(item).strip() for item in raw_answer])
            else:
                cleaned_answer = str(raw_answer).strip().strip('"\'')

            # Qwen models may prefix a <think>...</think> block; keep only what follows it
            if "</think>" in cleaned_answer:
                cleaned_answer = cleaned_answer.split("</think>")[-1].strip().strip('"\'')

            print(f"Agent final answer: {cleaned_answer}")
            return cleaned_answer
        except Exception as e:
            print(f"Error during agent execution: {e}")
            return f"ERROR: {e}"

def run_and_submit_all():
    # Local run: no Hugging Face login session, so the username comes from HF_USERNAME.
    space_id = os.getenv("SPACE_ID")
    username = HF_USERNAME
    print(f"Submitting as Hugging Face user: {username}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = SmolAgentsGAIAAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else AGENT_CODE_URL

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             return "Fetched questions list is empty.", None
    except Exception as e:
        return f"Error fetching questions: {e}", None

    results_log = []
    answers_payload = []
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")
        
        if not task_id or question_text is None:
            continue
            
        try:
            file_context = f"[attached file: {file_name}] " if file_name else ""
            full_prompt = f"[task_id: {task_id}] {file_context}{question_text}"
            
            submitted_answer = agent(full_prompt)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)

with gr.Blocks() as demo:
    gr.Markdown("# Smolagents GAIA Agent Evaluation Runner")
    gr.Button("Run Evaluation & Submit All Answers").click(fn=run_and_submit_all, outputs=[gr.Textbox(label="Status"), gr.DataFrame(label="Results")])

if __name__ == "__main__":
    demo.launch(debug=True, share=False)