from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import joblib, os, json, time
from pathlib import Path
from datetime import datetime
from train_transformer import train_transformer_model

# ---------------------------------------------------
# APP INITIALIZATION
# ---------------------------------------------------
app = FastAPI(title="Cyberbullying Detection System")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

security = HTTPBasic()


# ---------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------
try:
    FALLBACK = joblib.load(BASE_DIR / "model.pkl")
    print("✅ Loaded fallback model (model.pkl).")
except Exception as e:
    print(f"⚠️ Fallback model not found: {e}")
    FALLBACK = None

transformer_pipeline = None


# ---------------------------------------------------
# CYBERBULLYING WORD CONTEXT LISTS
# ---------------------------------------------------
SEXUAL_CONTEXT = {
    "sexy", "hot", "naughty", "babe", "send pic", "kiss", "boobs", "nude",
    "handsome", "beautiful", "pretty", "cute girl", "bed", "flirt", "babe",
    "date me", "love u", "strip", "naughty girl", "body", "lust", "porn"
}

VIOLENCE_CONTEXT = {
    "kill", "bomb", "murder", "attack", "shoot", "gun", "destroy", "stab",
    "die", "death", "blood", "terror", "explode", "threat"
}

HATE_CONTEXT = {
    "hate", "idiot", "stupid", "dumb", "moron", "worthless", "fat", "ugly",
    "trash", "loser", "jerk", "freak", "bitch", "bastard", "slut", "pig"
}

RELIGIOUS_CONTEXT = {
    "islam", "christian", "hindu", "jew", "allah", "jesus", "temple", "church",
    "religion", "quran", "bible", "prophet", "god", "priest", "muslim", "satan"
}

MENTAL_HEALTH_CONTEXT = {
    "suicide", "depress", "kill yourself", "go die", "hopeless", "worthless",
    "sad", "hurt", "anxiety", "alone", "mental", "die now"
}

# ---------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------
def datetimeformat(value):
    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "N/A"

templates.env.filters["datetimeformat"] = datetimeformat


# ---------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.environ.get("DEMO_USER", "admin")
    correct_password = os.environ.get("DEMO_PASS", "password")
    if credentials.username == correct_username and credentials.password == correct_password:
        return credentials.username
    raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------
# MAIN DETECTION FUNCTION
# ---------------------------------------------------
def predict_text(text: str, use_transformer: bool = False):
    """Detect cyberbullying content using rules + transformer + fallback model."""
    global transformer_pipeline
    model_used = "Fallback TF-IDF"
    lower_text = text.lower().strip()

    def match_any(phrases):
        return any(phrase in lower_text for phrase in phrases)

    # --- Enhanced Rule-based Filters ---
    if match_any(SEXUAL_CONTEXT):
        return "Cyberbullying", 0.99, "Sexual/Harassment Content"
    if match_any(VIOLENCE_CONTEXT):
        return "Cyberbullying", 0.98, "Violent/Threatening Language"
    if match_any(HATE_CONTEXT):
        return "Cyberbullying", 0.97, "Hate/Insult Language"
    if match_any(RELIGIOUS_CONTEXT):
        return "Cyberbullying", 0.96, "Religious Disrespect"
    if match_any(MENTAL_HEALTH_CONTEXT):
        return "Cyberbullying", 0.95, "Mental Health Abuse"

    # Context-based short phrases
    if any(word in lower_text for word in ["send pic", "naughty", "flirt", "babe", "hot girl", "hi sexy", "cute body"]):
        return "Cyberbullying", 0.97, "Flirty/Harassment Message"
    if any(word in lower_text for word in ["go die", "hate you", "you are ugly", "fool", "idiot", "fat", "loser"]):
        return "Cyberbullying", 0.96, "Insult/Abuse Message"

    # --- Transformer model ---
    if use_transformer:
        model_name = os.environ.get("HF_MODEL", "unitary/toxic-bert")
        try:
            from transformers import pipeline
            if transformer_pipeline is None:
                print(f"🔄 Loading transformer model: {model_name}")
                transformer_pipeline = pipeline("text-classification", model=model_name, device=-1)

            pred = transformer_pipeline(text, truncation=True)[0]
            label_raw = pred.get("label", "").lower()
            score = float(pred.get("score", 0.0))
            toxic_keywords = ["toxic", "abuse", "offensive", "hate", "insult"]

            if any(k in label_raw for k in toxic_keywords) or "label_1" in label_raw:
                label = "Cyberbullying" if score >= 0.45 else "Non-Cyberbullying"
            else:
                label = "Cyberbullying" if score >= 0.75 and label_raw.startswith("negative") else "Non-Cyberbullying"

            return label, score, f"Transformer ({model_name})"
        except Exception as e:
            print(f"⚠️ Transformer Error: {e}")
            return "Error", 0.0, f"Transformer ({model_name})"

    # --- Fallback Model (TF-IDF) ---
    if FALLBACK is not None:
        try:
            pred = FALLBACK.predict([text])[0]
            prob = float(FALLBACK.predict_proba([text])[0][1]) if hasattr(FALLBACK, "predict_proba") else 0.5
            label = "Cyberbullying" if pred == 1 else "Non-Cyberbullying"
            return label, prob, model_used
        except Exception as e:
            print(f"⚠️ Fallback model error: {e}")
            return "Error", 0.0, "Fallback Error"

    return "Error", 0.0, "No model available"


# ---------------------------------------------------
# LOGGING
# ---------------------------------------------------
def log_prediction_data(text, label):
    log_path = BASE_DIR / "predictions_log.json"
    entry = {"text": text, "label": label, "timestamp": time.time()}
    data = []
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
        except Exception:
            data = []
    data.append(entry)
    log_path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...), use_transformer: str = Form("no")):
    use_transformer = use_transformer.lower() in ["yes", "on", "true"]
    label, prob, model_used = predict_text(text, use_transformer)
    log_prediction_data(text, label)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": {"text": text, "label": label, "prob": prob, "model": model_used}},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, username: str = Depends(get_current_username)):
    """Admin dashboard view"""
    log_path = BASE_DIR / "predictions_log.json"
    stats = {"total": 0, "bullying": 0, "non": 0, "recent": []}
    if log_path.exists():
        data = json.loads(log_path.read_text())
        stats["total"] = len(data)
        stats["recent"] = data[-10:][::-1]
        stats["bullying"] = sum(1 for d in data if d["label"] == "Cyberbullying")
        stats["non"] = stats["total"] - stats["bullying"]
    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats, "user": username})


@app.get("/train_transformer", response_class=HTMLResponse)
async def train_transformer_page(request: Request, username: str = Depends(get_current_username)):
    return templates.TemplateResponse("train_notice.html", {"request": request, "user": username})


@app.post("/train_transformer", response_class=HTMLResponse)
async def train_transformer_post(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    username: str = Depends(get_current_username)
):
    """Train transformer using uploaded CSV"""
    content = await file.read()
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    csv_path = models_dir / f"uploaded_{int(time.time())}.csv"
    csv_path.write_bytes(content)

    def run_training():
        try:
            model_path = train_transformer_model(str(csv_path))
            with open(BASE_DIR / "train_status.log", "w", encoding="utf-8") as f:
                f.write(f"✅ Model trained successfully at {model_path}\n")
        except Exception as e:
            with open(BASE_DIR / "train_status.log", "w", encoding="utf-8") as f:
                f.write(f"❌ Training failed: {e}\n")

    background_tasks.add_task(run_training)
    message = "🚀 Training started in background. Please wait and refresh."
    return templates.TemplateResponse("train_notice.html", {"request": request, "message": message, "user": username})
