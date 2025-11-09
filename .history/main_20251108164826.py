from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
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
app.add_middleware(SessionMiddleware, secret_key="supersecret123")

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
# USER MANAGEMENT (login, register, logout)
# ---------------------------------------------------
USER_FILE = BASE_DIR / "users.json"
if not USER_FILE.exists():
    USER_FILE.write_text(json.dumps({"admin": {"password": "password"}}))


def get_users():
    try:
        return json.loads(USER_FILE.read_text())
    except Exception:
        return {}


def save_users(users):
    USER_FILE.write_text(json.dumps(users, indent=2))


def get_current_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


# ---------------------------------------------------
# BASIC WORD LISTS
# ---------------------------------------------------
SEXUAL_CONTEXT = {"sexy", "hot", "naughty", "babe", "kiss", "boobs", "nude", "bed", "flirt", "lust", "porn"}
VIOLENCE_CONTEXT = {"kill", "bomb", "attack", "shoot", "gun", "die", "murder"}
HATE_CONTEXT = {"harm","idiot", "stupid", "moron", "worthless", "fat", "ugly", "bitch", "slut", "trash", "loser"}
MENTAL_HEALTH_CONTEXT = {"suicide", "depress", "kill yourself", "hopeless", "worthless", "sad", "hurt"}

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def datetimeformat(value):
    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "N/A"


templates.env.filters["datetimeformat"] = datetimeformat


# ---------------------------------------------------
# DETECTION LOGIC
# ---------------------------------------------------
def predict_text(text: str, use_transformer: bool = False):
    global transformer_pipeline
    text_low = text.lower()
    model_used = "Fallback TF-IDF"

    def match_any(words):
        return any(w in text_low for w in words)

    if match_any(SEXUAL_CONTEXT):
        return "Cyberbullying", 0.99, "Sexual Harassment"
    if match_any(VIOLENCE_CONTEXT):
        return "Cyberbullying", 0.97, "Violence/Threat"
    if match_any(HATE_CONTEXT):
        return "Cyberbullying", 0.96, "Insult/Hate"
    if match_any(MENTAL_HEALTH_CONTEXT):
        return "Cyberbullying", 0.94, "Mental Health Abuse"

    # Transformer
    if use_transformer:
        try:
            from transformers import pipeline
            if transformer_pipeline is None:
                transformer_pipeline = pipeline("text-classification", model="unitary/toxic-bert", device=-1)
            pred = transformer_pipeline(text, truncation=True)[0]
            label = "Cyberbullying" if "toxic" in pred["label"].lower() else "Non-Cyberbullying"
            return label, pred["score"], "Transformer"
        except Exception as e:
            print("Transformer error:", e)

    if FALLBACK is None:
        return "Error", 0.0, "No model"

    try:
        pred = FALLBACK.predict([text])[0]
        prob = float(FALLBACK.predict_proba([text])[0][1])
        label = "Cyberbullying" if pred == 1 else "Non-Cyberbullying"
        return label, prob, model_used
    except Exception as e:
        print(f"Fallback error: {e}")
        return "Error", 0.0, "Fallback Error"


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
async def dashboard(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login")

    log_path = BASE_DIR / "predictions_log.json"
    stats = {"total": 0, "bullying": 0, "non": 0, "recent": []}
    if log_path.exists():
        data = json.loads(log_path.read_text())
        stats["total"] = len(data)
        stats["recent"] = data[-10:][::-1]
        stats["bullying"] = sum(1 for d in data if d["label"] == "Cyberbullying")
        stats["non"] = stats["total"] - stats["bullying"]

    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats, "user": user})


# ---------------------------------------------------
# AUTH ROUTES (LOGIN / LOGOUT / REGISTER)
# ---------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "message": ""})


@app.post("/login", response_class=HTMLResponse)
async def login_user(request: Request, username: str = Form(...), password: str = Form(...)):
    users = get_users()
    if username in users and users[username]["password"] == password:
        request.session["user"] = username
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "message": "❌ Invalid credentials"})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "message": ""})


@app.post("/register", response_class=HTMLResponse)
async def register_user(request: Request, username: str = Form(...), password: str = Form(...)):
    users = get_users()
    if username in users:
        return templates.TemplateResponse("register.html", {"request": request, "message": "⚠️ Username already exists"})
    users[username] = {"password": password}
    save_users(users)
    return templates.TemplateResponse("login.html", {"request": request, "message": "✅ Registered successfully! Please log in."})


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)
