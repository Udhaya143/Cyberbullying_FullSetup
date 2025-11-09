from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import joblib, os, json, time, asyncio
from pathlib import Path
from datetime import datetime
from train_transformer import train_transformer_model
import sys

# ---------------------------------------------------
# APP INITIALIZATION
# ---------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')
app = FastAPI(title="Cyberbullying Detection System")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.add_middleware(SessionMiddleware, secret_key="supersecret123")
TRAINING_LOG_FILE.write_text("🚀 Training started...\n", encoding="utf-8")

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
LOCAL_TRANSFORMER_DIR = BASE_DIR / "models" / "transformer_model"

# ---------------------------------------------------
# USER MANAGEMENT
# ---------------------------------------------------
USER_FILE = BASE_DIR / "users.json"
if not USER_FILE.exists():
    USER_FILE.write_text(json.dumps({"admin": {"password": "password"}}, indent=2))


def get_users():
    try:
        data = json.loads(USER_FILE.read_text())
        if isinstance(data, list):
            data = {u["username"]: {"password": u["password"]} for u in data}
        elif not isinstance(data, dict):
            data = {}
        return data
    except Exception:
        return {"admin": {"password": "password"}}


def save_users(users):
    USER_FILE.write_text(json.dumps(users, indent=2))


# ---------------------------------------------------
# CONTEXT WORD LISTS
# ---------------------------------------------------
# 🩶 DIRTY / SEXUAL / VIOLENCE / HATE / MENTAL HEALTH
DIRTY_WORDS = {
    "fuck", "fucking", "fucker", "motherfucker", "bullshit", "shit", "crap", "asshole",
    "bastard", "dickhead", "pussy", "cock", "cum", "cumming", "dildo", "vibrator", "porn",
    "porno", "pornography", "bitch", "slut", "whore", "hoe", "skank", "cunt", "fag",
    "faggot", "jerkoff", "handjob", "blowjob", "deepthroat", "anal", "anus", "ass",
    "booty", "butt", "butthole", "tits", "boobs", "nipple", "nips", "thong", "panties",
    "underwear", "bra", "strip", "stripper", "nude", "naked", "banging", "bang", "bed",
    "doggystyle", "69", "sex", "sexual", "intercourse", "masturbate", "orgasm", "suck",
    "lick", "moan", "spank", "fetish", "horny", "kinky", "erotic", "seduce", "sensual",
    "lust", "threesome", "hookup", "flirt", "naughty", "hottie", "sexy", "babe", "beautiful",
    "handsome", "curves", "legs", "thighs", "booty", "fuckboy", "fuckgirl", "slutty",
    "nudes", "onlyfans", "playboy", "playgirl", "camgirl", "escort", "hooker", "prostitute",
    "twerk", "dominatrix", "submissive", "bdsm", "roleplay", "sexslave", "cumshot",
    "vagina", "penis", "balls", "pussyjuice", "rimjob", "pegging", "gspot", "cocksucker",
    "nipslip", "lapdance", "whip", "chains", "bondage", "creamypie", "dominant", "slave",
    "penetration", "buttplug", "kamasutra", "lingerie", "tempting", "vulgar", "adult",
    "xxx", "wet", "moist", "pornhub", "xvideos", "xnxx"
}

SEXUAL_CONTEXT = {
    "sex", "sexual", "nude", "naked", "boobs", "breast", "tits", "bra", "panty", "underwear",
    "kiss", "kissing", "hot", "sexy", "horny", "babe", "lust", "adult", "erotic", "fetish",
    "porn", "camgirl", "strip", "stripper", "seduce", "aroused", "orgasm", "masturbate",
    "cock", "pussy", "vagina", "cum", "wet", "naughty", "foreplay", "lingerie", "body",
    "curves", "butt", "booty", "ass", "legs", "bikini", "flirt", "romantic", "kinky",
    "blowjob", "handjob", "anal", "threesome", "hookup", "makeout", "hottie", "nipslip",
    "lick", "moan", "spank", "dirty", "explicit", "vulgar", "sensual", "romance", "playboy",
    "onlyfans", "escort", "prostitute", "brothel", "twerk", "twerking", "suck", "licking"
}

VIOLENCE_CONTEXT = {
    "kill", "murder", "shoot", "stab", "bomb", "attack", "beat", "fight", "hurt", "cut", "knife",
    "blood", "die", "death", "explode", "massacre", "slaughter", "terror", "terrorist", "suicide",
    "execute", "hang", "choke", "strangle", "torture", "assault", "gun", "weapon", "grenade",
    "blast", "war", "killself", "rape", "threat", "kill you", "beat you", "shoot you", "stab you",
    "die soon", "go die", "choke you", "slap you", "kick you", "murderer", "killer", "gang",
    "terrorattack", "lynch", "genocide", "brutality", "bloodbath", "executioner", "battlefield"
}

HATE_CONTEXT = {
    "hate", "disgusting", "scum", "trash", "vermin", "filth", "worthless", "ugly", "stupid",
    "idiot", "freak", "loser", "vile", "toxic", "pig", "dog", "animal", "garbage", "moron",
    "monster", "sick", "nasty", "witch", "whore", "slut", "bastard", "jerk", "coward", "clown",
    "racist", "nigger", "chink", "gook", "cracker", "spic", "coon", "slave", "immigrant trash",
    "terrorist", "sandnigger", "blacktrash", "whitepig", "americandog", "asianpig", "feminazi",
    "gay", "faggot", "lesbo", "dyke", "queer", "transhate", "paki", "nazi", "kkk", "skinhead",
    "supremacist", "lynch", "no lives matter", "hatecrime"
}

MENTAL_HEALTH_CONTEXT = {
    "depressed", "sad", "hopeless", "worthless", "numb", "lonely", "crying", "pain", "hurt",
    "useless", "empty", "lost", "dark", "suffering", "hate myself", "broken", "failed", "failure",
    "anxiety", "panic", "fear", "worry", "suicide", "kill myself", "end my life", "want to die",
    "selfharm", "slit my wrist", "die", "jump off", "no reason to live", "goodbye forever",
    "take pills", "hang myself", "bleed out", "kill me", "overdose", "i want to disappear",
    "bullied", "humiliated", "excluded", "no friends", "hate everyone", "i’m ugly", "trauma",
    "abuse", "molested", "raped", "broken soul", "can’t sleep", "fear of people", "helpless",
    "adhd", "bipolar", "schizophrenia", "ocd", "addiction", "anorexia", "alcoholic",
    "mentallyill", "crazy", "psycho", "kill yourself", "you’re worthless", "no one loves you",
    "cut deeper", "just die", "hang yourself", "you’re a loser", "you deserve to die",
    "help me", "therapy", "mental health", "helpline", "please listen", "need someone to talk"
}

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def datetimeformat(value):
    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "N/A"

templates.env.filters["datetimeformat"] = datetimeformat


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
# DETECTION LOGIC
# ---------------------------------------------------
def predict_text(text: str, use_transformer: bool = False):
    global transformer_pipeline
    text_low = text.lower()
    model_used = "Fallback TF-IDF"

    def match_any(words):
        return any(w in text_low for w in words)

    if match_any(DIRTY_WORDS):
        return "Cyberbullying", 0.99, "Sexual/Obscene Language"
    if match_any(SEXUAL_CONTEXT):
        return "Cyberbullying", 0.99, "Sexual Harassment"
    if match_any(VIOLENCE_CONTEXT):
        return "Cyberbullying", 0.97, "Violence/Threat"
    if match_any(HATE_CONTEXT):
        return "Cyberbullying", 0.96, "Insult/Hate"
    if match_any(MENTAL_HEALTH_CONTEXT):
        return "Cyberbullying", 0.94, "Mental Health Abuse"

    if use_transformer:
        try:
            from transformers import pipeline
            model_name = str(LOCAL_TRANSFORMER_DIR) if LOCAL_TRANSFORMER_DIR.exists() else "unitary/toxic-bert"
            if transformer_pipeline is None:
                transformer_pipeline = pipeline("text-classification", model=model_name, device=-1)
            pred = transformer_pipeline(text, truncation=True)[0]
            label = "Cyberbullying" if "toxic" in pred["label"].lower() else "Non-Cyberbullying"
            return label, float(pred.get("score", 0.0)), f"Transformer ({model_name})"
        except Exception as e:
            print("Transformer error:", e)

    if FALLBACK is None:
        return "Error", 0.0, "No model"
    try:
        pred = FALLBACK.predict([text])[0]
        prob = float(FALLBACK.predict_proba([text])[0][1]) if hasattr(FALLBACK, "predict_proba") else 0.0
        label = "Cyberbullying" if pred == 1 else "Non-Cyberbullying"
        return label, prob, model_used
    except Exception as e:
        return "Error", 0.0, "Fallback Error"


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
    return templates.TemplateResponse("index.html", {"request": request, "result": {"text": text, "label": label, "prob": prob, "model": model_used}})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login")
    log_path = BASE_DIR / "predictions_log.json"
    stats = {"total": 0, "bullying": 0, "non": 0, "recent": []}
    if log_path.exists():
        data = json.loads(log_path.read_text())
        stats["total"] = len(data)
        stats["recent"] = data[-10:][::-1]
        stats["bullying"] = sum(1 for d in data if d["label"] == "Cyberbullying")
        stats["non"] = stats["total"] - stats["bullying"]
    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats, "user": request.session.get("user")})


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

@app.get("/train_transformer", response_class=HTMLResponse)
async def train_transformer_page(request: Request):
    """Page to upload dataset and start Transformer training"""
    if not request.session.get("user"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("train_transformer.html", {"request": request, "user": request.session.get("user")})


@app.post("/upload_train")
async def upload_and_train(request: Request, file: UploadFile = File(...)):
    """Handle dataset upload and trigger background training"""
    if not request.session.get("user"):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    TRAINING_LOG_FILE.write_text("🚀 Training started...\n")

    # Save uploaded CSV
    filename = f"uploaded_{int(time.time())}.csv"
    csv_path = BASE_DIR / "uploads" / filename
    csv_path.write_bytes(await file.read())

    def background_training():
        try:
            with open(TRAINING_LOG_FILE, "a", encoding="utf-8") as logf:
                logf.write(f"📂 Loading dataset: {csv_path}\n")
            train_transformer_model(str(csv_path))
            with open(TRAINING_LOG_FILE, "a", encoding="utf-8") as logf:
                logf.write("\n✅ Training completed successfully!\n")
        except Exception as e:
            with open(TRAINING_LOG_FILE, "a", encoding="utf-8") as logf:
                logf.write(f"\n❌ Training failed: {e}\n")

    thread = threading.Thread(target=background_training)
    thread.start()

    return JSONResponse({"message": f"Training started using {file.filename}"})


@app.get("/train_logs")
async def train_logs():
    """Stream training logs live to the browser"""
    def stream():
        last_pos = 0
        while True:
            try:
                if TRAINING_LOG_FILE.exists():
                    with open(TRAINING_LOG_FILE, "r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        lines = f.readlines()
                        last_pos = f.tell()
                        for line in lines:
                            yield f"data: {line.strip()}\n\n"
                time.sleep(1)
            except Exception as e:
                yield f"data: [Stream Error] {e}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")