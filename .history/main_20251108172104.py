# main.py
from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
app.add_middleware(SessionMiddleware, secret_key="supersecret123")  # keep a strong key in prod

# ---------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------
FALLBACK = None
try:
    FALLBACK = joblib.load(BASE_DIR / "model.pkl")
    print("✅ Loaded fallback model (model.pkl).")
except Exception as e:
    print(f"⚠️ Fallback model not found: {e}")

# transformer (lazy)
transformer_pipeline = None
LOCAL_TRANSFORMER_DIR = BASE_DIR / "models" / "transformer_model"  # where training saves

# ---------------------------------------------------
# USER MANAGEMENT (login, register, logout)
# ---------------------------------------------------
USER_FILE = BASE_DIR / "users.json"
if not USER_FILE.exists():
    USER_FILE.write_text(json.dumps({"admin": {"password": "password"}}, indent=2))


def get_users():
    try:
        data = json.loads(USER_FILE.read_text())
        if isinstance(data, list):  # ❌ Fix old format
            data = {u["username"]: {"password": u["password"]} for u in data}
        elif not isinstance(data, dict):  # If file corrupted
            data = {}
        return data
    except Exception:
        return {"admin": {"password": "password"}}


def save_users(users):
    USER_FILE.write_text(json.dumps(users, indent=2))


def get_current_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

# ---------------------------------------------------
# CONTEXT WORD LISTS (as provided)
# ---------------------------------------------------
DIRTY_WORDS = {
    # --- General Obscene Words ---
    "fuck", "fucking", "fucker", "motherfucker", "bullshit", "shit", "crap", "asshole",
    "bastard", "dickhead", "pussy", "cock", "cum", "cumming", "dildo", "vibrator", "porn",
    "porno", "pornography", "bitch", "slut", "whore", "hoe", "skank", "cunt", "fag",
    "faggot", "jerkoff", "handjob", "blowjob", "deepthroat", "anal", "anus", "ass",
    "booty", "butt", "butthole", "tits", "boobs", "nipple", "nips", "nipslip", "thong",
    "panties", "underwear", "bra", "strip", "stripper", "nude", "naked", "bare", "banging",
    "bang", "bed", "bedroom", "doggystyle", "69", "sex", "sexual", "intercourse",
    "masturbate", "masturbation", "orgasm", "suck", "sucking", "lick", "licking", "moan",
    "moaning", "spank", "spanking", "fetish", "horny", "kinky", "erotic", "seduce",
    "seducing", "seduction", "sensual", "lust", "lusting", "thirsty", "threesome",
    "one-night", "hookup", "crush", "dating", "flirt", "flirting", "naughty", "hottie",
    "sexy", "hot", "babe", "beautiful", "gorgeous", "handsome", "body", "curves", "figure",
    "legs", "thighs", "booty", "ass", "butt", "spank", "fuckboy", "fuckgirl", "slutty",
    "banging", "sexchat", "nudes", "onlyfans", "playboy", "playgirl", "camgirl", "escort",
    "hooker", "prostitute", "brothel", "twerk", "twerking", "dominatrix", "submissive",
    "dirtytalk", "bdsm", "roleplay", "sexslave", "cumshot", "facial", "orgy", "ejaculate",
    "ejaculation", "vagina", "clit", "clitoris", "penis", "balls", "testicles", "scrotum",
    "pussyjuice", "blow", "handjob", "suckoff", "rimjob", "analplay", "pegging", "gspot",
    "cocksucker", "dickpic", "nudes", "cam", "bodycam", "busty", "milf", "gilf", "teen",
    "collegegirl", "shemale", "ladyboy", "crossdresser", "tranny", "pornstar", "nipslip",
    "striptease", "lapdance", "whip", "chains", "bondage", "orgy", "deep", "cream",
    "creamypie", "gag", "gagging", "dominant", "slave", "sexslave", "fingering",
    "penetrate", "penetration", "vibrator", "buttplug", "kamasutra", "lingerie",
    "tempting", "temptation", "naughtiness", "explicit", "vulgar", "adult", "fetish",
    "dirty", "smexy", "lustful", "foreplay", "romance", "xxx", "69ing", "wet", "moist",
    "cumslut", "porns", "boobjob", "spitroast", "gangbang", "roughsex", "nips", "peep",
    "sexually", "touch", "caress", "grope", "fondle", "rub", "rubbing", "massage", "oral",
    "kiss", "makeout", "deepkiss", "fucktoy", "pornhub", "xvideos", "redtube", "xnxx"
}

SEXUAL_CONTEXT = {
    "sex", "sexual", "nude", "naked", "boobs", "breast", "tits", "bra", "panty", "underwear",
    "kiss", "kissing", "hot", "sexy", "horny", "babe", "lust", "adult", "erotic", "fetish",
    "porn", "porno", "pornography", "camgirl", "camguy", "strip", "stripper", "striptease",
    "seduce", "seducing", "seduction", "aroused", "arousal", "orgasm", "masturbate",
    "masturbation", "cock", "dick", "penis", "pussy", "vagina", "clit", "clitoris", "cum",
    "sperm", "ejaculate", "ejaculation", "wet", "naughty", "bedroom", "foreplay", "lingerie",
    "modeling", "handsome", "beautiful", "gorgeous", "pretty", "body", "curves", "butt",
    "booty", "ass", "thighs", "legs", "figure", "bikini", "thong", "dating", "love", "flirt",
    "flirting", "romantic", "kinky", "fetish", "xxx", "69", "blowjob", "handjob", "deepthroat",
    "anal", "doggy", "missionary", "threesome", "naughty", "hookup", "one-night", "crush",
    "makeout", "smexy", "hottie", "nips", "nipple", "nipslip", "lick", "licking", "touch",
    "caress", "moan", "moaning", "lustful", "intimate", "intimacy", "naughtiness",
    "provocative", "tempting", "temptation", "entice", "spank", "dominatrix", "submissive",
    "roleplay", "dirty", "explicit", "vulgar", "sensual", "romance", "playboy", "playgirl",
    "onlyfans", "hooker", "escort", "prostitute", "prostitution", "brothel", "lusting",
    "twerk", "twerking", "bodycam", "cam", "suck", "sucking", "licking", "thirsty"
}
VIOLENCE_CONTEXT = {
    "kill", "murder", "shoot", "stab", "bomb", "attack", "beat", "beating", "fight",
    "fighting", "hurt", "harm", "cut", "knife", "blood", "bleed", "die", "death",
    "dead", "explode", "explosion", "firebomb", "massacre", "slaughter", "terror",
    "terrorist", "terrorism", "suicide", "suicidal", "execute", "execution", "hang",
    "hanging", "poison", "choke", "strangle", "shooting", "blowup", "torture",
    "assault", "crush", "destroy", "hitman", "gun", "rifle", "pistol", "weapon",
    "grenade", "bullet", "mine", "blast", "riot", "war", "warfare", "killself",
    "jump", "bridge", "bloodshed", "knifeattack", "bombblast", "massshooting",
    "explode", "abuse", "burn", "burning", "kidnap", "rape", "rapist", "hangman",
    "threat", "threaten", "kill you", "beat you", "shoot you", "stab you",
    "crush you", "destroy you", "hurt you", "bomb you", "die soon", "go die",
    "choke you", "slap you", "kick you", "shootdown", "wipeout", "obliterate",
    "cutthroat", "hostage", "murderer", "killer", "arson", "gang", "mafia",
    "cartel", "assassin", "suicidebomber", "terrorattack", "executehim",
    "executeher", "beatup", "knockout", "take revenge", "revenge", "revengeful",
    "nazi", "hitler", "gaschamber", "lynch", "lynching", "massmurder",
    "genocide", "holocaust", "shootall", "slay", "slaying",
    "attackers", "terrorcell", "jihad", "isis", "alqaeda", "extremist",
    "terrorgroup", "militia", "warcriminal", "brutal", "brutality",
    "shootout", "hijack", "hostage", "suicideattack", "bloodbath",
    "vengeance", "executioner", "slaughterhouse", "battlefield", "murderous"
}
HATE_CONTEXT = {
    "hate", "hateful", "disgusting", "disgrace", "scum", "trash", "vermin", "filth",
    "worthless", "dirty", "rotten", "evil", "ugly", "stupid", "idiot", "freak", "loser",
    "disgust", "vile", "toxic", "pig", "dog", "animal", "filthy", "garbage", "dumb",
    "moron", "monster", "trashbag", "sick", "degenerate", "disgusted", "nasty", "witch",
    "whore", "slut", "bastard", "cunt", "jerk", "retard", "psychopath", "coward", "clown",
    "racist", "racism", "nigger", "negro", "chink", "gook", "beaner", "cracker",
    "redneck", "wetback", "spic", "coon", "gypsy", "ape", "monkey", "tribal",
    "savage", "uncivilized", "subhuman", "slave", "jungleman", "go back to africa",
    "immigrant trash", "white trash", "brownie", "halfbreed", "yellowman", "arab terrorist",
    "terrorist", "sandnigger", "foreign scum", "blackie", "indiot", "hindu freak",
    "muslimterrorist", "muslimdog", "jewtrash", "antisemite", "kike", "christfreak",
    "jesusfreak", "allahlover", "pagan", "infidel", "devilworshipper", "heathen",
    "islamophobic", "islamophobia", "christophobia", "zionistpig", "holytrash",
    "churchburner", "quranburner", "templetrash", "buddhafreak", "atheistdog",
    "gay", "faggot", "lesbo", "dyke", "tranny", "transfreak", "queer", "homo", "gaytrash",
    "gendertrash", "crossdresser", "she-male", "ladyboy", "sissy", "feminazi", "misogynist",
    "bitch", "whore", "slut", "hoe", "gold-digger", "skank", "femalescum", "manhater",
    "toxicman", "gaylover", "rainbowtrash", "shemale", "queerfreak", "transhate",
    "paki", "chinesevirus", "indiantrash", "filipinotrash", "mexicantrash", "japtrash",
    "arabtrash", "blacktrash", "whitepig", "americandog", "chinklover", "asianpig",
    "europeanscum", "africantrash", "indiandog", "russianpig", "ukrainiantrash",
    "italianmafia", "irishdrunk", "africanmonkey", "browntrash", "latinopig",
    "nazi", "fascist", "commie", "communistpig", "lefttrash", "righttrash", "wokefreak",
    "snowflake", "libtard", "reptilian", "marxist", "extremist", "terrorlover",
    "redtrash", "bluepig", "demofreak", "repubfreak", "dictatorlover",
    "burnthem", "killthem", "destroythem", "shootthem", "wipeout", "exterminate",
    "hangthem", "dieall", "bombthem", "nuke", "eradicate", "masskill", "go back",
    "send them home", "get out", "not welcome", "you people", "your kind", "they all stink",
    "hate group", "whitepower", "blmtrash", "kkk", "neo-nazi", "skinhead", "supremacist",
    "gaschamber", "lynch", "execute", "no lives matter", "hatecrime"
}
MENTAL_HEALTH_CONTEXT = {
    "depressed", "sad", "hopeless", "worthless", "numb", "lonely", "miserable", "exhausted",
    "tiredoflife", "crying", "pain", "hurt", "useless", "empty", "lost", "dark", "lifeless",
    "can’t go on", "pointless", "suffering", "no one cares", "hate myself", "broken", "meaningless",
    "failed", "failure", "donewithlife", "lowenergy", "mentalpain", "despair",
    "anxiety", "panic", "scared", "fear", "overthinking", "nervous", "shaking", "paranoid",
    "can’t breathe", "panicattack", "heart racing", "stress", "worry", "suffocating",
    "fearful", "terrified", "no control", "helpless", "triggered", "anxious", "nervousbreakdown",
    "suicide", "kill myself", "end my life", "want to die", "cut myself", "selfharm",
    "slit my wrist", "i’m done", "die", "jump off", "no reason to live", "goodbye forever",
    "it’s over", "drinking bleach", "take pills", "not waking up", "hang myself",
    "bleed out", "can’t handle", "done living", "kill me", "suffocate", "overdose",
    "i’m tired of living", "i want to disappear", "self destruction", "goodbye world",
    "bullied", "they mock me", "i’m nothing", "they hate me", "everyone laughs",
    "embarrassed", "humiliated", "excluded", "unwanted", "rejected", "ignored",
    "no friends", "outcast", "they call me names", "i hate school", "hate everyone",
    "they make fun of me", "i’m ugly", "i’m stupid", "no one loves me", "can’t fit in",
    "trauma", "abuse", "molested", "raped", "beaten", "hurt inside", "broken soul",
    "can’t sleep", "nightmares", "flashbacks", "traumatized", "crisis", "abandoned",
    "controlled", "gaslight", "manipulated", "trapped", "fear of people", "haunted",
    "can’t trust anyone", "no escape", "mentalbreak", "suffering inside",
    "adhd", "bipolar", "schizophrenia", "psychosis", "ocd", "addiction", "autistic",
    "borderline", "personalitydisorder", "eatingdisorder", "anorexia", "bulimia",
    "alcoholic", "drugaddict", "mentallyill", "crazy", "insane", "psycho", "mad", "delusional",
    "you should die", "nobody cares", "kill yourself", "drink bleach", "you’re worthless",
    "no one loves you", "cut deeper", "just die", "go kill yourself", "hang yourself",
    "nobody would miss you", "do it already", "you’re a failure", "you’re disgusting",
    "why are you alive", "you’re a loser", "you deserve to die", "nobody likes you",
    "you should disappear", "your life is a joke", "kill yourself now",
    "need help", "talk to someone", "therapy", "mental health", "counseling",
    "helpline", "i want to talk", "need support", "i feel weak", "can’t do this",
    "i want to see a therapist", "help me", "i’m scared", "save me", "please listen",
    "nobody understands", "crisis hotline", "need someone to talk to"
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
    """
    Rule-based context → Transformer (optional) → Fallback TF-IDF
    """
    global transformer_pipeline
    text_low = text.lower()
    model_used = "Fallback TF-IDF"

    def match_any(words):
        # substring match is intentional to catch plural / variants
        return any(w in text_low for w in words)

    # 1) Rule-based immediate flags
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

    # 2) Transformer (if asked)
    if use_transformer:
        try:
            from transformers import pipeline
            model_name = str(LOCAL_TRANSFORMER_DIR) if LOCAL_TRANSFORMER_DIR.exists() else "unitary/toxic-bert"
            if transformer_pipeline is None:
                print(f"🔄 Loading transformer model from: {model_name}")
                transformer_pipeline = pipeline("text-classification", model=model_name, device=-1)
            pred = transformer_pipeline(text, truncation=True)[0]
            label = "Cyberbullying" if "toxic" in pred["label"].lower() or pred["label"].lower() in {"label_1", "negative"} else "Non-Cyberbullying"
            return label, float(pred.get("score", 0.0)), f"Transformer ({model_name})"
        except Exception as e:
            print("Transformer error:", e)

    # 3) Fallback TF-IDF
    if FALLBACK is None:
        return "Error", 0.0, "No model"
    try:
        pred = FALLBACK.predict([text])[0]
        prob = float(FALLBACK.predict_proba([text])[0][1]) if hasattr(FALLBACK, "predict_proba") else 0.0
        label = "Cyberbullying" if pred == 1 else "Non-Cyberbullying"
        return label, prob, model_used
    except Exception as e:
        print(f"Fallback error: {e}")
        return "Error", 0.0, "Fallback Error"

# ---------------------------------------------------
# ROUTES: PAGES
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

# ---------------------------------------------------
# TRAINING ROUTES (kept)
# ---------------------------------------------------
@app.get("/train_transformer", response_class=HTMLResponse)
async def train_transformer_page(request: Request):
    # protect training behind login (optional)
    if not request.session.get("user"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("train_notice.html", {"request": request, "user": request.session.get("user")})

@app.post("/train_transformer", response_class=HTMLResponse)
async def train_transformer_post(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    if not request.session.get("user"):
        return RedirectResponse("/login")
    content = await file.read()
    csv_path = BASE_DIR / f"uploaded_{int(time.time())}.csv"
    csv_path.write_bytes(content)

    def run_training():
        status_log = BASE_DIR / "train_status.log"
        try:
            status_log.write_text("Training started...\n")
            model_path = train_transformer_model(str(csv_path))
            status_log.write_text(f"✅ Model trained successfully at: {model_path}\n")
        except Exception as e:
            status_log.write_text(f"❌ Training failed: {e}\n")

    background_tasks.add_task(run_training)
    message = "🚀 Training started. Check this page in a bit to see the status."
    return templates.TemplateResponse("train_notice.html", {"request": request, "message": message, "user": request.session.get("user")})

@app.get("/train_status", response_class=HTMLResponse)
async def train_status(request: Request):
    status_log = BASE_DIR / "train_status.log"
    msg = status_log.read_text(encoding="utf-8") if status_log.exists() else "Training not started or running..."
    return templates.TemplateResponse("train_notice.html", {"request": request, "message": msg, "user": request.session.get("user")})
