Cyberbullying Detection System - Corrected Full Setup

How to run:
1. Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate   (Windows)
   source venv/bin/activate  (macOS/Linux)

2. Install dependencies (GPU setup optional):
   pip install -r requirements.txt

3. (Optional) Train a fallback classical model for quick predictions:
   python train_model.py
   This will create 'model.pkl' used by the fallback predictor.

4. Run the FastAPI app:
   uvicorn main:app --host 127.0.0.1 --port 8000

5. Open http://127.0.0.1:8000 in your browser.
   - Use the checkbox to enable transformer predictions (downloads model first time).
   - Admin dashboard requires basic auth: default admin/password

Notes:
- Transformer training requires internet to download models the first time.
- If behind a firewall, download model files manually and place under models/transformer.
