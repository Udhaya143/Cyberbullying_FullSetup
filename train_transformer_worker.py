import sys, traceback
from train_transformer import train_transformer_model

if __name__ == '__main__':
    if len(sys.argv)<2:
        print('ERROR: expected csv path argument', file=sys.stderr)
        sys.exit(2)
    csv = sys.argv[1]
    try:
        print('=== TRAIN WORKER STARTED ===', flush=True)
        path = train_transformer_model(csv)
        print('=== TRAIN WORKER DONE ===', flush=True)
        print('MODEL_SAVED:' + str(path), flush=True)
    except Exception as e:
        print('=== TRAIN WORKER ERROR ===', flush=True)
        traceback.print_exc()
        sys.exit(1)
