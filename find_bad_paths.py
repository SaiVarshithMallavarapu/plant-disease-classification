import os

def find_bad_paths(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            full = os.path.join(dirpath, name)
            try:
                # force a round‑trip through UTF‑8
                full.encode('utf-8').decode('utf-8')
            except UnicodeDecodeError:
                print("⚠️ Invalid UTF‑8 path:", full)
    print("✅ All paths are valid UTF‑8.")            

if __name__ == "__main__":
    find_bad_paths("data")  # or "data/train" if you want to narrow it down
