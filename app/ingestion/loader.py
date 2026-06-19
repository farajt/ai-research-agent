import os


def load_text_files(raw_dir: str):
    """
    Week 1 keeps this simple: plain .txt files only.
    We'll extend this to PDFs and web pages once the core pipeline works end-to-end.
    """
    documents = []
    if not os.path.isdir(raw_dir):
        return documents

    for filename in os.listdir(raw_dir):
        if filename.endswith(".txt"):
            path = os.path.join(raw_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents.append({"text": f.read(), "source": filename})
    return documents
