import uuid
import os
import random
import markdown
import jinja2
import yaml
import shutil

from pathlib import Path
from functools import lru_cache
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException

app = FastAPI()

app.mount("/static", StaticFiles(directory="/data"), name="static")


@lru_cache()
def load_corpus(corpus):
    config = load_config(corpus)
    corpus_path = config["corpus"]["path"]
    pool = []

    with open(Path("/data") / corpus / corpus_path) as fp:
        for line in fp:
            pool.append(line.strip())

    if config["corpus"].get("shuffle", True):
        random.shuffle(pool)

    return pool


def load_config(corpus):
    path = Path("/data") / corpus

    with open(path / "config.yml") as fp:
        return yaml.load(fp.read())


def read_file(corpus, path):
    with open(Path("/data") / corpus / path) as fp:
        return fp.read()


@app.get("/{corpus}", response_class=HTMLResponse)
def index(corpus: str):
    config = load_config(corpus)
    readme = read_file(corpus, config["index"]["readme"])

    with open("/code/templates/index.html") as fp:
        template = jinja2.Template(fp.read())

    return HTMLResponse(template.render(readme=markdown.markdown(readme)))


@app.get("/{corpus}/annotate", response_class=HTMLResponse)
def annotate(corpus: str):
    with open("/code/templates/annotation.html") as fp:
        template = jinja2.Template(fp.read())

    return HTMLResponse(template.render(corpus=corpus))


@app.post("/{corpus}/pack/new")
def new_pack(corpus: str):
    return {"next_pack": ensure_pack(corpus)}


@app.post("/{corpus}/pack/submit")
def submit_pack(corpus: str, pack: str):
    check_pack(corpus, pack) 
    return {"next_pack": ensure_pack(corpus)}


def check_pack(corpus, pack):
    pack_path = Path("/data") / corpus / "packs" / "open" / (pack + ".ann")
    text_path = pack_path.with_suffix(".txt")

    if not pack_path.exists():
        raise HTTPException(400, "The current pack doesn't exists.")

    with open(pack_path) as fp:
        for line in fp:
            break
        else:
            raise HTTPException(400, "The current pack doesn't have any annotation.")

    shutil.move(pack_path, pack_path.parent.parent / "submitted" / pack_path.name)
    shutil.move(text_path, text_path.parent.parent / "submitted" / text_path.name)


def ensure_pack(corpus):
    config = load_config(corpus)
    pack = str(uuid.uuid4())
    pack_path = Path("/data") / corpus / "packs" / "open" / (pack + ".txt")
    ann_path = pack_path.with_suffix(".ann")

    raw = load_corpus(corpus)

    with open(pack_path, "w") as fp:
        for i in range(5):
            fp.write(raw.pop() + "\n")

    with open(ann_path, "w") as fp:
        pass

    os.chmod(str(pack_path), 0o777)
    os.chmod(ann_path, 0o777)

    return pack
