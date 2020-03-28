import os
import random
import markdown
import jinja2
import yaml
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException

app = FastAPI()

app.mount("/static", StaticFiles(directory=Path(__file__).absolute().parent.parent / "data" / "autobrat"), name='static')


def load_corpus(corpus):
    config = load_config(corpus)
    corpus_path = config['corpus'].get('path', 'corpus/pool.txt')
    pool = []

    with open(Path(__file__).absolute().parent.parent / "data" / "autobrat" / corpus / corpus_path) as fp:
        for line in fp:
            pool.append(line.strip())

    if config['corpus'].get('shuffle', True):
        random.shuffle(pool)

    return pool


def load_config(corpus):
    path = Path(__file__).absolute().parent.parent / "data" / "autobrat" / corpus
    
    with open(path / "config.yml") as fp:
        return yaml.load(fp.read())


def read_file(corpus, path):
    with open(Path(__file__).absolute().parent.parent / "data" / "autobrat" / corpus / path) as fp:
        return fp.read()


@app.get("/{corpus}", response_class=HTMLResponse)
def index(corpus: str):
    config = load_config(corpus)
    readme = read_file(corpus, config['index']['readme'])

    with open(Path(__file__).parent / "templates" / "index.html") as fp:
        template = jinja2.Template(fp.read())

    return HTMLResponse(template.render(readme=markdown.markdown(readme)))


@app.post("/navigate")
def navigate(current_pack: int, direction:str='next'):
    if direction == 'next':
        next_pack = current_pack + 1
    elif direction == 'previous':
        next_pack = current_pack - 1

    if next_pack <= 0:
        raise HTTPException(400, "Cannot navigate back pass zero.")

    if next_pack > current_pack:
        check_pack(current_pack)

    ensure_pack(next_pack)

    return {
        'next_pack': next_pack
    }


def check_pack(pack): 
    pack_path = Path(__file__).absolute().parent.parent / "data" / "autobrat" / "packs" / (str(pack) + ".ann")

    if not pack_path.exists():
        raise HTTPException(400, "The current pack doesn't exists.")

    with open(pack_path) as fp:
        for line in fp:
            break
        else:
            raise HTTPException(400, "The current pack doesn't have any annotation.")
            

def ensure_pack(pack):
    pack_path = Path(__file__).absolute().parent.parent / "data" / "autobrat" / "packs" / (str(pack) + ".txt")
    ann_path = pack_path.with_suffix(".ann")

    if not pack_path.exists():
        with open(pack_path, "w") as fp:
            for i in range(5):
                fp.write(POOL.pop() + "\n")

        with open(ann_path, "w") as fp:
            pass

        os.chmod(str(pack_path), 0o777)
        os.chmod(ann_path, 0o777)    
