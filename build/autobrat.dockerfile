FROM autogoal/autogoal:latest

RUN pip install fastapi
RUN pip install uvicorn
RUN pip install aiofiles
RUN spacy download es
RUN spacy download en

RUN ln -s /code/autogoal /usr/local/lib/python3.6/site-packages/autogoal
