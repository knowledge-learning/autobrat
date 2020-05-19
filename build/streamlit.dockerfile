FROM autogoal/autogoal:latest

RUN pip install streamlit
RUN pip install sklearn
RUN pip install spacy
RUN python -m spacy download es
RUN pip install nltk
RUN pip install fire
RUN pip install pyyaml

RUN ln -s /code/autogoal /usr/local/lib/python3.6/site-packages/autogoal
