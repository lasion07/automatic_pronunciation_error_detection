FROM python:3.9

WORKDIR /backend/
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# CMD [ "python","main.py" ]