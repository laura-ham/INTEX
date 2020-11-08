FROM python:3.8.1

ARG DOCKER_IMAGE
ENV DOCKER_IMAGE ${DOCKER_IMAGE}

RUN mkdir -p /usr/local/app
RUN mkdir -p /logs


WORKDIR /usr/local/app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD streamlit run streamlit_runner.py --server.port 80
