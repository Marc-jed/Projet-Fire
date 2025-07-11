FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


COPY dags /app/dags


WORKDIR /app/dags
CMD ["python", "app.py"]


