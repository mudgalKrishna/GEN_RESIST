FROM ubuntu:22.04

RUN apt update && apt install -y python3 python3-pip python3-dev build-essential git
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
