FROM ubuntu:latest

RUN apt-get update

RUN apt-get install -y python3-pip

RUN mkdir -p /usr/src/app

WORKDIR /usr/src/app

COPY requirements.txt .

COPY digit_prediction_api.py .

COPY mnist_trained_model.keras .

RUN pip install --upgrade pip --break-system-packages

RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt --break-system-packages

EXPOSE 8000

CMD ["python3", "digit_prediction_api.py","mnist_trained_model.keras"]
