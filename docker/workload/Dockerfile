FROM python:3.11-slim
WORKDIR /app
COPY workload.py /app/workload.py
ENV PYTHONUNBUFFERED=1
CMD ["python","-u","/app/workload.py"]
