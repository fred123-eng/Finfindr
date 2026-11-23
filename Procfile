web: gunicorn --bind 0.0.0.0:$PORT --timeout 120 --graceful-timeout 30 --keep-alive 5 --worker-tmp-dir /dev/shm --worker-class gthread --threads 4 finfindr_server:app
