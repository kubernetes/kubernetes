from flask import Flask
import redis
import os

app = Flask(__name__)

# Configure Redis connection
# The hostname 'redis-service' will be the name of our Kubernetes service for Redis
redis_host = os.environ.get('REDIS_HOST', 'redis-service')
redis_port = int(os.environ.get('REDIS_PORT', 6379))

try:
    r = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=1, socket_timeout=1, health_check_interval=30)
    r.ping() # Check connection
    print("Connected to Redis successfully!")
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis: {e}")
    r = None # Set r to None if connection fails

@app.route('/')
def hello():
    global r
    count = "N/A"
    try:
        if r:
            count = r.incr('visitor_count')
        else: # Try to reconnect if r is None
            print("Attempting to reconnect to Redis...")
            r = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=1, socket_timeout=1)
            r.ping()
            print("Reconnected to Redis successfully!")
            count = r.incr('visitor_count')
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error: {e}")
        r = None # Reset r to None on error
        count = "Error connecting to counter"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        count = "Error"
        
    return f"Hello World! Visitors: {count}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
