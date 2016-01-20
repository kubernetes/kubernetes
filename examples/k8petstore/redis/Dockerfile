#
# Redis Dockerfile
#
# https://github.com/dockerfile/redis
#

# Pull base image.
FROM redis

# Define mountable directories.
VOLUME ["/data"]

# Define working directory.
WORKDIR /data

ADD etc_redis_redis.conf /etc/redis/redis.conf

# Print redis configs and start.
# CMD "redis-server /etc/redis/redis.conf"

# Expose ports.
EXPOSE 6379
