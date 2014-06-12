#!/bin/bash

redis-server --slaveof $SERVICE_HOST $REDISMASTER_SERVICE_PORT
