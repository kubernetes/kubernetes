#!/bin/bash

# Start up the server in a way that won't block Travis.

cd testapp
node scripts/web-server.js &
sleep 1
echo Test application started
