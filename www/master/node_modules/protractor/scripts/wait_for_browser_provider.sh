#!/bin/bash


# Wait for Connect to be ready before exiting
while [ ! -f $BROWSER_PROVIDER_READY_FILE ]; do
  sleep .5
done
