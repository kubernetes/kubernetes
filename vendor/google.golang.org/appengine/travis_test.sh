#!/bin/bash
set -e

go version
go test -v google.golang.org/appengine/...
go test -v -race google.golang.org/appengine/...
if [[ $GOAPP == "true" ]]; then
  export PATH="$PATH:/tmp/sdk/go_appengine"
  export APPENGINE_DEV_APPSERVER=/tmp/sdk/go_appengine/dev_appserver.py
  goapp version
  goapp test -v google.golang.org/appengine/...
fi
