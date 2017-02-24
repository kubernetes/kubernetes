#!/bin/sh
set -ux

# We had to bump inotify limits once in the past, hence why this oddly specific
# script lives here in kubernetes-worker.

sysctl fs.inotify > $DEBUG_SCRIPT_DIR/sysctl-limits
ls -l /proc/*/fd/* | grep inotify > $DEBUG_SCRIPT_DIR/inotify-instances
