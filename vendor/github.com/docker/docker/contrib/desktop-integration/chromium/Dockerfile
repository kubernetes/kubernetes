# VERSION:        0.1
# DESCRIPTION:    Create chromium container with its dependencies
# AUTHOR:         Jessica Frazelle <jess@docker.com>
# COMMENTS:
#   This file describes how to build a Chromium container with all
#   dependencies installed. It uses native X11 unix socket.
#   Tested on Debian Jessie
# USAGE:
#   # Download Chromium Dockerfile
#   wget http://raw.githubusercontent.com/docker/docker/master/contrib/desktop-integration/chromium/Dockerfile
#
#   # Build chromium image
#   docker build -t chromium .
#
#   # Run stateful data-on-host chromium. For ephemeral, remove -v /data/chromium:/data
#   docker run -v /data/chromium:/data -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -e DISPLAY=unix$DISPLAY chromium

#   # To run stateful dockerized data containers
#   docker run --volumes-from chromium-data -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -e DISPLAY=unix$DISPLAY chromium

# Base docker image
FROM debian:jessie
MAINTAINER Jessica Frazelle <jess@docker.com>

# Install Chromium
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-l10n \
    libcanberra-gtk-module \
    libexif-dev \
    --no-install-recommends

# Autorun chromium
CMD ["/usr/bin/chromium", "--no-sandbox", "--user-data-dir=/data"]
