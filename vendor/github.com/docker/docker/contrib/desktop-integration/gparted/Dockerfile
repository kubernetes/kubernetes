# VERSION:        0.1
# DESCRIPTION:    Create gparted container with its dependencies
# AUTHOR:         Jessica Frazelle <jess@docker.com>
# COMMENTS:
#   This file describes how to build a gparted container with all
#   dependencies installed. It uses native X11 unix socket.
#   Tested on Debian Jessie
# USAGE:
#   # Download gparted Dockerfile
#   wget http://raw.githubusercontent.com/docker/docker/master/contrib/desktop-integration/gparted/Dockerfile
#
#   # Build gparted image
#   docker build -t gparted .
#
#   docker run -v /tmp/.X11-unix:/tmp/.X11-unix \
#     --device=/dev/sda:/dev/sda \
#     -e DISPLAY=unix$DISPLAY gparted
#

# Base docker image
FROM debian:jessie
MAINTAINER Jessica Frazelle <jess@docker.com>

# Install Gparted and its dependencies
RUN apt-get update && apt-get install -y \
    gparted \
    libcanberra-gtk-module \
    --no-install-recommends

# Autorun gparted
CMD ["/usr/sbin/gparted"]
