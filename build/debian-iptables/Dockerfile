FROM debian

# All apt-get's must be in one run command or the
# cleanup has no effect.
RUN apt-get update && \
    apt-get install -y iptables && \
    ls /var/lib/apt/lists/*debian* | xargs rm
