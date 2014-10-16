# This Dockerfile will build an image that is configured
# to use Fluentd to collect all Docker container log files
# and then cause them to be ingested using the Google Cloud
# Logging API. This configuration assumes that the host performning
# the collection is a VM that has been created with a logging.write
# scope.

FROM ubuntu:14.04
MAINTAINER Satnam Singh "satnam@google.com"

# Ensure there are enough file descriptors for running Fluentd.
RUN ulimit -n 65536

# Install prerequisites.
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get install -y -q libcurl4-openssl-dev make && \
    apt-get clean

# Install Fluentd.
RUN /usr/bin/curl -L http://toolbelt.treasuredata.com/sh/install-ubuntu-trusty-td-agent2.sh | sh

# Change the default user and group to root.
# Needed to allow access to /var/log/docker/... files.
RUN sed -i -e "s/USER=td-agent/USER=root/" -e "s/GROUP=td-agent/GROUP=root/" /etc/init.d/td-agent
# Supress use of V1 config.
RUN sed -i~ -e 's/ --use-v1-config//' /etc/init.d/td-agent

# Install GCP logging plug-in for Fluentd.
RUN gsutil cp gs://signals-agents/out_google_cloud.rb /etc/td-agent/plugin

# Copy the Fluentd configuration file.
COPY td-agent.conf /etc/td-agent/td-agent.conf

# Run Fluentd in the foreground.
ENTRYPOINT ["/usr/sbin/td-agent"]
