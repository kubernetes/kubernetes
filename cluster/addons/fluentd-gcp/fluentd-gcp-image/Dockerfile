# This Dockerfile will build an image that is configured
# to use Fluentd to collect all Docker container log files
# and then cause them to be ingested using the Google Cloud
# Logging API. This configuration assumes that the host performning
# the collection is a VM that has been created with a logging.write
# scope and that the Logging API has been enabled for the project
# in the Google Developer Console. 

FROM ubuntu:14.04
MAINTAINER Satnam Singh "satnam@google.com"

# Disable prompts from apt.
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -q update && \
    apt-get install -y curl && \
    apt-get clean && \
    curl -s https://storage.googleapis.com/signals-agents/logging/google-fluentd-install.sh | sudo bash

# Copy the Fluentd configuration file for logging Docker container logs.
COPY google-fluentd.conf /etc/google-fluentd/google-fluentd.conf

# Install the record reformer plugin.
RUN /usr/sbin/google-fluentd-gem install fluent-plugin-record-reformer

# Start Fluentd to pick up our config that watches Docker container logs.
CMD /usr/sbin/google-fluentd "$FLUENTD_ARGS" > /var/log/google-fluentd.log
