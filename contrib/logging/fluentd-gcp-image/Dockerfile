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
ENV OPTS_APT -y --force-yes --no-install-recommends

RUN apt-get -q update && \
    apt-get -y install apt-utils adduser && \
    apt-get clean

ADD google-fluentd_1.0.0-0_amd64.deb /etc/google-fluentd/pkg/google-fluentd_1.0.0-0_amd64.deb
RUN dpkg -i /etc/google-fluentd/pkg/google-fluentd_1.0.0-0_amd64.deb
RUN /opt/google-fluentd/embedded/bin/gem install google-api-client
ADD out_google_cloud.rb /etc/google-fluentd/plugin/out_google_cloud.rb 
ADD agent.conf /etc/google-fluentd/google-fluentd.conf
COPY catch-all-inputs.tar.gz /tmp/catch-all-inputs.tar.gz 
RUN tar -C /etc/google-fluentd -zxf /tmp/catch-all-inputs.tar.gz
RUN sed -i~ -e "s/\(USER\|GROUP\)=google-fluentd/\1=root/;" /etc/init.d/google-fluentd
RUN sed -i~ -e 's/ --use-v1-config//' /etc/init.d/google-fluentd

# Copy the Fluentd configuration file for logging Docker container logs.
COPY google-fluentd.conf /etc/google-fluentd/google-fluentd.conf

# Start Fluentd to pick up our config that watches Docker container logs.
CMD /usr/sbin/google-fluentd > /var/log/google-fluentd.log