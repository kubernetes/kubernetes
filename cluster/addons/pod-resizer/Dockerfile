# Copyright 2016 The Kubernetes Authors. All rights reserved.
FROM alpine:edge
MAINTAINER Quintin Lee "qlee@google.com"

# Disable prompts from apt.
#ENV DEBIAN_FRONTEND noninteractive
#ENV DO_NOT_INSTALL_CATCH_ALL_CONFIG true

ADD build/pod_nanny pod_nanny

CMD ./pod_nanny
