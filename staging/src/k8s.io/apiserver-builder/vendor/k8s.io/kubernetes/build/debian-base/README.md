# Kubernetes Debian Base

The Kubernetes debian-base image provides a common base for Kubernetes system images that require
external dependencies (such as `iptables`, `sh`, or anything that is more than a static go-binary).

This image differs from the standard debian image by removing a lot of packages and files that are
generally not necessary in containers. The end result is an image that is just over 40 MB, down from
123 MB.

The image also provides a convenience script `/usr/local/bin/clean-install` that encapsulates the
process of updating apt repositories, installing the packages, and then cleaning up unnecessary
caches & logs.
