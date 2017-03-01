# Collecting Docker Log Files with Fluentd and sending to GCP.
This directory contains the source files needed to make a Docker image
that collects Docker container log files using [Fluentd](http://www.fluentd.org/)
and sends them to GCP.
This image is designed to be used as part of the [Kubernetes](https://github.com/kubernetes/kubernetes)
cluster bring up process. The image resides at DockerHub under the name
[kubernetes/fluentd-gcp](https://registry.hub.docker.com/u/kubernetes/fluentd-gcp/).

# Usage

The image is built with its own set of plugins which you can later use
in the configuration. The set of plugin is enumerated in a Gemfile in the
image's directory. You can find details about fluentd configuration on the
[official site](http://docs.fluentd.org/articles/config-file).

In order to configure fluentd image, you should mount a directory with `.conf`
files to `/etc/fluent/config.d` or add files to that directory by building
a new image on top. All `.conf` files in the `/etc/fluent/config.d` directory
will be included to the final fluentd configuration.

Command line arguments to the fluentd executable are passed
via environment variable `FLUENTD_ARGS`.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/fluentd-gcp/fluentd-gcp-image/README.md?pixel)]()
