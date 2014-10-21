## Logging

**Experimental work in progress.**

### Logging with Fluentd and Elastiscsearch

To enable logging of the stdout and stderr output of every Docker container in
a Kubernetes cluster set the shell environment
variable ``FLUENTD_ELASTICSEARCH`` to ``true`` e.g. in bash:
```
export FLUENTD_ELASTICSEARCH=true
```
This will instantiate a [Fluentd](http://www.fluentd.org/) instance on each node which will
collect all the Dcoker container log files. The collected logs will
be targetted at an [Elasticsearch](http://www.elasticsearch.org/) instance assumed to be running on the
local node and accepting log information on port 9200. This can be accomplished
by writing a pod specification and service sepecificaiton to define an
Elasticsearch service (more informaiton to follow shortly in the contrib directory).

### Logging with Fluentd and Google Compute Platform

To enable logging of Docker contains in a cluster using Google Compute
Platfrom set the shell environment variable ``FLUENTD_GCP`` to ``true``.