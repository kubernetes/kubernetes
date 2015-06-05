## Logging

**Experimental work in progress.**

### Logging with Fluentd and Elastiscsearch

To enable logging of the stdout and stderr output of every Docker container in
a Kubernetes cluster set the shell environment variables
``ENABLE_NODE_LOGGING`` to ``true`` and ``LOGGING_DESTINATION`` to ``elasticsearch``.

e.g. in bash:
```
export ENABLE_NODE_LOGGING=true
export LOGGING_DESTINATION=elasticsearch
```

This will instantiate a [Fluentd](http://www.fluentd.org/) instance on each node which will
collect all the Dcoker container log files. The collected logs will
be targeted at an [Elasticsearch](http://www.elasticsearch.org/) instance assumed to be running on the
local node and accepting log information on port 9200. This can be accomplished
by writing a pod specification and service specification to define an
Elasticsearch service (more information to follow shortly in the contrib directory).

### Logging with Fluentd and Google Compute Platform

To enable logging of Docker contains in a cluster using Google Compute
Platform set the config flags ``ENABLE_NODE_LOGGING`` to ``true`` and
``LOGGING_DESTINATION`` to ``gcp``.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/logging.md?pixel)]()
