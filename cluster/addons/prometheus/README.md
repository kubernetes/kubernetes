# Prometheus in Kubernetes

This is an experimental [Prometheus](http://prometheus.io/) setup for monitoring
Kubernetes services that expose prometheus-friendly metrics through address
http://service_address:service_port/metrics.

The purpose of the setup is to gather performance-related metrics during load
tests and analyze them to find and fix bottlenecks.

The list of services to be monitored is passed as a command line aguments in
the yaml file. The startup scripts assumes that each service T will have
2 environment variables set ```T_SERVICE_HOST``` and ```T_SERVICE_PORT``` which
can be configured manually in yaml file if you want to monitor something
that is not a regular Kubernetes service. For regular Kubernetes services
the env variables are set up automatically.

By default the metrics are written to a temporary location (that can be changed
in the the volumes section of the yaml file). Prometheus' UI is available 
at port 9090.
