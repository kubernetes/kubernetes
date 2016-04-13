# Monitoring cAdvisor with Prometheus

cAdvisor exposes container statistics as [Prometheus](http://prometheus.io) metrics out of the box. By default, these metrics are served under the `/metrics` HTTP endpoint. This endpoint may be customized by setting the `-prometheus_endpoint` command-line flag.

To monitor cAdvisor with Prometheus, simply configure one or more jobs in Prometheus which scrape the relevant cAdvisor processes at that metrics endpoint. For details, see Prometheus's [Configuration](http://prometheus.io/docs/operating/configuration/) documentation, as well as the [Getting started](http://prometheus.io/docs/introduction/getting_started/) guide.

# Examples
[CenturyLink Labs](https://labs.ctl.io/) did an excellent write up on [Monitoring Docker services with Prometheus +cAdvisor](https://labs.ctl.io/monitoring-docker-services-with-prometheus/)
