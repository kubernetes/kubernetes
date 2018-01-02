# Monitoring cAdvisor with Prometheus

cAdvisor exposes container statistics as [Prometheus](https://prometheus.io) metrics out of the box. By default, these metrics are served under the `/metrics` HTTP endpoint. This endpoint may be customized by setting the `-prometheus_endpoint` command-line flag.

To monitor cAdvisor with Prometheus, simply configure one or more jobs in Prometheus which scrape the relevant cAdvisor processes at that metrics endpoint. For details, see Prometheus's [Configuration](https://prometheus.io/docs/operating/configuration/) documentation, as well as the [Getting started](https://prometheus.io/docs/introduction/getting_started/) guide.

# Examples

* [CenturyLink Labs](https://labs.ctl.io/) did an excellent write up on [Monitoring Docker services with Prometheus +cAdvisor](https://www.ctl.io/developers/blog/post/monitoring-docker-services-with-prometheus/), while it is great to get a better overview of cAdvisor integration with Prometheus, the PromDash GUI part is outdated as it has been deprecated for Grafana.

* [vegasbrianc](https://github.com/vegasbrianc) provides a [starter project](https://github.com/vegasbrianc/prometheus) for cAdvisor and Prometheus monitoring, alongide a ready-to-use [Grafana dashboard](https://github.com/vegasbrianc/grafana_dashboard).
