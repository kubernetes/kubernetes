package events

import "github.com/docker/go-metrics"

var (
	eventsCounter    metrics.Counter
	eventSubscribers metrics.Gauge
)

func init() {
	ns := metrics.NewNamespace("engine", "daemon", nil)
	eventsCounter = ns.NewCounter("events", "The number of events logged")
	eventSubscribers = ns.NewGauge("events_subscribers", "The number of current subscribers to events", metrics.Total)
	metrics.Register(ns)
}
