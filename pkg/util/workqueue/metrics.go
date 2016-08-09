package workqueue

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

type queueMetrics interface {
	add(item t)
	get(item t)
	done(item t)
}

type defaultQueueMetrics struct {
	depth                prometheus.Gauge
	adds                 prometheus.Counter
	latency              prometheus.Summary
	workDuration         prometheus.Summary
	addTimes             map[t]time.Time
	processingStartTimes map[t]time.Time
}

func newQueueMetrics(name string) queueMetrics {
	var ret *defaultQueueMetrics
	if len(name) == 0 {
		return ret
	}

	ret = &defaultQueueMetrics{
		depth: prometheus.NewGauge(prometheus.GaugeOpts{
			Subsystem: name,
			Name:      "depth",
			Help:      "Current depth of workqueue: " + name,
		}),
		adds: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: name,
			Name:      "adds",
			Help:      "Total number of adds handled by workqueue: " + name,
		}),
		latency: prometheus.NewSummary(prometheus.SummaryOpts{
			Subsystem: name,
			Name:      "queue_latency",
			Help:      "How long an item stays in workqueue" + name + " before being requested.",
		}),
		workDuration: prometheus.NewSummary(prometheus.SummaryOpts{
			Subsystem: name,
			Name:      "work_duration",
			Help:      "How long processing an item from workqueue" + name + " takes.",
		}),
		addTimes:             map[t]time.Time{},
		processingStartTimes: map[t]time.Time{},
	}

	prometheus.Register(ret.depth)
	prometheus.Register(ret.adds)
	prometheus.Register(ret.latency)
	prometheus.Register(ret.workDuration)

	return ret
}

func (m *defaultQueueMetrics) add(item t) {
	if m == nil {
		return
	}

	m.adds.Inc()
	m.depth.Inc()
	if _, exists := m.addTimes[item]; !exists {
		m.addTimes[item] = time.Now()
	}
}

func (m *defaultQueueMetrics) get(item t) {
	if m == nil {
		return
	}

	m.depth.Dec()
	m.processingStartTimes[item] = time.Now()
	if startTime, exists := m.addTimes[item]; exists {
		m.latency.Observe(sinceInMicroseconds(startTime))
		delete(m.addTimes, item)
	}
}

func (m *defaultQueueMetrics) done(item t) {
	if m == nil {
		return
	}

	if startTime, exists := m.processingStartTimes[item]; exists {
		m.workDuration.Observe(sinceInMicroseconds(startTime))
		delete(m.processingStartTimes, item)
	}
}

// Gets the time since the specified start in microseconds.
func sinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}

type retryMetrics interface {
	retry()
}

type defaultRetryMetrics struct {
	retries prometheus.Counter
}

func newRetryMetrics(name string) retryMetrics {
	var ret *defaultRetryMetrics
	if len(name) == 0 {
		return ret
	}

	ret = &defaultRetryMetrics{
		retries: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: name,
			Name:      "retries",
			Help:      "Total number of retries handled by workqueue: " + name,
		}),
	}

	prometheus.Register(ret.retries)

	return ret
}

func (m *defaultRetryMetrics) retry() {
	if m == nil {
		return
	}

	m.retries.Inc()
}
