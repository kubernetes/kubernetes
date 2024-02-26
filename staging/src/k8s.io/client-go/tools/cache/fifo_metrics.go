package cache

import (
	"sync"
)

type FifoGaugeMetric interface {
	Inc()
	Dec()
}

type fifoMetricsInterface interface {
	add()
	pop()
	cleanUp()
}

type fifoNoopMetrics struct{}

func (fifoNoopMetrics) Inc() {}
func (fifoNoopMetrics) Dec() {}

type fifoMetrics struct {
	// current depth of a workqueue
	depth FifoGaugeMetric

	// total number of adds handled by a workqueue
	adds CounterMetric

	mp FifoMetricsProvider

	name string
}

func (f *fifoMetrics) add() {
	if f == nil {
		return
	}

	f.adds.Inc()
	f.depth.Inc()
}

func (f *fifoMetrics) pop() {
	if f == nil {
		return
	}
	f.depth.Dec()
}

func (f *fifoMetrics) cleanUp() {
	if f == nil {
		return
	}
	f.mp.DeleteDepthMetric(f.name)
	f.mp.DeleteAddsMetric(f.name)
}

var fifoGlobalMetricsFactory = fifoMetricsFactory{
	metricsProvider: fifoNoopMetricsProvider{},
}

// FifoMetricsProvider generates various metrics used by the fifo.
type FifoMetricsProvider interface {
	NewAddsMetric(name string) CounterMetric
	DeleteAddsMetric(name string)
	NewDepthMetric(name string) FifoGaugeMetric
	DeleteDepthMetric(name string)
}

type fifoNoopMetricsProvider struct{}

func (fifoNoopMetricsProvider) NewAddsMetric(name string) CounterMetric { return fifoNoopMetrics{} }

func (fifoNoopMetricsProvider) DeleteAddsMetric(name string) {}

func (fifoNoopMetricsProvider) NewDepthMetric(name string) FifoGaugeMetric { return fifoNoopMetrics{} }

func (fifoNoopMetricsProvider) DeleteDepthMetric(name string) {}

type fifoMetricsFactory struct {
	metricsProvider FifoMetricsProvider

	onlyOnce sync.Once
}

func (f *fifoMetricsFactory) setProvider(mp FifoMetricsProvider) {
	f.onlyOnce.Do(func() {
		f.metricsProvider = mp
	})
}

func (f *fifoMetricsFactory) newFifoMetrics(name string) fifoMetricsInterface {
	mp := f.metricsProvider
	if len(name) == 0 {
		return nil
	}
	return &fifoMetrics{
		depth: mp.NewDepthMetric(name),
		adds:  mp.NewAddsMetric(name),
		mp:    mp,
		name:  name,
	}
}

// SetProvider sets the metrics provider for all subsequently created fifo.
// Only the first call has an effect.
func SetProvider(metricsProvider FifoMetricsProvider) {
	fifoGlobalMetricsFactory.setProvider(metricsProvider)
}
