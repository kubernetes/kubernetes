package csm

import (
	"sync/atomic"
)

const (
	runningEnum = iota
	pausedEnum
)

var (
	// MetricsChannelSize of metrics to hold in the channel
	MetricsChannelSize = 100
)

type metricChan struct {
	ch     chan metric
	paused int64
}

func newMetricChan(size int) metricChan {
	return metricChan{
		ch: make(chan metric, size),
	}
}

func (ch *metricChan) Pause() {
	atomic.StoreInt64(&ch.paused, pausedEnum)
}

func (ch *metricChan) Continue() {
	atomic.StoreInt64(&ch.paused, runningEnum)
}

func (ch *metricChan) IsPaused() bool {
	v := atomic.LoadInt64(&ch.paused)
	return v == pausedEnum
}

// Push will push metrics to the metric channel if the channel
// is not paused
func (ch *metricChan) Push(m metric) bool {
	if ch.IsPaused() {
		return false
	}

	select {
	case ch.ch <- m:
		return true
	default:
		return false
	}
}
