// Copyright 2021 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !go1.17
// +build !go1.17

package prometheus

import (
	"runtime"
	"sync"
	"time"
)

type goCollector struct {
	base baseGoCollector

	// ms... are memstats related.
	msLast          *runtime.MemStats // Previously collected memstats.
	msLastTimestamp time.Time
	msMtx           sync.Mutex // Protects msLast and msLastTimestamp.
	msMetrics       memStatsMetrics
	msRead          func(*runtime.MemStats) // For mocking in tests.
	msMaxWait       time.Duration           // Wait time for fresh memstats.
	msMaxAge        time.Duration           // Maximum allowed age of old memstats.
}

// NewGoCollector is the obsolete version of collectors.NewGoCollector.
// See there for documentation.
//
// Deprecated: Use collectors.NewGoCollector instead.
func NewGoCollector() Collector {
	msMetrics := goRuntimeMemStats()
	msMetrics = append(msMetrics, struct {
		desc    *Desc
		eval    func(*runtime.MemStats) float64
		valType ValueType
	}{
		// This metric is omitted in Go1.17+, see https://github.com/prometheus/client_golang/issues/842#issuecomment-861812034
		desc: NewDesc(
			memstatNamespace("gc_cpu_fraction"),
			"The fraction of this program's available CPU time used by the GC since the program started.",
			nil, nil,
		),
		eval:    func(ms *runtime.MemStats) float64 { return ms.GCCPUFraction },
		valType: GaugeValue,
	})
	return &goCollector{
		base:      newBaseGoCollector(),
		msLast:    &runtime.MemStats{},
		msRead:    runtime.ReadMemStats,
		msMaxWait: time.Second,
		msMaxAge:  5 * time.Minute,
		msMetrics: msMetrics,
	}
}

// Describe returns all descriptions of the collector.
func (c *goCollector) Describe(ch chan<- *Desc) {
	c.base.Describe(ch)
	for _, i := range c.msMetrics {
		ch <- i.desc
	}
}

// Collect returns the current state of all metrics of the collector.
func (c *goCollector) Collect(ch chan<- Metric) {
	var (
		ms   = &runtime.MemStats{}
		done = make(chan struct{})
	)
	// Start reading memstats first as it might take a while.
	go func() {
		c.msRead(ms)
		c.msMtx.Lock()
		c.msLast = ms
		c.msLastTimestamp = time.Now()
		c.msMtx.Unlock()
		close(done)
	}()

	// Collect base non-memory metrics.
	c.base.Collect(ch)

	timer := time.NewTimer(c.msMaxWait)
	select {
	case <-done: // Our own ReadMemStats succeeded in time. Use it.
		timer.Stop() // Important for high collection frequencies to not pile up timers.
		c.msCollect(ch, ms)
		return
	case <-timer.C: // Time out, use last memstats if possible. Continue below.
	}
	c.msMtx.Lock()
	if time.Since(c.msLastTimestamp) < c.msMaxAge {
		// Last memstats are recent enough. Collect from them under the lock.
		c.msCollect(ch, c.msLast)
		c.msMtx.Unlock()
		return
	}
	// If we are here, the last memstats are too old or don't exist. We have
	// to wait until our own ReadMemStats finally completes. For that to
	// happen, we have to release the lock.
	c.msMtx.Unlock()
	<-done
	c.msCollect(ch, ms)
}

func (c *goCollector) msCollect(ch chan<- Metric, ms *runtime.MemStats) {
	for _, i := range c.msMetrics {
		ch <- MustNewConstMetric(i.desc, i.valType, i.eval(ms))
	}
}
