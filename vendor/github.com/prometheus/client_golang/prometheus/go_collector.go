// Copyright 2018 The Prometheus Authors
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

package prometheus

import (
	"runtime"
	"runtime/debug"
	"sync"
	"time"
)

type goCollector struct {
	goroutinesDesc *Desc
	threadsDesc    *Desc
	gcDesc         *Desc
	goInfoDesc     *Desc

	// ms... are memstats related.
	msLast          *runtime.MemStats // Previously collected memstats.
	msLastTimestamp time.Time
	msMtx           sync.Mutex // Protects msLast and msLastTimestamp.
	msMetrics       memStatsMetrics
	msRead          func(*runtime.MemStats) // For mocking in tests.
	msMaxWait       time.Duration           // Wait time for fresh memstats.
	msMaxAge        time.Duration           // Maximum allowed age of old memstats.
}

// NewGoCollector returns a collector that exports metrics about the current Go
// process. This includes memory stats. To collect those, runtime.ReadMemStats
// is called. This requires to “stop the world”, which usually only happens for
// garbage collection (GC). Take the following implications into account when
// deciding whether to use the Go collector:
//
// 1. The performance impact of stopping the world is the more relevant the more
// frequently metrics are collected. However, with Go1.9 or later the
// stop-the-world time per metrics collection is very short (~25µs) so that the
// performance impact will only matter in rare cases. However, with older Go
// versions, the stop-the-world duration depends on the heap size and can be
// quite significant (~1.7 ms/GiB as per
// https://go-review.googlesource.com/c/go/+/34937).
//
// 2. During an ongoing GC, nothing else can stop the world. Therefore, if the
// metrics collection happens to coincide with GC, it will only complete after
// GC has finished. Usually, GC is fast enough to not cause problems. However,
// with a very large heap, GC might take multiple seconds, which is enough to
// cause scrape timeouts in common setups. To avoid this problem, the Go
// collector will use the memstats from a previous collection if
// runtime.ReadMemStats takes more than 1s. However, if there are no previously
// collected memstats, or their collection is more than 5m ago, the collection
// will block until runtime.ReadMemStats succeeds. (The problem might be solved
// in Go1.13, see https://github.com/golang/go/issues/19812 for the related Go
// issue.)
func NewGoCollector() Collector {
	return &goCollector{
		goroutinesDesc: NewDesc(
			"go_goroutines",
			"Number of goroutines that currently exist.",
			nil, nil),
		threadsDesc: NewDesc(
			"go_threads",
			"Number of OS threads created.",
			nil, nil),
		gcDesc: NewDesc(
			"go_gc_duration_seconds",
			"A summary of the pause duration of garbage collection cycles.",
			nil, nil),
		goInfoDesc: NewDesc(
			"go_info",
			"Information about the Go environment.",
			nil, Labels{"version": runtime.Version()}),
		msLast:    &runtime.MemStats{},
		msRead:    runtime.ReadMemStats,
		msMaxWait: time.Second,
		msMaxAge:  5 * time.Minute,
		msMetrics: memStatsMetrics{
			{
				desc: NewDesc(
					memstatNamespace("alloc_bytes"),
					"Number of bytes allocated and still in use.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.Alloc) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("alloc_bytes_total"),
					"Total number of bytes allocated, even if freed.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.TotalAlloc) },
				valType: CounterValue,
			}, {
				desc: NewDesc(
					memstatNamespace("sys_bytes"),
					"Number of bytes obtained from system.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.Sys) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("lookups_total"),
					"Total number of pointer lookups.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.Lookups) },
				valType: CounterValue,
			}, {
				desc: NewDesc(
					memstatNamespace("mallocs_total"),
					"Total number of mallocs.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.Mallocs) },
				valType: CounterValue,
			}, {
				desc: NewDesc(
					memstatNamespace("frees_total"),
					"Total number of frees.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.Frees) },
				valType: CounterValue,
			}, {
				desc: NewDesc(
					memstatNamespace("heap_alloc_bytes"),
					"Number of heap bytes allocated and still in use.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.HeapAlloc) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("heap_sys_bytes"),
					"Number of heap bytes obtained from system.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.HeapSys) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("heap_idle_bytes"),
					"Number of heap bytes waiting to be used.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.HeapIdle) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("heap_inuse_bytes"),
					"Number of heap bytes that are in use.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.HeapInuse) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("heap_released_bytes"),
					"Number of heap bytes released to OS.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.HeapReleased) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("heap_objects"),
					"Number of allocated objects.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.HeapObjects) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("stack_inuse_bytes"),
					"Number of bytes in use by the stack allocator.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.StackInuse) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("stack_sys_bytes"),
					"Number of bytes obtained from system for stack allocator.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.StackSys) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("mspan_inuse_bytes"),
					"Number of bytes in use by mspan structures.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.MSpanInuse) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("mspan_sys_bytes"),
					"Number of bytes used for mspan structures obtained from system.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.MSpanSys) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("mcache_inuse_bytes"),
					"Number of bytes in use by mcache structures.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.MCacheInuse) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("mcache_sys_bytes"),
					"Number of bytes used for mcache structures obtained from system.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.MCacheSys) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("buck_hash_sys_bytes"),
					"Number of bytes used by the profiling bucket hash table.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.BuckHashSys) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("gc_sys_bytes"),
					"Number of bytes used for garbage collection system metadata.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.GCSys) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("other_sys_bytes"),
					"Number of bytes used for other system allocations.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.OtherSys) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("next_gc_bytes"),
					"Number of heap bytes when next garbage collection will take place.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.NextGC) },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("last_gc_time_seconds"),
					"Number of seconds since 1970 of last garbage collection.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return float64(ms.LastGC) / 1e9 },
				valType: GaugeValue,
			}, {
				desc: NewDesc(
					memstatNamespace("gc_cpu_fraction"),
					"The fraction of this program's available CPU time used by the GC since the program started.",
					nil, nil,
				),
				eval:    func(ms *runtime.MemStats) float64 { return ms.GCCPUFraction },
				valType: GaugeValue,
			},
		},
	}
}

func memstatNamespace(s string) string {
	return "go_memstats_" + s
}

// Describe returns all descriptions of the collector.
func (c *goCollector) Describe(ch chan<- *Desc) {
	ch <- c.goroutinesDesc
	ch <- c.threadsDesc
	ch <- c.gcDesc
	ch <- c.goInfoDesc
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

	ch <- MustNewConstMetric(c.goroutinesDesc, GaugeValue, float64(runtime.NumGoroutine()))
	n, _ := runtime.ThreadCreateProfile(nil)
	ch <- MustNewConstMetric(c.threadsDesc, GaugeValue, float64(n))

	var stats debug.GCStats
	stats.PauseQuantiles = make([]time.Duration, 5)
	debug.ReadGCStats(&stats)

	quantiles := make(map[float64]float64)
	for idx, pq := range stats.PauseQuantiles[1:] {
		quantiles[float64(idx+1)/float64(len(stats.PauseQuantiles)-1)] = pq.Seconds()
	}
	quantiles[0.0] = stats.PauseQuantiles[0].Seconds()
	ch <- MustNewConstSummary(c.gcDesc, uint64(stats.NumGC), stats.PauseTotal.Seconds(), quantiles)

	ch <- MustNewConstMetric(c.goInfoDesc, GaugeValue, 1)

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

// memStatsMetrics provide description, value, and value type for memstat metrics.
type memStatsMetrics []struct {
	desc    *Desc
	eval    func(*runtime.MemStats) float64
	valType ValueType
}

// NewBuildInfoCollector returns a collector collecting a single metric
// "go_build_info" with the constant value 1 and three labels "path", "version",
// and "checksum". Their label values contain the main module path, version, and
// checksum, respectively. The labels will only have meaningful values if the
// binary is built with Go module support and from source code retrieved from
// the source repository (rather than the local file system). This is usually
// accomplished by building from outside of GOPATH, specifying the full address
// of the main package, e.g. "GO111MODULE=on go run
// github.com/prometheus/client_golang/examples/random". If built without Go
// module support, all label values will be "unknown". If built with Go module
// support but using the source code from the local file system, the "path" will
// be set appropriately, but "checksum" will be empty and "version" will be
// "(devel)".
//
// This collector uses only the build information for the main module. See
// https://github.com/povilasv/prommod for an example of a collector for the
// module dependencies.
func NewBuildInfoCollector() Collector {
	path, version, sum := readBuildInfo()
	c := &selfCollector{MustNewConstMetric(
		NewDesc(
			"go_build_info",
			"Build information about the main Go module.",
			nil, Labels{"path": path, "version": version, "checksum": sum},
		),
		GaugeValue, 1)}
	c.init(c.self)
	return c
}
