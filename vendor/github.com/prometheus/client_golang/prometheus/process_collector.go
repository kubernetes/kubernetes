// Copyright 2015 The Prometheus Authors
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
	"errors"
	"os"

	"github.com/prometheus/procfs"
)

type processCollector struct {
	collectFn       func(chan<- Metric)
	pidFn           func() (int, error)
	reportErrors    bool
	cpuTotal        *Desc
	openFDs, maxFDs *Desc
	vsize, maxVsize *Desc
	rss             *Desc
	startTime       *Desc
}

// ProcessCollectorOpts defines the behavior of a process metrics collector
// created with NewProcessCollector.
type ProcessCollectorOpts struct {
	// PidFn returns the PID of the process the collector collects metrics
	// for. It is called upon each collection. By default, the PID of the
	// current process is used, as determined on construction time by
	// calling os.Getpid().
	PidFn func() (int, error)
	// If non-empty, each of the collected metrics is prefixed by the
	// provided string and an underscore ("_").
	Namespace string
	// If true, any error encountered during collection is reported as an
	// invalid metric (see NewInvalidMetric). Otherwise, errors are ignored
	// and the collected metrics will be incomplete. (Possibly, no metrics
	// will be collected at all.) While that's usually not desired, it is
	// appropriate for the common "mix-in" of process metrics, where process
	// metrics are nice to have, but failing to collect them should not
	// disrupt the collection of the remaining metrics.
	ReportErrors bool
}

// NewProcessCollector returns a collector which exports the current state of
// process metrics including CPU, memory and file descriptor usage as well as
// the process start time. The detailed behavior is defined by the provided
// ProcessCollectorOpts. The zero value of ProcessCollectorOpts creates a
// collector for the current process with an empty namespace string and no error
// reporting.
//
// Currently, the collector depends on a Linux-style proc filesystem and
// therefore only exports metrics for Linux.
//
// Note: An older version of this function had the following signature:
//
//     NewProcessCollector(pid int, namespace string) Collector
//
// Most commonly, it was called as
//
//     NewProcessCollector(os.Getpid(), "")
//
// The following call of the current version is equivalent to the above:
//
//     NewProcessCollector(ProcessCollectorOpts{})
func NewProcessCollector(opts ProcessCollectorOpts) Collector {
	ns := ""
	if len(opts.Namespace) > 0 {
		ns = opts.Namespace + "_"
	}

	c := &processCollector{
		reportErrors: opts.ReportErrors,
		cpuTotal: NewDesc(
			ns+"process_cpu_seconds_total",
			"Total user and system CPU time spent in seconds.",
			nil, nil,
		),
		openFDs: NewDesc(
			ns+"process_open_fds",
			"Number of open file descriptors.",
			nil, nil,
		),
		maxFDs: NewDesc(
			ns+"process_max_fds",
			"Maximum number of open file descriptors.",
			nil, nil,
		),
		vsize: NewDesc(
			ns+"process_virtual_memory_bytes",
			"Virtual memory size in bytes.",
			nil, nil,
		),
		maxVsize: NewDesc(
			ns+"process_virtual_memory_max_bytes",
			"Maximum amount of virtual memory available in bytes.",
			nil, nil,
		),
		rss: NewDesc(
			ns+"process_resident_memory_bytes",
			"Resident memory size in bytes.",
			nil, nil,
		),
		startTime: NewDesc(
			ns+"process_start_time_seconds",
			"Start time of the process since unix epoch in seconds.",
			nil, nil,
		),
	}

	if opts.PidFn == nil {
		pid := os.Getpid()
		c.pidFn = func() (int, error) { return pid, nil }
	} else {
		c.pidFn = opts.PidFn
	}

	// Set up process metric collection if supported by the runtime.
	if _, err := procfs.NewDefaultFS(); err == nil {
		c.collectFn = c.processCollect
	} else {
		c.collectFn = func(ch chan<- Metric) {
			c.reportError(ch, nil, errors.New("process metrics not supported on this platform"))
		}
	}

	return c
}

// Describe returns all descriptions of the collector.
func (c *processCollector) Describe(ch chan<- *Desc) {
	ch <- c.cpuTotal
	ch <- c.openFDs
	ch <- c.maxFDs
	ch <- c.vsize
	ch <- c.maxVsize
	ch <- c.rss
	ch <- c.startTime
}

// Collect returns the current state of all metrics of the collector.
func (c *processCollector) Collect(ch chan<- Metric) {
	c.collectFn(ch)
}

func (c *processCollector) processCollect(ch chan<- Metric) {
	pid, err := c.pidFn()
	if err != nil {
		c.reportError(ch, nil, err)
		return
	}

	p, err := procfs.NewProc(pid)
	if err != nil {
		c.reportError(ch, nil, err)
		return
	}

	if stat, err := p.Stat(); err == nil {
		ch <- MustNewConstMetric(c.cpuTotal, CounterValue, stat.CPUTime())
		ch <- MustNewConstMetric(c.vsize, GaugeValue, float64(stat.VirtualMemory()))
		ch <- MustNewConstMetric(c.rss, GaugeValue, float64(stat.ResidentMemory()))
		if startTime, err := stat.StartTime(); err == nil {
			ch <- MustNewConstMetric(c.startTime, GaugeValue, startTime)
		} else {
			c.reportError(ch, c.startTime, err)
		}
	} else {
		c.reportError(ch, nil, err)
	}

	if fds, err := p.FileDescriptorsLen(); err == nil {
		ch <- MustNewConstMetric(c.openFDs, GaugeValue, float64(fds))
	} else {
		c.reportError(ch, c.openFDs, err)
	}

	if limits, err := p.Limits(); err == nil {
		ch <- MustNewConstMetric(c.maxFDs, GaugeValue, float64(limits.OpenFiles))
		ch <- MustNewConstMetric(c.maxVsize, GaugeValue, float64(limits.AddressSpace))
	} else {
		c.reportError(ch, nil, err)
	}
}

func (c *processCollector) reportError(ch chan<- Metric, desc *Desc, err error) {
	if !c.reportErrors {
		return
	}
	if desc == nil {
		desc = NewInvalidDesc(err)
	}
	ch <- NewInvalidMetric(desc, err)
}
