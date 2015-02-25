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

// +build linux,cgo plan9,cgo solaris,cgo

package prometheus

import "github.com/prometheus/procfs"

func processCollectSupported() bool {
	if _, err := procfs.NewStat(); err == nil {
		return true
	}
	return false
}

func (c *processCollector) processCollect(ch chan<- Metric) {
	pid, err := c.pidFn()
	if err != nil {
		c.reportCollectErrors(ch, err)
		return
	}

	p, err := procfs.NewProc(pid)
	if err != nil {
		c.reportCollectErrors(ch, err)
		return
	}

	if stat, err := p.NewStat(); err != nil {
		// Report collect errors for metrics depending on stat.
		ch <- NewInvalidMetric(c.vsize.Desc(), err)
		ch <- NewInvalidMetric(c.rss.Desc(), err)
		ch <- NewInvalidMetric(c.startTime.Desc(), err)
		ch <- NewInvalidMetric(c.cpuTotal.Desc(), err)
	} else {
		c.cpuTotal.Set(stat.CPUTime())
		ch <- c.cpuTotal
		c.vsize.Set(float64(stat.VirtualMemory()))
		ch <- c.vsize
		c.rss.Set(float64(stat.ResidentMemory()))
		ch <- c.rss

		if startTime, err := stat.StartTime(); err != nil {
			ch <- NewInvalidMetric(c.startTime.Desc(), err)
		} else {
			c.startTime.Set(startTime)
			ch <- c.startTime
		}
	}

	if fds, err := p.FileDescriptorsLen(); err != nil {
		ch <- NewInvalidMetric(c.openFDs.Desc(), err)
	} else {
		c.openFDs.Set(float64(fds))
		ch <- c.openFDs
	}

	if limits, err := p.NewLimits(); err != nil {
		ch <- NewInvalidMetric(c.maxFDs.Desc(), err)
	} else {
		c.maxFDs.Set(float64(limits.OpenFiles))
		ch <- c.maxFDs
	}
}

func (c *processCollector) reportCollectErrors(ch chan<- Metric, err error) {
	ch <- NewInvalidMetric(c.cpuTotal.Desc(), err)
	ch <- NewInvalidMetric(c.openFDs.Desc(), err)
	ch <- NewInvalidMetric(c.maxFDs.Desc(), err)
	ch <- NewInvalidMetric(c.vsize.Desc(), err)
	ch <- NewInvalidMetric(c.rss.Desc(), err)
	ch <- NewInvalidMetric(c.startTime.Desc(), err)
}
