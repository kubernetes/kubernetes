/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package loadmetrics

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"time"

	"k8s.io/klog/v2"
)

const (
	// defaultWindow is the sliding window duration over which CPU utilization and
	// request rate are averaged. A 5-second window sampled every 250ms (20 intervals)
	// smooths out short scheduling and request bursts while updating frequently
	// enough for load balancers that recalculate endpoint weights every 1s.
	defaultWindow = 5 * time.Second
	// defaultSampleInterval is the interval between consecutive load samples.
	defaultSampleInterval = 250 * time.Millisecond
)

// NewLoadReporter creates a Reporter configured with the process CPU core limit
// and OS load sampler, or returns an error if process CPU time measurement is not
// supported on the current platform.
//
// It uses runtime.GOMAXPROCS(0) as the CPU core limit because since Go 1.25
// (containermaxprocs) GOMAXPROCS automatically reflects the minimum of host logical
// CPUs, process CPU affinity, and Linux cgroup CPU quota. The value is read once at
// initialization because GOMAXPROCS(0) acquires the global Go scheduler lock
// (sched.lock) on every call.
func NewLoadReporter() (*Reporter, error) {
	if !cpuTimeSupported() {
		return nil, fmt.Errorf("process CPU time measurement is not supported on %s", runtime.GOOS)
	}
	cpuLimitCores := float64(runtime.GOMAXPROCS(0))
	if cpuLimitCores <= 0 {
		return nil, fmt.Errorf("GOMAXPROCS must be positive, got %f", cpuLimitCores)
	}
	return newSampler(cpuLimitCores, &realSampler{}), nil
}

func newSampler(cpuLimitCores float64, sampler LoadSampler) *Reporter {
	maxSamples := int(defaultWindow/defaultSampleInterval) + 1
	return &Reporter{
		sampleInterval: defaultSampleInterval,
		cpuLimitCores:  cpuLimitCores,
		sampler:        sampler,
		samples:        make([]*loadSample, 0, maxSamples),
	}
}

// Reporter periodically samples process CPU usage and request counts over a sliding
// window and publishes the latest ORCA load report for lock-free reads on the
// HTTP request path.
type Reporter struct {
	sampleInterval  time.Duration
	cpuLimitCores   float64
	sampler         LoadSampler
	samples         []*loadSample
	nextSampleIndex int
	latestReport    atomic.Pointer[ReportORCA]
}

// Run periodically samples server load until stopCh is closed. Once a full sliding
// window of samples is collected, it updates the published ORCA load report on each tick.
func (s *Reporter) Run(stopCh <-chan struct{}) {
	ticker := time.NewTicker(s.sampleInterval)
	defer ticker.Stop()

	for {
		select {
		case <-stopCh:
			return
		case <-ticker.C:
			sample, err := s.sampler.Sample()
			if err != nil {
				klog.ErrorS(err, "Failed to sample server load, clearing samples and will not report until success")
				s.reset()
				continue
			}
			s.recordSample(sample)
			if len(s.samples) < cap(s.samples) {
				continue
			}
			report, err := computeLoad(s.samples[s.nextSampleIndex], sample, s.cpuLimitCores)
			if err != nil {
				klog.ErrorS(err, "Failed to compute server load, clearing samples and will not report until success")
				s.reset()
				continue
			}
			s.latestReport.Store(report)
		}
	}
}

func (s *Reporter) recordSample(sample *loadSample) {
	if len(s.samples) < cap(s.samples) {
		s.samples = append(s.samples, sample)
		return
	}
	s.samples[s.nextSampleIndex] = sample
	s.nextSampleIndex = (s.nextSampleIndex + 1) % len(s.samples)
}

func (s *Reporter) reset() {
	s.samples = s.samples[:0]
	s.nextSampleIndex = 0
	s.latestReport.Store(nil)
}

// LoadReport returns the most recently computed ORCA load report, or nil if a full
// sample window has not yet been collected.
func (s *Reporter) LoadReport() *ReportORCA {
	return s.latestReport.Load()
}

// ReportORCA holds the ORCA load metrics exposed in endpoint-load-metrics headers.
type ReportORCA struct {
	// CPUUtilization is the fraction of available CPU capacity (GOMAXPROCS) used
	// over the sample window. Per the CNCF xDS ORCA specification, cpu_utilization
	// is non-negative and may exceed 1.0 when usage exceeds soft limits.
	CPUUtilization float64
	// RequestsPerSecond is the average request rate (rps_fractional) over the sample window.
	RequestsPerSecond float64
}

// RecordRequest increments the cumulative request counter used to calculate RequestsPerSecond.
func (s *Reporter) RecordRequest() {
	s.sampler.RecordRequest()
}

// computeLoad calculates the average CPU utilization and request rate between oldest
// and current over a sliding window.
//
// cumulativeCPUSeconds is measured in core-seconds (1-core CPU time accumulated
// across all threads), so dividing its delta by wall-clock elapsedSeconds yields
// the average number of active CPU cores (usedCores) over the window. Dividing
// usedCores by cpuLimitCores (GOMAXPROCS) compares cores to cores;
func computeLoad(oldest, current *loadSample, cpuLimitCores float64) (*ReportORCA, error) {
	elapsedSeconds := current.timestamp.Sub(oldest.timestamp).Seconds()
	if elapsedSeconds <= 0 {
		return nil, fmt.Errorf("current sample timestamp is not after oldest sample timestamp")
	}
	cpuUsedSeconds := current.cumulativeCPUSeconds - oldest.cumulativeCPUSeconds
	if cpuUsedSeconds < 0 {
		return nil, fmt.Errorf("current cumulative CPU usage is less than oldest cumulative CPU usage")
	}
	requestsPerSecond := float64(current.cumulativeRequestCount-oldest.cumulativeRequestCount) / elapsedSeconds
	cpuUtilization := cpuUsedSeconds / cpuLimitCores / elapsedSeconds
	// Per the CNCF xDS ORCA specification, cpu_utilization is constrained to >= 0 and
	// may exceed 1.0 when OS thread CPU usage exceeds the GOMAXPROCS soft limit.
	cpuUtilization = max(cpuUtilization, 0)

	return &ReportORCA{
		CPUUtilization:    cpuUtilization,
		RequestsPerSecond: requestsPerSecond,
	}, nil
}
