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
	// Averaging over a 5-second sliding window sampled every 250ms (20 intervals)
	// smooths out short scheduling and request bursts while updating frequently
	// enough for load balancers that recalculate endpoint weights every 1s.
	defaultWindow         = 5 * time.Second
	defaultSampleInterval = 250 * time.Millisecond
)

// We use runtime.GOMAXPROCS(0) as the CPU core limit because since Go 1.25
// (containermaxprocs) it automatically reflects the minimum of host logical
// CPUs, process CPU affinity, and Linux cgroup CPU quota. We read it once at
// initialization because GOMAXPROCS(0) acquires the global Go scheduler lock
// (sched.lock) on every call.
func NewLoadReporter() (*Reporter, error) {
	return newSampler(float64(runtime.GOMAXPROCS(0)), &realSampler{})
}

func newSampler(cpuLimitCores float64, sampler LoadSampler) (*Reporter, error) {
	maxSamples := int(defaultWindow/defaultSampleInterval) + 1
	return &Reporter{
		sampleInterval: defaultSampleInterval,
		cpuLimitCores:  cpuLimitCores,
		sampler:        sampler,
		samples:        make([]*loadSample, 0, maxSamples),
	}, nil
}

type Reporter struct {
	sampleInterval  time.Duration
	cpuLimitCores   float64
	sampler         LoadSampler
	samples         []*loadSample
	nextSampleIndex int
	latestReport    atomic.Pointer[ReportORCA]
}

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

func (s *Reporter) LoadReport() *ReportORCA {
	return s.latestReport.Load()
}

type ReportORCA struct {
	CPUUtilization    float64
	RequestsPerSecond float64
}

func (s *Reporter) RecordRequest() {
	s.sampler.RecordRequest()
}

// cumulativeCPUSeconds is measured in core-seconds (1-core CPU time accumulated
// across all threads), so dividing its delta by wall-clock elapsedSeconds yields
// the average number of active CPU cores (usedCores) over the window. Dividing
// usedCores by cpuLimitCores compares cores to cores, clamped to 1.0 because
// the CNCF xDS ORCA specification constrains cpu_utilization to [0.0, 1.0].
func computeLoad(oldest, current *loadSample, cpuLimitCores float64) (*ReportORCA, error) {
	elapsedSeconds := current.timestamp.Sub(oldest.timestamp).Seconds()
	if elapsedSeconds <= 0 {
		return nil, fmt.Errorf("current sample timestamp is not after oldest sample timestamp")
	}
	usedCores := (current.cumulativeCPUSeconds - oldest.cumulativeCPUSeconds) / elapsedSeconds
	requestsPerSecond := float64(current.cumulativeRequestCount-oldest.cumulativeRequestCount) / elapsedSeconds

	return &ReportORCA{
		CPUUtilization:    min(usedCores/cpuLimitCores, 1.0),
		RequestsPerSecond: requestsPerSecond,
	}, nil
}
