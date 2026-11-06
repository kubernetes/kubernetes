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
	"sync/atomic"
	"time"
)

// LoadSampler collects point-in-time process CPU and request count snapshots.
type LoadSampler interface {
	Sample() (*loadSample, error)
	RecordRequest()
}

// loadSample is a point-in-time snapshot of cumulative process load counters.
type loadSample struct {
	timestamp time.Time
	// cumulativeCPUSeconds is the cumulative 1-core CPU time (seconds * CPU cores)
	// consumed by the process since startup.
	cumulativeCPUSeconds   float64
	cumulativeRequestCount uint64
}

type realSampler struct {
	requestCounter atomic.Uint64
}

func (s *realSampler) RecordRequest() {
	s.requestCounter.Add(1)
}

func (s *realSampler) Sample() (*loadSample, error) {
	cpuSeconds, err := getProcessCPUTimeSeconds()
	if err != nil {
		return nil, err
	}
	return &loadSample{
		cumulativeRequestCount: s.requestCounter.Load(),
		timestamp:              time.Now(),
		cumulativeCPUSeconds:   cpuSeconds,
	}, nil
}
