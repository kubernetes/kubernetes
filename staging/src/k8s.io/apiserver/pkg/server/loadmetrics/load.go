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

	"golang.org/x/sys/unix"
)

type LoadSampler interface {
	Sample() (*loadSample, error)
	RecordRequest()
}

// cumulativeCPUSeconds is the cumulative 1-core CPU time (seconds * CPU cores)
// consumed by the process since startup.
type loadSample struct {
	timestamp              time.Time
	cumulativeCPUSeconds   float64
	cumulativeRequestCount uint64
}

type realSampler struct {
	requestCounter atomic.Uint64
}

func (s *realSampler) RecordRequest() {
	s.requestCounter.Add(1)
}

// We use clock_gettime(CLOCK_PROCESS_CPUTIME_ID) (matching grpc-go's GetCPUTime in
// https://github.com/grpc/grpc-go/blob/master/internal/syscall/syscall_linux.go)
// to read cumulative process CPU time because the Linux kernel sums per-thread
// nanosecond scheduler counters (sum_exec_runtime) and live running task deltas
// in a single call with nanosecond resolution.
//
// Alternatives rejected:
//   - /proc/self/stat requires file I/O per tick and quantizes CPU time to
//     CLK_TCK jiffies (typically 10ms).
//   - runtime/metrics (/cpu/classes/*:cpu-seconds) only updates during GC mark
//     termination and omits kernel/syscall execution time.
func (s *realSampler) Sample() (*loadSample, error) {
	var ts unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_PROCESS_CPUTIME_ID, &ts); err != nil {
		return nil, err
	}
	return &loadSample{
		cumulativeRequestCount: s.requestCounter.Load(),
		timestamp:              time.Now(),
		cumulativeCPUSeconds:   time.Duration(ts.Nano()).Seconds(),
	}, nil
}
