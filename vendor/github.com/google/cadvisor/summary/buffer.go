// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package summary

import (
	info "github.com/google/cadvisor/info/v2"
)

// Manages a buffer of usage samples.
// This is similar to stats buffer in cache/memory.
// The main difference is that we do not pre-allocate the buffer as most containers
// won't live that long.
type SamplesBuffer struct {
	// list of collected samples.
	samples []info.Usage
	// maximum size this buffer can grow to.
	maxSize int
	// index for the latest sample.
	index int
}

// Initializes an empty buffer.
func NewSamplesBuffer(size int) *SamplesBuffer {
	return &SamplesBuffer{
		index:   -1,
		maxSize: size,
	}
}

// Returns the current number of samples in the buffer.
func (s *SamplesBuffer) Size() int {
	return len(s.samples)
}

// Add an element to the buffer. Oldest one is overwritten if required.
func (s *SamplesBuffer) Add(stat info.Usage) {
	if len(s.samples) < s.maxSize {
		s.samples = append(s.samples, stat)
		s.index++
		return
	}
	s.index = (s.index + 1) % s.maxSize
	s.samples[s.index] = stat
}

// Returns pointers to the last 'n' stats.
func (s *SamplesBuffer) RecentStats(n int) []*info.Usage {
	if n > len(s.samples) {
		n = len(s.samples)
	}
	start := s.index - (n - 1)
	if start < 0 {
		start += len(s.samples)
	}

	out := make([]*info.Usage, n)
	for i := 0; i < n; i++ {
		index := (start + i) % len(s.samples)
		out[i] = &s.samples[index]
	}
	return out
}
