// Copyright 2017, OpenCensus Authors
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

package trace

import (
	"time"
)

// samplePeriod is the minimum time between accepting spans in a single bucket.
const samplePeriod = time.Second

// defaultLatencies contains the default latency bucket bounds.
// TODO: consider defaults, make configurable
var defaultLatencies = [...]time.Duration{
	10 * time.Microsecond,
	100 * time.Microsecond,
	time.Millisecond,
	10 * time.Millisecond,
	100 * time.Millisecond,
	time.Second,
	10 * time.Second,
	time.Minute,
}

// bucket is a container for a set of spans for a particular error code or latency range.
type bucket struct {
	nextTime  time.Time   // next time we can accept a span
	buffer    []*SpanData // circular buffer of spans
	nextIndex int         // location next SpanData should be placed in buffer
	overflow  bool        // whether the circular buffer has wrapped around
}

func makeBucket(bufferSize int) bucket {
	return bucket{
		buffer: make([]*SpanData, bufferSize),
	}
}

// add adds a span to the bucket, if nextTime has been reached.
func (b *bucket) add(s *SpanData) {
	if s.EndTime.Before(b.nextTime) {
		return
	}
	if len(b.buffer) == 0 {
		return
	}
	b.nextTime = s.EndTime.Add(samplePeriod)
	b.buffer[b.nextIndex] = s
	b.nextIndex++
	if b.nextIndex == len(b.buffer) {
		b.nextIndex = 0
		b.overflow = true
	}
}

// size returns the number of spans in the bucket.
func (b *bucket) size() int {
	if b.overflow {
		return len(b.buffer)
	}
	return b.nextIndex
}

// span returns the ith span in the bucket.
func (b *bucket) span(i int) *SpanData {
	if !b.overflow {
		return b.buffer[i]
	}
	if i < len(b.buffer)-b.nextIndex {
		return b.buffer[b.nextIndex+i]
	}
	return b.buffer[b.nextIndex+i-len(b.buffer)]
}

// resize changes the size of the bucket to n, keeping up to n existing spans.
func (b *bucket) resize(n int) {
	cur := b.size()
	newBuffer := make([]*SpanData, n)
	if cur < n {
		for i := 0; i < cur; i++ {
			newBuffer[i] = b.span(i)
		}
		b.buffer = newBuffer
		b.nextIndex = cur
		b.overflow = false
		return
	}
	for i := 0; i < n; i++ {
		newBuffer[i] = b.span(i + cur - n)
	}
	b.buffer = newBuffer
	b.nextIndex = 0
	b.overflow = true
}

// latencyBucket returns the appropriate bucket number for a given latency.
func latencyBucket(latency time.Duration) int {
	i := 0
	for i < len(defaultLatencies) && latency >= defaultLatencies[i] {
		i++
	}
	return i
}

// latencyBucketBounds returns the lower and upper bounds for a latency bucket
// number.
//
// The lower bound is inclusive, the upper bound is exclusive (except for the
// last bucket.)
func latencyBucketBounds(index int) (lower time.Duration, upper time.Duration) {
	if index == 0 {
		return 0, defaultLatencies[index]
	}
	if index == len(defaultLatencies) {
		return defaultLatencies[index-1], 1<<63 - 1
	}
	return defaultLatencies[index-1], defaultLatencies[index]
}
