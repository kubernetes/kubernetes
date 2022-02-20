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
	"sync"
	"time"

	"go.opencensus.io/internal"
)

const (
	maxBucketSize     = 100000
	defaultBucketSize = 10
)

var (
	ssmu       sync.RWMutex // protects spanStores
	spanStores = make(map[string]*spanStore)
)

// This exists purely to avoid exposing internal methods used by z-Pages externally.
type internalOnly struct{}

func init() {
	//TODO(#412): remove
	internal.Trace = &internalOnly{}
}

// ReportActiveSpans returns the active spans for the given name.
func (i internalOnly) ReportActiveSpans(name string) []*SpanData {
	s := spanStoreForName(name)
	if s == nil {
		return nil
	}
	var out []*SpanData
	s.mu.Lock()
	defer s.mu.Unlock()
	for activeSpan := range s.active {
		if s, ok := activeSpan.(*span); ok {
			out = append(out, s.makeSpanData())
		}
	}
	return out
}

// ReportSpansByError returns a sample of error spans.
//
// If code is nonzero, only spans with that status code are returned.
func (i internalOnly) ReportSpansByError(name string, code int32) []*SpanData {
	s := spanStoreForName(name)
	if s == nil {
		return nil
	}
	var out []*SpanData
	s.mu.Lock()
	defer s.mu.Unlock()
	if code != 0 {
		if b, ok := s.errors[code]; ok {
			for _, sd := range b.buffer {
				if sd == nil {
					break
				}
				out = append(out, sd)
			}
		}
	} else {
		for _, b := range s.errors {
			for _, sd := range b.buffer {
				if sd == nil {
					break
				}
				out = append(out, sd)
			}
		}
	}
	return out
}

// ConfigureBucketSizes sets the number of spans to keep per latency and error
// bucket for different span names.
func (i internalOnly) ConfigureBucketSizes(bcs []internal.BucketConfiguration) {
	for _, bc := range bcs {
		latencyBucketSize := bc.MaxRequestsSucceeded
		if latencyBucketSize < 0 {
			latencyBucketSize = 0
		}
		if latencyBucketSize > maxBucketSize {
			latencyBucketSize = maxBucketSize
		}
		errorBucketSize := bc.MaxRequestsErrors
		if errorBucketSize < 0 {
			errorBucketSize = 0
		}
		if errorBucketSize > maxBucketSize {
			errorBucketSize = maxBucketSize
		}
		spanStoreSetSize(bc.Name, latencyBucketSize, errorBucketSize)
	}
}

// ReportSpansPerMethod returns a summary of what spans are being stored for each span name.
func (i internalOnly) ReportSpansPerMethod() map[string]internal.PerMethodSummary {
	out := make(map[string]internal.PerMethodSummary)
	ssmu.RLock()
	defer ssmu.RUnlock()
	for name, s := range spanStores {
		s.mu.Lock()
		p := internal.PerMethodSummary{
			Active: len(s.active),
		}
		for code, b := range s.errors {
			p.ErrorBuckets = append(p.ErrorBuckets, internal.ErrorBucketSummary{
				ErrorCode: code,
				Size:      b.size(),
			})
		}
		for i, b := range s.latency {
			min, max := latencyBucketBounds(i)
			p.LatencyBuckets = append(p.LatencyBuckets, internal.LatencyBucketSummary{
				MinLatency: min,
				MaxLatency: max,
				Size:       b.size(),
			})
		}
		s.mu.Unlock()
		out[name] = p
	}
	return out
}

// ReportSpansByLatency returns a sample of successful spans.
//
// minLatency is the minimum latency of spans to be returned.
// maxLatency, if nonzero, is the maximum latency of spans to be returned.
func (i internalOnly) ReportSpansByLatency(name string, minLatency, maxLatency time.Duration) []*SpanData {
	s := spanStoreForName(name)
	if s == nil {
		return nil
	}
	var out []*SpanData
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, b := range s.latency {
		min, max := latencyBucketBounds(i)
		if i+1 != len(s.latency) && max <= minLatency {
			continue
		}
		if maxLatency != 0 && maxLatency < min {
			continue
		}
		for _, sd := range b.buffer {
			if sd == nil {
				break
			}
			if minLatency != 0 || maxLatency != 0 {
				d := sd.EndTime.Sub(sd.StartTime)
				if d < minLatency {
					continue
				}
				if maxLatency != 0 && d > maxLatency {
					continue
				}
			}
			out = append(out, sd)
		}
	}
	return out
}

// spanStore keeps track of spans stored for a particular span name.
//
// It contains all active spans; a sample of spans for failed requests,
// categorized by error code; and a sample of spans for successful requests,
// bucketed by latency.
type spanStore struct {
	mu                     sync.Mutex // protects everything below.
	active                 map[SpanInterface]struct{}
	errors                 map[int32]*bucket
	latency                []bucket
	maxSpansPerErrorBucket int
}

// newSpanStore creates a span store.
func newSpanStore(name string, latencyBucketSize int, errorBucketSize int) *spanStore {
	s := &spanStore{
		active:                 make(map[SpanInterface]struct{}),
		latency:                make([]bucket, len(defaultLatencies)+1),
		maxSpansPerErrorBucket: errorBucketSize,
	}
	for i := range s.latency {
		s.latency[i] = makeBucket(latencyBucketSize)
	}
	return s
}

// spanStoreForName returns the spanStore for the given name.
//
// It returns nil if it doesn't exist.
func spanStoreForName(name string) *spanStore {
	var s *spanStore
	ssmu.RLock()
	s, _ = spanStores[name]
	ssmu.RUnlock()
	return s
}

// spanStoreForNameCreateIfNew returns the spanStore for the given name.
//
// It creates it if it didn't exist.
func spanStoreForNameCreateIfNew(name string) *spanStore {
	ssmu.RLock()
	s, ok := spanStores[name]
	ssmu.RUnlock()
	if ok {
		return s
	}
	ssmu.Lock()
	defer ssmu.Unlock()
	s, ok = spanStores[name]
	if ok {
		return s
	}
	s = newSpanStore(name, defaultBucketSize, defaultBucketSize)
	spanStores[name] = s
	return s
}

// spanStoreSetSize resizes the spanStore for the given name.
//
// It creates it if it didn't exist.
func spanStoreSetSize(name string, latencyBucketSize int, errorBucketSize int) {
	ssmu.RLock()
	s, ok := spanStores[name]
	ssmu.RUnlock()
	if ok {
		s.resize(latencyBucketSize, errorBucketSize)
		return
	}
	ssmu.Lock()
	defer ssmu.Unlock()
	s, ok = spanStores[name]
	if ok {
		s.resize(latencyBucketSize, errorBucketSize)
		return
	}
	s = newSpanStore(name, latencyBucketSize, errorBucketSize)
	spanStores[name] = s
}

func (s *spanStore) resize(latencyBucketSize int, errorBucketSize int) {
	s.mu.Lock()
	for i := range s.latency {
		s.latency[i].resize(latencyBucketSize)
	}
	for _, b := range s.errors {
		b.resize(errorBucketSize)
	}
	s.maxSpansPerErrorBucket = errorBucketSize
	s.mu.Unlock()
}

// add adds a span to the active bucket of the spanStore.
func (s *spanStore) add(span SpanInterface) {
	s.mu.Lock()
	s.active[span] = struct{}{}
	s.mu.Unlock()
}

// finished removes a span from the active set, and adds a corresponding
// SpanData to a latency or error bucket.
func (s *spanStore) finished(span SpanInterface, sd *SpanData) {
	latency := sd.EndTime.Sub(sd.StartTime)
	if latency < 0 {
		latency = 0
	}
	code := sd.Status.Code

	s.mu.Lock()
	delete(s.active, span)
	if code == 0 {
		s.latency[latencyBucket(latency)].add(sd)
	} else {
		if s.errors == nil {
			s.errors = make(map[int32]*bucket)
		}
		if b := s.errors[code]; b != nil {
			b.add(sd)
		} else {
			b := makeBucket(s.maxSpansPerErrorBucket)
			s.errors[code] = &b
			b.add(sd)
		}
	}
	s.mu.Unlock()
}
