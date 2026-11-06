// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"cel.dev/cel-go/common/types/ref"
)

const (
	defaultMemoryTrackerSampleInterval = 1
)

// MemoryTrackerOption configures a MemoryTracker instance.
type MemoryTrackerOption func(*MemoryTracker)

// MemoryTrackerLimit sets a limit on the peak aggregate memory observed during tracking.
//
// The tracker does not enforce the limit itself; callers should consult ExceedsLimit after
// tracking observations and terminate evaluation as appropriate.
func MemoryTrackerLimit(limit uint32) MemoryTrackerOption {
	return func(t *MemoryTracker) {
		t.limit = limit
		t.hasLimit = true
	}
}

// MemoryTrackerSampleInterval configures how frequently Sample observations compute a size.
//
// An interval of N means every Nth call to Sample performs a size computation; intervening
// calls are skipped. Values less than 1 are treated as 1, meaning every sample is computed.
// Sampling bounds the tracking overhead for high-frequency observation points such as
// comprehension loops and bind initializers.
func MemoryTrackerSampleInterval(interval uint32) MemoryTrackerOption {
	return func(t *MemoryTracker) {
		if interval < 1 {
			interval = 1
		}
		t.sampleInterval = interval
	}
}

// MemoryTrackerSizeCalculator overrides the SizeCalculator used to compute value sizes.
func MemoryTrackerSizeCalculator(calc *SizeCalculator) MemoryTrackerOption {
	return func(t *MemoryTracker) {
		t.calc = calc
	}
}

// MemoryTracker records the peak aggregate memory observed during an evaluation.
//
// Memory is measured in aggregate element counts as computed by a SizeCalculator, with all
// arithmetic saturating at math.MaxUint32. The tracker is independent of any interpreter
// implementation; evaluators feed it observations at the points where values materialize
// during evaluation, such as resolved attributes, call results, constructed aggregates, and
// values built up within comprehensions or bind initializers.
//
// The peak is the largest single observation, where one Track call observes a value
// as a watermark.
//
// A MemoryTracker is stateful and intended for use by a single evaluation at a time; it is
// not safe for concurrent use.
type MemoryTracker struct {
	version        int
	calc           *SizeCalculator
	limit          uint32
	hasLimit       bool
	sampleInterval uint32

	sampleCounts      map[int64]uint32
	peak              uint32
	calcLimitExceeded bool
}

// NewMemoryTracker returns a new MemoryTracker configured with optional MemoryTrackerOption
// settings, using a default SizeCalculator when one is not provided.
func NewMemoryTracker(opts ...MemoryTrackerOption) *MemoryTracker {
	t := &MemoryTracker{
		version:        0,
		sampleInterval: defaultMemoryTrackerSampleInterval,
	}
	for _, opt := range opts {
		opt(t)
	}
	if t.calc == nil {
		t.calc = NewSizeCalculator()
	}
	return t
}

// Version returns the tracking version.
func (t *MemoryTracker) Version() int {
	return t.version
}

// Track observes a value as a watermark, returning its aggregate size with saturation at math.MaxUint32.
func (t *MemoryTracker) Track(val ref.Val) uint32 {
	if val == nil {
		return 0
	}
	switch v := val.(type) {
	case Bool, Int, Uint, Double, Duration, Timestamp, Null, *Type, *Err, *Unknown:
		if 1 > t.peak {
			t.peak = 1
		}
		return 1
	case String:
		sz := t.calc.stringSize(len(v))
		if sz > t.peak {
			t.peak = sz
		}
		return sz
	case Bytes:
		sz := t.calc.stringSize(len(v))
		if sz > t.peak {
			t.peak = sz
		}
		return sz
	}

	est := t.calc.ApproximateAggregateSize(val)
	if est.LimitExceeded {
		t.calcLimitExceeded = true
	}
	if est.Size > t.peak {
		t.peak = est.Size
	}
	return est.Size
}

// Sample observes a value for a specific node ID subject to the tracker's sample interval,
// returning the value's aggregate size when computed, or zero when the observation is skipped.
//
// The first observation of any node ID is always tracked; subsequent observations are sampled
// at multiples of the sample interval.
//
// Sample is intended for high-frequency observation points, such as accumulator values built
// up by comprehension loops or bind initializers, where sizing every iteration would be
// prohibitively expensive.
func (t *MemoryTracker) Sample(id int64, val ref.Val) uint32 {
	if t.sampleCounts == nil {
		t.sampleCounts = make(map[int64]uint32)
	}
	t.sampleCounts[id]++
	count := t.sampleCounts[id]
	if count == 1 || t.sampleInterval <= 1 || count%t.sampleInterval == 0 {
		return t.Track(val)
	}
	return 0
}

// Peak returns the largest single watermark observed, saturating at math.MaxUint32.
func (t *MemoryTracker) Peak() uint32 {
	return t.peak
}

// ExceedsLimit indicates whether the peak observed memory exceeds the configured limit.
// When no limit is configured, ExceedsLimit always returns false.
func (t *MemoryTracker) ExceedsLimit() bool {
	return t.hasLimit && t.peak > t.limit
}

// CalculationLimitExceeded indicates whether any tracked value was too expensive to size,
// causing the size computation to abort at the SizeCalculator's depth or traversal limits.
//
// Such observations saturate to math.MaxUint32; this signal distinguishes values which were
// too costly to measure from values whose measured size genuinely saturated uint32.
func (t *MemoryTracker) CalculationLimitExceeded() bool {
	return t.calcLimitExceeded
}
