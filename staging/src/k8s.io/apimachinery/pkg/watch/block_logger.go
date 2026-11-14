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

package watch

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const (
	// DefaultBlockLoggerInterval is the default rate-limiting window for BlockLogger warnings.
	DefaultBlockLoggerInterval = 5 * time.Second
	// DefaultBlockLoggerThreshold is the default cumulative blocking threshold (10% of 5s window)
	// for channel sends and lock-bound store operations.
	DefaultBlockLoggerThreshold = 500 * time.Millisecond
	// DefaultDecodeLoggerThreshold is the default budget (50% of the 5s window) for the time the
	// watch stream spends waiting on object decoding, before the decoder risks falling behind
	// the incoming stream rate.
	DefaultDecodeLoggerThreshold = 2500 * time.Millisecond
	// DefaultBlockLoggerSingleThreshold triggers a warning when any single event blocks or decodes
	// longer than 200ms (even on low-rate resources where cumulative window wait stays below Threshold).
	DefaultBlockLoggerSingleThreshold = 200 * time.Millisecond
)

// BlockLoggerOptions configures attribution and threshold parameters for a BlockLogger.
type BlockLoggerOptions struct {
	Interval        time.Duration
	Threshold       time.Duration
	SingleThreshold time.Duration
	Source          string
	ResourceType    string
	Logger          *klog.Logger
	Clock           clock.Clock
}

// BlockLogger is a rate-limited logger that tracks cumulative duration spent in an operation
// or waiting on a channel send over a sliding interval. It logs warnings when cumulative wait
// within the interval exceeds threshold or any single event exceeds singleThreshold.
type BlockLogger struct {
	interval        time.Duration
	bucketWidth     time.Duration
	threshold       time.Duration
	singleThreshold time.Duration
	msg             string
	source          string
	logger          klog.Logger
	clock           clock.Clock

	// resourceType transitions once from nil to a concrete Go type string;
	// atomic.Pointer avoids locking mu or dirtying cache lines on every event.
	resourceType atomic.Pointer[string]

	mu sync.Mutex
	// origin is the start of bucket 0, and newest is the index of the bucket holding the latest wait.
	origin  time.Time
	newest  int64
	buckets [blockLoggerBuckets]blockLoggerBucket
	// timeWaiting and eventCount are the totals across the live buckets.
	timeWaiting time.Duration
	eventCount  int
	// pendingSince is when the current episode first exceeded a threshold.
	pendingSince time.Time
	pending      blockLoggerReport
	hasPending   bool
}

// blockLoggerBuckets is the number of slices the sliding interval is split into, so the
// evaluated window always covers between (blockLoggerBuckets-1)/blockLoggerBuckets and all of it.
const blockLoggerBuckets = 5

// blockLoggerBucket aggregates the waits that ended within one slice of the sliding window.
type blockLoggerBucket struct {
	timeWaiting time.Duration
	eventCount  int
	maxWait     time.Duration
}

// blockLoggerReport is the busiest sliding window and longest wait observed during an episode.
type blockLoggerReport struct {
	timeWaiting time.Duration
	eventCount  int
	maxWait     time.Duration
	window      time.Duration
}

// NewBlockLogger creates a BlockLogger with the specified interval, threshold, warning message, resourceType, and clock.
func NewBlockLogger(interval time.Duration, threshold time.Duration, msg string, resourceType string, clock clock.Clock) *BlockLogger {
	return NewBlockLoggerWithOptions(msg, BlockLoggerOptions{
		Interval:        interval,
		Threshold:       threshold,
		SingleThreshold: DefaultBlockLoggerSingleThreshold,
		ResourceType:    resourceType,
		Clock:           clock,
	})
}

// NewDefaultBlockLogger creates a BlockLogger with default blocking thresholds and the specified resourceType.
func NewDefaultBlockLogger(msg string, resourceType string, clock clock.Clock) *BlockLogger {
	return NewBlockLogger(DefaultBlockLoggerInterval, DefaultBlockLoggerThreshold, msg, resourceType, clock)
}

// NewBlockLoggerWithOptions creates a BlockLogger with explicit attribution and threshold options.
func NewBlockLoggerWithOptions(msg string, opts BlockLoggerOptions) *BlockLogger {
	interval := opts.Interval
	if interval <= 0 {
		interval = DefaultBlockLoggerInterval
	}

	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = DefaultBlockLoggerThreshold
	}

	clk := opts.Clock
	if clk == nil {
		clk = clock.RealClock{}
	}

	logger := klog.Background()
	if opts.Logger != nil {
		logger = *opts.Logger
	}

	l := &BlockLogger{
		interval:        interval,
		threshold:       threshold,
		singleThreshold: opts.SingleThreshold,
		msg:             msg,
		source:          opts.Source,
		logger:          logger,
		clock:           clk,
	}
	if opts.ResourceType != "" {
		l.SetResourceType(opts.ResourceType)
	}
	l.bucketWidth = interval / blockLoggerBuckets
	l.origin = l.clock.Now()
	return l
}

// SetResourceType updates the resource type tracked by the logger.
func (l *BlockLogger) SetResourceType(resourceType string) {
	if resourceType == "" {
		l.resourceType.Store(nil)
		return
	}
	s := resourceType
	l.resourceType.Store(&s)
}

// EnsureResourceType lazily sets the concrete Go type of obj once without acquiring mu.
func (l *BlockLogger) EnsureResourceType(obj any) {
	if obj == nil || l.resourceType.Load() != nil {
		return
	}

	s := fmt.Sprintf("%T", obj)
	l.resourceType.CompareAndSwap(nil, &s)
}

// ResourceType returns the resource type tracked by the logger.
func (l *BlockLogger) ResourceType() string {
	p := l.resourceType.Load()
	if p == nil {
		return ""
	}
	return *p
}

// Now returns the current timestamp from the logger's clock.
func (l *BlockLogger) Now() time.Time {
	return l.clock.Now()
}

// RecordSince records clock.Now().Sub(start) and checks the interval using one end-of-call clock read.
func (l *BlockLogger) RecordSince(start time.Time) {
	now := l.clock.Now()
	l.recordAt(now.Sub(start), now)
}

// RecordWait records a pre-measured duration ending now. When the waits ending within the sliding
// interval exceed threshold, or a single wait exceeds singleThreshold, an episode starts; it is
// logged at klog.V(2) one interval later with its busiest window, so at most once per interval.
func (l *BlockLogger) RecordWait(waitTime time.Duration) {
	l.recordAt(waitTime, l.clock.Now())
}

func (l *BlockLogger) recordAt(waitTime time.Duration, now time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()

	b := l.advanceLocked(now)
	b.timeWaiting += waitTime
	b.eventCount++
	b.maxWait = max(b.maxWait, waitTime)
	l.timeWaiting += waitTime
	l.eventCount++

	if l.timeWaiting > l.threshold || (l.singleThreshold > 0 && waitTime > l.singleThreshold) {
		l.notePendingLocked(now)
	}
	if l.hasPending && now.Sub(l.pendingSince) >= l.interval {
		l.reportLocked()
	}
}

// advanceLocked expires buckets that left the sliding interval and returns the bucket for now.
// Waits recorded out of order by concurrent callers land in the newest bucket.
func (l *BlockLogger) advanceLocked(now time.Time) *blockLoggerBucket {
	idx := int64(now.Sub(l.origin) / l.bucketWidth)
	switch {
	case idx-l.newest >= blockLoggerBuckets:
		l.buckets = [blockLoggerBuckets]blockLoggerBucket{}
		l.timeWaiting, l.eventCount = 0, 0
		l.newest = idx
	case idx > l.newest:
		for l.newest < idx {
			l.newest++
			expired := &l.buckets[l.newest%blockLoggerBuckets]
			l.timeWaiting -= expired.timeWaiting
			l.eventCount -= expired.eventCount
			*expired = blockLoggerBucket{}
		}
	}
	return &l.buckets[l.newest%blockLoggerBuckets]
}

// notePendingLocked starts an episode on the first threshold crossing and keeps its busiest
// window and longest wait until the episode is reported.
func (l *BlockLogger) notePendingLocked(now time.Time) {
	var maxWait time.Duration
	for i := range l.buckets {
		maxWait = max(maxWait, l.buckets[i].maxWait)
	}

	if !l.hasPending {
		l.hasPending = true
		l.pendingSince = now
	}
	l.pending.maxWait = max(l.pending.maxWait, maxWait)
	if l.timeWaiting >= l.pending.timeWaiting {
		oldest := max(l.newest-blockLoggerBuckets+1, 0)
		l.pending.timeWaiting = l.timeWaiting
		l.pending.eventCount = l.eventCount
		l.pending.window = now.Sub(l.origin.Add(time.Duration(oldest) * l.bucketWidth))
	}
}

// Flush emits a pending warning when a watch stream closes, bypassing the rate limit.
func (l *BlockLogger) Flush() {
	l.mu.Lock()
	defer l.mu.Unlock()

	if !l.hasPending {
		return
	}
	l.reportLocked()
}

func (l *BlockLogger) reportLocked() {
	p := l.pending
	l.pending, l.hasPending = blockLoggerReport{}, false
	if !l.logger.V(2).Enabled() {
		return
	}

	fields := []any{
		"resourceType", l.ResourceType(),
		"eventCount", p.eventCount,
		"timeWaiting", p.timeWaiting,
		"avgWait", p.timeWaiting / time.Duration(p.eventCount),
		"maxWait", p.maxWait,
		"window", p.window,
	}
	if l.source != "" {
		fields = append(fields, "source", l.source)
	}
	l.logger.V(2).Info(l.msg, fields...)
}

// TimeWaiting returns the wait recorded within the sliding interval (for testing).
func (l *BlockLogger) TimeWaiting() time.Duration {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.advanceLocked(l.clock.Now())
	return l.timeWaiting
}

// EventCount returns the number of waits recorded within the sliding interval (for testing).
func (l *BlockLogger) EventCount() int {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.advanceLocked(l.clock.Now())
	return l.eventCount
}
