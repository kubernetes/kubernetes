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
	"math/rand/v2"
	"reflect"
	"sync"
	"testing"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/clock"
	clocktesting "k8s.io/utils/clock/testing"
)

const testBlockLoggerMsg = "slow test pipeline"

var testBlockLoggerEpoch = time.Date(2026, time.January, 1, 0, 0, 0, 0, time.UTC)

type blockLoggerTestPod struct{}

type blockLoggerTestNode struct{}

// capturedBlockLog is one BlockLogger warning recorded by a buffered ktesting logger.
type capturedBlockLog struct {
	verbosity int
	values    map[string]any
}

func newBufferedTestLogger(t *testing.T, verbosity int) klog.Logger {
	return ktesting.NewLogger(t, ktesting.NewConfig(ktesting.Verbosity(verbosity), ktesting.BufferLogs(true)))
}

func capturedBlockLogs(t *testing.T, logger klog.Logger) []capturedBlockLog {
	t.Helper()

	var logs []capturedBlockLog
	for _, entry := range logger.GetSink().(ktesting.Underlier).GetBuffer().Data() {
		if entry.Message != testBlockLoggerMsg {
			t.Fatalf("Unexpected log message %q", entry.Message)
		}
		values := make(map[string]any, len(entry.ParameterKVList)/2)
		for i := 0; i+1 < len(entry.ParameterKVList); i += 2 {
			values[fmt.Sprint(entry.ParameterKVList[i])] = entry.ParameterKVList[i+1]
		}
		logs = append(logs, capturedBlockLog{verbosity: entry.Verbosity, values: values})
	}
	return logs
}

func (c capturedBlockLog) duration(t *testing.T, key string) time.Duration {
	t.Helper()

	d, ok := c.values[key].(time.Duration)
	if !ok {
		t.Fatalf("Expected %q to be a time.Duration, got %T in %v", key, c.values[key], c.values)
	}
	return d
}

func (c capturedBlockLog) eventCount(t *testing.T) int {
	t.Helper()

	n, ok := c.values["eventCount"].(int)
	if !ok {
		t.Fatalf("Expected eventCount to be an int, got %T in %v", c.values["eventCount"], c.values)
	}
	return n
}

func newTestBlockLogger(t *testing.T, opts BlockLoggerOptions) (*BlockLogger, klog.Logger, *clocktesting.FakeClock) {
	t.Helper()

	fakeClock := clocktesting.NewFakeClock(testBlockLoggerEpoch)
	logger := newBufferedTestLogger(t, 2)
	opts.Clock = fakeClock
	opts.Logger = &logger
	return NewBlockLoggerWithOptions(testBlockLoggerMsg, opts), logger, fakeClock
}

func TestNewBlockLoggerWithOptions_Defaults(t *testing.T) {
	l := NewBlockLoggerWithOptions(testBlockLoggerMsg, BlockLoggerOptions{Interval: -time.Second, Threshold: -time.Second})
	if l.interval != DefaultBlockLoggerInterval {
		t.Errorf("Expected interval %v, got %v", DefaultBlockLoggerInterval, l.interval)
	}
	if l.threshold != DefaultBlockLoggerThreshold {
		t.Errorf("Expected threshold %v, got %v", DefaultBlockLoggerThreshold, l.threshold)
	}
	if l.singleThreshold != 0 {
		t.Errorf("Expected the single-event threshold to stay disabled, got %v", l.singleThreshold)
	}
	if _, ok := l.clock.(clock.RealClock); !ok {
		t.Errorf("Expected a nil clock to default to clock.RealClock, got %T", l.clock)
	}
	if got := l.ResourceType(); got != "" {
		t.Errorf("Expected an empty resource type, got %q", got)
	}

	l = NewDefaultBlockLogger(testBlockLoggerMsg, "", nil)
	if l.interval != DefaultBlockLoggerInterval || l.threshold != DefaultBlockLoggerThreshold || l.singleThreshold != DefaultBlockLoggerSingleThreshold {
		t.Errorf("Expected default interval/threshold/single threshold %v/%v/%v, got %v/%v/%v",
			DefaultBlockLoggerInterval, DefaultBlockLoggerThreshold, DefaultBlockLoggerSingleThreshold, l.interval, l.threshold, l.singleThreshold)
	}
	if _, ok := l.clock.(clock.RealClock); !ok {
		t.Errorf("Expected a nil clock to default to clock.RealClock, got %T", l.clock)
	}

	fakeClock := clocktesting.NewFakeClock(testBlockLoggerEpoch)
	l = NewBlockLogger(time.Second, time.Millisecond, testBlockLoggerMsg, "*v1.Pod", fakeClock)
	if l.interval != time.Second || l.threshold != time.Millisecond || l.singleThreshold != DefaultBlockLoggerSingleThreshold {
		t.Errorf("Expected interval/threshold/single threshold 1s/1ms/%v, got %v/%v/%v",
			DefaultBlockLoggerSingleThreshold, l.interval, l.threshold, l.singleThreshold)
	}
	if got := l.ResourceType(); got != "*v1.Pod" {
		t.Errorf("Expected resource type '*v1.Pod', got %q", got)
	}
	if !l.origin.Equal(testBlockLoggerEpoch) || l.bucketWidth != time.Second/blockLoggerBuckets {
		t.Errorf("Expected buckets of %v starting at %v, got %v starting at %v", time.Second/blockLoggerBuckets, testBlockLoggerEpoch, l.bucketWidth, l.origin)
	}
}

func TestBlockLogger_Thresholds(t *testing.T) {
	testCases := []struct {
		name            string
		singleThreshold time.Duration
		waits           []time.Duration
		wantLog         bool
	}{
		{
			name:            "cumulative and single waits within thresholds",
			singleThreshold: 200 * time.Millisecond,
			waits:           []time.Duration{100 * time.Millisecond, 150 * time.Millisecond, 200 * time.Millisecond},
		},
		{
			name:            "cumulative wait equal to threshold",
			singleThreshold: 200 * time.Millisecond,
			waits:           []time.Duration{100 * time.Millisecond, 200 * time.Millisecond, 200 * time.Millisecond},
		},
		{
			name:            "cumulative wait exceeds threshold",
			singleThreshold: 200 * time.Millisecond,
			waits:           []time.Duration{150 * time.Millisecond, 150 * time.Millisecond, 150 * time.Millisecond, 50*time.Millisecond + time.Nanosecond},
			wantLog:         true,
		},
		{
			name:            "single wait exceeds single threshold",
			singleThreshold: 200 * time.Millisecond,
			waits:           []time.Duration{200*time.Millisecond + time.Nanosecond},
			wantLog:         true,
		},
		{
			name:  "single threshold disabled",
			waits: []time.Duration{450 * time.Millisecond},
		},
		{
			name:    "single threshold disabled but cumulative wait exceeds threshold",
			waits:   []time.Duration{300 * time.Millisecond, 300 * time.Millisecond},
			wantLog: true,
		},
	}

	for _, tc := range testCases {
		for _, closeBy := range []string{"interval", "flush"} {
			t.Run(tc.name+"/"+closeBy, func(t *testing.T) {
				l, logger, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{
					Threshold:       500 * time.Millisecond,
					SingleThreshold: tc.singleThreshold,
				})
				for _, wait := range tc.waits {
					l.RecordWait(wait)
				}
				if logs := capturedBlockLogs(t, logger); len(logs) != 0 {
					t.Fatalf("Expected no warning before the window closes, got %d", len(logs))
				}

				fakeClock.Step(DefaultBlockLoggerInterval)
				if closeBy == "flush" {
					l.Flush()
				} else {
					l.RecordWait(0)
				}

				logs := capturedBlockLogs(t, logger)
				if len(logs) > 1 || (len(logs) == 1) != tc.wantLog {
					t.Fatalf("Expected warning=%v, got %d warnings: %v", tc.wantLog, len(logs), logs)
				}
			})
		}
	}
}

func TestBlockLogger_LogFields(t *testing.T) {
	for _, source := range []string{"", "pods-reflector"} {
		t.Run(fmt.Sprintf("source=%q", source), func(t *testing.T) {
			l, logger, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{
				Threshold:       500 * time.Millisecond,
				SingleThreshold: 200 * time.Millisecond,
				Source:          source,
				ResourceType:    "*v1.Pod",
			})
			l.RecordWait(100 * time.Millisecond)
			// The 300ms spike starts the episode, and the 200ms wait makes the window exceed the threshold.
			fakeClock.Step(time.Second)
			l.RecordWait(300 * time.Millisecond)
			fakeClock.Step(time.Second)
			l.RecordWait(200 * time.Millisecond)
			if logs := capturedBlockLogs(t, logger); len(logs) != 0 {
				t.Fatalf("Expected the warning to wait for the end of the episode, got %d", len(logs))
			}

			fakeClock.Step(4 * time.Second)
			l.RecordWait(0)

			logs := capturedBlockLogs(t, logger)
			if len(logs) != 1 {
				t.Fatalf("Expected exactly one warning, got %d", len(logs))
			}
			if logs[0].verbosity != 2 {
				t.Errorf("Expected the warning at verbosity 2, got %d", logs[0].verbosity)
			}
			want := map[string]any{
				"resourceType": "*v1.Pod",
				"eventCount":   3,
				"timeWaiting":  600 * time.Millisecond,
				"avgWait":      200 * time.Millisecond,
				"maxWait":      300 * time.Millisecond,
				"window":       2 * time.Second,
			}
			if source != "" {
				want["source"] = source
			}
			if !reflect.DeepEqual(want, logs[0].values) {
				t.Errorf("Expected warning fields %v, got %v", want, logs[0].values)
			}
		})
	}
}

func TestBlockLogger_WaitsExpireAfterSlidingInterval(t *testing.T) {
	l, _, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{})
	fakeClock.Step(999 * time.Millisecond)
	l.RecordWait(time.Millisecond)

	// The wait stays counted until its bucket is older than the interval, between 4s and 5s later.
	fakeClock.SetTime(testBlockLoggerEpoch.Add(DefaultBlockLoggerInterval - time.Nanosecond))
	if got := l.TimeWaiting(); got != time.Millisecond {
		t.Fatalf("Expected the wait to be counted just before its bucket expires, got %v", got)
	}

	fakeClock.SetTime(testBlockLoggerEpoch.Add(DefaultBlockLoggerInterval))
	if got := l.TimeWaiting(); got != 0 {
		t.Fatalf("Expected the wait to expire with its bucket, got %v", got)
	}
	if got := l.EventCount(); got != 0 {
		t.Fatalf("Expected the event to expire with its bucket, got event count %d", got)
	}
}

func TestBlockLogger_Flush(t *testing.T) {
	t.Run("without a pending episode is a no-op", func(t *testing.T) {
		l, logger, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{SingleThreshold: DefaultBlockLoggerSingleThreshold})
		l.RecordWait(100 * time.Millisecond)
		fakeClock.Step(2 * time.Second)
		l.Flush()
		if logs := capturedBlockLogs(t, logger); len(logs) != 0 {
			t.Fatalf("Expected no warning, got %d", len(logs))
		}
		if l.EventCount() != 1 || l.TimeWaiting() != 100*time.Millisecond {
			t.Fatalf("Expected Flush to keep the sliding totals, got event count %d and time waiting %v", l.EventCount(), l.TimeWaiting())
		}
	})

	t.Run("reports a pending episode exactly once", func(t *testing.T) {
		l, logger, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{SingleThreshold: DefaultBlockLoggerSingleThreshold})
		fakeClock.Step(time.Second)
		l.RecordWait(300 * time.Millisecond)
		fakeClock.Step(time.Second)
		l.Flush()
		l.Flush()

		logs := capturedBlockLogs(t, logger)
		if len(logs) != 1 {
			t.Fatalf("Expected exactly one warning, got %d", len(logs))
		}
		if got := logs[0].duration(t, "window"); got != time.Second {
			t.Errorf("Expected the 1s window that held the spike, got %v", got)
		}
		if got := logs[0].duration(t, "maxWait"); got != 300*time.Millisecond {
			t.Errorf("Expected maxWait 300ms, got %v", got)
		}
		if got := logs[0].eventCount(t); got != 1 {
			t.Errorf("Expected event count 1, got %d", got)
		}

		// The reported spike is still in the sliding window, but it must not start a new episode.
		fakeClock.Step(500 * time.Millisecond)
		l.RecordWait(time.Millisecond)
		fakeClock.Step(DefaultBlockLoggerInterval)
		l.RecordWait(time.Millisecond)
		if logs := capturedBlockLogs(t, logger); len(logs) != 1 {
			t.Errorf("Expected the spike to be reported once, got %d warnings", len(logs))
		}
	})
}

func TestBlockLogger_SaturatedStreamLogsOncePerInterval(t *testing.T) {
	l, logger, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{
		Threshold:       DefaultBlockLoggerThreshold,
		SingleThreshold: DefaultBlockLoggerSingleThreshold,
	})
	const (
		duration     = 10 * time.Minute
		interArrival = 10 * time.Millisecond
	)

	// Every interval is saturated (5ms per 10ms) and contains 300ms spikes, which is the worst case for log volume.
	events := int(duration / interArrival)
	for i := 1; i <= events; i++ {
		fakeClock.Step(interArrival)
		wait := 5 * time.Millisecond
		if i%100 == 0 {
			wait = 300 * time.Millisecond
		}
		l.RecordWait(wait)
	}
	l.Flush()

	logs := capturedBlockLogs(t, logger)
	if maxLogs := int(duration/DefaultBlockLoggerInterval) + 1; len(logs) > maxLogs || len(logs) < maxLogs-2 {
		t.Fatalf("Expected one warning per %v interval (%d-%d), got %d", DefaultBlockLoggerInterval, maxLogs-2, maxLogs, len(logs))
	}
	minWindow := DefaultBlockLoggerInterval * (blockLoggerBuckets - 1) / blockLoggerBuckets
	for i, log := range logs {
		if got := log.duration(t, "window"); got < minWindow || got > DefaultBlockLoggerInterval {
			t.Errorf("Warning %d: expected a window between %v and %v, got %v", i, minWindow, DefaultBlockLoggerInterval, got)
		}
		if got := log.duration(t, "maxWait"); got != 300*time.Millisecond {
			t.Errorf("Warning %d: expected maxWait 300ms, got %v", i, got)
		}
		if got := log.duration(t, "timeWaiting"); got <= DefaultBlockLoggerThreshold {
			t.Errorf("Warning %d: expected timeWaiting above %v, got %v", i, DefaultBlockLoggerThreshold, got)
		}
	}
}

func TestBlockLogger_BurstAcrossIntervalBoundary(t *testing.T) {
	// Fully busy decoding (back-to-back 70us decodes) centered on the 5s mark used to split into two
	// sub-threshold halves with fixed windows. The sliding window must report it exactly once.
	testCases := []struct {
		burst     time.Duration
		wantWarns int
	}{
		{burst: 2400 * time.Millisecond},
		{burst: 2600 * time.Millisecond, wantWarns: 1},
		{burst: 4900 * time.Millisecond, wantWarns: 1},
		{burst: 5200 * time.Millisecond, wantWarns: 1},
	}

	for _, tc := range testCases {
		t.Run(tc.burst.String(), func(t *testing.T) {
			const decodeCost = 70 * time.Microsecond
			l, logger, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{
				Threshold:       DefaultDecodeLoggerThreshold,
				SingleThreshold: DefaultBlockLoggerSingleThreshold,
			})

			start := DefaultBlockLoggerInterval - tc.burst/2
			for next := 100 * time.Millisecond; next < start; next += 100 * time.Millisecond {
				fakeClock.SetTime(testBlockLoggerEpoch.Add(next))
				l.RecordWait(time.Microsecond)
			}
			fakeClock.SetTime(testBlockLoggerEpoch.Add(start))
			for end := fakeClock.Now().Add(tc.burst); fakeClock.Now().Before(end); {
				begin := l.Now()
				fakeClock.Step(decodeCost)
				l.RecordSince(begin)
			}
			for i := 0; i < 200; i++ {
				fakeClock.Step(100 * time.Millisecond)
				l.RecordWait(time.Microsecond)
			}
			l.Flush()

			logs := capturedBlockLogs(t, logger)
			if len(logs) != tc.wantWarns {
				t.Fatalf("Expected %d warnings, got %d: %v", tc.wantWarns, len(logs), logs)
			}
			for _, log := range logs {
				if got := log.duration(t, "timeWaiting"); got <= DefaultDecodeLoggerThreshold {
					t.Errorf("Expected timeWaiting above %v, got %v", DefaultDecodeLoggerThreshold, got)
				}
			}
		})
	}
}

func TestBlockLogger_PendingEpisodeSurvivesSilence(t *testing.T) {
	l, logger, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{SingleThreshold: DefaultBlockLoggerSingleThreshold})
	fakeClock.Step(time.Second)
	l.RecordWait(300 * time.Millisecond)

	// The spike has expired from the sliding window by the next event, but the episode must still be reported.
	fakeClock.Step(time.Minute)
	l.RecordWait(time.Millisecond)

	logs := capturedBlockLogs(t, logger)
	if len(logs) != 1 {
		t.Fatalf("Expected exactly one warning, got %d", len(logs))
	}
	if got := logs[0].duration(t, "maxWait"); got != 300*time.Millisecond {
		t.Errorf("Expected maxWait 300ms, got %v", got)
	}
}

func TestBlockLogger_RandomizedTrafficMatchesExactWindows(t *testing.T) {
	type recordedWait struct {
		end  time.Time
		wait time.Duration
	}
	lookBack := DefaultBlockLoggerInterval * (blockLoggerBuckets - 1) / blockLoggerBuckets

	for seed := uint64(1); seed <= 20; seed++ {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(seed, seed))
			l, logger, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{
				Threshold:       DefaultBlockLoggerThreshold,
				SingleThreshold: DefaultBlockLoggerSingleThreshold,
			})

			// An exact rolling window is the reference model: the bucketed total must stay between
			// the exact sums over the shortest and longest spans the buckets can cover.
			var recent []recordedWait
			for i := 0; i < 5000; i++ {
				gap := time.Duration(rng.Int64N(int64(200 * time.Millisecond)))
				if rng.IntN(50) == 0 {
					gap = time.Duration(rng.Int64N(int64(3 * DefaultBlockLoggerInterval)))
				}
				fakeClock.Step(gap)
				wait := time.Duration(rng.Int64N(int64(300 * time.Millisecond)))
				l.RecordWait(wait)

				now := fakeClock.Now()
				recent = append(recent, recordedWait{end: now, wait: wait})
				for now.Sub(recent[0].end) >= DefaultBlockLoggerInterval {
					recent = recent[1:]
				}
				var exactLookBack, exactInterval time.Duration
				for _, r := range recent {
					exactInterval += r.wait
					if now.Sub(r.end) < lookBack {
						exactLookBack += r.wait
					}
				}
				if got := l.TimeWaiting(); got < exactLookBack || got > exactInterval {
					t.Fatalf("Event %d: sliding total %v outside the exact range [%v, %v]", i, got, exactLookBack, exactInterval)
				}
			}
			elapsed := fakeClock.Since(testBlockLoggerEpoch)
			l.Flush()

			logs := capturedBlockLogs(t, logger)
			if maxLogs := int(elapsed/DefaultBlockLoggerInterval) + 1; len(logs) > maxLogs {
				t.Fatalf("Expected at most %d warnings over %v, got %d", maxLogs, elapsed, len(logs))
			}
			for i, log := range logs {
				count := log.eventCount(t)
				timeWaiting := log.duration(t, "timeWaiting")
				if window := log.duration(t, "window"); window > DefaultBlockLoggerInterval {
					t.Errorf("Warning %d: expected a window of at most %v, got %v", i, DefaultBlockLoggerInterval, window)
				}
				if got := log.duration(t, "avgWait"); got != timeWaiting/time.Duration(count) {
					t.Errorf("Warning %d: expected avgWait %v, got %v", i, timeWaiting/time.Duration(count), got)
				}
				if got := log.duration(t, "maxWait"); got > timeWaiting {
					t.Errorf("Warning %d: maxWait %v exceeds timeWaiting %v", i, got, timeWaiting)
				}
			}
		})
	}
}

func TestBlockLogger_ConcurrentRecordingKeepsTotalsConsistent(t *testing.T) {
	l, _, fakeClock := newTestBlockLogger(t, BlockLoggerOptions{SingleThreshold: DefaultBlockLoggerSingleThreshold})
	const (
		recorders         = 8
		eventsPerRecorder = 2000
		clockSteps        = 500
		flushes           = 500
	)

	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < recorders; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for j := 0; j < eventsPerRecorder; j++ {
				// RecordSince reads the clock before taking the lock, so concurrent waits can arrive out of order.
				l.RecordSince(l.Now().Add(-time.Millisecond))
			}
		}()
	}
	// The clock driver rotates buckets mid-flight, and the flusher mimics Decoder.Close racing with the watch goroutine.
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		for j := 0; j < clockSteps; j++ {
			fakeClock.Step(100 * time.Millisecond)
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		for j := 0; j < flushes; j++ {
			l.Flush()
		}
	}()
	close(start)
	wg.Wait()

	l.mu.Lock()
	var bucketWait time.Duration
	var bucketCount int
	for _, b := range l.buckets {
		bucketWait += b.timeWaiting
		bucketCount += b.eventCount
	}
	if l.timeWaiting != bucketWait || l.eventCount != bucketCount {
		t.Errorf("Expected totals %v/%d to match the buckets %v/%d", l.timeWaiting, l.eventCount, bucketWait, bucketCount)
	}
	l.mu.Unlock()

	// Rotating every bucket out one at a time must bring the running totals back to exactly zero.
	for i := 0; i < blockLoggerBuckets; i++ {
		fakeClock.Step(DefaultBlockLoggerInterval / blockLoggerBuckets)
		l.TimeWaiting()
	}
	if l.TimeWaiting() != 0 || l.EventCount() != 0 {
		t.Errorf("Expected empty totals after every bucket expired, got %v/%d", l.TimeWaiting(), l.EventCount())
	}
}

func TestBlockLogger_ResourceType(t *testing.T) {
	l, _, _ := newTestBlockLogger(t, BlockLoggerOptions{ResourceType: "*v1.Pod"})

	steps := []struct {
		action func()
		want   string
	}{
		{action: func() { l.EnsureResourceType(&blockLoggerTestNode{}) }, want: "*v1.Pod"},
		{action: func() { l.SetResourceType("") }, want: ""},
		{action: func() { l.EnsureResourceType(nil) }, want: ""},
		{action: func() { l.EnsureResourceType(&blockLoggerTestNode{}) }, want: "*watch.blockLoggerTestNode"},
		{action: func() { l.EnsureResourceType(&blockLoggerTestPod{}) }, want: "*watch.blockLoggerTestNode"},
		{action: func() { l.SetResourceType("custom") }, want: "custom"},
	}
	for i, step := range steps {
		step.action()
		if got := l.ResourceType(); got != step.want {
			t.Errorf("Step %d: expected resource type %q, got %q", i, step.want, got)
		}
	}
}
