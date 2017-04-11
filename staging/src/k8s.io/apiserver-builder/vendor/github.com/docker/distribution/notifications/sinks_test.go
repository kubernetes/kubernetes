package notifications

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/Sirupsen/logrus"

	"testing"
)

func TestBroadcaster(t *testing.T) {
	const nEvents = 1000
	var sinks []Sink

	for i := 0; i < 10; i++ {
		sinks = append(sinks, &testSink{})
	}

	b := NewBroadcaster(sinks...)

	var block []Event
	var wg sync.WaitGroup
	for i := 1; i <= nEvents; i++ {
		block = append(block, createTestEvent("push", "library/test", "blob"))

		if i%10 == 0 && i > 0 {
			wg.Add(1)
			go func(block ...Event) {
				if err := b.Write(block...); err != nil {
					t.Fatalf("error writing block of length %d: %v", len(block), err)
				}
				wg.Done()
			}(block...)

			block = nil
		}
	}

	wg.Wait() // Wait until writes complete
	checkClose(t, b)

	// Iterate through the sinks and check that they all have the expected length.
	for _, sink := range sinks {
		ts := sink.(*testSink)
		ts.mu.Lock()
		defer ts.mu.Unlock()

		if len(ts.events) != nEvents {
			t.Fatalf("not all events ended up in testsink: len(testSink) == %d, not %d", len(ts.events), nEvents)
		}

		if !ts.closed {
			t.Fatalf("sink should have been closed")
		}
	}

}

func TestEventQueue(t *testing.T) {
	const nevents = 1000
	var ts testSink
	metrics := newSafeMetrics()
	eq := newEventQueue(
		// delayed sync simulates destination slower than channel comms
		&delayedSink{
			Sink:  &ts,
			delay: time.Millisecond * 1,
		}, metrics.eventQueueListener())

	var wg sync.WaitGroup
	var block []Event
	for i := 1; i <= nevents; i++ {
		block = append(block, createTestEvent("push", "library/test", "blob"))
		if i%10 == 0 && i > 0 {
			wg.Add(1)
			go func(block ...Event) {
				if err := eq.Write(block...); err != nil {
					t.Fatalf("error writing event block: %v", err)
				}
				wg.Done()
			}(block...)

			block = nil
		}
	}

	wg.Wait()
	checkClose(t, eq)

	ts.mu.Lock()
	defer ts.mu.Unlock()
	metrics.Lock()
	defer metrics.Unlock()

	if len(ts.events) != nevents {
		t.Fatalf("events did not make it to the sink: %d != %d", len(ts.events), 1000)
	}

	if !ts.closed {
		t.Fatalf("sink should have been closed")
	}

	if metrics.Events != nevents {
		t.Fatalf("unexpected ingress count: %d != %d", metrics.Events, nevents)
	}

	if metrics.Pending != 0 {
		t.Fatalf("unexpected egress count: %d != %d", metrics.Pending, 0)
	}
}

func TestRetryingSink(t *testing.T) {

	// Make a sync that fails most of the time, ensuring that all the events
	// make it through.
	var ts testSink
	flaky := &flakySink{
		rate: 1.0, // start out always failing.
		Sink: &ts,
	}
	s := newRetryingSink(flaky, 3, 10*time.Millisecond)

	var wg sync.WaitGroup
	var block []Event
	for i := 1; i <= 100; i++ {
		block = append(block, createTestEvent("push", "library/test", "blob"))

		// Above 50, set the failure rate lower
		if i > 50 {
			s.mu.Lock()
			flaky.rate = 0.90
			s.mu.Unlock()
		}

		if i%10 == 0 && i > 0 {
			wg.Add(1)
			go func(block ...Event) {
				defer wg.Done()
				if err := s.Write(block...); err != nil {
					t.Fatalf("error writing event block: %v", err)
				}
			}(block...)

			block = nil
		}
	}

	wg.Wait()
	checkClose(t, s)

	ts.mu.Lock()
	defer ts.mu.Unlock()

	if len(ts.events) != 100 {
		t.Fatalf("events not propagated: %d != %d", len(ts.events), 100)
	}
}

type testSink struct {
	events []Event
	mu     sync.Mutex
	closed bool
}

func (ts *testSink) Write(events ...Event) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	ts.events = append(ts.events, events...)
	return nil
}

func (ts *testSink) Close() error {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	ts.closed = true

	logrus.Infof("closing testSink")
	return nil
}

type delayedSink struct {
	Sink
	delay time.Duration
}

func (ds *delayedSink) Write(events ...Event) error {
	time.Sleep(ds.delay)
	return ds.Sink.Write(events...)
}

type flakySink struct {
	Sink
	rate float64
}

func (fs *flakySink) Write(events ...Event) error {
	if rand.Float64() < fs.rate {
		return fmt.Errorf("error writing %d events", len(events))
	}

	return fs.Sink.Write(events...)
}

func checkClose(t *testing.T, sink Sink) {
	if err := sink.Close(); err != nil {
		t.Fatalf("unexpected error closing: %v", err)
	}

	// second close should not crash but should return an error.
	if err := sink.Close(); err == nil {
		t.Fatalf("no error on double close")
	}

	// Write after closed should be an error
	if err := sink.Write([]Event{}...); err == nil {
		t.Fatalf("write after closed did not have an error")
	} else if err != ErrSinkClosed {
		t.Fatalf("error should be ErrSinkClosed")
	}
}
