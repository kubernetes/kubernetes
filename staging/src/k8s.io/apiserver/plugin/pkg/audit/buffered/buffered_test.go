/*
Copyright 2018 The Kubernetes Authors.

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

package buffered

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

type options struct {
	withWaitGroup bool
	withChannel   bool
}

func newBufferedBackend(opts options) (*bufferedBackend, *fakeBackend) {
	fakeBackend := &fakeBackend{}
	switch {
	case opts.withChannel == true:
		fakeBackend.ch = make(chan struct{})
	case opts.withWaitGroup == true:
		fakeBackend.wg = new(sync.WaitGroup)
	default:
	}

	config := NewDefaultBatchConfig()
	backend := NewBackend(fakeBackend, config)
	return backend.(*bufferedBackend), fakeBackend
}

func newEvents(number int) []*auditinternal.Event {
	events := make([]*auditinternal.Event, number)
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	return events
}

// waitForEmptyBuffer indicates when the sendBatchEvents method has read from the
// existing buffer. This lets test coordinate closing a timer and stop channel
// until the for loop has read from the buffer.
func waitForEmptyBuffer(b *bufferedBackend) {
	for len(b.buffer) != 0 {
		time.Sleep(time.Millisecond)
	}
}

func TestBufferedBackendBatchMaxEvents(t *testing.T) {
	stopCh := make(chan struct{})
	timer := make(chan time.Time)

	backend, delegateBackend := newBufferedBackend(options{})

	events := newEvents(backend.maxBatchSize + 1) // greater than max size.

	backend.ProcessEvents(events...)

	delegateBackend.ProcessEvents(backend.collectEvents(timer, stopCh)...)
	require.Equal(t, 1, int(atomic.LoadInt32(&delegateBackend.got)), "did not get batch max size")

	go func() {
		waitForEmptyBuffer(backend) // wait for the buffer to empty
		close(timer)                // Trigger the wait timeout
	}()

	delegateBackend.ProcessEvents(backend.collectEvents(timer, stopCh)...)
	require.Equal(t, 2, int(atomic.LoadInt32(&delegateBackend.got)), "failed to get the rest of the events")
}

func TestBufferedBackendStopCh(t *testing.T) {
	events := newEvents(1) // less than max size.
	stopCh := make(chan struct{})
	timer := make(chan time.Time)

	backend, delegateBackend := newBufferedBackend(options{})

	backend.ProcessEvents(events...)

	go func() {
		waitForEmptyBuffer(backend)
		close(stopCh) // stop channel has stopped
	}()

	delegateBackend.ProcessEvents(backend.collectEvents(timer, stopCh)...)
	require.Equal(t, 1, int(atomic.LoadInt32(&delegateBackend.got)), "get queued events after timer expires")
}

func TestBufferedBackendProcessEventsAfterStop(t *testing.T) {
	events := newEvents(1) // less than max size.

	backend, _ := newBufferedBackend(options{})
	stopCh := make(chan struct{})

	backend.Run(stopCh)
	close(stopCh)
	<-backend.shutdownCh
	backend.ProcessEvents(events...)
	assert.Equal(t, 0, len(backend.buffer), "processed events after the backed has been stopped")
}

func TestBufferedBackendShutdown(t *testing.T) {
	shutdownCh := make(chan struct{})
	backend, delegateBackend := newBufferedBackend(options{withChannel: true})
	events := newEvents(1)
	backend.ProcessEvents(events...)

	go func() {
		// Assume stopCh was closed.
		close(backend.buffer)
		backend.processEvents(backend.collectEvents(make(chan time.Time), make(chan struct{})))
	}()

	for atomic.LoadInt32(&delegateBackend.got) == 0 {
		time.Sleep(time.Millisecond)
	}

	go func() {
		close(backend.shutdownCh)
		backend.Shutdown()
		close(shutdownCh)
	}()

	// Wait for some time in case there's a bug that allows for the Shutdown
	// method to exit before all requests has been completed.
	time.Sleep(1 * time.Second)
	select {
	case <-shutdownCh:
		t.Fatal("Backend shut down before all requests finished")
	default:
		// Continue.
	}

	close(delegateBackend.ch)
	<-shutdownCh
}

func TestBufferedBackendEmptyBuffer(t *testing.T) {
	events := newEvents(1) // less than max size.
	stopCh := make(chan struct{})
	timer := make(chan time.Time, 1)

	backend, delegateBackend := newBufferedBackend(options{})

	timer <- time.Now() // Timer is done.

	// Buffer is empty, no events have been queued. This should exit but send no events.
	backend.processEvents(backend.collectEvents(timer, stopCh))

	// Send additional events after the sendBatchEvents has been called.
	backend.ProcessEvents(events...)
	go func() {
		waitForEmptyBuffer(backend)
		timer <- time.Now()
	}()

	delegateBackend.ProcessEvents(backend.collectEvents(timer, stopCh)...)

	require.Equal(t, 1, int(atomic.LoadInt32(&delegateBackend.got)), "expected one batch")
}

func TestBufferedBackendBufferFull(t *testing.T) {
	backend, _ := newBufferedBackend(options{})
	events := newEvents(len(backend.buffer) + 1) // More than buffered size

	// Make sure this doesn't block.
	backend.ProcessEvents(events...)
}

// Test BufferedBackend.Run
func TestBufferedBackendRun(t *testing.T) {
	done := make(chan struct{})
	backend, delegateBackend := newBufferedBackend(options{withWaitGroup: true})
	events := newEvents(backend.maxBatchSize * 3)
	want := 3

	delegateBackend.wg.Add(want)

	go func() {
		delegateBackend.wg.Wait()
		// When the expected number of events have been received, close the channel.
		close(done)
	}()

	stopCh := make(chan struct{})
	defer close(stopCh)

	// Test the Run codepath. E.g. that the spawned goroutines behave correctly.
	backend.Run(stopCh)

	backend.ProcessEvents(events...)

	select {
	case <-done:
		// Received all the events.
	case <-time.After(2 * time.Minute):
		t.Errorf("expected %d batch events got %d", want, atomic.LoadInt32(&delegateBackend.got))
	}
}

// Test concurrently processing events
func TestBufferedBackendConcurrentRequests(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	backend, delegateBackend := newBufferedBackend(options{withWaitGroup: true})
	events := newEvents(len(backend.buffer))

	delegateBackend.wg.Add(len(events) / backend.maxBatchSize)

	backend.Run(stopCh)

	go backend.ProcessEvents(events[:len(events)/2]...)
	go backend.ProcessEvents(events[len(events)/2:]...)
	// Wait for the underlying backend to receive all events.
	delegateBackend.wg.Wait()
}

type fakeBackend struct {
	wg  *sync.WaitGroup
	got int32
	ch  chan struct{} // used for test shutdown
}

var _ audit.Backend = &fakeBackend{}

func (b *fakeBackend) Run(stopCh <-chan struct{}) error {
	return nil
}

func (b *fakeBackend) Shutdown() {
	// nothing to do here
	return
}

func (b *fakeBackend) ProcessEvents(ev ...*auditinternal.Event) {
	atomic.AddInt32(&b.got, 1)
	if b.wg != nil {
		b.wg.Done()
	}
	if b.ch != nil {
		<-b.ch
	}
}
