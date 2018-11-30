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
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/plugin/pkg/audit/fake"
)

var (
	infiniteTimeCh <-chan time.Time
)

func newEvents(number int) []*auditinternal.Event {
	events := make([]*auditinternal.Event, number)
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	return events
}

func testBatchConfig() BatchConfig {
	return BatchConfig{
		BufferSize:     100,
		MaxBatchSize:   10,
		MaxBatchWait:   wait.ForeverTestTimeout,
		ThrottleEnable: false,
		AsyncDelegate:  true,
	}
}

func TestBatchedBackendCollectEvents(t *testing.T) {
	config := testBatchConfig()
	batchSize := config.MaxBatchSize
	backend := NewBackend(&fake.Backend{}, config).(*bufferedBackend)

	t.Log("Max batch size encountered.")
	backend.ProcessEvents(newEvents(batchSize + 1)...)
	batch := backend.collectEvents(nil, nil)
	assert.Len(t, batch, batchSize, "Expected full batch")

	t.Log("Partial batch should hang until timer expires.")
	backend.ProcessEvents(newEvents(1)...)
	tc := make(chan time.Time)
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		batch = backend.collectEvents(tc, nil)
	}()
	// Wait for the queued events to be collected.
	err := wait.Poll(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		return len(backend.buffer) == 0, nil
	})
	require.NoError(t, err)

	tc <- time.Now() // Trigger "timeout"
	wg.Wait()
	assert.Len(t, batch, 2, "Expected partial batch")

	t.Log("Collected events should be delivered when stop channel is closed.")
	backend.ProcessEvents(newEvents(3)...)
	stopCh := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		batch = backend.collectEvents(nil, stopCh)
	}()
	// Wait for the queued events to be collected.
	err = wait.Poll(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		return len(backend.buffer) == 0, nil
	})
	require.NoError(t, err)

	close(stopCh)
	wg.Wait()
	assert.Len(t, batch, 3, "Expected partial batch")
}

func TestUnbatchedBackendCollectEvents(t *testing.T) {
	config := testBatchConfig()
	config.MaxBatchSize = 1 // No batching.
	backend := NewBackend(&fake.Backend{}, config).(*bufferedBackend)

	t.Log("Max batch size encountered.")
	backend.ProcessEvents(newEvents(3)...)
	batch := backend.collectEvents(nil, nil)
	assert.Len(t, batch, 1, "Expected single event")

	t.Log("Queue should always be drained.")
	for len(backend.buffer) > 0 {
		batch = backend.collectEvents(nil, nil)
		assert.Len(t, batch, 1, "Expected single event")
	}

	t.Log("Collection should hault when stop channel is closed.")
	stopCh := make(chan struct{})
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		batch = backend.collectEvents(nil, stopCh)
	}()
	close(stopCh)
	wg.Wait()
	assert.Empty(t, batch, "Empty final batch")
}

func TestBufferedBackendProcessEventsAfterStop(t *testing.T) {
	t.Parallel()

	backend := NewBackend(&fake.Backend{}, testBatchConfig()).(*bufferedBackend)

	closedStopCh := make(chan struct{})
	close(closedStopCh)
	backend.Run(closedStopCh)
	backend.Shutdown()
	backend.ProcessEvents(newEvents(1)...)
	batch := backend.collectEvents(infiniteTimeCh, wait.NeverStop)

	require.Empty(t, batch, "processed events after the backed has been stopped")
}

func TestBufferedBackendProcessEventsBufferFull(t *testing.T) {
	t.Parallel()

	config := testBatchConfig()
	config.BufferSize = 1
	backend := NewBackend(&fake.Backend{}, config).(*bufferedBackend)

	backend.ProcessEvents(newEvents(2)...)

	require.Len(t, backend.buffer, 1, "buffed contains more elements than it should")
}

func TestBufferedBackendShutdownWaitsForDelegatedCalls(t *testing.T) {
	t.Parallel()

	delegatedCallStartCh := make(chan struct{})
	delegatedCallEndCh := make(chan struct{})
	delegateBackend := &fake.Backend{
		OnRequest: func(_ []*auditinternal.Event) {
			close(delegatedCallStartCh)
			<-delegatedCallEndCh
		},
	}
	config := testBatchConfig()
	backend := NewBackend(delegateBackend, config)

	// Run backend, process events, wait for them to be batched and for delegated call to start.
	stopCh := make(chan struct{})
	backend.Run(stopCh)
	backend.ProcessEvents(newEvents(config.MaxBatchSize)...)
	<-delegatedCallStartCh

	// Start shutdown procedure.
	shutdownEndCh := make(chan struct{})
	go func() {
		close(stopCh)
		backend.Shutdown()
		close(shutdownEndCh)
	}()

	// Wait for some time and then check whether Shutdown has exited. Can give false positive,
	// but never false negative.
	time.Sleep(100 * time.Millisecond)
	select {
	case <-shutdownEndCh:
		t.Fatalf("Shutdown exited before delegated call ended")
	default:
	}

	// Wait for Shutdown to exit after delegated call has exited.
	close(delegatedCallEndCh)
	<-shutdownEndCh
}

func TestDelegateProcessEvents(t *testing.T) {
	for _, async := range []bool{true, false} {
		t.Run(fmt.Sprintf("async:%t", async), func(t *testing.T) {
			config := testBatchConfig()
			config.AsyncDelegate = async
			wg := sync.WaitGroup{}
			delegate := &fake.Backend{
				OnRequest: func(events []*auditinternal.Event) {
					assert.Len(t, events, config.MaxBatchSize, "Unexpected batch")
					wg.Done()
				},
			}
			b := NewBackend(delegate, config).(*bufferedBackend)
			wg.Add(5)
			for i := 0; i < 5; i++ {
				b.processEvents(newEvents(config.MaxBatchSize))
			}
			wg.Wait()
		})
	}
}
