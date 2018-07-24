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
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/plugin/pkg/audit/fake"
)

var (
	closedStopCh = func() <-chan struct{} {
		ch := make(chan struct{})
		close(ch)
		return ch
	}()
	infiniteTimeCh <-chan time.Time = make(chan time.Time)
	closedTimeCh                    = func() <-chan time.Time {
		ch := make(chan time.Time)
		close(ch)
		return ch
	}()
)

func newEvents(number int) []*auditinternal.Event {
	events := make([]*auditinternal.Event, number)
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	return events
}

func TestBufferedBackendCollectEvents(t *testing.T) {
	config := NewDefaultBatchConfig()

	testCases := []struct {
		desc          string
		timer         <-chan time.Time
		stopCh        <-chan struct{}
		numEvents     int
		wantBatchSize int
	}{
		{
			desc:          "max batch size encountered",
			timer:         infiniteTimeCh,
			stopCh:        wait.NeverStop,
			numEvents:     config.MaxBatchSize + 1,
			wantBatchSize: config.MaxBatchSize,
		},
		{
			desc:   "timer expired",
			timer:  closedTimeCh,
			stopCh: wait.NeverStop,
		},
		{
			desc:   "chanel closed",
			timer:  infiniteTimeCh,
			stopCh: closedStopCh,
		},
	}
	for _, tc := range testCases {
		tc := tc
		t.Run(tc.desc, func(t *testing.T) {
			t.Parallel()

			backend := NewBackend(&fake.Backend{}, config).(*bufferedBackend)

			backend.ProcessEvents(newEvents(tc.numEvents)...)
			batch := backend.collectEvents(tc.timer, tc.stopCh)

			require.Equal(t, tc.wantBatchSize, len(batch), "unexpected batch size")
		})
	}
}

func TestBufferedBackendProcessEventsAfterStop(t *testing.T) {
	t.Parallel()

	backend := NewBackend(&fake.Backend{}, NewDefaultBatchConfig()).(*bufferedBackend)

	backend.Run(closedStopCh)
	backend.Shutdown()
	backend.ProcessEvents(newEvents(1)...)
	batch := backend.collectEvents(infiniteTimeCh, wait.NeverStop)

	require.Equal(t, 0, len(batch), "processed events after the backed has been stopped")
}

func TestBufferedBackendProcessEventsBufferFull(t *testing.T) {
	t.Parallel()

	config := NewDefaultBatchConfig()
	config.BufferSize = 1
	backend := NewBackend(&fake.Backend{}, config).(*bufferedBackend)

	backend.ProcessEvents(newEvents(2)...)

	require.Equal(t, 1, len(backend.buffer), "buffed contains more elements than it should")
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
	config := NewDefaultBatchConfig()
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
