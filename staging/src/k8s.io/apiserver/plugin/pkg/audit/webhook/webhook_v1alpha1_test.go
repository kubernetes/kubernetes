/*
Copyright 2017 The Kubernetes Authors.

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

package webhook

import (
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1alpha1 "k8s.io/apiserver/pkg/apis/audit/v1alpha1"
)

func TestBatchWebhookMaxEventsV1Alpha1(t *testing.T) {
	nRest := 10
	events := make([]*auditinternal.Event, defaultBatchMaxSize+nRest) // greater than max size.
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	got := make(chan int, 2)
	s := httptest.NewServer(newWebhookHandler(t, &auditv1alpha1.EventList{}, func(events runtime.Object) {
		got <- len(events.(*auditv1alpha1.EventList).Items)
	}))
	defer s.Close()

	backend := newTestBatchWebhook(t, s.URL, []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion})

	backend.ProcessEvents(events...)

	stopCh := make(chan struct{})
	timer := make(chan time.Time, 1)

	backend.sendBatchEvents(backend.collectEvents(stopCh, timer))
	require.Equal(t, defaultBatchMaxSize, <-got, "did not get batch max size")

	go func() {
		waitForEmptyBuffer(backend) // wait for the buffer to empty
		timer <- time.Now()         // Trigger the wait timeout
	}()

	backend.sendBatchEvents(backend.collectEvents(stopCh, timer))
	require.Equal(t, nRest, <-got, "failed to get the rest of the events")
}

func TestBatchWebhookStopChV1Alpha1(t *testing.T) {
	events := make([]*auditinternal.Event, 1) // less than max size.
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	expected := len(events)
	got := make(chan int, 2)
	s := httptest.NewServer(newWebhookHandler(t, &auditv1alpha1.EventList{}, func(events runtime.Object) {
		got <- len(events.(*auditv1alpha1.EventList).Items)
	}))
	defer s.Close()

	backend := newTestBatchWebhook(t, s.URL, []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion})
	backend.ProcessEvents(events...)

	stopCh := make(chan struct{})
	timer := make(chan time.Time)

	go func() {
		waitForEmptyBuffer(backend)
		close(stopCh) // stop channel has stopped
	}()
	backend.sendBatchEvents(backend.collectEvents(stopCh, timer))
	require.Equal(t, expected, <-got, "get queued events after timer expires")
}

func TestBatchWebhookProcessEventsAfterStopV1Alpha1(t *testing.T) {
	events := make([]*auditinternal.Event, 1) // less than max size.
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	got := make(chan struct{})
	s := httptest.NewServer(newWebhookHandler(t, &auditv1alpha1.EventList{}, func(events runtime.Object) {
		close(got)
	}))
	defer s.Close()

	backend := newTestBatchWebhook(t, s.URL, []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion})
	stopCh := make(chan struct{})

	backend.Run(stopCh)
	close(stopCh)
	<-backend.shutdownCh
	backend.ProcessEvents(events...)
	assert.Equal(t, 0, len(backend.buffer), "processed events after the backed has been stopped")
}

func TestBatchWebhookShutdownV1Alpha1(t *testing.T) {
	events := make([]*auditinternal.Event, 1)
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	got := make(chan struct{})
	contReqCh := make(chan struct{})
	shutdownCh := make(chan struct{})
	s := httptest.NewServer(newWebhookHandler(t, &auditv1alpha1.EventList{}, func(events runtime.Object) {
		close(got)
		<-contReqCh
	}))
	defer s.Close()

	backend := newTestBatchWebhook(t, s.URL, []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion})
	backend.ProcessEvents(events...)

	go func() {
		// Assume stopCh was closed.
		close(backend.buffer)
		backend.sendBatchEvents(backend.collectLastEvents())
	}()

	<-got

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

	close(contReqCh)
	<-shutdownCh
}

func TestBatchWebhookEmptyBufferV1Alpha1(t *testing.T) {
	events := make([]*auditinternal.Event, 1) // less than max size.
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	expected := len(events)
	got := make(chan int, 2)
	s := httptest.NewServer(newWebhookHandler(t, &auditv1alpha1.EventList{}, func(events runtime.Object) {
		got <- len(events.(*auditv1alpha1.EventList).Items)
	}))
	defer s.Close()

	backend := newTestBatchWebhook(t, s.URL, []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion})

	stopCh := make(chan struct{})
	timer := make(chan time.Time, 1)

	timer <- time.Now() // Timer is done.

	// Buffer is empty, no events have been queued. This should exit but send no events.
	backend.sendBatchEvents(backend.collectEvents(stopCh, timer))

	// Send additional events after the sendBatchEvents has been called.
	backend.ProcessEvents(events...)
	go func() {
		waitForEmptyBuffer(backend)
		timer <- time.Now()
	}()

	backend.sendBatchEvents(backend.collectEvents(stopCh, timer))

	// Make sure we didn't get a POST with zero events.
	require.Equal(t, expected, <-got, "expected one event")
}

func TestBatchBufferFullV1Alpha1(t *testing.T) {
	events := make([]*auditinternal.Event, defaultBatchBufferSize+1) // More than buffered size
	for i := range events {
		events[i] = &auditinternal.Event{}
	}
	s := httptest.NewServer(newWebhookHandler(t, &auditv1alpha1.EventList{}, func(events runtime.Object) {
		// Do nothing.
	}))
	defer s.Close()

	backend := newTestBatchWebhook(t, s.URL, []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion})

	// Make sure this doesn't block.
	backend.ProcessEvents(events...)
}

func TestBatchRunV1Alpha1(t *testing.T) {

	// Divisable by max batch size so we don't have to wait for a minute for
	// the test to finish.
	events := make([]*auditinternal.Event, defaultBatchMaxSize*3)
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	got := new(int64)
	want := len(events)

	wg := new(sync.WaitGroup)
	wg.Add(want)
	done := make(chan struct{})

	go func() {
		wg.Wait()
		// When the expected number of events have been received, close the channel.
		close(done)
	}()

	s := httptest.NewServer(newWebhookHandler(t, &auditv1alpha1.EventList{}, func(obj runtime.Object) {
		events := obj.(*auditv1alpha1.EventList)
		atomic.AddInt64(got, int64(len(events.Items)))
		wg.Add(-len(events.Items))
	}))
	defer s.Close()

	stopCh := make(chan struct{})
	defer close(stopCh)

	backend := newTestBatchWebhook(t, s.URL, []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion})

	// Test the Run codepath. E.g. that the spawned goroutines behave correctly.
	backend.Run(stopCh)

	backend.ProcessEvents(events...)

	select {
	case <-done:
		// Received all the events.
	case <-time.After(2 * time.Minute):
		t.Errorf("expected %d events got %d", want, atomic.LoadInt64(got))
	}
}

func TestBatchConcurrentRequestsV1Alpha1(t *testing.T) {
	events := make([]*auditinternal.Event, defaultBatchBufferSize) // Don't drop events
	for i := range events {
		events[i] = &auditinternal.Event{}
	}

	wg := new(sync.WaitGroup)
	wg.Add(len(events))

	s := httptest.NewServer(newWebhookHandler(t, &auditv1alpha1.EventList{}, func(events runtime.Object) {
		wg.Add(-len(events.(*auditv1alpha1.EventList).Items))

		// Since the webhook makes concurrent requests, blocking on the webhook response
		// shouldn't block the webhook from sending more events.
		//
		// Wait for all responses to be received before sending the response.
		wg.Wait()
	}))
	defer s.Close()

	stopCh := make(chan struct{})
	defer close(stopCh)

	backend := newTestBatchWebhook(t, s.URL, []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion})
	backend.Run(stopCh)

	backend.ProcessEvents(events...)
	// Wait for the webhook to receive all events.
	wg.Wait()
}
