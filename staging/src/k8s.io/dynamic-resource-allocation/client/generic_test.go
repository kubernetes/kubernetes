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

package client

import (
	"context"
	"testing"
	"time"

	resourcev1 "k8s.io/api/resource/v1"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
)

// stalledWatch is an upstream watch whose result channel stays open after Stop.
// The wrapper must terminate on its own stopChan instead of relying on the
// upstream closing its channel.
type stalledWatch struct {
	ch chan watch.Event
}

func (s *stalledWatch) Stop() {}

func (s *stalledWatch) ResultChan() <-chan watch.Event { return s.ch }

// TestWatchSomethingStopTerminatesRun verifies that stopping the watch makes run
// return, which closes the result channel that consumers of watch.Interface wait
// for. A plain break inside the select would only leave the select, leaving run to
// spin on the upstream watch, converting and discarding events nobody can receive.
func TestWatchSomethingStopTerminatesRun(t *testing.T) {
	const queued = 5

	upstream := &stalledWatch{ch: make(chan watch.Event, queued)}
	for range queued {
		upstream.ch <- watch.Event{
			Type:   watch.Added,
			Object: &resourcev1.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "slice"}},
		}
	}

	w := &watchSomething[*resourcev1.ResourceSlice, resourcev1.ResourceSlice, *resourcev1beta1.ResourceSlice]{
		upstream:   upstream,
		resultChan: make(chan watch.Event),
		stopChan:   make(chan struct{}),
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		w.run()
	}()

	// The consumer never reads from w.ResultChan. Wait until run has taken the
	// first event off the upstream channel and blocked trying to deliver it, so
	// that the remaining count below is deterministic.
	ctx := context.Background()
	if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		return len(upstream.ch) == queued-1, nil
	}); err != nil {
		t.Fatalf("timed out waiting for the watch to start delivering: %v", err)
	}

	w.Stop()

	select {
	case <-done:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("run did not return after Stop")
	}

	if _, open := <-w.ResultChan(); open {
		t.Error("expected the result channel to be closed after Stop")
	}

	if remaining := len(upstream.ch); remaining != queued-1 {
		t.Errorf("expected the upstream watch to still hold %d events after Stop, it holds %d", queued-1, remaining)
	}
}
