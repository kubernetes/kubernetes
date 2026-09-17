/*
Copyright 2015 The Kubernetes Authors.

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

package results

import (
	"fmt"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestCacheOperations(t *testing.T) {
	m := NewManager()

	unsetID := kubecontainer.ContainerID{Type: "test", ID: "unset"}
	setID := kubecontainer.ContainerID{Type: "test", ID: "set"}

	_, found := m.Get(unsetID)
	assert.False(t, found, "unset result found")

	m.Set(setID, Success, &corev1.Pod{})
	result, found := m.Get(setID)
	assert.Equal(t, Success, result, "set result")
	assert.True(t, found, "set result found")

	m.Remove(setID)
	_, found = m.Get(setID)
	assert.False(t, found, "removed result found")
}

func TestMutableUpdates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	m := NewManager()

	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-pod"}}
	fooID := kubecontainer.ContainerID{Type: "test", ID: "foo"}
	barID := kubecontainer.ContainerID{Type: "test", ID: "bar"}

	expectUpdate := func(expected Update, msg string) {
		select {
		case u := <-m.Updates():
			if expected != u {
				t.Errorf("Expected update %v, received %v: %s", expected, u, msg)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Timed out waiting for update %v: %s", expected, msg)
		}
	}

	expectNoUpdate := func(msg string) {
		// NOTE: Since updates are accumulated asynchronously, this method is not guaranteed to fail
		// when it should. In the event it misses a failure, the following calls to expectUpdate should
		// still fail.
		select {
		case u := <-m.Updates():
			t.Errorf("Unexpected update %v: %s", u, msg)
		default:
			// Pass
		}
	}

	// New result should always push an update.
	m.Set(fooID, Success, pod)
	expectUpdate(Update{ContainerID: fooID, Result: Success, PodUID: pod.UID, Version: 1}, "new success")

	m.Set(barID, Failure, pod)
	expectUpdate(Update{ContainerID: barID, Result: Failure, PodUID: pod.UID, Version: 2}, "new failure")

	// Unchanged results should not send an update.
	m.Set(fooID, Success, pod)
	expectNoUpdate("unchanged foo")

	m.Set(barID, Failure, pod)
	expectNoUpdate("unchanged bar")

	// Changed results should send an update.
	m.Set(fooID, Failure, pod)
	expectUpdate(Update{ContainerID: fooID, Result: Failure, PodUID: pod.UID, Version: 3}, "changed foo")

	m.Set(barID, Success, pod)
	expectUpdate(Update{ContainerID: barID, Result: Success, PodUID: pod.UID, Version: 4}, "changed bar")

	// Re-enabling a probe must publish its new identity even when the health result is unchanged.
	metadata := Metadata{ContainerName: "bar", ProbeID: 2}
	m.SetWithMetadata(barID, Success, pod, metadata)
	expectUpdate(Update{
		ContainerID:   barID,
		Result:        Success,
		PodUID:        pod.UID,
		ContainerName: metadata.ContainerName,
		ProbeID:       metadata.ProbeID,
		Version:       5,
	}, "new probe instance")
}

func TestIsCurrent(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	m := NewManager()
	id := kubecontainer.ContainerID{Type: "test", ID: "container"}
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod"}}
	metadata := Metadata{ContainerName: "container", ProbeID: 1}
	t.Cleanup(func() { m.Remove(id) })

	m.SetWithMetadata(id, Success, pod, metadata)
	first := receiveUpdate(t, m)

	assert.True(t, m.IsCurrent(first))

	m.SetWithMetadata(id, Failure, pod, metadata)
	second := receiveUpdate(t, m)

	assert.False(t, m.IsCurrent(first))
	assert.True(t, m.IsCurrent(second))

	// Returning to the same value must not make an earlier notification current again.
	m.SetWithMetadata(id, Success, pod, metadata)
	third := receiveUpdate(t, m)

	assert.False(t, m.IsCurrent(first))
	assert.False(t, m.IsCurrent(second))
	assert.True(t, m.IsCurrent(third))
	assert.Greater(t, third.Version, first.Version)

	// Publishing an unchanged result preserves the existing publication identity.
	m.SetWithMetadata(id, Success, pod, metadata)

	assert.True(t, m.IsCurrent(third))

	for name, change := range map[string]func(*Update){
		"container ID":   func(u *Update) { u.ContainerID.ID = "other" },
		"result":         func(u *Update) { u.Result = Failure },
		"pod UID":        func(u *Update) { u.PodUID = "other" },
		"container name": func(u *Update) { u.ContainerName = "other" },
		"probe instance": func(u *Update) { u.ProbeID++ },
		"version":        func(u *Update) { u.Version++ },
	} {
		t.Run(name, func(t *testing.T) {
			update := third
			change(&update)

			assert.False(t, m.IsCurrent(update))
		})
	}
}

func TestUpdatesDoNotBlockPublishers(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	m := NewManager().(*mutableManager)
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod"}}
	const count = 100
	ids := make([]kubecontainer.ContainerID, count)
	for i := range ids {
		ids[i] = kubecontainer.ContainerID{Type: "test", ID: fmt.Sprintf("container-%d", i)}
	}

	t.Cleanup(func() {
		for _, id := range ids {
			m.Remove(id)
		}

		waitForDispatcherIdle(t, m)
	})

	done := make(chan struct{})
	go func() {
		defer close(done)

		for _, id := range ids {
			m.SetWithMetadata(id, Success, pod, Metadata{ContainerName: id.ID, ProbeID: 1})
		}
	}()
	select {
	case <-done:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("publishing results blocked while the updates channel was not consumed")
	}

	seen := make(map[kubecontainer.ContainerID]bool)
	for len(seen) < count {
		update := receiveUpdate(t, m)

		assert.True(t, m.IsCurrent(update))
		assert.False(t, seen[update.ContainerID], "duplicate notification for %v", update.ContainerID)

		seen[update.ContainerID] = true
	}
}

func TestPendingUpdatesCoalesce(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	m := NewManager().(*mutableManager)

	// Keep every publication pending until the consumer starts, regardless of dispatcher timing.
	m.updates = make(chan Update)
	id := kubecontainer.ContainerID{Type: "test", ID: "container"}
	pod := &corev1.Pod{}
	metadata := Metadata{ContainerName: "container", ProbeID: 1}
	t.Cleanup(func() {
		m.Remove(id)
		waitForDispatcherIdle(t, m)
	})

	for i := 0; i < 1000; i++ {
		m.SetWithMetadata(id, Result(i%2), pod, metadata)
	}

	m.RLock()

	assert.Len(t, m.pending, 1, "a slow consumer must not accumulate every transition")

	expected := m.cache[id]
	m.RUnlock()

	assert.Equal(t, uint64(1000), expected.Version)
	assert.Equal(t, Failure, expected.Result)

	for {
		update := receiveUpdate(t, m)
		if m.IsCurrent(update) {
			assert.Equal(t, expected, update)

			break
		}
	}

	waitForDispatcherIdle(t, m)
}

func TestRemoveAndReaddInvalidatesQueuedUpdates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	m := NewManager().(*mutableManager)
	id := kubecontainer.ContainerID{Type: "test", ID: "container"}
	pod := &corev1.Pod{}
	t.Cleanup(func() {
		m.Remove(id)
		waitForDispatcherIdle(t, m)
	})

	m.SetWithMetadata(id, Success, pod, Metadata{ContainerName: "container", ProbeID: 1})

	if !assert.Eventually(t, func() bool { return len(m.updates) == 1 }, wait.ForeverTestTimeout, time.Millisecond) {
		t.Fatal("initial notification was not queued")
	}

	m.Remove(id)
	m.SetWithMetadata(id, Success, pod, Metadata{ContainerName: "container", ProbeID: 2})

	old := receiveUpdate(t, m)

	assert.Equal(t, uint64(1), old.ProbeID)
	assert.False(t, m.IsCurrent(old))

	current := receiveUpdate(t, m)

	assert.Equal(t, uint64(2), current.ProbeID)
	assert.True(t, m.IsCurrent(current))
	assert.Greater(t, current.Version, old.Version)
}

func TestRemoveStopsBlockedDispatcher(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	m := NewManager().(*mutableManager)
	m.updates = make(chan Update)
	id := kubecontainer.ContainerID{Type: "test", ID: "container"}
	pod := &corev1.Pod{}
	m.Set(id, Success, pod)
	m.Remove(id)
	waitForDispatcherIdle(t, m)

	// A later publication starts a dispatcher again after the previous one exited.
	m.Set(id, Failure, pod)
	update := receiveUpdate(t, m)

	assert.Equal(t, Failure, update.Result)
	assert.True(t, m.IsCurrent(update))

	waitForDispatcherIdle(t, m)
}

func receiveUpdate(t *testing.T, m Manager) Update {
	t.Helper()
	select {
	case update := <-m.Updates():
		return update
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("timed out waiting for a probe result notification")
		return Update{}
	}
}

func waitForDispatcherIdle(t *testing.T, m *mutableManager) {
	t.Helper()

	assert.Eventually(t, func() bool {
		m.RLock()
		defer m.RUnlock()

		return !m.dispatching
	}, wait.ForeverTestTimeout, time.Millisecond, "dispatcher did not exit after pending notifications were cleared")
}

func TestResult_ToPrometheusType(t *testing.T) {
	tests := []struct {
		name     string
		result   Result
		expected float64
	}{
		{
			name:     "result is Success",
			result:   Success,
			expected: 0,
		},
		{
			name:     "result is Failure",
			result:   Failure,
			expected: 1,
		},
		{
			name:     "result is other",
			result:   123,
			expected: -1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.result.ToPrometheusType(); got != test.expected {
				t.Errorf("Result.ToPrometheusType() = %v, expected %v", got, test.expected)
			}
		})
	}
}

func TestResult_String(t *testing.T) {
	tests := []struct {
		name     string
		result   Result
		expected string
	}{
		{
			name:     "result is Success",
			result:   Success,
			expected: "Success",
		},
		{
			name:     "result is Failure",
			result:   Failure,
			expected: "Failure",
		},
		{
			name:     "result is other",
			result:   -123,
			expected: "UNKNOWN",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.result.String(); got != test.expected {
				t.Errorf("Result.String() = %v, expected %v", got, test.expected)
			}
		})
	}
}

func TestUpdates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, false)

	m := NewManager()

	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-pod"}}
	fooID := kubecontainer.ContainerID{Type: "test", ID: "foo"}
	barID := kubecontainer.ContainerID{Type: "test", ID: "bar"}

	expectUpdate := func(expected Update, msg string) {
		select {
		case u := <-m.Updates():
			if expected != u {
				t.Errorf("Expected update %v, received %v: %s", expected, u, msg)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Timed out waiting for update %v: %s", expected, msg)
		}
	}

	expectNoUpdate := func(msg string) {
		// NOTE: Since updates are accumulated asynchronously, this method is not guaranteed to fail
		// when it should. In the event it misses a failure, the following calls to expectUpdate should
		// still fail.
		select {
		case u := <-m.Updates():
			t.Errorf("Unexpected update %v: %s", u, msg)
		default:
			// Pass
		}
	}

	// New result should always push an update.
	m.Set(fooID, Success, pod)
	expectUpdate(Update{ContainerID: fooID, Result: Success, PodUID: pod.UID}, "new success")

	m.Set(barID, Failure, pod)
	expectUpdate(Update{ContainerID: barID, Result: Failure, PodUID: pod.UID}, "new failure")

	// Unchanged results should not send an update.
	m.Set(fooID, Success, pod)
	expectNoUpdate("unchanged foo")

	m.Set(barID, Failure, pod)
	expectNoUpdate("unchanged bar")

	// Changed results should send an update.
	m.Set(fooID, Failure, pod)
	expectUpdate(Update{ContainerID: fooID, Result: Failure, PodUID: pod.UID}, "changed foo")

	m.Set(barID, Success, pod)
	expectUpdate(Update{ContainerID: barID, Result: Success, PodUID: pod.UID}, "changed bar")
}

func TestLegacyUpdatesPreserveBackpressureAndTransitions(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, false)

	synctest.Test(t, func(t *testing.T) {
		m := NewManager().(*manager)
		m.updates = make(chan Update, 1)
		id := kubecontainer.ContainerID{Type: "test", ID: "container"}
		pod := &corev1.Pod{}
		m.Set(id, Success, pod)
		done := make(chan struct{})
		go func() { defer close(done); m.Set(id, Failure, pod) }()
		synctest.Wait()
		select {
		case <-done:
			t.Fatal("legacy Set did not wait for the full notification channel")
		default:
		}

		m.Remove(id)
		first := <-m.Updates()
		<-done
		second := <-m.Updates()
		if first.Result != Success || second.Result != Failure {
			t.Fatalf("legacy transitions were coalesced or reordered: first=%v second=%v", first, second)
		}
		if first.Version != 0 || second.Version != 0 || !m.IsCurrent(first) || !m.IsCurrent(second) {
			t.Fatal("legacy notifications were subject to mutable publication versioning")
		}
	})
}
