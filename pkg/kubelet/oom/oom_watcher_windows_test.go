//go:build windows

/*
Copyright 2025 The Kubernetes Authors.

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

package oom

import (
	"context"
	"fmt"
	"testing"
	"time"

	"k8s.io/client-go/tools/record"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	kubelettypes "k8s.io/kubelet/pkg/types"
	"k8s.io/kubernetes/test/utils/ktesting"

	"github.com/stretchr/testify/assert"
)

// fakeRuntimeGetter is a test double for the subset of the CRI runtime service
// the Windows OOM watcher needs.
type fakeRuntimeGetter struct {
	containers []*runtimeapi.Container
	statuses   map[string]*runtimeapi.ContainerStatus
}

func (f *fakeRuntimeGetter) ListContainers(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	return f.containers, nil
}

func (f *fakeRuntimeGetter) ContainerStatus(_ context.Context, id string, _ bool) (*runtimeapi.ContainerStatusResponse, error) {
	status, ok := f.statuses[id]
	if !ok {
		return nil, fmt.Errorf("container %q not found", id)
	}
	return &runtimeapi.ContainerStatusResponse{Status: status}, nil
}

func oomKilledStatus(id, name string) *runtimeapi.ContainerStatus {
	return &runtimeapi.ContainerStatus{
		Id:     id,
		Reason: oomKilledExitReason,
		Labels: map[string]string{
			kubelettypes.KubernetesPodNameLabel:       "test-pod",
			kubelettypes.KubernetesPodNamespaceLabel:  "default",
			kubelettypes.KubernetesPodUIDLabel:        "pod-uid",
			kubelettypes.KubernetesContainerNameLabel: name,
		},
		Metadata: &runtimeapi.ContainerMetadata{Name: name},
	}
}

func newTestWindowsWatcher(runtime containerStatusGetter) (*windowsWatcher, *record.FakeRecorder) {
	fakeRecorder := record.NewFakeRecorder(10)
	return &windowsWatcher{
		recorder:   fakeRecorder,
		containers: runtime,
		seen:       make(map[string]struct{}),
	}, fakeRecorder
}

// TestWindowsWatcherReportsOOMKilled verifies that a container which crossed
// into the OOMKilled CRI state emits a Kubernetes event exactly once and
// increments the container_oom_events_total counter it backs.
func TestWindowsWatcherReportsOOMKilled(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := klog.FromContext(tCtx)
	containerName := fmt.Sprintf("app-%s", t.Name())
	cid := "container-1"

	runtime := &fakeRuntimeGetter{
		containers: []*runtimeapi.Container{
			{Id: cid, State: runtimeapi.ContainerState_CONTAINER_EXITED},
		},
		statuses: map[string]*runtimeapi.ContainerStatus{cid: oomKilledStatus(cid, containerName)},
	}
	w, fakeRecorder := newTestWindowsWatcher(runtime)

	w.reconcile(tCtx, logger)

	select {
	case ev := <-fakeRecorder.Events:
		assert.Contains(t, ev, windowsOOMEventReason)
		assert.Contains(t, ev, containerName)
	case <-time.After(time.Second):
		t.Fatal("expected an OOMKilled Kubernetes event")
	}
	// The counter metric is backed by recordOOMKill, keyed by container name.
	assert.Equal(t, uint64(1), OOMEventsForContainer(containerName))

	// Reconciling again must not double-emit the event or inflate the counter.
	w.reconcile(tCtx, logger)
	select {
	case <-fakeRecorder.Events:
		t.Fatal("did not expect a second event for the same OOMKilled container")
	case <-time.After(100 * time.Millisecond):
	}
	assert.Equal(t, uint64(1), OOMEventsForContainer(containerName))
}

// TestWindowsWatcherSkipsRunningContainers verifies that a running container is
// never treated as OOMKilled, regardless of its status payload.
func TestWindowsWatcherSkipsRunningContainers(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := klog.FromContext(tCtx)
	containerName := fmt.Sprintf("app-%s", t.Name())
	cid := "container-running"

	runtime := &fakeRuntimeGetter{
		containers: []*runtimeapi.Container{
			{Id: cid, State: runtimeapi.ContainerState_CONTAINER_RUNNING},
		},
		statuses: map[string]*runtimeapi.ContainerStatus{cid: oomKilledStatus(cid, containerName)},
	}
	w, fakeRecorder := newTestWindowsWatcher(runtime)

	w.reconcile(tCtx, logger)

	select {
	case <-fakeRecorder.Events:
		t.Fatal("did not expect an OOMKilled event for a running container")
	case <-time.After(100 * time.Millisecond):
	}
	assert.Equal(t, uint64(0), OOMEventsForContainer(containerName))
}

// TestWindowsWatcherSeedsOOMKilled verifies that containers already OOMKilled
// when the watcher starts are not replayed as fresh events on the first poll.
func TestWindowsWatcherSeedsOOMKilled(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := klog.FromContext(tCtx)
	containerName := fmt.Sprintf("app-%s", t.Name())
	cid := "container-seed"

	runtime := &fakeRuntimeGetter{
		containers: []*runtimeapi.Container{
			{Id: cid, State: runtimeapi.ContainerState_CONTAINER_EXITED},
		},
		statuses: map[string]*runtimeapi.ContainerStatus{cid: oomKilledStatus(cid, containerName)},
	}
	w, fakeRecorder := newTestWindowsWatcher(runtime)

	seedOOMKills(tCtx, w.containers, w.seen, logger)

	// The container was already OOMKilled at start, so the first reconcile must
	// not surface it again.
	w.reconcile(tCtx, logger)

	select {
	case <-fakeRecorder.Events:
		t.Fatal("did not expect a replayed event for a pre-existing OOMKilled container")
	case <-time.After(100 * time.Millisecond):
	}
	assert.Equal(t, uint64(0), OOMEventsForContainer(containerName))
}

// TestWindowsWatcherNoRuntimeIsSafe verifies that a nil runtime service does
// not panic or emit anything during a reconcile.
func TestWindowsWatcherNoRuntimeIsSafe(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := klog.FromContext(tCtx)

	w, _ := newTestWindowsWatcher(nil)
	w.reconcile(tCtx, logger)

	assert.Equal(t, uint64(0), OOMEventsForContainer(t.Name()))
}
