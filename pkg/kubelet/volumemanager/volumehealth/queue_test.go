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

package volumehealth

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume/csi"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// blockingHealthClient records concurrency of NodeGetVolumeHealth calls. When
// block is set, calls wait until release is closed or the context ends.
type blockingHealthClient struct {
	block   bool
	release chan struct{}
	started chan string

	mu          sync.Mutex
	inFlight    int
	maxInFlight int
	calls       int
	requests    []volumeProbeCall
}

type volumeProbeCall struct {
	volumeHandle string
	stagingPath  string
	publishPath  string
}

func newBlockingHealthClient(block bool) *blockingHealthClient {
	return &blockingHealthClient{
		block:   block,
		release: make(chan struct{}),
		started: make(chan string, 100),
	}
}

func (c *blockingHealthClient) NodeGetVolumeHealth(ctx context.Context, volID, stagingTargetPath, volumePublishPath string) ([]v1.VolumeHealthCondition, error) {
	c.mu.Lock()
	c.inFlight++
	c.calls++
	c.requests = append(c.requests, volumeProbeCall{volID, stagingTargetPath, volumePublishPath})
	c.maxInFlight = max(c.maxInFlight, c.inFlight)
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		c.inFlight--
		c.mu.Unlock()
	}()

	c.started <- volID
	if !c.block {
		return nil, nil
	}
	select {
	case <-c.release:
		return nil, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *blockingHealthClient) NodeGetStorageHealth(ctx context.Context, secrets map[string]string) ([]storagev1.StorageHealthCondition, error) {
	return nil, errors.New("not implemented")
}

func (c *blockingHealthClient) NodeSupportsVolumeHealth(ctx context.Context) (bool, error) {
	return true, nil
}

func (c *blockingHealthClient) NodeSupportsStorageHealth(ctx context.Context) (bool, error) {
	return false, nil
}

func (c *blockingHealthClient) stats() (calls, maxInFlight int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls, c.maxInFlight
}

// waitStarted waits for n NodeGetVolumeHealth calls to begin.
func (c *blockingHealthClient) waitStarted(t *testing.T, n int) {
	t.Helper()
	for i := range n {
		select {
		case <-c.started:
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatalf("timed out waiting for probe %d of %d to start", i+1, n)
		}
	}
}

func testVolume(driverName string, i int) cache.VolumeToHealthCheck {
	name := fmt.Sprintf("%s-%d", driverName, i)
	return cache.VolumeToHealthCheck{
		VolumeName: v1.UniqueVolumeName("volume-" + name),
		PodName:    volumetypes.UniquePodName("pod-" + name),
		Pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID: types.UID("uid-" + name),
			},
		},
		OuterVolumeName: "vol-" + name,
		DriverName:      driverName,
		CSIVolumeHandle: "handle-" + name,
	}
}

func testVolumes(driverName string, n int) []cache.VolumeToHealthCheck {
	volumes := make([]cache.VolumeToHealthCheck, 0, n)
	for i := range n {
		volumes = append(volumes, testVolume(driverName, i))
	}
	return volumes
}

// newQueueTestManager returns a manager whose probes go to clients[driverName]
// and that probes exactly volumes, with no storage health.
func newQueueTestManager(status StatusUpdater, workers int, volumes []cache.VolumeToHealthCheck, clients map[string]csi.HealthClient) *manager {
	m := newManager(status, time.Hour, workers)
	m.listVolumes = func() []cache.VolumeToHealthCheck { return volumes }
	m.listDrivers = func() []string { return nil }
	m.clientFactory = func(driverName string) csi.HealthClient { return clients[driverName] }
	m.csiNodeUpdater = func() CSINodeUpdater { return nil }
	return m
}

// runManager starts m.Run and returns a function that stops it and waits for
// Run to return.
func runManager(t *testing.T, m *manager) func() {
	t.Helper()
	tCtx := ktesting.Init(t)
	ctx, cancel := context.WithCancel(tCtx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		m.Run(ctx)
	}()
	stop := func() {
		cancel()
		select {
		case <-done:
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatal("Run did not return after context cancellation")
		}
	}
	t.Cleanup(stop)
	return stop
}

func (t *testStatusUpdater) callsWithUIDPrefix(prefix string) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	count := 0
	for _, c := range t.calls {
		if strings.HasPrefix(string(c.podUID), prefix) {
			count++
		}
	}
	return count
}

func waitForStatusUpdates(t *testing.T, status *testStatusUpdater, uidPrefix string, want int) {
	t.Helper()
	err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		return status.callsWithUIDPrefix(uidPrefix) >= want, nil
	})
	if err != nil {
		t.Fatalf("expected %d status updates for %q, got %d", want, uidPrefix, status.callsWithUIDPrefix(uidPrefix))
	}
}

func TestWorkersProbeInParallel(t *testing.T) {
	const workers = 3
	client := newBlockingHealthClient(true)
	status := &testStatusUpdater{}
	m := newQueueTestManager(status, workers, testVolumes("driver", 2*workers), map[string]csi.HealthClient{"driver": client})
	runManager(t, m)

	client.waitStarted(t, workers)
	close(client.release)
	waitForStatusUpdates(t, status, "uid-driver", 2*workers)

	if _, maxInFlight := client.stats(); maxInFlight != workers {
		t.Errorf("expected %d probes in flight at most, got %d", workers, maxInFlight)
	}
}

func TestSameVolumeIsNotProbedConcurrently(t *testing.T) {
	tCtx := ktesting.Init(t)
	vol := testVolume("driver", 0)
	client := newBlockingHealthClient(true)
	status := &testStatusUpdater{}
	m := newQueueTestManager(status, 2, []cache.VolumeToHealthCheck{vol}, map[string]csi.HealthClient{"driver": client})
	runManager(t, m)

	client.waitStarted(t, 1)
	vol.StagingPath = "/staging"
	vol.PublishPath = "/publish"
	m.listVolumes = func() []cache.VolumeToHealthCheck { return []cache.VolumeToHealthCheck{vol} }
	m.enqueueVolumes(tCtx)
	if got := m.queue.Len(); got != 0 {
		t.Fatalf("expected updated volume to wait for its in-flight probe, got %d queued items", got)
	}
	close(client.release)
	client.waitStarted(t, 1)
	waitForStatusUpdates(t, status, "uid-driver", 2)

	if calls, maxInFlight := client.stats(); calls != 2 || maxInFlight != 1 {
		t.Errorf("expected 2 sequential probes, got %d calls with %d in flight at most", calls, maxInFlight)
	}
	client.mu.Lock()
	defer client.mu.Unlock()
	want := []volumeProbeCall{
		{volumeHandle: vol.CSIVolumeHandle},
		{volumeHandle: vol.CSIVolumeHandle, stagingPath: vol.StagingPath, publishPath: vol.PublishPath},
	}
	if !reflect.DeepEqual(client.requests, want) {
		t.Errorf("expected probes with old then updated paths: got %+v, want %+v", client.requests, want)
	}
}

func TestQueuedVolumeUsesLatestParameters(t *testing.T) {
	tCtx := ktesting.Init(t)
	volumes := []cache.VolumeToHealthCheck{testVolume("driver", 0)}
	oldClient := newBlockingHealthClient(false)
	client := newBlockingHealthClient(false)
	status := &testStatusUpdater{}
	m := newQueueTestManager(status, 1, volumes, map[string]csi.HealthClient{"driver": oldClient, "new-driver": client})
	t.Cleanup(m.queue.ShutDown)
	m.enqueueVolumes(tCtx)

	volumes[0].DriverName = "new-driver"
	volumes[0].CSIVolumeHandle = "new-handle"
	volumes[0].StagingPath = "/staging"
	volumes[0].PublishPath = "/publish"
	m.enqueueVolumes(tCtx)
	if got := m.queue.Len(); got != 1 {
		t.Fatalf("expected path changes to coalesce into one queued item, got %d", got)
	}
	m.processNextVolume(tCtx)
	if calls, _ := oldClient.stats(); calls != 0 {
		t.Errorf("expected no probes using the old driver, got %d", calls)
	}
	want := []volumeProbeCall{{volumeHandle: "new-handle", stagingPath: "/staging", publishPath: "/publish"}}
	if !reflect.DeepEqual(client.requests, want) {
		t.Errorf("expected latest probe parameters: got %+v, want %+v", client.requests, want)
	}
	if got := status.callCount(); got != 1 {
		t.Errorf("expected one status update, got %d", got)
	}
}

func TestRemovedVolumeIsNotProbed(t *testing.T) {
	tCtx := ktesting.Init(t)
	vol := testVolume("driver", 0)
	client := newBlockingHealthClient(false)
	status := &testStatusUpdater{}
	m := newQueueTestManager(status, 1, []cache.VolumeToHealthCheck{vol}, map[string]csi.HealthClient{"driver": client})
	t.Cleanup(m.queue.ShutDown)
	m.enqueueVolumes(tCtx)
	m.listVolumes = func() []cache.VolumeToHealthCheck { return nil }
	m.enqueueVolumes(tCtx)
	m.processNextVolume(tCtx)
	if calls, _ := client.stats(); calls != 0 {
		t.Errorf("expected no probes for removed volume, got %d", calls)
	}
	if got := status.callCount(); got != 0 {
		t.Errorf("expected no status updates for removed volume, got %d", got)
	}
	m.volumesMu.Lock()
	defer m.volumesMu.Unlock()
	if got := len(m.volumes); got != 0 {
		t.Errorf("expected removed volume to be forgotten, got %d retained volumes", got)
	}
}

func TestRunStopsWithBlockedProbes(t *testing.T) {
	client := newBlockingHealthClient(true)
	m := newQueueTestManager(&testStatusUpdater{}, 2, testVolumes("driver", 2), map[string]csi.HealthClient{"driver": client})
	stop := runManager(t, m)

	client.waitStarted(t, 2)
	stop()
}

func TestProbeStorageHealthStopsOnCancel(t *testing.T) {
	tCtx := ktesting.Init(t)
	ctx, cancel := context.WithCancel(tCtx)
	cancel()

	client := &fakeHealthClient{supportsStorageHealth: true}
	updater := &fakeCSINodeUpdater{}
	m := newQueueTestManager(&testStatusUpdater{}, 1, nil, map[string]csi.HealthClient{"driver": client})
	t.Cleanup(m.queue.ShutDown)
	m.listDrivers = func() []string { return []string{"driver"} }
	m.csiNodeUpdater = func() CSINodeUpdater { return updater }

	m.probeStorageHealth(ctx)
	if client.storageCalls != 0 || len(updater.calls) != 0 {
		t.Errorf("expected no storage probes after cancellation, got %d RPCs and %d CSINode updates", client.storageCalls, len(updater.calls))
	}
}
