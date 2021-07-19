/*
Copyright 2014 The Kubernetes Authors.

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

package kubelet

import (
	"context"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/queue"
)

// fakePodWorkers runs sync pod function in serial, so we can have
// deterministic behaviour in testing.
type fakePodWorkers struct {
	lock      sync.Mutex
	syncPodFn syncPodFnType
	cache     kubecontainer.Cache
	t         TestingInterface

	triggeredDeletion []types.UID

	statusLock            sync.Mutex
	running               map[types.UID]bool
	terminating           map[types.UID]bool
	terminationRequested  map[types.UID]bool
	removeRuntime         map[types.UID]bool
	removeContent         map[types.UID]bool
	terminatingStaticPods map[string]bool
}

func (f *fakePodWorkers) UpdatePod(options UpdatePodOptions) {
	f.lock.Lock()
	defer f.lock.Unlock()
	var uid types.UID
	switch {
	case options.Pod != nil:
		uid = options.Pod.UID
	case options.RunningPod != nil:
		uid = options.RunningPod.ID
	default:
		return
	}
	status, err := f.cache.Get(uid)
	if err != nil {
		f.t.Errorf("Unexpected error: %v", err)
	}
	switch options.UpdateType {
	case kubetypes.SyncPodKill:
		f.triggeredDeletion = append(f.triggeredDeletion, uid)
	default:
		if err := f.syncPodFn(context.Background(), options.UpdateType, options.Pod, options.MirrorPod, status); err != nil {
			f.t.Errorf("Unexpected error: %v", err)
		}
	}
}

func (f *fakePodWorkers) SyncKnownPods(desiredPods []*v1.Pod) map[types.UID]PodWorkType {
	return nil
}

func (f *fakePodWorkers) CouldHaveRunningContainers(uid types.UID) bool {
	f.statusLock.Lock()
	defer f.statusLock.Unlock()
	return f.running[uid]
}
func (f *fakePodWorkers) IsPodTerminationRequested(uid types.UID) bool {
	f.statusLock.Lock()
	defer f.statusLock.Unlock()
	return f.terminationRequested[uid]
}
func (f *fakePodWorkers) ShouldPodContainersBeTerminating(uid types.UID) bool {
	f.statusLock.Lock()
	defer f.statusLock.Unlock()
	return f.terminating[uid]
}
func (f *fakePodWorkers) ShouldPodRuntimeBeRemoved(uid types.UID) bool {
	f.statusLock.Lock()
	defer f.statusLock.Unlock()
	return f.removeRuntime[uid]
}
func (f *fakePodWorkers) ShouldPodContentBeRemoved(uid types.UID) bool {
	f.statusLock.Lock()
	defer f.statusLock.Unlock()
	return f.removeContent[uid]
}
func (f *fakePodWorkers) IsPodForMirrorPodTerminatingByFullName(podFullname string) bool {
	f.statusLock.Lock()
	defer f.statusLock.Unlock()
	return f.terminatingStaticPods[podFullname]
}

type TestingInterface interface {
	Errorf(format string, args ...interface{})
}

func newPod(uid, name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:  types.UID(uid),
			Name: name,
		},
	}
}

// syncPodRecord is a record of a sync pod call
type syncPodRecord struct {
	name       string
	updateType kubetypes.SyncPodType
	runningPod *kubecontainer.Pod
	terminated bool
}

func createPodWorkers() (*podWorkers, map[types.UID][]syncPodRecord) {
	lock := sync.Mutex{}
	processed := make(map[types.UID][]syncPodRecord)
	fakeRecorder := &record.FakeRecorder{}
	fakeRuntime := &containertest.FakeRuntime{}
	fakeCache := containertest.NewFakeCache(fakeRuntime)
	w := newPodWorkers(
		func(ctx context.Context, updateType kubetypes.SyncPodType, pod, mirrorPod *v1.Pod, podStatus *kubecontainer.PodStatus) error {
			func() {
				lock.Lock()
				defer lock.Unlock()
				pod := pod
				processed[pod.UID] = append(processed[pod.UID], syncPodRecord{
					name:       pod.Name,
					updateType: updateType,
				})
			}()
			return nil
		},
		func(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus, runningPod *kubecontainer.Pod, gracePeriod *int64, podStatusFn func(*v1.PodStatus)) error {
			func() {
				lock.Lock()
				defer lock.Unlock()
				processed[pod.UID] = append(processed[pod.UID], syncPodRecord{
					name:       pod.Name,
					updateType: kubetypes.SyncPodKill,
					runningPod: runningPod,
				})
			}()
			return nil
		},
		func(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus) error {
			func() {
				lock.Lock()
				defer lock.Unlock()
				processed[pod.UID] = append(processed[pod.UID], syncPodRecord{
					name:       pod.Name,
					terminated: true,
				})
			}()
			return nil
		},
		fakeRecorder,
		queue.NewBasicWorkQueue(&clock.RealClock{}),
		time.Second,
		time.Second,
		fakeCache,
	)
	return w.(*podWorkers), processed
}

func drainWorkers(podWorkers *podWorkers, numPods int) {
	for {
		stillWorking := false
		podWorkers.podLock.Lock()
		for i := 0; i < numPods; i++ {
			if s, ok := podWorkers.podSyncStatuses[types.UID(strconv.Itoa(i))]; ok && s.working {
				stillWorking = true
				break
			}
		}
		podWorkers.podLock.Unlock()
		if !stillWorking {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func drainAllWorkers(podWorkers *podWorkers) {
	for {
		stillWorking := false
		podWorkers.podLock.Lock()
		for _, worker := range podWorkers.podSyncStatuses {
			if worker.working {
				stillWorking = true
				break
			}
		}
		podWorkers.podLock.Unlock()
		if !stillWorking {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func TestUpdatePod(t *testing.T) {
	podWorkers, processed := createPodWorkers()

	numPods := 20
	for i := 0; i < numPods; i++ {
		for j := i; j < numPods; j++ {
			podWorkers.UpdatePod(UpdatePodOptions{
				Pod:        newPod(strconv.Itoa(j), strconv.Itoa(i)),
				UpdateType: kubetypes.SyncPodCreate,
			})
		}
	}
	drainWorkers(podWorkers, numPods)

	if len(processed) != numPods {
		t.Errorf("Not all pods processed: %v", len(processed))
		return
	}
	for i := 0; i < numPods; i++ {
		uid := types.UID(strconv.Itoa(i))
		if len(processed[uid]) < 1 || len(processed[uid]) > i+1 {
			t.Errorf("Pod %v processed %v times", i, len(processed[uid]))
			continue
		}

		// PodWorker guarantees the first and the last event will be processed
		first := 0
		last := len(processed[uid]) - 1
		if processed[uid][first].name != "0" {
			t.Errorf("Pod %v: incorrect order %v, %v", i, first, processed[uid][first])

		}
		if processed[uid][last].name != strconv.Itoa(i) {
			t.Errorf("Pod %v: incorrect order %v, %v", i, last, processed[uid][last])
		}
	}
}

func TestUpdatePodForRuntimePod(t *testing.T) {
	podWorkers, processed := createPodWorkers()

	// ignores running pod of wrong sync type
	podWorkers.UpdatePod(UpdatePodOptions{
		UpdateType: kubetypes.SyncPodCreate,
		RunningPod: &kubecontainer.Pod{ID: "1", Name: "1", Namespace: "test"},
	})
	drainAllWorkers(podWorkers)
	if len(processed) != 0 {
		t.Fatalf("Not all pods processed: %v", len(processed))
	}

	// creates synthetic pod
	podWorkers.UpdatePod(UpdatePodOptions{
		UpdateType: kubetypes.SyncPodKill,
		RunningPod: &kubecontainer.Pod{ID: "1", Name: "1", Namespace: "test"},
	})
	drainAllWorkers(podWorkers)
	if len(processed) != 1 {
		t.Fatalf("Not all pods processed: %v", processed)
	}
	updates := processed["1"]
	if len(updates) != 1 {
		t.Fatalf("unexpected updates: %v", updates)
	}
	if updates[0].runningPod == nil || updates[0].updateType != kubetypes.SyncPodKill || updates[0].name != "1" {
		t.Fatalf("unexpected update: %v", updates)
	}
}

func TestUpdatePodForTerminatedRuntimePod(t *testing.T) {
	podWorkers, processed := createPodWorkers()

	now := time.Now()
	podWorkers.podSyncStatuses[types.UID("1")] = &podSyncStatus{
		startedTerminating: true,
		terminatedAt:       now.Add(-time.Second),
		terminatingAt:      now.Add(-2 * time.Second),
		gracePeriod:        1,
	}

	// creates synthetic pod
	podWorkers.UpdatePod(UpdatePodOptions{
		UpdateType: kubetypes.SyncPodKill,
		RunningPod: &kubecontainer.Pod{ID: "1", Name: "1", Namespace: "test"},
	})
	drainAllWorkers(podWorkers)
	if len(processed) != 0 {
		t.Fatalf("Not all pods processed: %v", processed)
	}
	updates := processed["1"]
	if len(updates) != 0 {
		t.Fatalf("unexpected updates: %v", updates)
	}
	if len(podWorkers.lastUndeliveredWorkUpdate) != 0 {
		t.Fatalf("Unexpected undelivered work")
	}
}

func TestUpdatePodDoesNotForgetSyncPodKill(t *testing.T) {
	podWorkers, processed := createPodWorkers()
	numPods := 20
	for i := 0; i < numPods; i++ {
		pod := newPod(strconv.Itoa(i), strconv.Itoa(i))
		podWorkers.UpdatePod(UpdatePodOptions{
			Pod:        pod,
			UpdateType: kubetypes.SyncPodCreate,
		})
		podWorkers.UpdatePod(UpdatePodOptions{
			Pod:        pod,
			UpdateType: kubetypes.SyncPodKill,
		})
		podWorkers.UpdatePod(UpdatePodOptions{
			Pod:        pod,
			UpdateType: kubetypes.SyncPodUpdate,
		})
	}
	drainWorkers(podWorkers, numPods)
	if len(processed) != numPods {
		t.Errorf("Not all pods processed: %v", len(processed))
		return
	}
	for i := 0; i < numPods; i++ {
		uid := types.UID(strconv.Itoa(i))
		// each pod should be processed two times (create, kill, but not update)
		syncPodRecords := processed[uid]
		if len(syncPodRecords) < 2 {
			t.Errorf("Pod %v processed %v times, but expected at least 2", i, len(syncPodRecords))
			continue
		}
		if syncPodRecords[0].updateType != kubetypes.SyncPodCreate {
			t.Errorf("Pod %v event was %v, but expected %v", i, syncPodRecords[0].updateType, kubetypes.SyncPodCreate)
		}
		if syncPodRecords[1].updateType != kubetypes.SyncPodKill {
			t.Errorf("Pod %v event was %v, but expected %v", i, syncPodRecords[1].updateType, kubetypes.SyncPodKill)
		}
	}
}

func TestSyncKnownPods(t *testing.T) {
	podWorkers, _ := createPodWorkers()

	numPods := 20
	for i := 0; i < numPods; i++ {
		podWorkers.UpdatePod(UpdatePodOptions{
			Pod:        newPod(strconv.Itoa(i), "name"),
			UpdateType: kubetypes.SyncPodUpdate,
		})
	}
	drainWorkers(podWorkers, numPods)

	if len(podWorkers.podUpdates) != numPods {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}

	desiredPods := map[types.UID]sets.Empty{}
	desiredPods[types.UID("2")] = sets.Empty{}
	desiredPods[types.UID("14")] = sets.Empty{}
	desiredPodList := []*v1.Pod{newPod("2", "name"), newPod("14", "name")}

	// kill all but the requested pods
	for i := 0; i < numPods; i++ {
		pod := newPod(strconv.Itoa(i), "name")
		if _, ok := desiredPods[pod.UID]; ok {
			continue
		}
		if (i % 2) == 0 {
			now := metav1.Now()
			pod.DeletionTimestamp = &now
		}
		podWorkers.UpdatePod(UpdatePodOptions{
			Pod:        pod,
			UpdateType: kubetypes.SyncPodKill,
		})
	}
	drainWorkers(podWorkers, numPods)

	if !podWorkers.ShouldPodContainersBeTerminating(types.UID("0")) {
		t.Errorf("Expected pod to be terminating")
	}
	if !podWorkers.ShouldPodContainersBeTerminating(types.UID("1")) {
		t.Errorf("Expected pod to be terminating")
	}
	if podWorkers.ShouldPodContainersBeTerminating(types.UID("2")) {
		t.Errorf("Expected pod to not be terminating")
	}
	if !podWorkers.IsPodTerminationRequested(types.UID("0")) {
		t.Errorf("Expected pod to be terminating")
	}
	if podWorkers.IsPodTerminationRequested(types.UID("2")) {
		t.Errorf("Expected pod to not be terminating")
	}

	if podWorkers.CouldHaveRunningContainers(types.UID("0")) {
		t.Errorf("Expected pod to be terminated (deleted and terminated)")
	}
	if podWorkers.CouldHaveRunningContainers(types.UID("1")) {
		t.Errorf("Expected pod to be terminated")
	}
	if !podWorkers.CouldHaveRunningContainers(types.UID("2")) {
		t.Errorf("Expected pod to not be terminated")
	}

	if !podWorkers.ShouldPodContentBeRemoved(types.UID("0")) {
		t.Errorf("Expected pod to be suitable for removal (deleted and terminated)")
	}
	if podWorkers.ShouldPodContentBeRemoved(types.UID("1")) {
		t.Errorf("Expected pod to not be suitable for removal (terminated but not deleted)")
	}
	if podWorkers.ShouldPodContentBeRemoved(types.UID("2")) {
		t.Errorf("Expected pod to not be suitable for removal (not terminated)")
	}

	if podWorkers.ShouldPodContainersBeTerminating(types.UID("abc")) {
		t.Errorf("Expected pod to not be known to be terminating (does not exist but not yet synced)")
	}
	if !podWorkers.CouldHaveRunningContainers(types.UID("abc")) {
		t.Errorf("Expected pod to potentially have running containers (does not exist but not yet synced)")
	}
	if podWorkers.ShouldPodContentBeRemoved(types.UID("abc")) {
		t.Errorf("Expected pod to not be suitable for removal (does not exist but not yet synced)")
	}

	podWorkers.SyncKnownPods(desiredPodList)
	if len(podWorkers.podUpdates) != 2 {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}
	if _, exists := podWorkers.podUpdates[types.UID("2")]; !exists {
		t.Errorf("No updates channel for pod 2")
	}
	if _, exists := podWorkers.podUpdates[types.UID("14")]; !exists {
		t.Errorf("No updates channel for pod 14")
	}
	if podWorkers.IsPodTerminationRequested(types.UID("2")) {
		t.Errorf("Expected pod termination request to be cleared after sync")
	}

	if !podWorkers.ShouldPodContainersBeTerminating(types.UID("abc")) {
		t.Errorf("Expected pod to be expected to terminate containers (does not exist and synced at least once)")
	}
	if podWorkers.CouldHaveRunningContainers(types.UID("abc")) {
		t.Errorf("Expected pod to be known not to have running containers (does not exist and synced at least once)")
	}
	if !podWorkers.ShouldPodContentBeRemoved(types.UID("abc")) {
		t.Errorf("Expected pod to be suitable for removal (does not exist and synced at least once)")
	}

	// verify workers that are not terminated stay open even if config no longer
	// sees them
	podWorkers.SyncKnownPods(nil)
	if len(podWorkers.podUpdates) != 2 {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}
	if len(podWorkers.podSyncStatuses) != 2 {
		t.Errorf("Incorrect number of tracked statuses: %#v", podWorkers.podSyncStatuses)
	}
	if len(podWorkers.lastUndeliveredWorkUpdate) != 0 {
		t.Errorf("Incorrect number of tracked statuses: %#v", podWorkers.lastUndeliveredWorkUpdate)
	}

	for uid := range desiredPods {
		pod := newPod(string(uid), "name")
		podWorkers.UpdatePod(UpdatePodOptions{
			Pod:        pod,
			UpdateType: kubetypes.SyncPodKill,
		})
	}
	drainWorkers(podWorkers, numPods)

	// verify once those pods terminate (via some other flow) the workers are cleared
	podWorkers.SyncKnownPods(nil)
	if len(podWorkers.podUpdates) != 0 {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}
	if len(podWorkers.podSyncStatuses) != 0 {
		t.Errorf("Incorrect number of tracked statuses: %#v", podWorkers.podSyncStatuses)
	}
	if len(podWorkers.lastUndeliveredWorkUpdate) != 0 {
		t.Errorf("Incorrect number of tracked statuses: %#v", podWorkers.lastUndeliveredWorkUpdate)
	}
}

type simpleFakeKubelet struct {
	pod       *v1.Pod
	mirrorPod *v1.Pod
	podStatus *kubecontainer.PodStatus
	wg        sync.WaitGroup
}

func (kl *simpleFakeKubelet) syncPod(ctx context.Context, updateType kubetypes.SyncPodType, pod, mirrorPod *v1.Pod, podStatus *kubecontainer.PodStatus) error {
	kl.pod, kl.mirrorPod, kl.podStatus = pod, mirrorPod, podStatus
	return nil
}

func (kl *simpleFakeKubelet) syncPodWithWaitGroup(ctx context.Context, updateType kubetypes.SyncPodType, pod, mirrorPod *v1.Pod, podStatus *kubecontainer.PodStatus) error {
	kl.pod, kl.mirrorPod, kl.podStatus = pod, mirrorPod, podStatus
	kl.wg.Done()
	return nil
}

func (kl *simpleFakeKubelet) syncTerminatingPod(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus, runningPod *kubecontainer.Pod, gracePeriod *int64, podStatusFn func(*v1.PodStatus)) error {
	return nil
}

func (kl *simpleFakeKubelet) syncTerminatedPod(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus) error {
	return nil
}

// TestFakePodWorkers verifies that the fakePodWorkers behaves the same way as the real podWorkers
// for their invocation of the syncPodFn.
func TestFakePodWorkers(t *testing.T) {
	fakeRecorder := &record.FakeRecorder{}
	fakeRuntime := &containertest.FakeRuntime{}
	fakeCache := containertest.NewFakeCache(fakeRuntime)

	kubeletForRealWorkers := &simpleFakeKubelet{}
	kubeletForFakeWorkers := &simpleFakeKubelet{}

	realPodWorkers := newPodWorkers(
		kubeletForRealWorkers.syncPodWithWaitGroup,
		kubeletForRealWorkers.syncTerminatingPod,
		kubeletForRealWorkers.syncTerminatedPod,
		fakeRecorder, queue.NewBasicWorkQueue(&clock.RealClock{}), time.Second, time.Second, fakeCache)
	fakePodWorkers := &fakePodWorkers{
		syncPodFn: kubeletForFakeWorkers.syncPod,
		cache:     fakeCache,
		t:         t,
	}

	tests := []struct {
		pod       *v1.Pod
		mirrorPod *v1.Pod
	}{
		{
			&v1.Pod{},
			&v1.Pod{},
		},
		{
			podWithUIDNameNs("12345678", "foo", "new"),
			podWithUIDNameNs("12345678", "fooMirror", "new"),
		},
		{
			podWithUIDNameNs("98765", "bar", "new"),
			podWithUIDNameNs("98765", "barMirror", "new"),
		},
	}

	for i, tt := range tests {
		kubeletForRealWorkers.wg.Add(1)
		realPodWorkers.UpdatePod(UpdatePodOptions{
			Pod:        tt.pod,
			MirrorPod:  tt.mirrorPod,
			UpdateType: kubetypes.SyncPodUpdate,
		})
		fakePodWorkers.UpdatePod(UpdatePodOptions{
			Pod:        tt.pod,
			MirrorPod:  tt.mirrorPod,
			UpdateType: kubetypes.SyncPodUpdate,
		})

		kubeletForRealWorkers.wg.Wait()

		if !reflect.DeepEqual(kubeletForRealWorkers.pod, kubeletForFakeWorkers.pod) {
			t.Errorf("%d: Expected: %#v, Actual: %#v", i, kubeletForRealWorkers.pod, kubeletForFakeWorkers.pod)
		}

		if !reflect.DeepEqual(kubeletForRealWorkers.mirrorPod, kubeletForFakeWorkers.mirrorPod) {
			t.Errorf("%d: Expected: %#v, Actual: %#v", i, kubeletForRealWorkers.mirrorPod, kubeletForFakeWorkers.mirrorPod)
		}

		if !reflect.DeepEqual(kubeletForRealWorkers.podStatus, kubeletForFakeWorkers.podStatus) {
			t.Errorf("%d: Expected: %#v, Actual: %#v", i, kubeletForRealWorkers.podStatus, kubeletForFakeWorkers.podStatus)
		}
	}
}

// TestKillPodNowFunc tests the blocking kill pod function works with pod workers as expected.
func TestKillPodNowFunc(t *testing.T) {
	fakeRecorder := &record.FakeRecorder{}
	podWorkers, processed := createPodWorkers()
	killPodFunc := killPodNow(podWorkers, fakeRecorder)
	pod := newPod("test", "test")
	gracePeriodOverride := int64(0)
	err := killPodFunc(pod, false, &gracePeriodOverride, func(status *v1.PodStatus) {
		status.Phase = v1.PodFailed
		status.Reason = "reason"
		status.Message = "message"
	})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	drainAllWorkers(podWorkers)
	if len(processed) != 1 {
		t.Fatalf("len(processed) expected: %v, actual: %#v", 1, processed)
	}
	syncPodRecords := processed[pod.UID]
	if len(syncPodRecords) != 2 {
		t.Fatalf("Pod processed expected %v times, got %#v", 1, syncPodRecords)
	}
	if syncPodRecords[0].updateType != kubetypes.SyncPodKill {
		t.Errorf("Pod update type was %v, but expected %v", syncPodRecords[0].updateType, kubetypes.SyncPodKill)
	}
	if !syncPodRecords[1].terminated {
		t.Errorf("Pod terminated %v, but expected %v", syncPodRecords[1].terminated, true)
	}
}
