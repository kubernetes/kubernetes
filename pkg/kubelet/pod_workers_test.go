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
	"reflect"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
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
	syncPodFn syncPodFnType
	cache     kubecontainer.Cache
	t         TestingInterface
}

func (f *fakePodWorkers) UpdatePod(options *UpdatePodOptions) {
	status, err := f.cache.Get(options.Pod.UID)
	if err != nil {
		f.t.Errorf("Unexpected error: %v", err)
	}
	if err := f.syncPodFn(syncPodOptions{
		mirrorPod:      options.MirrorPod,
		pod:            options.Pod,
		podStatus:      status,
		updateType:     options.UpdateType,
		killPodOptions: options.KillPodOptions,
	}); err != nil {
		f.t.Errorf("Unexpected error: %v", err)
	}
}

func (f *fakePodWorkers) ForgetNonExistingPodWorkers(desiredPods map[types.UID]sets.Empty) {}

func (f *fakePodWorkers) ForgetWorker(uid types.UID) {}

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
}

func createPodWorkers() (*podWorkers, map[types.UID][]syncPodRecord) {
	lock := sync.Mutex{}
	processed := make(map[types.UID][]syncPodRecord)
	fakeRecorder := &record.FakeRecorder{}
	fakeRuntime := &containertest.FakeRuntime{}
	fakeCache := containertest.NewFakeCache(fakeRuntime)
	podWorkers := newPodWorkers(
		func(options syncPodOptions) error {
			func() {
				lock.Lock()
				defer lock.Unlock()
				pod := options.pod
				processed[pod.UID] = append(processed[pod.UID], syncPodRecord{
					name:       pod.Name,
					updateType: options.updateType,
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
	return podWorkers, processed
}

func drainWorkers(podWorkers *podWorkers, numPods int) {
	for {
		stillWorking := false
		podWorkers.podLock.Lock()
		for i := 0; i < numPods; i++ {
			if podWorkers.isWorking[types.UID(string(i))] {
				stillWorking = true
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
			podWorkers.UpdatePod(&UpdatePodOptions{
				Pod:        newPod(string(j), string(i)),
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
		uid := types.UID(i)
		if len(processed[uid]) < 1 || len(processed[uid]) > i+1 {
			t.Errorf("Pod %v processed %v times", i, len(processed[uid]))
			continue
		}

		// PodWorker guarantees the first and the last event will be processed
		first := 0
		last := len(processed[uid]) - 1
		if processed[uid][first].name != string(0) {
			t.Errorf("Pod %v: incorrect order %v, %v", i, first, processed[uid][first])

		}
		if processed[uid][last].name != string(i) {
			t.Errorf("Pod %v: incorrect order %v, %v", i, last, processed[uid][last])
		}
	}
}

func TestUpdatePodDoesNotForgetSyncPodKill(t *testing.T) {
	podWorkers, processed := createPodWorkers()
	numPods := 20
	for i := 0; i < numPods; i++ {
		pod := newPod(string(i), string(i))
		podWorkers.UpdatePod(&UpdatePodOptions{
			Pod:        pod,
			UpdateType: kubetypes.SyncPodCreate,
		})
		podWorkers.UpdatePod(&UpdatePodOptions{
			Pod:        pod,
			UpdateType: kubetypes.SyncPodKill,
		})
		podWorkers.UpdatePod(&UpdatePodOptions{
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
		uid := types.UID(i)
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

func TestForgetNonExistingPodWorkers(t *testing.T) {
	podWorkers, _ := createPodWorkers()

	numPods := 20
	for i := 0; i < numPods; i++ {
		podWorkers.UpdatePod(&UpdatePodOptions{
			Pod:        newPod(string(i), "name"),
			UpdateType: kubetypes.SyncPodUpdate,
		})
	}
	drainWorkers(podWorkers, numPods)

	if len(podWorkers.podUpdates) != numPods {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}

	desiredPods := map[types.UID]sets.Empty{}
	desiredPods[types.UID(2)] = sets.Empty{}
	desiredPods[types.UID(14)] = sets.Empty{}
	podWorkers.ForgetNonExistingPodWorkers(desiredPods)
	if len(podWorkers.podUpdates) != 2 {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}
	if _, exists := podWorkers.podUpdates[types.UID(2)]; !exists {
		t.Errorf("No updates channel for pod 2")
	}
	if _, exists := podWorkers.podUpdates[types.UID(14)]; !exists {
		t.Errorf("No updates channel for pod 14")
	}

	podWorkers.ForgetNonExistingPodWorkers(map[types.UID]sets.Empty{})
	if len(podWorkers.podUpdates) != 0 {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}
}

type simpleFakeKubelet struct {
	pod       *v1.Pod
	mirrorPod *v1.Pod
	podStatus *kubecontainer.PodStatus
	wg        sync.WaitGroup
}

func (kl *simpleFakeKubelet) syncPod(options syncPodOptions) error {
	kl.pod, kl.mirrorPod, kl.podStatus = options.pod, options.mirrorPod, options.podStatus
	return nil
}

func (kl *simpleFakeKubelet) syncPodWithWaitGroup(options syncPodOptions) error {
	kl.pod, kl.mirrorPod, kl.podStatus = options.pod, options.mirrorPod, options.podStatus
	kl.wg.Done()
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

	realPodWorkers := newPodWorkers(kubeletForRealWorkers.syncPodWithWaitGroup, fakeRecorder, queue.NewBasicWorkQueue(&clock.RealClock{}), time.Second, time.Second, fakeCache)
	fakePodWorkers := &fakePodWorkers{kubeletForFakeWorkers.syncPod, fakeCache, t}

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
		realPodWorkers.UpdatePod(&UpdatePodOptions{
			Pod:        tt.pod,
			MirrorPod:  tt.mirrorPod,
			UpdateType: kubetypes.SyncPodUpdate,
		})
		fakePodWorkers.UpdatePod(&UpdatePodOptions{
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
	err := killPodFunc(pod, v1.PodStatus{Phase: v1.PodFailed, Reason: "reason", Message: "message"}, &gracePeriodOverride)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(processed) != 1 {
		t.Errorf("len(processed) expected: %v, actual: %v", 1, len(processed))
		return
	}
	syncPodRecords := processed[pod.UID]
	if len(syncPodRecords) != 1 {
		t.Errorf("Pod processed %v times, but expected %v", len(syncPodRecords), 1)
	}
	if syncPodRecords[0].updateType != kubetypes.SyncPodKill {
		t.Errorf("Pod update type was %v, but expected %v", syncPodRecords[0].updateType, kubetypes.SyncPodKill)
	}
}
