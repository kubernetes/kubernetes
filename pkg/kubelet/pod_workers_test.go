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

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/queue"
	"k8s.io/utils/clock"
)

// fakePodWorkers runs sync pod function in serial, so we can have
// deterministic behaviour in testing.
type fakePodWorkers struct {
	lock      sync.Mutex
	syncPodFn syncPodFnType
	cache     kubecontainer.Cache
	t         TestingInterface

	triggeredDeletion []types.UID
	triggeredTerminal []types.UID

	statusLock            sync.Mutex
	running               map[types.UID]bool
	terminating           map[types.UID]bool
	terminated            map[types.UID]bool
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
		isTerminal, err := f.syncPodFn(context.Background(), options.UpdateType, options.Pod, options.MirrorPod, status)
		if err != nil {
			f.t.Errorf("Unexpected error: %v", err)
		}
		if isTerminal {
			f.triggeredTerminal = append(f.triggeredTerminal, uid)
		}
	}
}

func (f *fakePodWorkers) SyncKnownPods(desiredPods []*v1.Pod) map[types.UID]PodWorkerState {
	return nil
}

func (f *fakePodWorkers) IsPodKnownTerminated(uid types.UID) bool {
	f.statusLock.Lock()
	defer f.statusLock.Unlock()
	return f.terminated[uid]
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
func (f *fakePodWorkers) setPodRuntimeBeRemoved(uid types.UID) {
	f.statusLock.Lock()
	defer f.statusLock.Unlock()
	f.removeRuntime = map[types.UID]bool{uid: true}
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

func newPodWithPhase(uid, name string, phase v1.PodPhase) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:  types.UID(uid),
			Name: name,
		},
		Status: v1.PodStatus{
			Phase: phase,
		},
	}
}

func newStaticPod(uid, name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:  types.UID(uid),
			Name: name,
			Annotations: map[string]string{
				kubetypes.ConfigSourceAnnotationKey: kubetypes.FileSource,
			},
		},
	}
}

func newNamedPod(uid, namespace, name string, isStatic bool) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID(uid),
			Namespace: namespace,
			Name:      name,
		},
	}
	if isStatic {
		pod.Annotations = map[string]string{
			kubetypes.ConfigSourceAnnotationKey: kubetypes.FileSource,
		}
	}
	return pod
}

// syncPodRecord is a record of a sync pod call
type syncPodRecord struct {
	name       string
	updateType kubetypes.SyncPodType
	runningPod *kubecontainer.Pod
	terminated bool
}

type FakeQueueItem struct {
	UID   types.UID
	Delay time.Duration
}

type fakeQueue struct {
	lock         sync.Mutex
	queue        []FakeQueueItem
	currentStart int
}

func (q *fakeQueue) Empty() bool {
	q.lock.Lock()
	defer q.lock.Unlock()
	return (len(q.queue) - q.currentStart) == 0
}

func (q *fakeQueue) Items() []FakeQueueItem {
	q.lock.Lock()
	defer q.lock.Unlock()
	return append(make([]FakeQueueItem, 0, len(q.queue)), q.queue...)
}

func (q *fakeQueue) Set() sets.String {
	q.lock.Lock()
	defer q.lock.Unlock()
	work := sets.NewString()
	for _, item := range q.queue[q.currentStart:] {
		work.Insert(string(item.UID))
	}
	return work
}

func (q *fakeQueue) Enqueue(uid types.UID, delay time.Duration) {
	q.lock.Lock()
	defer q.lock.Unlock()
	q.queue = append(q.queue, FakeQueueItem{UID: uid, Delay: delay})
}

func (q *fakeQueue) GetWork() []types.UID {
	q.lock.Lock()
	defer q.lock.Unlock()
	work := make([]types.UID, 0, len(q.queue)-q.currentStart)
	for _, item := range q.queue[q.currentStart:] {
		work = append(work, item.UID)
	}
	q.currentStart = len(q.queue)
	return work
}

func createPodWorkers() (*podWorkers, map[types.UID][]syncPodRecord) {
	lock := sync.Mutex{}
	processed := make(map[types.UID][]syncPodRecord)
	fakeRecorder := &record.FakeRecorder{}
	fakeRuntime := &containertest.FakeRuntime{}
	fakeCache := containertest.NewFakeCache(fakeRuntime)
	fakeQueue := &fakeQueue{}
	w := newPodWorkers(
		func(ctx context.Context, updateType kubetypes.SyncPodType, pod, mirrorPod *v1.Pod, podStatus *kubecontainer.PodStatus) (bool, error) {
			func() {
				lock.Lock()
				defer lock.Unlock()
				pod := pod
				processed[pod.UID] = append(processed[pod.UID], syncPodRecord{
					name:       pod.Name,
					updateType: updateType,
				})
			}()
			return false, nil
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
		fakeQueue,
		time.Second,
		time.Millisecond,
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

func drainWorkersExcept(podWorkers *podWorkers, uids ...types.UID) {
	set := sets.NewString()
	for _, uid := range uids {
		set.Insert(string(uid))
	}
	for {
		stillWorking := false
		podWorkers.podLock.Lock()
		for k, v := range podWorkers.podSyncStatuses {
			if set.Has(string(k)) {
				continue
			}
			if v.working {
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

func TestUpdatePodWithTerminatedPod(t *testing.T) {
	podWorkers, _ := createPodWorkers()
	terminatedPod := newPodWithPhase("0000-0000-0000", "done-pod", v1.PodSucceeded)
	orphanedPod := &kubecontainer.Pod{ID: "0000-0000-0001", Name: "orphaned-pod"}
	pod := newPod("0000-0000-0002", "running-pod")

	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        terminatedPod,
		UpdateType: kubetypes.SyncPodCreate,
	})
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        pod,
		UpdateType: kubetypes.SyncPodCreate,
	})
	podWorkers.UpdatePod(UpdatePodOptions{
		UpdateType: kubetypes.SyncPodKill,
		RunningPod: orphanedPod,
	})
	drainAllWorkers(podWorkers)
	if podWorkers.IsPodKnownTerminated(pod.UID) == true {
		t.Errorf("podWorker state should not be terminated")
	}
	if podWorkers.IsPodKnownTerminated(terminatedPod.UID) == false {
		t.Errorf("podWorker state should be terminated")
	}
	if podWorkers.IsPodKnownTerminated(orphanedPod.ID) == false {
		t.Errorf("podWorker state should be terminated for orphaned pod")
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

func newUIDSet(uids ...types.UID) sets.String {
	set := sets.NewString()
	for _, uid := range uids {
		set.Insert(string(uid))
	}
	return set
}

type terminalPhaseSync struct {
	lock     sync.Mutex
	fn       syncPodFnType
	terminal sets.String
}

func (s *terminalPhaseSync) SyncPod(ctx context.Context, updateType kubetypes.SyncPodType, pod *v1.Pod, mirrorPod *v1.Pod, podStatus *kubecontainer.PodStatus) (bool, error) {
	isTerminal, err := s.fn(ctx, updateType, pod, mirrorPod, podStatus)
	if err != nil {
		return false, err
	}
	if !isTerminal {
		s.lock.Lock()
		defer s.lock.Unlock()
		isTerminal = s.terminal.Has(string(pod.UID))
	}
	return isTerminal, nil
}

func (s *terminalPhaseSync) SetTerminal(uid types.UID) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.terminal.Insert(string(uid))
}

func newTerminalPhaseSync(fn syncPodFnType) *terminalPhaseSync {
	return &terminalPhaseSync{
		fn:       fn,
		terminal: sets.NewString(),
	}
}

func TestTerminalPhaseTransition(t *testing.T) {
	podWorkers, _ := createPodWorkers()
	var channels WorkChannel
	podWorkers.workerChannelFn = channels.Intercept
	terminalPhaseSyncer := newTerminalPhaseSync(podWorkers.syncPodFn)
	podWorkers.syncPodFn = terminalPhaseSyncer.SyncPod

	// start pod
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("1", "test1", "pod1", false),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)

	// should observe pod running
	pod1 := podWorkers.podSyncStatuses[types.UID("1")]
	if pod1.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod1)
	}

	// send another update to the pod
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("1", "test1", "pod1", false),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)

	// should observe pod still running
	pod1 = podWorkers.podSyncStatuses[types.UID("1")]
	if pod1.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod1)
	}

	// the next sync should result in a transition to terminal
	terminalPhaseSyncer.SetTerminal(types.UID("1"))
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("1", "test1", "pod1", false),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)

	// should observe pod terminating
	pod1 = podWorkers.podSyncStatuses[types.UID("1")]
	if !pod1.IsTerminationRequested() || !pod1.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod1)
	}
}

func TestStaticPodExclusion(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	podWorkers, processed := createPodWorkers()
	var channels WorkChannel
	podWorkers.workerChannelFn = channels.Intercept

	testPod := newNamedPod("2-static", "test1", "pod1", true)
	if !kubetypes.IsStaticPod(testPod) {
		t.Fatalf("unable to test static pod")
	}

	// start two pods with the same name, one static, one apiserver
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("1-normal", "test1", "pod1", false),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("2-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)

	// should observe both pods running
	pod1 := podWorkers.podSyncStatuses[types.UID("1-normal")]
	if pod1.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod1)
	}
	pod2 := podWorkers.podSyncStatuses[types.UID("2-static")]
	if pod2.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod2)
	}

	if len(processed) != 2 {
		t.Fatalf("unexpected synced pods: %#v", processed)
	}
	if e, a :=
		[]syncPodRecord{{name: "pod1", updateType: kubetypes.SyncPodUpdate}},
		processed[types.UID("2-static")]; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected sync pod calls: %s", cmp.Diff(e, a))
	}
	if e, a := map[string]types.UID{"pod1_test1": "2-static"}, podWorkers.startedStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected started static pods: %s", cmp.Diff(e, a))
	}

	// attempt to start a second and third static pod, which should not start
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("3-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("4-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)

	// should observe both pods running but last pod shouldn't have synced
	pod1 = podWorkers.podSyncStatuses[types.UID("1-normal")]
	if pod1.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod1)
	}
	pod2 = podWorkers.podSyncStatuses[types.UID("2-static")]
	if pod2.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod2)
	}
	pod3 := podWorkers.podSyncStatuses[types.UID("3-static")]
	if pod3.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod3)
	}
	pod4 := podWorkers.podSyncStatuses[types.UID("4-static")]
	if pod4.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod4)
	}

	if len(processed) != 2 {
		t.Fatalf("unexpected synced pods: %#v", processed)
	}
	if expected, actual :=
		[]syncPodRecord{{name: "pod1", updateType: kubetypes.SyncPodUpdate}},
		processed[types.UID("2-static")]; !reflect.DeepEqual(expected, actual) {
		t.Fatalf("unexpected sync pod calls: %s", cmp.Diff(expected, actual))
	}
	if expected, actual :=
		[]syncPodRecord(nil),
		processed[types.UID("3-static")]; !reflect.DeepEqual(expected, actual) {
		t.Fatalf("unexpected sync pod calls: %s", cmp.Diff(expected, actual))
	}
	if expected, actual :=
		[]syncPodRecord(nil),
		processed[types.UID("4-static")]; !reflect.DeepEqual(expected, actual) {
		t.Fatalf("unexpected sync pod calls: %s", cmp.Diff(expected, actual))
	}
	if e, a := map[string]types.UID{"pod1_test1": "2-static"}, podWorkers.startedStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected started static pods: %s", cmp.Diff(e, a))
	}
	if e, a := map[string][]types.UID{"pod1_test1": {"3-static", "4-static"}}, podWorkers.waitingToStartStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected waiting static pods: %s", cmp.Diff(e, a))
	}
	// verify all are enqueued
	if e, a := sets.NewString("1-normal", "2-static", "4-static", "3-static"), podWorkers.workQueue.(*fakeQueue).Set(); !e.Equal(a) {
		t.Fatalf("unexpected queued items: %s", cmp.Diff(e, a))
	}

	// send a basic update for 3-static
	podWorkers.workQueue.GetWork()
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("3-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)

	// 3-static should not be started because 2-static is still running
	if e, a := map[string]types.UID{"pod1_test1": "2-static"}, podWorkers.startedStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected started static pods: %s", cmp.Diff(e, a))
	}
	if e, a := map[string][]types.UID{"pod1_test1": {"3-static", "4-static"}}, podWorkers.waitingToStartStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected waiting static pods: %s", cmp.Diff(e, a))
	}
	// the queue should include a single item for 3-static (indicating we need to retry later)
	if e, a := sets.NewString("3-static"), newUIDSet(podWorkers.workQueue.GetWork()...); !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected queued items: %s", cmp.Diff(e, a))
	}

	// mark 3-static as deleted while 2-static is still running
	podWorkers.workQueue.GetWork()
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("3-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodKill,
	})
	drainAllWorkers(podWorkers)

	// should observe 3-static as terminated because it has never started, but other state should be a no-op
	pod3 = podWorkers.podSyncStatuses[types.UID("3-static")]
	if !pod3.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod3)
	}
	// the queue should be empty because the worker is now done
	if e, a := sets.NewString(), newUIDSet(podWorkers.workQueue.GetWork()...); !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected queued items: %s", cmp.Diff(e, a))
	}
	// 2-static is still running
	if e, a := map[string]types.UID{"pod1_test1": "2-static"}, podWorkers.startedStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected started static pods: %s", cmp.Diff(e, a))
	}
	// 3-static and 4-static are both still queued
	if e, a := map[string][]types.UID{"pod1_test1": {"3-static", "4-static"}}, podWorkers.waitingToStartStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected waiting static pods: %s", cmp.Diff(e, a))
	}

	// terminate 2-static
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("2-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodKill,
	})
	drainAllWorkers(podWorkers)

	// should observe 2-static as terminated, and 2-static should no longer be reported as the started static pod
	pod2 = podWorkers.podSyncStatuses[types.UID("2-static")]
	if !pod2.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod3)
	}
	if e, a := map[string]types.UID{}, podWorkers.startedStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected started static pods: %s", cmp.Diff(e, a))
	}
	if e, a := map[string][]types.UID{"pod1_test1": {"3-static", "4-static"}}, podWorkers.waitingToStartStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected waiting static pods: %s", cmp.Diff(e, a))
	}

	// simulate a periodic event from the work queue for 4-static
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("4-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)

	// 4-static should be started because 3-static has already terminated
	pod4 = podWorkers.podSyncStatuses[types.UID("4-static")]
	if pod4.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", pod3)
	}
	if e, a := map[string]types.UID{"pod1_test1": "4-static"}, podWorkers.startedStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected started static pods: %s", cmp.Diff(e, a))
	}
	if e, a := map[string][]types.UID{}, podWorkers.waitingToStartStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected waiting static pods: %s", cmp.Diff(e, a))
	}

	// initiate a sync with all pods remaining
	state := podWorkers.SyncKnownPods([]*v1.Pod{
		newNamedPod("1-normal", "test1", "pod1", false),
		newNamedPod("2-static", "test1", "pod1", true),
		newNamedPod("3-static", "test1", "pod1", true),
		newNamedPod("4-static", "test1", "pod1", true),
	})
	drainAllWorkers(podWorkers)

	// 2-static and 3-static should both be listed as terminated
	if e, a := map[types.UID]PodWorkerState{
		"1-normal": SyncPod,
		"2-static": TerminatedPod,
		"3-static": TerminatedPod,
		"4-static": SyncPod,
	}, state; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected actual state: %s", cmp.Diff(e, a))
	}
	// 3-static is still in the config, it should still be in our status
	if status, ok := podWorkers.podSyncStatuses["3-static"]; !ok || status.terminatedAt.IsZero() || !status.finished || status.working {
		t.Fatalf("unexpected post termination status: %#v", status)
	}

	// initiate a sync with 3-static removed
	state = podWorkers.SyncKnownPods([]*v1.Pod{
		newNamedPod("1-normal", "test1", "pod1", false),
		newNamedPod("2-static", "test1", "pod1", true),
		newNamedPod("4-static", "test1", "pod1", true),
	})
	drainAllWorkers(podWorkers)

	// expect sync to put 3-static into final state and remove the status
	if e, a := map[types.UID]PodWorkerState{
		"1-normal": SyncPod,
		"2-static": TerminatedPod,
		"3-static": TerminatedPod,
		"4-static": SyncPod,
	}, state; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected actual state: %s", cmp.Diff(e, a))
	}
	if status, ok := podWorkers.podSyncStatuses["3-static"]; ok {
		t.Fatalf("unexpected post termination status: %#v", status)
	}

	// start a static pod, kill it, then add another one, but ensure the pod worker
	// for pod 5 doesn't see the kill event (so it remains waiting to start)
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("5-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	channels.Channel("5-static").Hold()
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("5-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodKill,
	})
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("6-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainWorkersExcept(podWorkers, "5-static")

	// pod 5 should have termination requested, but hasn't cleaned up
	pod5 := podWorkers.podSyncStatuses[types.UID("5-static")]
	if !pod5.IsTerminationRequested() || pod5.IsTerminated() {
		t.Fatalf("unexpected status for pod 5: %#v", pod5)
	}
	if e, a := map[string]types.UID{"pod1_test1": "4-static"}, podWorkers.startedStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected started static pods: %s", cmp.Diff(e, a))
	}
	if e, a := map[string][]types.UID{"pod1_test1": {"5-static", "6-static"}}, podWorkers.waitingToStartStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected waiting static pods: %s", cmp.Diff(e, a))
	}

	// terminate 4-static and wake 6-static
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("4-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodKill,
	})
	drainWorkersExcept(podWorkers, "5-static")
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("6-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainWorkersExcept(podWorkers, "5-static")

	// 5-static should still be waiting, 6-static should have started and synced
	pod5 = podWorkers.podSyncStatuses[types.UID("5-static")]
	if !pod5.IsTerminationRequested() || pod5.IsTerminated() {
		t.Fatalf("unexpected status for pod 5: %#v", pod5)
	}
	if e, a := map[string]types.UID{"pod1_test1": "6-static"}, podWorkers.startedStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected started static pods: %s", cmp.Diff(e, a))
	}
	// no static pods shoud be waiting
	if e, a := map[string][]types.UID{}, podWorkers.waitingToStartStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected waiting static pods: %s", cmp.Diff(e, a))
	}
	// prove 6-static synced
	if expected, actual :=
		[]syncPodRecord{{name: "pod1", updateType: kubetypes.SyncPodUpdate}},
		processed[types.UID("6-static")]; !reflect.DeepEqual(expected, actual) {
		t.Fatalf("unexpected sync pod calls: %s", cmp.Diff(expected, actual))
	}

	// ensure 5-static exits when we deliver the event out of order
	channels.Channel("5-static").Release()
	drainAllWorkers(podWorkers)
	pod5 = podWorkers.podSyncStatuses[types.UID("5-static")]
	if !pod5.IsTerminated() {
		t.Fatalf("unexpected status for pod 5: %#v", pod5)
	}

	// start three more static pods, kill the previous static pod blocking start,
	// and simulate the second pod of three (8) getting to run first
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("7-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("8-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("9-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("6-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodKill,
	})
	drainAllWorkers(podWorkers)
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("8-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)

	// 7 and 8 should both be waiting still with no syncs
	if e, a := map[string]types.UID{}, podWorkers.startedStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected started static pods: %s", cmp.Diff(e, a))
	}
	// only 7-static can start now, but it hasn't received an event
	if e, a := map[string][]types.UID{"pod1_test1": {"7-static", "8-static", "9-static"}}, podWorkers.waitingToStartStaticPodsByFullname; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected waiting static pods: %s", cmp.Diff(e, a))
	}
	// none of the new pods have synced
	if expected, actual :=
		[]syncPodRecord(nil),
		processed[types.UID("7-static")]; !reflect.DeepEqual(expected, actual) {
		t.Fatalf("unexpected sync pod calls: %s", cmp.Diff(expected, actual))
	}
	if expected, actual :=
		[]syncPodRecord(nil),
		processed[types.UID("8-static")]; !reflect.DeepEqual(expected, actual) {
		t.Fatalf("unexpected sync pod calls: %s", cmp.Diff(expected, actual))
	}
	if expected, actual :=
		[]syncPodRecord(nil),
		processed[types.UID("9-static")]; !reflect.DeepEqual(expected, actual) {
		t.Fatalf("unexpected sync pod calls: %s", cmp.Diff(expected, actual))
	}

	// terminate 7-static and wake 8-static
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("7-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodKill,
	})
	drainAllWorkers(podWorkers)
	podWorkers.UpdatePod(UpdatePodOptions{
		Pod:        newNamedPod("8-static", "test1", "pod1", true),
		UpdateType: kubetypes.SyncPodUpdate,
	})
	drainAllWorkers(podWorkers)

	// 8 should have synced
	if expected, actual :=
		[]syncPodRecord{{name: "pod1", updateType: kubetypes.SyncPodUpdate}},
		processed[types.UID("8-static")]; !reflect.DeepEqual(expected, actual) {
		t.Fatalf("unexpected sync pod calls: %s", cmp.Diff(expected, actual))
	}
}

type WorkChannelItem struct {
	out   chan podWork
	lock  sync.Mutex
	pause bool
	queue []podWork
}

func (item *WorkChannelItem) Handle(work podWork) {
	item.lock.Lock()
	defer item.lock.Unlock()
	if item.pause {
		item.queue = append(item.queue, work)
		return
	}
	item.out <- work
}

func (item *WorkChannelItem) Hold() {
	item.lock.Lock()
	defer item.lock.Unlock()
	item.pause = true
}

func (item *WorkChannelItem) Close() {
	item.lock.Lock()
	defer item.lock.Unlock()
	if item.out != nil {
		close(item.out)
		item.out = nil
	}
}

// Release blocks until all work is passed on the chain
func (item *WorkChannelItem) Release() {
	item.lock.Lock()
	defer item.lock.Unlock()
	item.pause = false
	for _, work := range item.queue {
		item.out <- work
	}
	item.queue = nil
}

// WorkChannel intercepts podWork channels between the pod worker and its child
// goroutines and allows tests to pause or release the flow of podWork to the
// workers.
type WorkChannel struct {
	lock     sync.Mutex
	channels map[types.UID]*WorkChannelItem
}

func (w *WorkChannel) Channel(uid types.UID) *WorkChannelItem {
	w.lock.Lock()
	defer w.lock.Unlock()
	if w.channels == nil {
		w.channels = make(map[types.UID]*WorkChannelItem)
	}
	channel, ok := w.channels[uid]
	if !ok {
		channel = &WorkChannelItem{
			out: make(chan podWork, 1),
		}
		w.channels[uid] = channel
	}
	return channel
}

func (w *WorkChannel) Intercept(uid types.UID, ch chan podWork) (outCh <-chan podWork) {
	channel := w.Channel(uid)
	w.lock.Lock()

	defer w.lock.Unlock()
	go func() {
		defer func() {
			channel.Close()
			w.lock.Lock()
			defer w.lock.Unlock()
			delete(w.channels, uid)
		}()
		for w := range ch {
			channel.Handle(w)
		}
	}()
	return channel.out
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

func Test_removeTerminatedWorker(t *testing.T) {
	podUID := types.UID("pod-uid")

	testCases := []struct {
		desc                               string
		podSyncStatus                      *podSyncStatus
		startedStaticPodsByFullname        map[string]types.UID
		waitingToStartStaticPodsByFullname map[string][]types.UID
		removed                            bool
	}{
		{
			desc: "finished worker",
			podSyncStatus: &podSyncStatus{
				finished: true,
			},
			removed: true,
		},
		{
			desc: "waiting to start worker because of another started pod with the same fullname",
			podSyncStatus: &podSyncStatus{
				finished: false,
				fullname: "fake-fullname",
			},
			startedStaticPodsByFullname: map[string]types.UID{
				"fake-fullname": "another-pod-uid",
			},
			waitingToStartStaticPodsByFullname: map[string][]types.UID{
				"fake-fullname": {podUID},
			},
			removed: false,
		},
		{
			desc: "not yet started worker",
			podSyncStatus: &podSyncStatus{
				finished: false,
				fullname: "fake-fullname",
			},
			startedStaticPodsByFullname: make(map[string]types.UID),
			waitingToStartStaticPodsByFullname: map[string][]types.UID{
				"fake-fullname": {podUID},
			},
			removed: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			podWorkers, _ := createPodWorkers()
			podWorkers.podSyncStatuses[podUID] = tc.podSyncStatus
			podWorkers.startedStaticPodsByFullname = tc.startedStaticPodsByFullname
			podWorkers.waitingToStartStaticPodsByFullname = tc.waitingToStartStaticPodsByFullname

			podWorkers.removeTerminatedWorker(podUID)
			_, exists := podWorkers.podSyncStatuses[podUID]
			if tc.removed && exists {
				t.Errorf("Expected pod worker to be removed")
			}
			if !tc.removed && !exists {
				t.Errorf("Expected pod worker to not be removed")
			}
		})
	}
}

type simpleFakeKubelet struct {
	pod       *v1.Pod
	mirrorPod *v1.Pod
	podStatus *kubecontainer.PodStatus
	wg        sync.WaitGroup
}

func (kl *simpleFakeKubelet) syncPod(ctx context.Context, updateType kubetypes.SyncPodType, pod, mirrorPod *v1.Pod, podStatus *kubecontainer.PodStatus) (bool, error) {
	kl.pod, kl.mirrorPod, kl.podStatus = pod, mirrorPod, podStatus
	return false, nil
}

func (kl *simpleFakeKubelet) syncPodWithWaitGroup(ctx context.Context, updateType kubetypes.SyncPodType, pod, mirrorPod *v1.Pod, podStatus *kubecontainer.PodStatus) (bool, error) {
	kl.pod, kl.mirrorPod, kl.podStatus = pod, mirrorPod, podStatus
	kl.wg.Done()
	return false, nil
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

func Test_allowPodStart(t *testing.T) {
	testCases := []struct {
		desc                               string
		pod                                *v1.Pod
		podSyncStatuses                    map[types.UID]*podSyncStatus
		startedStaticPodsByFullname        map[string]types.UID
		waitingToStartStaticPodsByFullname map[string][]types.UID

		expectedStartedStaticPodsByFullname        map[string]types.UID
		expectedWaitingToStartStaticPodsByFullname map[string][]types.UID
		allowed                                    bool
		allowedEver                                bool
	}{
		{
			// TODO: Do we want to allow non-static pods with the same full name?
			// Note that it may disable the force deletion of pods.
			desc: "non-static pod",
			pod:  newPod("uid-0", "test"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "test_",
				},
				"uid-1": {
					fullname: "test_",
				},
			},
			allowed:     true,
			allowedEver: true,
		},
		{
			// TODO: Do we want to allow a non-static pod with the same full name
			// as the started static pod?
			desc: "non-static pod when there is a started static pod with the same full name",
			pod:  newPod("uid-0", "test"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "test_",
				},
				"uid-1": {
					fullname: "test_",
				},
			},
			startedStaticPodsByFullname: map[string]types.UID{
				"test_": types.UID("uid-1"),
			},
			expectedStartedStaticPodsByFullname: map[string]types.UID{
				"test_": types.UID("uid-1"),
			},
			allowed:     true,
			allowedEver: true,
		},
		{
			// TODO: Do we want to allow a static pod with the same full name as the
			// started non-static pod?
			desc: "static pod when there is a started non-static pod with the same full name",
			pod:  newPod("uid-0", "test"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "test_",
				},
				"uid-1": {
					fullname: "test_",
				},
			},
			allowed:     true,
			allowedEver: true,
		},
		{
			desc: "static pod when there are no started static pods with the same full name",
			pod:  newStaticPod("uid-0", "foo"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "foo_",
				},
				"uid-1": {
					fullname: "bar_",
				},
			},
			startedStaticPodsByFullname: map[string]types.UID{
				"bar_": types.UID("uid-1"),
			},
			expectedStartedStaticPodsByFullname: map[string]types.UID{
				"foo_": types.UID("uid-0"),
				"bar_": types.UID("uid-1"),
			},
			allowed:     true,
			allowedEver: true,
		},
		{
			desc: "static pod when there is a started static pod with the same full name",
			pod:  newStaticPod("uid-0", "foo"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "foo_",
				},
				"uid-1": {
					fullname: "foo_",
				},
			},
			startedStaticPodsByFullname: map[string]types.UID{
				"foo_": types.UID("uid-1"),
			},
			expectedStartedStaticPodsByFullname: map[string]types.UID{
				"foo_": types.UID("uid-1"),
			},
			allowed:     false,
			allowedEver: true,
		},
		{
			desc: "static pod if the static pod has already started",
			pod:  newStaticPod("uid-0", "foo"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "foo_",
				},
			},
			startedStaticPodsByFullname: map[string]types.UID{
				"foo_": types.UID("uid-0"),
			},
			expectedStartedStaticPodsByFullname: map[string]types.UID{
				"foo_": types.UID("uid-0"),
			},
			allowed:     true,
			allowedEver: true,
		},
		{
			desc: "static pod if the static pod is the first pod waiting to start",
			pod:  newStaticPod("uid-0", "foo"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "foo_",
				},
			},
			waitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-0"),
				},
			},
			expectedStartedStaticPodsByFullname: map[string]types.UID{
				"foo_": types.UID("uid-0"),
			},
			expectedWaitingToStartStaticPodsByFullname: make(map[string][]types.UID),
			allowed:     true,
			allowedEver: true,
		},
		{
			desc: "static pod if the static pod is not the first pod waiting to start",
			pod:  newStaticPod("uid-0", "foo"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "foo_",
				},
				"uid-1": {
					fullname: "foo_",
				},
			},
			waitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-1"),
					types.UID("uid-0"),
				},
			},
			expectedStartedStaticPodsByFullname: make(map[string]types.UID),
			expectedWaitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-1"),
					types.UID("uid-0"),
				},
			},
			allowed:     false,
			allowedEver: true,
		},
		{
			desc: "static pod if the static pod is the first valid pod waiting to start / clean up until picking the first valid pod",
			pod:  newStaticPod("uid-0", "foo"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "foo_",
				},
				"uid-1": {
					fullname: "foo_",
				},
			},
			waitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-2"),
					types.UID("uid-2"),
					types.UID("uid-3"),
					types.UID("uid-0"),
					types.UID("uid-1"),
				},
			},
			expectedStartedStaticPodsByFullname: map[string]types.UID{
				"foo_": types.UID("uid-0"),
			},
			expectedWaitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-1"),
				},
			},
			allowed:     true,
			allowedEver: true,
		},
		{
			desc: "static pod if the static pod is the first pod that is not termination requested and waiting to start",
			pod:  newStaticPod("uid-0", "foo"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname: "foo_",
				},
				"uid-1": {
					fullname: "foo_",
				},
				"uid-2": {
					fullname:      "foo_",
					terminatingAt: time.Now(),
				},
				"uid-3": {
					fullname:     "foo_",
					terminatedAt: time.Now(),
				},
			},
			waitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-2"),
					types.UID("uid-3"),
					types.UID("uid-0"),
					types.UID("uid-1"),
				},
			},
			expectedStartedStaticPodsByFullname: map[string]types.UID{
				"foo_": types.UID("uid-0"),
			},
			expectedWaitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-1"),
				},
			},
			allowed:     true,
			allowedEver: true,
		},
		{
			desc: "static pod if there is no sync status for the pod should be denied",
			pod:  newStaticPod("uid-0", "foo"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-1": {
					fullname: "foo_",
				},
				"uid-2": {
					fullname:      "foo_",
					terminatingAt: time.Now(),
				},
				"uid-3": {
					fullname:     "foo_",
					terminatedAt: time.Now(),
				},
			},
			waitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-1"),
				},
			},
			expectedStartedStaticPodsByFullname: map[string]types.UID{},
			expectedWaitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-1"),
				},
			},
			allowed:     false,
			allowedEver: false,
		},
		{
			desc: "static pod if the static pod is terminated should not be allowed",
			pod:  newStaticPod("uid-0", "foo"),
			podSyncStatuses: map[types.UID]*podSyncStatus{
				"uid-0": {
					fullname:      "foo_",
					terminatingAt: time.Now(),
				},
			},
			waitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-2"),
					types.UID("uid-3"),
					types.UID("uid-0"),
					types.UID("uid-1"),
				},
			},
			expectedStartedStaticPodsByFullname: map[string]types.UID{},
			expectedWaitingToStartStaticPodsByFullname: map[string][]types.UID{
				"foo_": {
					types.UID("uid-2"),
					types.UID("uid-3"),
					types.UID("uid-0"),
					types.UID("uid-1"),
				},
			},
			allowed:     false,
			allowedEver: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			podWorkers, _ := createPodWorkers()
			if tc.podSyncStatuses != nil {
				podWorkers.podSyncStatuses = tc.podSyncStatuses
			}
			if tc.startedStaticPodsByFullname != nil {
				podWorkers.startedStaticPodsByFullname = tc.startedStaticPodsByFullname
			}
			if tc.waitingToStartStaticPodsByFullname != nil {
				podWorkers.waitingToStartStaticPodsByFullname = tc.waitingToStartStaticPodsByFullname
			}
			allowed, allowedEver := podWorkers.allowPodStart(tc.pod)
			if allowed != tc.allowed {
				if tc.allowed {
					t.Errorf("Pod should be allowed")
				} else {
					t.Errorf("Pod should not be allowed")
				}
			}

			if allowedEver != tc.allowedEver {
				if tc.allowedEver {
					t.Errorf("Pod should be allowed ever")
				} else {
					t.Errorf("Pod should not be allowed ever")
				}
			}

			// if maps are neither nil nor empty
			if len(podWorkers.startedStaticPodsByFullname) != 0 ||
				len(podWorkers.startedStaticPodsByFullname) != len(tc.expectedStartedStaticPodsByFullname) {
				if !reflect.DeepEqual(
					podWorkers.startedStaticPodsByFullname,
					tc.expectedStartedStaticPodsByFullname) {
					t.Errorf("startedStaticPodsByFullname: expected %v, got %v",
						tc.expectedStartedStaticPodsByFullname,
						podWorkers.startedStaticPodsByFullname)
				}
			}

			// if maps are neither nil nor empty
			if len(podWorkers.waitingToStartStaticPodsByFullname) != 0 ||
				len(podWorkers.waitingToStartStaticPodsByFullname) != len(tc.expectedWaitingToStartStaticPodsByFullname) {
				if !reflect.DeepEqual(
					podWorkers.waitingToStartStaticPodsByFullname,
					tc.expectedWaitingToStartStaticPodsByFullname) {
					t.Errorf("waitingToStartStaticPodsByFullname: expected %v, got %v",
						tc.expectedWaitingToStartStaticPodsByFullname,
						podWorkers.waitingToStartStaticPodsByFullname)
				}
			}
		})
	}
}
