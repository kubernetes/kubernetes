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

package preemption

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	corelisters "k8s.io/client-go/listers/core/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	apicache "k8s.io/kubernetes/pkg/scheduler/backend/api_cache"
	apidispatcher "k8s.io/kubernetes/pkg/scheduler/backend/api_dispatcher"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	apicalls "k8s.io/kubernetes/pkg/scheduler/framework/api_calls"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

// fakePodLister helps test IsPodRunningPreemption logic without worrying about cache synchronization issues.
// Current list of pods is set using field pods.
type fakePodLister struct {
	corelisters.PodLister
	pods map[string]*v1.Pod
}

func (m *fakePodLister) Pods(namespace string) corelisters.PodNamespaceLister {
	return &fakePodNamespaceLister{pods: m.pods}
}

// fakePodNamespaceLister helps test IsPodRunningPreemption logic without worrying about cache synchronization issues.
// Current list of pods is set using field pods.
type fakePodNamespaceLister struct {
	corelisters.PodNamespaceLister
	pods map[string]*v1.Pod
}

func (m *fakePodNamespaceLister) Get(name string) (*v1.Pod, error) {
	if pod, ok := m.pods[name]; ok {
		return pod, nil
	}
	// Important: Return the standard IsNotFound error for a fake cache miss.
	return nil, apierrors.NewNotFound(v1.Resource("pods"), name)
}

func TestIsPodRunningPreemption(t *testing.T) {
	var (
		victim1 = st.MakePod().Name("victim1").UID("victim1").
			Node("node").SchedulerName("sch").Priority(midPriority).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()

		victim2 = st.MakePod().Name("victim2").UID("victim2").
			Node("node").SchedulerName("sch").Priority(midPriority).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()

		victimWithDeletionTimestamp = st.MakePod().Name("victim-deleted").UID("victim-deleted").
						Node("node").SchedulerName("sch").Priority(midPriority).
						Terminating().
						Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
						Obj()
	)

	tests := []struct {
		name            string
		preemptorUID    types.UID
		preemptingSet   sets.Set[types.UID]
		lastVictimSet   map[types.UID]pendingVictim
		podsInPodLister map[string]*v1.Pod
		expectedResult  bool
	}{
		{
			name:           "preemptor not in preemptingSet",
			preemptorUID:   "preemptor",
			preemptingSet:  sets.New[types.UID](),
			lastVictimSet:  map[types.UID]pendingVictim{},
			expectedResult: false,
		},
		{
			name:          "preemptor not in preemptingSet, lastVictimSet not empty",
			preemptorUID:  "preemptor",
			preemptingSet: sets.New[types.UID](),
			lastVictimSet: map[types.UID]pendingVictim{
				"preemptor": {
					namespace: "ns",
					name:      "victim1",
				},
			},
			expectedResult: false,
		},
		{
			name:          "preemptor in preemptingSet, no lastVictim for preemptor",
			preemptorUID:  "preemptor",
			preemptingSet: sets.New[types.UID]("preemptor"),
			lastVictimSet: map[types.UID]pendingVictim{
				"otherPod": {
					namespace: "ns",
					name:      "victim1",
				},
			},
			expectedResult: true,
		},
		{
			name:          "preemptor in preemptingSet, victim in lastVictimSet, not in PodLister",
			preemptorUID:  "preemptor",
			preemptingSet: sets.New[types.UID]("preemptor"),
			lastVictimSet: map[types.UID]pendingVictim{
				"preemptor": {
					namespace: "ns",
					name:      "victim1",
				},
			},
			podsInPodLister: map[string]*v1.Pod{},
			expectedResult:  false,
		},
		{
			name:          "preemptor in preemptingSet, victim in lastVictimSet and in PodLister",
			preemptorUID:  "preemptor",
			preemptingSet: sets.New[types.UID]("preemptor"),
			lastVictimSet: map[types.UID]pendingVictim{
				"preemptor": {
					namespace: "ns",
					name:      "victim1",
				},
			},
			podsInPodLister: map[string]*v1.Pod{
				"victim1": victim1,
				"victim2": victim2,
			},
			expectedResult: true,
		},
		{
			name:          "preemptor in preemptingSet, victim in lastVictimSet and in PodLister with deletion timestamp",
			preemptorUID:  "preemptor",
			preemptingSet: sets.New[types.UID]("preemptor"),
			lastVictimSet: map[types.UID]pendingVictim{
				"preemptor": {
					namespace: "ns",
					name:      "victim-deleted",
				},
			},
			podsInPodLister: map[string]*v1.Pod{
				"victim1":        victim1,
				"victim-deleted": victimWithDeletionTimestamp,
			},
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("%v", tt.name), func(t *testing.T) {

			fakeLister := &fakePodLister{
				pods: tt.podsInPodLister,
			}
			a := &Executor{
				podLister:                    fakeLister,
				preempting:                   tt.preemptingSet,
				lastVictimsPendingPreemption: tt.lastVictimSet,
			}

			if result := a.IsPodRunningPreemption(tt.preemptorUID); tt.expectedResult != result {
				t.Errorf("Expected IsPodRunningPreemption to return %v but got %v", tt.expectedResult, result)
			}
		})
	}
}

type fakePodActivator struct {
	activatedPods map[string]*v1.Pod
	mu            *sync.RWMutex
}

func (f *fakePodActivator) Activate(logger klog.Logger, pods map[string]*v1.Pod) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for name, pod := range pods {
		f.activatedPods[name] = pod
	}
}

type fakePodNominator struct {
	// embed it so that we can only override NominatedPodsForNode
	internalqueue.SchedulingQueue

	// fakePodNominator doesn't respond to NominatedPodsForNode() until the channel is closed.
	requestStopper chan struct{}
}

func (f *fakePodNominator) NominatedPodsForNode(nodeName string) []fwk.PodInfo {
	<-f.requestStopper
	return nil
}

func TestPrepareCandidate(t *testing.T) {
	var (
		node1Name            = "node1"
		defaultSchedulerName = "default-scheduler"
	)
	condition := v1.PodCondition{
		Type:    v1.DisruptionTarget,
		Status:  v1.ConditionTrue,
		Reason:  v1.PodReasonPreemptionByScheduler,
		Message: fmt.Sprintf("%s: preempting to accommodate a higher priority pod", defaultSchedulerName),
	}

	var (
		victim1 = st.MakePod().Name("victim1").UID("victim1").
			Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()

		notFoundVictim1 = st.MakePod().Name("not-found-victim").UID("victim1").
				Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
				Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
				Obj()

		failVictim = st.MakePod().Name("fail-victim").UID("victim1").
				Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
				Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
				Obj()

		victim2 = st.MakePod().Name("victim2").UID("victim2").
			Node(node1Name).SchedulerName(defaultSchedulerName).Priority(50000).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()

		victim1WithMatchingCondition = st.MakePod().Name("victim1").UID("victim1").
						Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
						Conditions([]v1.PodCondition{condition}).
						Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
						Obj()

		failVictim1WithMatchingCondition = st.MakePod().Name("fail-victim").UID("victim1").
							Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
							Conditions([]v1.PodCondition{condition}).
							Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
							Obj()

		preemptor = st.MakePod().Name("preemptor").UID("preemptor").
				SchedulerName(defaultSchedulerName).Priority(highPriority).
				Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
				Obj()

		errDeletePodFailed   = errors.New("delete pod failed")
		errPatchStatusFailed = errors.New("patch pod status failed")
	)

	victimWithDeletionTimestamp := victim1.DeepCopy()
	victimWithDeletionTimestamp.Name = "victim1-with-deletion-timestamp"
	victimWithDeletionTimestamp.UID = "victim1-with-deletion-timestamp"
	victimWithDeletionTimestamp.DeletionTimestamp = &metav1.Time{Time: time.Now().Add(-100 * time.Second)}
	victimWithDeletionTimestamp.Finalizers = []string{"test"}

	tests := []struct {
		name      string
		nodeNames []string
		candidate Candidate
		preemptor *v1.Pod
		testPods  []*v1.Pod
		// expectedDeletedPod is the pod name that is expected to be deleted.
		//
		// You can set multiple pod name if there're multiple possibilities.
		// Both empty and "" means no pod is expected to be deleted.
		expectedDeletedPod    []string
		expectedDeletionError bool
		expectedPatchError    bool
		// Only compared when async preemption is disabled.
		expectedStatus *fwk.Status
		// Only compared when async preemption is enabled.
		expectedPreemptingMap sets.Set[types.UID]
		expectedActivatedPods map[string]*v1.Pod
	}{
		{
			name: "no victims",
			candidate: &candidate{
				victims: &extenderv1.Victims{},
			},
			preemptor: preemptor,
			testPods: []*v1.Pod{
				victim1,
			},
			nodeNames:      []string{node1Name},
			expectedStatus: nil,
		},
		{
			name: "one victim without condition",

			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1,
					},
				},
			},
			preemptor: preemptor,
			testPods: []*v1.Pod{
				victim1,
			},
			nodeNames:             []string{node1Name},
			expectedDeletedPod:    []string{"victim1"},
			expectedStatus:        nil,
			expectedPreemptingMap: sets.New(types.UID("preemptor")),
		},
		{
			name: "one victim, but victim is already being deleted",

			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victimWithDeletionTimestamp,
					},
				},
			},
			preemptor: preemptor,
			testPods: []*v1.Pod{
				victimWithDeletionTimestamp,
			},
			nodeNames:      []string{node1Name},
			expectedStatus: nil,
		},
		{
			name: "one victim, but victim is already deleted",

			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						notFoundVictim1,
					},
				},
			},
			preemptor:             preemptor,
			testPods:              []*v1.Pod{},
			nodeNames:             []string{node1Name},
			expectedStatus:        nil,
			expectedPreemptingMap: sets.New(types.UID("preemptor")),
		},
		{
			name: "one victim with same condition",

			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1WithMatchingCondition,
					},
				},
			},
			preemptor: preemptor,
			testPods: []*v1.Pod{
				victim1WithMatchingCondition,
			},
			nodeNames:             []string{node1Name},
			expectedDeletedPod:    []string{"victim1"},
			expectedStatus:        nil,
			expectedPreemptingMap: sets.New(types.UID("preemptor")),
		},
		{
			name: "one victim, not-found victim error is ignored when patching",

			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1WithMatchingCondition,
					},
				},
			},
			preemptor:             preemptor,
			testPods:              []*v1.Pod{},
			nodeNames:             []string{node1Name},
			expectedDeletedPod:    []string{"victim1"},
			expectedStatus:        nil,
			expectedPreemptingMap: sets.New(types.UID("preemptor")),
		},
		{
			name: "one victim, but pod deletion failed",

			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						failVictim1WithMatchingCondition,
					},
				},
			},
			preemptor:             preemptor,
			testPods:              []*v1.Pod{},
			expectedDeletionError: true,
			nodeNames:             []string{node1Name},
			expectedStatus:        fwk.AsStatus(errDeletePodFailed),
			expectedPreemptingMap: sets.New(types.UID("preemptor")),
			expectedActivatedPods: map[string]*v1.Pod{preemptor.Name: preemptor},
		},
		{
			name: "one victim, not-found victim error is ignored when deleting",

			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1,
					},
				},
			},
			preemptor:             preemptor,
			testPods:              []*v1.Pod{},
			nodeNames:             []string{node1Name},
			expectedDeletedPod:    []string{"victim1"},
			expectedStatus:        nil,
			expectedPreemptingMap: sets.New(types.UID("preemptor")),
		},
		{
			name: "one victim, but patch pod failed",

			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						failVictim,
					},
				},
			},
			preemptor:             preemptor,
			testPods:              []*v1.Pod{},
			expectedPatchError:    true,
			nodeNames:             []string{node1Name},
			expectedStatus:        fwk.AsStatus(errPatchStatusFailed),
			expectedPreemptingMap: sets.New(types.UID("preemptor")),
			expectedActivatedPods: map[string]*v1.Pod{preemptor.Name: preemptor},
		},
		{
			name: "two victims without condition, one passes successfully and the second fails",

			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						failVictim,
						victim2,
					},
				},
			},
			preemptor: preemptor,
			testPods: []*v1.Pod{
				victim1,
			},
			nodeNames:          []string{node1Name},
			expectedPatchError: true,
			expectedDeletedPod: []string{
				"victim2",
				// The first victim could fail before the deletion of the second victim happens,
				// which results in the second victim not being deleted.
				"",
			},
			expectedStatus:        fwk.AsStatus(errPatchStatusFailed),
			expectedPreemptingMap: sets.New(types.UID("preemptor")),
			expectedActivatedPods: map[string]*v1.Pod{preemptor.Name: preemptor},
		},
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, asyncAPICallsEnabled := range []bool{true, false} {
			for _, tt := range tests {
				t.Run(fmt.Sprintf("%v (Async preemption enabled: %v, Async API calls enabled: %v)", tt.name, asyncPreemptionEnabled, asyncAPICallsEnabled), func(t *testing.T) {
					metrics.Register()
					logger, ctx := ktesting.NewTestContext(t)
					ctx, cancel := context.WithCancel(ctx)
					defer cancel()

					nodes := make([]*v1.Node, len(tt.nodeNames))
					for i, nodeName := range tt.nodeNames {
						nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeRes).Obj()
					}
					registeredPlugins := append([]tf.RegisterPluginFunc{
						tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
						tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
					)
					var objs []runtime.Object
					for _, pod := range tt.testPods {
						objs = append(objs, pod)
					}

					mu := &sync.RWMutex{}
					deletedPods := sets.New[string]()
					deletionFailure := false // whether any request to delete pod failed
					patchFailure := false    // whether any request to patch pod status failed

					cs := clientsetfake.NewClientset(objs...)
					cs.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
						mu.Lock()
						defer mu.Unlock()
						name := action.(clienttesting.DeleteAction).GetName()
						if name == "fail-victim" {
							deletionFailure = true
							return true, nil, errDeletePodFailed
						}
						// fake clientset does not return an error for not-found pods, so we simulate it here.
						if name == "not-found-victim" {
							// Simulate a not-found error.
							return true, nil, apierrors.NewNotFound(v1.Resource("pods"), name)
						}

						deletedPods.Insert(name)
						return true, nil, nil
					})

					cs.PrependReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
						mu.Lock()
						defer mu.Unlock()
						if action.(clienttesting.PatchAction).GetName() == "fail-victim" {
							patchFailure = true
							return true, nil, errPatchStatusFailed
						}
						// fake clientset does not return an error for not-found pods, so we simulate it here.
						if action.(clienttesting.PatchAction).GetName() == "not-found-victim" {
							return true, nil, apierrors.NewNotFound(v1.Resource("pods"), "not-found-victim")
						}
						return true, nil, nil
					})

					informerFactory := informers.NewSharedInformerFactory(cs, 0)
					eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: cs.EventsV1()})
					fakeActivator := &fakePodActivator{activatedPods: make(map[string]*v1.Pod), mu: mu}

					// Note: NominatedPodsForNode is called at the beginning of the goroutine in any case.
					// fakePodNominator can delay the response of NominatedPodsForNode until the channel is closed,
					// which allows us to test the preempting map before the goroutine does nothing yet.
					requestStopper := make(chan struct{})
					nominator := &fakePodNominator{
						SchedulingQueue: internalqueue.NewSchedulingQueue(nil, informerFactory),
						requestStopper:  requestStopper,
					}
					var apiDispatcher *apidispatcher.APIDispatcher
					if asyncAPICallsEnabled {
						apiDispatcher = apidispatcher.New(cs, 16, apicalls.Relevances)
						apiDispatcher.Run(logger)
						defer apiDispatcher.Close()
					}

					fwk, err := tf.NewFramework(
						ctx,
						registeredPlugins, "",
						frameworkruntime.WithClientSet(cs),
						frameworkruntime.WithAPIDispatcher(apiDispatcher),
						frameworkruntime.WithLogger(logger),
						frameworkruntime.WithInformerFactory(informerFactory),
						frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
						frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(tt.testPods, nodes)),
						frameworkruntime.WithPodNominator(nominator),
						frameworkruntime.WithEventRecorder(eventBroadcaster.NewRecorder(scheme.Scheme, "test-scheduler")),
						frameworkruntime.WithPodActivator(fakeActivator),
					)
					if err != nil {
						t.Fatal(err)
					}
					informerFactory.Start(ctx.Done())
					informerFactory.WaitForCacheSync(ctx.Done())
					if asyncAPICallsEnabled {
						cache := internalcache.New(ctx, apiDispatcher)
						fwk.SetAPICacher(apicache.New(nil, cache))
					}

					executor := newExecutor(fwk)

					if asyncPreemptionEnabled {
						executor.prepareCandidateAsync(tt.candidate, tt.preemptor, "test-plugin")
						executor.mu.Lock()
						// The preempting map should be registered synchronously
						// so we don't need wait.Poll.
						if !tt.expectedPreemptingMap.Equal(executor.preempting) {
							t.Errorf("expected preempting map %v, got %v", tt.expectedPreemptingMap, executor.preempting)
							close(requestStopper)
							executor.mu.Unlock()
							return
						}
						executor.mu.Unlock()
						// make the requests complete
						close(requestStopper)
					} else {
						close(requestStopper) // no need to stop requests
						status := executor.prepareCandidate(ctx, tt.candidate, tt.preemptor, "test-plugin")
						if tt.expectedStatus == nil {
							if status != nil {
								t.Errorf("expect nil status, but got %v", status)
							}
						} else {
							if !cmp.Equal(status, tt.expectedStatus) {
								t.Errorf("expect status %v, but got %v", tt.expectedStatus, status)
							}
						}
					}

					var lastErrMsg string
					if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
						mu.RLock()
						defer mu.RUnlock()

						executor.mu.Lock()
						defer executor.mu.Unlock()
						if len(executor.preempting) != 0 {
							// The preempting map should be empty after the goroutine in all test cases.
							lastErrMsg = fmt.Sprintf("expected no preempting pods, got %v", executor.preempting)
							return false, nil
						}

						if tt.expectedDeletionError != deletionFailure {
							lastErrMsg = fmt.Sprintf("expected deletion error %v, got %v", tt.expectedDeletionError, deletionFailure)
							return false, nil
						}
						if tt.expectedPatchError != patchFailure {
							lastErrMsg = fmt.Sprintf("expected patch error %v, got %v", tt.expectedPatchError, patchFailure)
							return false, nil
						}

						if asyncPreemptionEnabled {
							if diff := cmp.Diff(tt.expectedActivatedPods, fakeActivator.activatedPods); tt.expectedActivatedPods != nil && diff != "" {
								lastErrMsg = fmt.Sprintf("Unexpected activated pods (-want,+got):\n%s", diff)
								return false, nil
							}
							if tt.expectedActivatedPods == nil && len(fakeActivator.activatedPods) != 0 {
								lastErrMsg = fmt.Sprintf("expected no activated pods, got %v", fakeActivator.activatedPods)
								return false, nil
							}
						}

						if deletedPods.Len() > 1 {
							// For now, we only expect at most one pod to be deleted in all test cases.
							// If we need to test multiple pods deletion, we need to update the test table definition.
							return false, fmt.Errorf("expected at most one pod to be deleted, got %v", deletedPods.UnsortedList())
						}

						if len(tt.expectedDeletedPod) == 0 {
							if deletedPods.Len() != 0 {
								// When tt.expectedDeletedPod is empty, we expect no pod to be deleted.
								return false, fmt.Errorf("expected no pod to be deleted, got %v", deletedPods.UnsortedList())
							}
							// nothing further to check.
							return true, nil
						}

						found := false
						for _, podName := range tt.expectedDeletedPod {
							if deletedPods.Has(podName) ||
								// If podName is empty, we expect no pod to be deleted.
								(deletedPods.Len() == 0 && podName == "") {
								found = true
							}
						}
						if !found {
							lastErrMsg = fmt.Sprintf("expected pod %v to be deleted, but %v is deleted", strings.Join(tt.expectedDeletedPod, " or "), deletedPods.UnsortedList())
							return false, nil
						}

						return true, nil
					}); err != nil {
						t.Fatal(lastErrMsg)
					}
				})
			}
		}
	}
}

func TestPrepareCandidateAsyncSetsPreemptingSets(t *testing.T) {
	var (
		node1Name            = "node1"
		defaultSchedulerName = "default-scheduler"
	)

	var (
		victim1 = st.MakePod().Name("victim1").UID("victim1").
			Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()

		victim2 = st.MakePod().Name("victim2").UID("victim2").
			Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()

		preemptor = st.MakePod().Name("preemptor").UID("preemptor").
				SchedulerName(defaultSchedulerName).Priority(highPriority).
				Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
				Obj()
		testPods = []*v1.Pod{
			victim1,
			victim2,
		}
		nodeNames = []string{node1Name}
	)

	tests := []struct {
		name       string
		candidate  Candidate
		lastVictim *v1.Pod
		preemptor  *v1.Pod
	}{
		{
			name: "no victims",
			candidate: &candidate{
				victims: &extenderv1.Victims{},
			},
			lastVictim: nil,
			preemptor:  preemptor,
		},
		{
			name: "one victim",
			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1,
					},
				},
			},
			lastVictim: victim1,
			preemptor:  preemptor,
		},
		{
			name: "two victims",
			candidate: &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1,
						victim2,
					},
				},
			},
			lastVictim: victim2,
			preemptor:  preemptor,
		},
	}

	for _, asyncAPICallsEnabled := range []bool{true, false} {
		for _, tt := range tests {
			t.Run(fmt.Sprintf("%v (Async API calls enabled: %v)", tt.name, asyncAPICallsEnabled), func(t *testing.T) {
				metrics.Register()
				logger, ctx := ktesting.NewTestContext(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()

				nodes := make([]*v1.Node, len(nodeNames))
				for i, nodeName := range nodeNames {
					nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeRes).Obj()
				}
				registeredPlugins := append([]tf.RegisterPluginFunc{
					tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
					tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				)
				var objs []runtime.Object
				for _, pod := range testPods {
					objs = append(objs, pod)
				}

				cs := clientsetfake.NewClientset(objs...)

				informerFactory := informers.NewSharedInformerFactory(cs, 0)
				eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: cs.EventsV1()})

				var apiDispatcher *apidispatcher.APIDispatcher
				if asyncAPICallsEnabled {
					apiDispatcher = apidispatcher.New(cs, 16, apicalls.Relevances)
					apiDispatcher.Run(logger)
					defer apiDispatcher.Close()
				}

				fwk, err := tf.NewFramework(
					ctx,
					registeredPlugins, "",
					frameworkruntime.WithClientSet(cs),
					frameworkruntime.WithAPIDispatcher(apiDispatcher),
					frameworkruntime.WithLogger(logger),
					frameworkruntime.WithInformerFactory(informerFactory),
					frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
					frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(testPods, nodes)),
					frameworkruntime.WithEventRecorder(eventBroadcaster.NewRecorder(scheme.Scheme, "test-scheduler")),
					frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				)
				if err != nil {
					t.Fatal(err)
				}
				informerFactory.Start(ctx.Done())
				if asyncAPICallsEnabled {
					cache := internalcache.New(ctx, apiDispatcher)
					fwk.SetAPICacher(apicache.New(nil, cache))
				}

				executor := newExecutor(fwk)
				// preemptPodCallsCounter helps verify if the last victim pod gets preempted after other victims.
				preemptPodCallsCounter := 0
				preemptFunc := executor.PreemptPod
				executor.PreemptPod = func(ctx context.Context, c Candidate, preemptor, victim *v1.Pod, pluginName string) error {
					// Verify contents of the sets: preempting and lastVictimsPendingPreemption before preemption of subsequent pods.
					executor.mu.RLock()
					preemptPodCallsCounter++

					if !executor.preempting.Has(tt.preemptor.UID) {
						t.Errorf("Expected preempting set to be contain %v before preempting victim %v but got set: %v", tt.preemptor.UID, victim.Name, executor.preempting)
					}

					victimCount := len(tt.candidate.Victims().Pods)
					if victim.Name == tt.lastVictim.Name {
						if victimCount != preemptPodCallsCounter {
							t.Errorf("Expected PreemptPod for last victim %v to be called last (call no. %v), but it was called as no. %v", victim.Name, victimCount, preemptPodCallsCounter)
						}
						if v, ok := executor.lastVictimsPendingPreemption[tt.preemptor.UID]; !ok || tt.lastVictim.Name != v.name {
							t.Errorf("Expected lastVictimsPendingPreemption map to contain victim %v for preemptor UID %v when preempting the last victim, but got map: %v",
								tt.lastVictim.Name, tt.preemptor.UID, executor.lastVictimsPendingPreemption)
						}
					} else {
						if preemptPodCallsCounter >= victimCount {
							t.Errorf("Expected PreemptPod for victim %v to be called earlier, but it was called as last - no. %v", victim.Name, preemptPodCallsCounter)
						}
						if _, ok := executor.lastVictimsPendingPreemption[tt.preemptor.UID]; ok {
							t.Errorf("Expected lastVictimsPendingPreemption map to not contain values for preemptor UID %v when not preempting the last victim, but got map: %v",
								tt.preemptor.UID, executor.lastVictimsPendingPreemption)
						}
					}
					executor.mu.RUnlock()

					return preemptFunc(ctx, c, preemptor, victim, pluginName)
				}

				executor.mu.RLock()
				if len(executor.preempting) > 0 {
					t.Errorf("Expected preempting set to be empty before prepareCandidateAsync but got %v", executor.preempting)
				}
				if len(executor.lastVictimsPendingPreemption) > 0 {
					t.Errorf("Expected lastVictimsPendingPreemption map to be empty before prepareCandidateAsync but got %v", executor.lastVictimsPendingPreemption)
				}
				executor.mu.RUnlock()

				executor.prepareCandidateAsync(tt.candidate, tt.preemptor, "test-plugin")

				// Perform the checks when there are no victims left to preempt.
				t.Log("Waiting for async preemption goroutine to finish cleanup...")
				err = wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 2*time.Second, false, func(ctx context.Context) (bool, error) {
					// Check if the preemptor is removed from the ev.preempting set.
					executor.mu.RLock()
					defer executor.mu.RUnlock()
					return !executor.preempting.Has(tt.preemptor.UID), nil
				})
				if err != nil {
					t.Errorf("Timed out waiting for preemptingSet to become empty. %v", err)
				}

				executor.mu.RLock()
				if _, ok := executor.lastVictimsPendingPreemption[tt.preemptor.UID]; ok {
					t.Errorf("Expected lastVictimsPendingPreemption map to not contain values for %v after completing preemption, but got map: %v",
						tt.preemptor.UID, executor.lastVictimsPendingPreemption)
				}
				if victimCount := len(tt.candidate.Victims().Pods); victimCount != preemptPodCallsCounter {
					t.Errorf("Expected PreemptPod to be called %v times during prepareCandidateAsync but got %v", victimCount, preemptPodCallsCounter)
				}
				executor.mu.RUnlock()
			})
		}
	}
}

func TestAsyncPreemptionFailure(t *testing.T) {
	metrics.Register()
	var (
		node1Name            = "node1"
		defaultSchedulerName = "default-scheduler"
		failVictimNamePrefix = "fail-victim"
	)

	makePod := func(name string, priority int32) *v1.Pod {
		return st.MakePod().Name(name).UID(name).
			Node(node1Name).SchedulerName(defaultSchedulerName).Priority(priority).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()
	}

	preemptor := makePod("preemptor", highPriority)

	makeVictim := func(name string) *v1.Pod {
		return makePod(name, midPriority)
	}

	tests := []struct {
		name                                 string
		victims                              []*v1.Pod
		expectSuccessfulPreemption           bool
		expectPreemptionAttemptForLastVictim bool
	}{
		{
			name: "Failure with a single victim",
			victims: []*v1.Pod{
				makeVictim(failVictimNamePrefix),
			},
			expectSuccessfulPreemption:           false,
			expectPreemptionAttemptForLastVictim: true,
		},
		{
			name: "Success with a single victim",
			victims: []*v1.Pod{
				makeVictim("victim1"),
			},
			expectSuccessfulPreemption:           true,
			expectPreemptionAttemptForLastVictim: true,
		},
		{
			name: "Failure in first of three victims",
			victims: []*v1.Pod{
				makeVictim(failVictimNamePrefix),
				makeVictim("victim2"),
				makeVictim("victim3"),
			},
			expectSuccessfulPreemption:           false,
			expectPreemptionAttemptForLastVictim: false,
		},
		{
			name: "Failure in second of three victims",
			victims: []*v1.Pod{
				makeVictim("victim1"),
				makeVictim(failVictimNamePrefix),
				makeVictim("victim3"),
			},
			expectSuccessfulPreemption:           false,
			expectPreemptionAttemptForLastVictim: false,
		},
		{
			name: "Failure in first two of three victims",
			victims: []*v1.Pod{
				makeVictim(failVictimNamePrefix + "1"),
				makeVictim(failVictimNamePrefix + "2"),
				makeVictim("victim3"),
			},
			expectSuccessfulPreemption:           false,
			expectPreemptionAttemptForLastVictim: false,
		},
		{
			name: "Failure in third of three victims",
			victims: []*v1.Pod{
				makeVictim("victim1"),
				makeVictim("victim2"),
				makeVictim(failVictimNamePrefix),
			},
			expectSuccessfulPreemption:           false,
			expectPreemptionAttemptForLastVictim: true,
		},
		{
			name: "Success with three victims",
			victims: []*v1.Pod{
				makeVictim("victim1"),
				makeVictim("victim2"),
				makeVictim("victim3"),
			},
			expectSuccessfulPreemption:           true,
			expectPreemptionAttemptForLastVictim: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			candidate := &candidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: tt.victims,
				},
			}

			// Set up the fake clientset.
			preemptionAttemptedPods := sets.New[string]()
			deletedPods := sets.New[string]()
			mu := &sync.RWMutex{}
			objs := []runtime.Object{preemptor}
			for _, v := range tt.victims {
				objs = append(objs, v)
			}

			cs := clientsetfake.NewClientset(objs...)
			cs.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				mu.Lock()
				defer mu.Unlock()
				name := action.(clienttesting.DeleteAction).GetName()
				preemptionAttemptedPods.Insert(name)
				if strings.HasPrefix(name, failVictimNamePrefix) {
					return true, nil, errors.New("delete pod failed")
				}
				deletedPods.Insert(name)
				return true, nil, nil
			})

			// Set up the framework.
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: cs.EventsV1()})
			fakeActivator := &fakePodActivator{activatedPods: make(map[string]*v1.Pod), mu: mu}

			registeredPlugins := append([]tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)

			snapshotPods := append([]*v1.Pod{preemptor}, tt.victims...)
			fwk, err := tf.NewFramework(
				ctx,
				registeredPlugins, "",
				frameworkruntime.WithClientSet(cs),
				frameworkruntime.WithLogger(logger),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithEventRecorder(eventBroadcaster.NewRecorder(scheme.Scheme, "test-scheduler")),
				frameworkruntime.WithPodActivator(fakeActivator),
				frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
				frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(snapshotPods, []*v1.Node{st.MakeNode().Name(node1Name).Obj()})),
			)
			if err != nil {
				t.Fatal(err)
			}
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			executor := newExecutor(fwk)

			// Run the actual preemption.
			executor.prepareCandidateAsync(candidate, preemptor, "test-plugin")

			// Wait for the async preemption to finish.
			err = wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
				// Check if the preemptor is removed from the executor.preempting set.
				executor.mu.RLock()
				defer executor.mu.RUnlock()
				return len(executor.preempting) == 0, nil
			})
			if err != nil {
				t.Fatalf("Timed out waiting for async preemption to finish: %v", err)
			}

			mu.RLock()
			defer mu.RUnlock()

			lastVictimName := tt.victims[len(tt.victims)-1].Name
			if tt.expectPreemptionAttemptForLastVictim != preemptionAttemptedPods.Has(lastVictimName) {
				t.Errorf("Last victim's preemption attempted - wanted: %v, got: %v", tt.expectPreemptionAttemptForLastVictim, preemptionAttemptedPods.Has(lastVictimName))
			}
			// Verify that the preemption of the last victim is attempted if and only if
			// the preemption of all of the preceding victims succeeds.
			precedingVictimsPreempted := true
			for _, victim := range tt.victims[:len(tt.victims)-1] {
				if !preemptionAttemptedPods.Has(victim.Name) || !deletedPods.Has(victim.Name) {
					precedingVictimsPreempted = false
				}
			}
			if precedingVictimsPreempted != preemptionAttemptedPods.Has(lastVictimName) {
				t.Errorf("Last victim's preemption attempted - wanted: %v, got: %v", precedingVictimsPreempted, preemptionAttemptedPods.Has(lastVictimName))
			}

			// Verify that the preemptor is activated if and only if the async preemption fails.
			if _, ok := fakeActivator.activatedPods[preemptor.Name]; ok != !tt.expectSuccessfulPreemption {
				t.Errorf("Preemptor activated - wanted: %v, got: %v", !tt.expectSuccessfulPreemption, ok)
			}

			// Verify if the last victim got deleted as expected.
			if tt.expectSuccessfulPreemption != deletedPods.Has(lastVictimName) {
				t.Errorf("Last victim deleted - wanted: %v, got: %v", tt.expectSuccessfulPreemption, deletedPods.Has(lastVictimName))
			}
		})
	}
}

func TestRemoveNominatedNodeName(t *testing.T) {
	tests := []struct {
		name                     string
		currentNominatedNodeName string
		newNominatedNodeName     string
		expectPatchRequest       bool
		expectedPatchData        string
	}{
		{
			name:                     "Should make patch request to clear node name",
			currentNominatedNodeName: "node1",
			expectPatchRequest:       true,
			expectedPatchData:        `{"status":{"nominatedNodeName":null}}`,
		},
		{
			name:                     "Should not make patch request if nominated node is already cleared",
			currentNominatedNodeName: "",
			expectPatchRequest:       false,
		},
	}
	for _, asyncAPICallsEnabled := range []bool{true, false} {
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				logger, ctx := ktesting.NewTestContext(t)
				actualPatchRequests := 0
				var actualPatchData string
				cs := &clientsetfake.Clientset{}
				patchCalled := make(chan struct{}, 1)
				cs.AddReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
					actualPatchRequests++
					patch := action.(clienttesting.PatchAction)
					actualPatchData = string(patch.GetPatch())
					patchCalled <- struct{}{}
					// For this test, we don't care about the result of the patched pod, just that we got the expected
					// patch request, so just returning &v1.Pod{} here is OK because scheduler doesn't use the response.
					return true, &v1.Pod{}, nil
				})

				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: "foo"},
					Status:     v1.PodStatus{NominatedNodeName: test.currentNominatedNodeName},
				}

				ctx, cancel := context.WithCancel(ctx)
				defer cancel()

				var apiCacher fwk.APICacher
				if asyncAPICallsEnabled {
					apiDispatcher := apidispatcher.New(cs, 16, apicalls.Relevances)
					apiDispatcher.Run(logger)
					defer apiDispatcher.Close()

					informerFactory := informers.NewSharedInformerFactory(cs, 0)
					queue := internalqueue.NewSchedulingQueue(nil, informerFactory, internalqueue.WithAPIDispatcher(apiDispatcher))
					apiCacher = apicache.New(queue, nil)
				}

				if err := clearNominatedNodeName(ctx, cs, apiCacher, pod); err != nil {
					t.Fatalf("Error calling removeNominatedNodeName: %v", err)
				}

				if test.expectPatchRequest {
					select {
					case <-patchCalled:
					case <-time.After(time.Second):
						t.Fatalf("Timed out while waiting for patch to be called")
					}
					if actualPatchData != test.expectedPatchData {
						t.Fatalf("Patch data mismatch: Actual was %v, but expected %v", actualPatchData, test.expectedPatchData)
					}
				} else {
					select {
					case <-patchCalled:
						t.Fatalf("Expected patch not to be called, actual patch data: %v", actualPatchData)
					case <-time.After(time.Second):
					}
				}
			})
		}
	}
}
