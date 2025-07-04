package queue

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util"
	testingclock "k8s.io/utils/clock/testing"
)

func getUnschedulablePod(p *PriorityQueue, pod *v1.Pod) *v1.Pod {
	pInfo := p.unschedulablePods.get(pod)
	if pInfo != nil {
		return pInfo.Pod
	}
	return nil
}

// TestPriorityQueue_AddUnschedulableIfNotPresent_Backoff tests the scenarios when
// AddUnschedulableIfNotPresent is called asynchronously.
// Pods in and before current scheduling cycle will be put back to activeQueue
// if we were trying to schedule them when we received move request.

func TestPriorityQueue_AddUnschedulableIfNotPresent(t *testing.T) {
	objs := []runtime.Object{highPriNominatedPodInfo.Pod, unschedulablePodInfo.Pod}
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs)
	// insert unschedulablePodInfo and pop right after that
	// because the scheduling queue records unschedulablePod as in-flight Pod.    q.Add(logger, unschedulablePodInfo.Pod)
	if p, err := q.Pop(logger); err != nil || p.Pod != unschedulablePodInfo.Pod {
		t.Errorf("Expected: %v after Pop, but got: %v", unschedulablePodInfo.Pod.Name, p.Pod.Name)
	}

	q.Add(logger, highPriNominatedPodInfo.Pod)
	err := q.AddUnschedulableIfNotPresent(logger, newQueuedPodInfoForLookup(unschedulablePodInfo.Pod, "plugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulableIfNotPresent: %v", err)
	}
	if p, err := q.Pop(logger); err != nil || p.Pod != highPriNominatedPodInfo.Pod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriNominatedPodInfo.Pod.Name, p.Pod.Name)
	}
	if len(q.nominator.nominatedPods) != 1 {
		t.Errorf("Expected nominatedPods to have one element: %v", q.nominator)
	}
	// unschedulablePodInfo is inserted to unschedulable pod pool because no events happened during scheduling.
	if getUnschedulablePod(q, unschedulablePodInfo.Pod) != unschedulablePodInfo.Pod {
		t.Errorf("Pod %v was not found in the unschedulablePods.", unschedulablePodInfo.Pod.Name)
	}
}

func TestPriorityQueue_AddUnschedulableIfNotPresent_Backoff(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(testingclock.NewFakeClock(time.Now())))
	totalNum := 10
	expectedPods := make([]v1.Pod, 0, totalNum)
	for i := 0; i < totalNum; i++ {
		priority := int32(i)
		p := st.MakePod().Name(fmt.Sprintf("pod%d", i)).Namespace(fmt.Sprintf("ns%d", i)).UID(fmt.Sprintf("upns%d", i)).Priority(priority).Obj()
		expectedPods = append(expectedPods, *p)
		// priority is to make pods ordered in the PriorityQueue
		q.Add(logger, p)
	}

	// Pop all pods except for the first one
	for i := totalNum - 1; i > 0; i-- {
		p, _ := q.Pop(logger)
		if diff := cmp.Diff(&expectedPods[i], p.Pod); diff != "" {
			t.Errorf("Unexpected pod (-want, +got):\n%s", diff)
		}
	}

	// move all pods to active queue when we were trying to schedule them
	q.MoveAllToActiveOrBackoffQueue(logger, framework.EventUnschedulableTimeout, nil, nil, nil)
	oldCycle := q.SchedulingCycle()

	firstPod, _ := q.Pop(logger)
	if diff := cmp.Diff(&expectedPods[0], firstPod.Pod); diff != "" {
		t.Errorf("Unexpected pod (-want, +got):\n%s", diff)
	}

	// mark pods[1] ~ pods[totalNum-1] as unschedulable and add them back
	for i := 1; i < totalNum; i++ {
		unschedulablePod := expectedPods[i].DeepCopy()
		unschedulablePod.Status = v1.PodStatus{
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodScheduled,
					Status: v1.ConditionFalse,
					Reason: v1.PodReasonUnschedulable,
				},
			},
		}

		err := q.AddUnschedulableIfNotPresent(logger, newQueuedPodInfoForLookup(unschedulablePod, "plugin"), oldCycle)
		if err != nil {
			t.Fatalf("unexpected error from AddUnschedulableIfNotPresent: %v", err)
		}
	}

	// Since there was a move request at the same cycle as "oldCycle", these pods
	// should be in the backoff queue.
	for i := 1; i < totalNum; i++ {
		if !q.backoffQ.has(newQueuedPodInfoForLookup(&expectedPods[i])) {
			t.Errorf("Expected %v to be added to backoffQ.", expectedPods[i].Name)
		}
	}
}

func TestUnschedulablePodsMap(t *testing.T) {
	var pods = []*v1.Pod{
		st.MakePod().Name("p0").Namespace("ns1").Annotation("annot1", "val1").NominatedNodeName("node1").Obj(),
		st.MakePod().Name("p1").Namespace("ns1").Annotation("annot", "val").Obj(),
		st.MakePod().Name("p2").Namespace("ns2").Annotation("annot2", "val2").Annotation("annot3", "val3").NominatedNodeName("node3").Obj(),
		st.MakePod().Name("p3").Namespace("ns4").Annotation("annot2", "val2").Annotation("annot3", "val3").NominatedNodeName("node1").Obj(),
	}
	var updatedPods = make([]*v1.Pod, len(pods))
	updatedPods[0] = pods[0].DeepCopy()
	updatedPods[1] = pods[1].DeepCopy()
	updatedPods[3] = pods[3].DeepCopy()

	tests := []struct {
		name                   string
		podsToAdd              []*v1.Pod
		expectedMapAfterAdd    map[string]*framework.QueuedPodInfo
		podsToUpdate           []*v1.Pod
		expectedMapAfterUpdate map[string]*framework.QueuedPodInfo
		podsToDelete           []*v1.Pod
		expectedMapAfterDelete map[string]*framework.QueuedPodInfo
	}{
		{
			name:      "create, update, delete subset of pods",
			podsToAdd: []*v1.Pod{pods[0], pods[1], pods[2], pods[3]},
			expectedMapAfterAdd: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[0]): {PodInfo: mustNewTestPodInfo(t, pods[0]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[1]): {PodInfo: mustNewTestPodInfo(t, pods[1]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[2]): {PodInfo: mustNewTestPodInfo(t, pods[2]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[3]): {PodInfo: mustNewTestPodInfo(t, pods[3]), UnschedulablePlugins: sets.New[string]()},
			},
			podsToUpdate: []*v1.Pod{updatedPods[0]},
			expectedMapAfterUpdate: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[0]): {PodInfo: mustNewTestPodInfo(t, updatedPods[0]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[1]): {PodInfo: mustNewTestPodInfo(t, pods[1]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[2]): {PodInfo: mustNewTestPodInfo(t, pods[2]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[3]): {PodInfo: mustNewTestPodInfo(t, pods[3]), UnschedulablePlugins: sets.New[string]()},
			},
			podsToDelete: []*v1.Pod{pods[0], pods[1]},
			expectedMapAfterDelete: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[2]): {PodInfo: mustNewTestPodInfo(t, pods[2]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[3]): {PodInfo: mustNewTestPodInfo(t, pods[3]), UnschedulablePlugins: sets.New[string]()},
			},
		},
		{
			name:      "create, update, delete all",
			podsToAdd: []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterAdd: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[0]): {PodInfo: mustNewTestPodInfo(t, pods[0]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[3]): {PodInfo: mustNewTestPodInfo(t, pods[3]), UnschedulablePlugins: sets.New[string]()},
			},
			podsToUpdate: []*v1.Pod{updatedPods[3]},
			expectedMapAfterUpdate: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[0]): {PodInfo: mustNewTestPodInfo(t, pods[0]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[3]): {PodInfo: mustNewTestPodInfo(t, updatedPods[3]), UnschedulablePlugins: sets.New[string]()},
			},
			podsToDelete:           []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterDelete: map[string]*framework.QueuedPodInfo{},
		},
		{
			name:      "delete non-existing and existing pods",
			podsToAdd: []*v1.Pod{pods[1], pods[2]},
			expectedMapAfterAdd: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[1]): {PodInfo: mustNewTestPodInfo(t, pods[1]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[2]): {PodInfo: mustNewTestPodInfo(t, pods[2]), UnschedulablePlugins: sets.New[string]()},
			},
			podsToUpdate: []*v1.Pod{updatedPods[1]},
			expectedMapAfterUpdate: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[1]): {PodInfo: mustNewTestPodInfo(t, updatedPods[1]), UnschedulablePlugins: sets.New[string]()},
				util.GetPodFullName(pods[2]): {PodInfo: mustNewTestPodInfo(t, pods[2]), UnschedulablePlugins: sets.New[string]()},
			},
			podsToDelete: []*v1.Pod{pods[2], pods[3]},
			expectedMapAfterDelete: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[1]): {PodInfo: mustNewTestPodInfo(t, updatedPods[1]), UnschedulablePlugins: sets.New[string]()},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upm := newUnschedulablePods(nil, nil)
			for _, p := range test.podsToAdd {
				upm.addOrUpdate(newQueuedPodInfoForLookup(p), framework.EventUnscheduledPodAdd.Label())
			}
			if diff := cmp.Diff(test.expectedMapAfterAdd, upm.podInfoMap, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected map after adding pods(-want, +got):\n%s", diff)
			}

			if len(test.podsToUpdate) > 0 {
				for _, p := range test.podsToUpdate {
					upm.addOrUpdate(newQueuedPodInfoForLookup(p), framework.EventUnscheduledPodUpdate.Label())
				}
				if diff := cmp.Diff(test.expectedMapAfterUpdate, upm.podInfoMap, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
					t.Errorf("Unexpected map after updating pods (-want, +got):\n%s", diff)
				}
			}
			for _, p := range test.podsToDelete {
				upm.delete(p, false)
			}
			if diff := cmp.Diff(test.expectedMapAfterDelete, upm.podInfoMap, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected map after deleting pods (-want, +got):\n%s", diff)
			}
			upm.clear()
			if len(upm.podInfoMap) != 0 {
				t.Errorf("Expected the map to be empty, but has %v elements.", len(upm.podInfoMap))
			}
		})
	}
}

func TestPriorityQueue_initPodMaxInUnschedulablePodsDuration(t *testing.T) {
	pod1 := st.MakePod().Name("test-pod-1").Namespace("ns1").UID("tp-1").NominatedNodeName("node1").Obj()
	pod2 := st.MakePod().Name("test-pod-2").Namespace("ns2").UID("tp-2").NominatedNodeName("node2").Obj()

	var timestamp = time.Now()
	pInfo1 := &framework.QueuedPodInfo{
		PodInfo:   mustNewTestPodInfo(t, pod1),
		Timestamp: timestamp.Add(-time.Second),
	}
	pInfo2 := &framework.QueuedPodInfo{
		PodInfo:   mustNewTestPodInfo(t, pod2),
		Timestamp: timestamp.Add(-2 * time.Second),
	}

	tests := []struct {
		name                              string
		podMaxInUnschedulablePodsDuration time.Duration
		operations                        []operation
		operands                          []*framework.QueuedPodInfo
		expected                          []*framework.QueuedPodInfo
	}{
		{
			name: "New priority queue by the default value of podMaxInUnschedulablePodsDuration",
			operations: []operation{
				addPodUnschedulablePods,
				addPodUnschedulablePods,
				flushUnscheduledQ,
			},
			operands: []*framework.QueuedPodInfo{pInfo1, pInfo2, nil},
			expected: []*framework.QueuedPodInfo{pInfo2, pInfo1},
		},
		{
			name:                              "New priority queue by user-defined value of podMaxInUnschedulablePodsDuration",
			podMaxInUnschedulablePodsDuration: 30 * time.Second,
			operations: []operation{
				addPodUnschedulablePods,
				addPodUnschedulablePods,
				flushUnscheduledQ,
			},
			operands: []*framework.QueuedPodInfo{pInfo1, pInfo2, nil},
			expected: []*framework.QueuedPodInfo{pInfo2, pInfo1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			var queue *PriorityQueue
			if test.podMaxInUnschedulablePodsDuration > 0 {
				queue = NewTestQueue(ctx, newDefaultQueueSort(),
					WithClock(testingclock.NewFakeClock(timestamp)),
					WithPodMaxInUnschedulablePodsDuration(test.podMaxInUnschedulablePodsDuration))
			} else {
				queue = NewTestQueue(ctx, newDefaultQueueSort(),
					WithClock(testingclock.NewFakeClock(timestamp)))
			}

			var podInfoList []*framework.QueuedPodInfo

			for i, op := range test.operations {
				op(t, logger, queue, test.operands[i])
			}

			expectedLen := len(test.expected)
			if queue.activeQ.len() != expectedLen {
				t.Fatalf("Expected %v items to be in activeQ, but got: %v", expectedLen, queue.activeQ.len())
			}

			for i := 0; i < expectedLen; i++ {
				if pInfo, err := queue.activeQ.pop(logger); err != nil {
					t.Errorf("Error while popping the head of the queue: %v", err)
				} else {
					podInfoList = append(podInfoList, pInfo)
					// Cleanup attempts counter incremented in activeQ.pop()
					pInfo.Attempts = 0
				}
			}

			if diff := cmp.Diff(test.expected, podInfoList, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected QueuedPodInfo list (-want, +got):\n%s", diff)
			}
		})
	}
}
