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

package queue

import (
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testingclock "k8s.io/utils/clock/testing"
)

// TestAddUnschedulablePodClearsInFlightPodGroupMarker verifies the queue state at
// the end of a pod group scheduling cycle when a single (non-member) pod is
// requeued concurrently with that cycle.
//
// The scheduling loop is single-threaded, but binding goroutines call
// AddUnschedulablePodIfNotPresent asynchronously. Pop() of a pod group records
// the group in activeQ.lastPoppedEntityKey so that member pods failing during
// the cycle park in pendingPodGroupPods and are handed back to
// AddAttemptedPodGroupIfNeeded, which owns requeueing the attempted group with
// its accumulated backoff counters.
//
// AddUnschedulablePodIfNotPresent calls activeQ.clearPoppedEntity()
// unconditionally on the single-pod path. When that call belongs to a binding
// goroutine while the scheduling loop is inside an unrelated pod group's cycle,
// it wipes the in-flight group's marker. The group's failing member pods then
// take the "first member of a new group" path and are requeued immediately into
// activeQ with a freshly built root entity (backoff counters reset), and the
// subsequent AddAttemptedPodGroupIfNeeded fails with "already present in the
// active queue" instead of performing the authoritative requeue.
func TestAddUnschedulablePodClearsInFlightPodGroupMarker(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)

	const pgName = "gang"
	podG1 := st.MakePod().Name("g1").Namespace("ns1").UID("g1").Priority(midPriority).PodGroupName(pgName).Obj()
	podG2 := st.MakePod().Name("g2").Namespace("ns1").UID("g2").Priority(midPriority).PodGroupName(pgName).Obj()
	// Higher priority so that Pop() returns the single pod before the group,
	// leaving the group as the last popped entity while the single pod's
	// binding goroutine is still running.
	single := st.MakePod().Name("single").Namespace("ns1").UID("single").Priority(highPriority).Obj()

	type outcome struct {
		requeued      *framework.QueuedPodGroupInfo
		inBackoffQ    bool
		inActiveQ     bool
		addAttemptErr error
	}

	runScenario := func(t *testing.T, requeueSinglePodMidCycle bool) outcome {
		tCtx := ktesting.Init(t)
		logger := tCtx.Logger()
		c := testingclock.NewFakeClock(time.Now())
		q := NewTestQueue(tCtx, newDefaultQueueSort(), WithClock(c),
			WithQueueingHintMapPerProfile(makeEmptyQueueingHintMapPerProfile()),
			WithPreEnqueuePluginMap(make(map[string]map[string]fwk.PreEnqueuePlugin)))

		podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()
		q.AddGenericPodGroup(logger, fwk.NewGenericPodGroup(podGroup))

		q.Add(tCtx, single)
		q.Add(tCtx, podG1)
		q.Add(tCtx, podG2)

		singleEntity, err := q.Pop(logger)
		if err != nil {
			tCtx.Fatalf("Failed to pop single pod: %v", err)
		}
		singleInfo := singleEntity.(*framework.QueuedPodInfo)

		groupEntity, err := q.Pop(logger)
		if err != nil {
			tCtx.Fatalf("Failed to pop pod group: %v", err)
		}
		pgInfo := groupEntity.(*framework.QueuedPodGroupInfo)

		if requeueSinglePodMidCycle {
			// The single pod's binding cycle fails while the scheduling loop
			// is processing the pod group's cycle.
			failedSingle := singleInfo.DeepCopy()
			failedSingle.UnschedulablePlugins = sets.New("fakePlugin")
			if err := q.AddUnschedulablePodIfNotPresent(tCtx, failedSingle, q.SchedulingCycle()); err != nil {
				tCtx.Fatalf("Failed to requeue single pod: %v", err)
			}
		}

		// The pod group's scheduling cycle fails; the scheduler requeues the
		// member pods through the failure handler, as submitPodGroupResult does.
		for _, pInfo := range pgInfo.PodInfosForGroup(pgInfo.PodGroupInfo.GetKey()) {
			failed := pInfo.DeepCopy()
			failed.UnschedulablePlugins = sets.New("fakePlugin")
			if err := q.AddUnschedulablePodIfNotPresent(tCtx, failed, q.SchedulingCycle()); err != nil {
				tCtx.Fatalf("Failed to requeue member pod %s: %v", pInfo.Pod.Name, err)
			}
		}

		// Contract: the outcome of the pod group scheduling cycle must not
		// depend on an unrelated single pod being requeued concurrently. The
		// failed member pods must be parked in pendingPodGroupPods, not
		// requeued as a standalone group entity.
		if got := len(q.PendingPodGroupPods()); got != 2 {
			tCtx.Errorf("Expected 2 pods in pendingPodGroupPods, got %d", got)
		}
		if got := len(q.PodsInActiveQ()); got != 0 {
			tCtx.Errorf("Expected no pods in activeQ before AddAttemptedPodGroupIfNeeded, got %d", got)
		}

		addAttemptErr := q.AddAttemptedPodGroupIfNeeded(logger, pgInfo, q.SchedulingCycle(), fwk.NewStatus(fwk.Unschedulable))

		requeued, ok := q.GetPodGroup(pgName, "ns1")
		if !ok {
			tCtx.Fatalf("Pod group not found in any queue after requeue")
		}
		return outcome{
			requeued:      requeued,
			inBackoffQ:    q.backoffQ.has(requeued),
			inActiveQ:     len(q.PodsInActiveQ()) > 0,
			addAttemptErr: addAttemptErr,
		}
	}

	// Both scenarios must produce the same, correct outcome: the attempted
	// group is requeued by AddAttemptedPodGroupIfNeeded into backoffQ with its
	// backoff counters carried over.
	for name, requeueSinglePodMidCycle := range map[string]bool{
		"control: no concurrent single-pod requeue":            false,
		"race: single pod requeued during the pod group cycle": true,
	} {
		t.Run(name, func(t *testing.T) {
			o := runScenario(t, requeueSinglePodMidCycle)
			if o.addAttemptErr != nil {
				t.Errorf("AddAttemptedPodGroupIfNeeded returned unexpected error: %v", o.addAttemptErr)
			}
			if !o.inBackoffQ || o.inActiveQ {
				t.Errorf("Expected the requeued pod group to be in backoffQ (inBackoffQ=%v, inActiveQ=%v)", o.inBackoffQ, o.inActiveQ)
			}
			if o.requeued.UnschedulableCount != 1 {
				t.Errorf("Expected UnschedulableCount=1 on the requeued group, got %d", o.requeued.UnschedulableCount)
			}
		})
	}
}
