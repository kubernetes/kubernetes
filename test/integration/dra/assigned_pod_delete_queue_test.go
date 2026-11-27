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

package dra

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testingclock "k8s.io/utils/clock/testing"
)

func TestAssignedPodDeleteRequeueOrdering(t *testing.T) {
	metrics.Register()
	orders := []string{"event-before-requeue", "event-after-requeue"}
	for i := range 30 {
		orders = append(orders, fmt.Sprintf("concurrent-%02d", i))
	}
	for _, order := range orders {
		for _, registered := range []bool{false, true} {
			for _, hasClaims := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/registered=%t/claims=%t", order, registered, hasClaims), func(t *testing.T) {
					logger, ctx := ktesting.NewTestContext(t)
					ctx, cancel := context.WithCancel(ctx)
					defer cancel()
					holder := st.MakePod().Name("holder").UID("holder").Obj()
					if hasClaims {
						holder.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "device"}}
					}
					waiter := st.MakePod().Name("waiter").UID("waiter").Obj()
					actualHint := assignedPodDeletionHint(t, false, nil)
					var hintCalls atomic.Int32
					var wrongObjects atomic.Bool
					hintMap := queue.QueueingHintMapPerProfile{"": {}}
					if registered {
						hintMap[""][framework.EventAssignedPodDelete] = []*queue.QueueingHintFunction{{
							PluginName: "DynamicResources",
							QueueingHintFn: func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
								hintCalls.Add(1)
								if pod.UID != waiter.UID || oldObj != holder || newObj != nil {
									wrongObjects.Store(true)
								}
								return actualHint(logger, pod, oldObj, newObj)
							},
						}}
					}
					// No queue workers or clock advancement: neither a flush nor backoff can rescue the waiter.
					q := queue.NewTestQueue(ctx, (&queuesort.PrioritySort{}).Less, queue.WithClock(testingclock.NewFakeClock(time.Now())),
						queue.WithPodInitialBackoffDuration(0), queue.WithPodMaxBackoffDuration(0), queue.WithQueueingHintMapPerProfile(hintMap))
					defer q.Close()
					q.Add(ctx, waiter)
					info := popDRAWaiter(t, logger, q, waiter)
					info.UnschedulablePlugins = sets.New("DynamicResources")
					cycle := q.SchedulingCycle()
					event := func() { q.MoveAllToActiveOrBackoffQueue(logger, framework.EventAssignedPodDelete, holder, nil, nil) }
					requeue := func() error { return q.AddUnschedulablePodIfNotPresent(ctx, info, cycle) }
					switch order {
					case "event-before-requeue":
						event()
						require.Zero(t, hintCalls.Load(), "the hint must wait until the failed plugin is known")
						require.Len(t, q.InFlightPods(), 1)
						require.Equal(t, waiter.UID, q.InFlightPods()[0].UID)
						require.NoError(t, requeue())
					case "event-after-requeue":
						require.NoError(t, requeue())
						require.Empty(t, q.PodsInActiveQ())
						require.Len(t, q.UnschedulablePods(), 1)
						event()
					default:
						start := make(chan struct{})
						var wg sync.WaitGroup
						var enqueueErr error
						wg.Add(2)
						go func() { defer wg.Done(); <-start; event() }()
						go func() { defer wg.Done(); <-start; enqueueErr = requeue() }()
						close(start)
						wg.Wait()
						require.NoError(t, enqueueErr)
					}
					require.False(t, wrongObjects.Load(), "hint must evaluate this holder's deletion for this waiter")
					require.Empty(t, q.InFlightPods())
					require.Empty(t, q.PodsInBackoffQ())
					if registered {
						require.EqualValues(t, 1, hintCalls.Load())
					} else {
						require.Zero(t, hintCalls.Load())
					}
					if registered && hasClaims {
						require.Len(t, q.PodsInActiveQ(), 1)
						require.Empty(t, q.UnschedulablePods())
						second := popDRAWaiter(t, logger, q, waiter)
						require.Equal(t, 2, second.Attempts)
						q.Done(waiter.UID)
					} else {
						require.Empty(t, q.PodsInActiveQ())
						require.Len(t, q.UnschedulablePods(), 1, "skip and unregistered events must leave the waiter unschedulable")
					}
				})
			}
		}
	}
}

func popDRAWaiter(t *testing.T, logger klog.Logger, q *queue.PriorityQueue, pod *v1.Pod) *framework.QueuedPodInfo {
	t.Helper()
	entity, err := q.Pop(logger)
	require.NoError(t, err)
	info, ok := entity.(*framework.QueuedPodInfo)
	require.True(t, ok, "expected a single Pod, got %T", entity)
	require.Equal(t, pod.UID, info.Pod.UID)
	return info
}
