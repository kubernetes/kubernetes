/*
Copyright 2024 The Kubernetes Authors.

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

	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/backend/heap"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestClose(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	rr := metrics.NewMetricsAsyncRecorder(10, time.Second, ctx.Done())
	aq := newActiveQueue(heap.NewWithRecorder(podInfoKeyFunc, heap.LessFunc[*framework.QueuedPodInfo](newDefaultQueueSort()), metrics.NewActivePodsRecorder()), true, *rr)

	aq.underLock(func(unlockedActiveQ unlockedActiveQueuer) {
		unlockedActiveQ.AddOrUpdate(&framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: st.MakePod().Namespace("foo").Name("p1").UID("p1").Obj()}})
		unlockedActiveQ.AddOrUpdate(&framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: st.MakePod().Namespace("bar").Name("p2").UID("p2").Obj()}})
	})

	_, err := aq.pop(logger)
	if err != nil {
		t.Fatalf("unexpected error while pop(): %v", err)
	}
	_, err = aq.pop(logger)
	if err != nil {
		t.Fatalf("unexpected error while pop(): %v", err)
	}
	aq.addEventIfAnyInFlight(nil, nil, framework.NodeAdd)
	aq.addEventIfAnyInFlight(nil, nil, framework.NodeConditionChange)

	if len(aq.listInFlightEvents()) != 4 {
		t.Fatalf("unexpected number of in-flight events: %v", len(aq.listInFlightEvents()))
	}
	if len(aq.listInFlightPods()) != 2 {
		t.Fatalf("unexpected number of in-flight pods: %v", len(aq.listInFlightPods()))
	}

	aq.close()

	// make sure the in-flight events and pods are cleaned up by close()

	if len(aq.listInFlightEvents()) != 0 {
		t.Fatalf("in-flight events should be cleaned up, but %v item(s) is remaining", len(aq.listInFlightEvents()))
	}

	if len(aq.listInFlightPods()) != 0 {
		t.Fatalf("in-flight pods should be cleaned up, but %v pod(s) is remaining", len(aq.listInFlightPods()))
	}
}
