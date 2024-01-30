/*
Copyright 2019 The Kubernetes Authors.

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

package util

import (
	"runtime"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

var (
	t0 = time.Date(2019, 01, 01, 0, 0, 0, 0, time.UTC)
	t1 = t0.Add(time.Second)
	t2 = t1.Add(time.Second)
	t3 = t2.Add(time.Second)
	t4 = t3.Add(time.Second)
	t5 = t4.Add(time.Second)

	ttNamespace   = "ttNamespace1"
	ttServiceName = "my-service"
)

func TestNewServiceNoPods(t *testing.T) {
	tester := newTester(t)

	service := createService(ttNamespace, ttServiceName, t2)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service).expect(t2)
}

func TestNewServiceExistingPods(t *testing.T) {
	tester := newTester(t)

	service := createService(ttNamespace, ttServiceName, t3)
	pod1 := createPod(ttNamespace, "pod1", t0)
	pod2 := createPod(ttNamespace, "pod2", t1)
	pod3 := createPod(ttNamespace, "pod3", t5)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2, pod3).
		// Pods were created before service, but trigger time is the time when service was created.
		expect(t3)
}

func TestPodsAdded(t *testing.T) {
	tester := newTester(t)

	service := createService(ttNamespace, ttServiceName, t0)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service).expect(t0)

	pod1 := createPod(ttNamespace, "pod1", t2)
	pod2 := createPod(ttNamespace, "pod2", t1)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2).expect(t1)
}

func TestPodsUpdated(t *testing.T) {
	tester := newTester(t)

	service := createService(ttNamespace, ttServiceName, t0)
	pod1 := createPod(ttNamespace, "pod1", t1)
	pod2 := createPod(ttNamespace, "pod2", t2)
	pod3 := createPod(ttNamespace, "pod3", t3)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2, pod3).expect(t0)

	pod1 = createPod(ttNamespace, "pod1", t5)
	pod2 = createPod(ttNamespace, "pod2", t4)
	// pod3 doesn't change.
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2, pod3).expect(t4)
}

func TestPodsUpdatedNoOp(t *testing.T) {
	tester := newTester(t)

	service := createService(ttNamespace, ttServiceName, t0)
	pod1 := createPod(ttNamespace, "pod1", t1)
	pod2 := createPod(ttNamespace, "pod2", t2)
	pod3 := createPod(ttNamespace, "pod3", t3)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2, pod3).expect(t0)

	// Nothing has changed.
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2, pod3).expectNil()
}

func TestPodDeletedThenAdded(t *testing.T) {
	tester := newTester(t)

	service := createService(ttNamespace, ttServiceName, t0)
	pod1 := createPod(ttNamespace, "pod1", t1)
	pod2 := createPod(ttNamespace, "pod2", t2)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2).expect(t0)

	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1).expectNil()

	pod2 = createPod(ttNamespace, "pod2", t4)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2).expect(t4)
}

func TestServiceDeletedThenAdded(t *testing.T) {
	tester := newTester(t)

	service := createService(ttNamespace, ttServiceName, t0)
	pod1 := createPod(ttNamespace, "pod1", t1)
	pod2 := createPod(ttNamespace, "pod2", t2)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2).expect(t0)

	tester.DeleteService(ttNamespace, ttServiceName)

	service = createService(ttNamespace, ttServiceName, t3)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2).expect(t3)
}

func TestServiceUpdatedNoPodChange(t *testing.T) {
	tester := newTester(t)

	service := createService(ttNamespace, ttServiceName, t0)
	pod1 := createPod(ttNamespace, "pod1", t1)
	pod2 := createPod(ttNamespace, "pod2", t2)
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2).expect(t0)

	// service's ports have changed.
	service.Spec = v1.ServiceSpec{
		Selector: map[string]string{},
		Ports:    []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: "TCP"}},
	}

	// Currently we're not able to calculate trigger time for service updates, hence the returned
	// value is a nil time.
	tester.whenComputeEndpointLastChangeTriggerTime(ttNamespace, service, pod1, pod2).expectNil()
}

// ------- Test Utils -------

type tester struct {
	*TriggerTimeTracker
	t *testing.T
}

func newTester(t *testing.T) *tester {
	return &tester{NewTriggerTimeTracker(), t}
}

func (t *tester) whenComputeEndpointLastChangeTriggerTime(
	namespace string, service *v1.Service, pods ...*v1.Pod) subject {
	return subject{t.ComputeEndpointLastChangeTriggerTime(namespace, service, pods), t.t}
}

type subject struct {
	got time.Time
	t   *testing.T
}

func (s subject) expect(expected time.Time) {
	s.doExpect(expected)
}

func (s subject) expectNil() {
	s.doExpect(time.Time{})
}

func (s subject) doExpect(expected time.Time) {
	if s.got != expected {
		_, fn, line, _ := runtime.Caller(2)
		s.t.Errorf("Wrong trigger time in %s:%d expected %s, got %s", fn, line, expected, s.got)
	}
}

func createPod(namespace, ttServiceName string, readyTime time.Time) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: ttServiceName},
		Status: v1.PodStatus{Conditions: []v1.PodCondition{
			{
				Type:               v1.PodReady,
				Status:             v1.ConditionTrue,
				LastTransitionTime: metav1.NewTime(readyTime),
			},
		},
		},
	}
}

func createService(namespace, ttServiceName string, creationTime time.Time) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         namespace,
			Name:              ttServiceName,
			CreationTimestamp: metav1.NewTime(creationTime),
		},
	}
}
