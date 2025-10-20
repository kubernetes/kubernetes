/*
Copyright 2025 The Kubernetes Authors.

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

package workloadmanager

import (
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

func TestPodGroupInfo_AssumeForget(t *testing.T) {
	pgi := newPodGroupInfo()
	pod := st.MakePod().Namespace("ns1").Name("p1").UID("p1").
		WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()

	pgi.addPod(pod)
	if pgi.AssumedPods().Has(pod.UID) {
		t.Fatal("AssumedPods should be initially empty")
	}
	if !pgi.unscheduledPods.Has(pod.UID) {
		t.Fatal("Pod should be initially in UnscheduledPods")
	}

	pgi.AssumePod(pod.UID)
	if !pgi.AssumedPods().Has(pod.UID) {
		t.Fatal("Pod should be in AssumedPods after AssumePod")
	}
	if pgi.unscheduledPods.Has(pod.UID) {
		t.Fatal("UnscheduledPods should be empty after AssumePod")
	}

	pgi.ForgetPod(pod.UID)
	if pgi.AssumedPods().Has(pod.UID) {
		t.Fatal("Pod should not be in AssumedPods after ForgetPod")
	}
	if !pgi.unscheduledPods.Has(pod.UID) {
		t.Fatal("Pod should be in UnscheduledPods after ForgetPod")
	}
}

func TestPodGroupInfo_SchedulingTimeout(t *testing.T) {
	pgi := newPodGroupInfo()

	timeout := pgi.SchedulingTimeout()
	if pgi.schedulingDeadline == nil {
		t.Fatal("Scheduling deadline should be set after SchedulingTimeout call, but is nil")
	}
	if timeout <= 0 {
		t.Errorf("Expected positive timeout duration, got %v", timeout)
	}

	deadline := *pgi.schedulingDeadline
	newTimeout := pgi.SchedulingTimeout()
	if !deadline.Equal(*pgi.schedulingDeadline) {
		t.Errorf("Previous deadline should not be changed: previous: %v, current: %v", deadline, *pgi.schedulingDeadline)
	}
	if newTimeout >= timeout {
		t.Errorf("Expected lower timeout duration: previous: %v, current: %v", timeout, newTimeout)
	}

	pgi.schedulingDeadline = ptr.To(time.Now().Add(-1 * time.Second))
	newTimeout = pgi.SchedulingTimeout()
	if deadline.Equal(*pgi.schedulingDeadline) {
		t.Error("Deadline should be reset after it has expired, but it wasn't")
	}
	if newTimeout <= 0 {
		t.Errorf("Expected positive timeout duration after reset, got %v", timeout)
	}
}
