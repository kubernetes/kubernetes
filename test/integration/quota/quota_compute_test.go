/*
Copyright 2026 The Kubernetes Authors.

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

package quota

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestQuotaPodCountReplenishOnDelete verifies that a quota on the pods object count blocks creation
// past the limit, and that deleting a pod replenishes usage so a create succeeds again.
func TestQuotaPodCountReplenishOnDelete(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-pod-count", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	quota := newResourceQuota(ns.Name, "quota", v1.ResourceList{
		v1.ResourcePods: resource.MustParse("2"),
	})
	waitForQuota(t, quota, srv.client)

	// Fill the quota: two pods are allowed, the third is denied.
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, newPod("pod-1", nil, nil), t)
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, newPod("pod-2", nil, nil), t)
	expectedQuotaUsed := v1.ResourceList{
		v1.ResourcePods: resource.MustParse("2"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)
	createPodExpectForbidden(srv.ctx, srv.client, ns.Name, newPod("pod-3", nil, nil), t)

	// Deleting a pod replenishes the quota, so the previously-denied pod can be created.
	if err := srv.client.CoreV1().Pods(ns.Name).Delete(srv.ctx, "pod-1", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("deleting pod-1: %v", err)
	}
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, newPod("pod-3", nil, nil), t)
	expectedQuotaUsed = v1.ResourceList{
		v1.ResourcePods: resource.MustParse("2"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)
}

// TestQuotaComputeResources verifies that compute request/limit quotas (cpu and memory) are charged
// from pod specs and deny pods that would exceed the hard limit.
func TestQuotaComputeResources(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-compute", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	quota := newResourceQuota(ns.Name, "quota", v1.ResourceList{
		v1.ResourceRequestsCPU:    resource.MustParse("1"),
		v1.ResourceRequestsMemory: resource.MustParse("1Gi"),
		v1.ResourceLimitsCPU:      resource.MustParse("2"),
		v1.ResourceLimitsMemory:   resource.MustParse("2Gi"),
	})
	waitForQuota(t, quota, srv.client)

	// A pod within budget is charged for all four compute dimensions.
	pod := newPod("pod-1",
		v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m"), v1.ResourceMemory: resource.MustParse("512Mi")},
		v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("1Gi")},
	)
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, pod, t)
	expectedQuotaUsed := v1.ResourceList{
		v1.ResourceRequestsCPU:    resource.MustParse("500m"),
		v1.ResourceRequestsMemory: resource.MustParse("512Mi"),
		v1.ResourceLimitsCPU:      resource.MustParse("1"),
		v1.ResourceLimitsMemory:   resource.MustParse("1Gi"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)

	// A second pod pushing requests.cpu over the hard limit (500m + 600m > 1) is denied. The other
	// dimensions stay within budget, so the only possible denial cause is requests.cpu.
	overBudget := newPod("pod-2",
		v1.ResourceList{v1.ResourceCPU: resource.MustParse("600m"), v1.ResourceMemory: resource.MustParse("256Mi")},
		v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("512Mi")},
	)
	createPodExpectForbidden(srv.ctx, srv.client, ns.Name, overBudget, t)
}

// TestQuotaPodConstraintsDenial verifies the admission Constraints check: once a quota tracks a
// compute resource, a pod that omits that request is denied ("must specify ...").
//
// Guard: this namespace must have no LimitRange, otherwise the LimitRanger admission plugin would
// default the missing request and the pod would pass, masking the denial.
func TestQuotaPodConstraintsDenial(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-constraints", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	// Hard limit is generous so a denial can only be the Constraints check, not an over-quota.
	quota := newResourceQuota(ns.Name, "quota", v1.ResourceList{
		v1.ResourceRequestsCPU: resource.MustParse("10"),
	})
	waitForQuota(t, quota, srv.client)

	// A pod that omits the cpu request is denied.
	createPodExpectForbidden(srv.ctx, srv.client, ns.Name, newPod("no-request", nil, nil), t)

	// A pod that specifies the cpu request is admitted.
	withRequest := newPod("with-request",
		v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
		nil,
	)
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, withRequest, t)
}

// TestQuotaHugepages verifies that a quota on requests.hugepages-2Mi is charged from a pod's
// hugepages request and denies pods that would exceed it.
func TestQuotaHugepages(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-hugepages", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	hugepages := v1.ResourceName("hugepages-2Mi")
	requestsHugepages := v1.ResourceName("requests.hugepages-2Mi")

	quota := newResourceQuota(ns.Name, "quota", v1.ResourceList{
		requestsHugepages: resource.MustParse("2Mi"),
	})
	waitForQuota(t, quota, srv.client)

	// Hugepages validation requires limit == request and that the container also sets cpu or memory.
	newHugepagePod := func(name string) *v1.Pod {
		return newPod(name,
			v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m"), hugepages: resource.MustParse("2Mi")},
			v1.ResourceList{hugepages: resource.MustParse("2Mi")},
		)
	}

	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, newHugepagePod("pod-1"), t)
	expectedQuotaUsed := v1.ResourceList{
		requestsHugepages: resource.MustParse("2Mi"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)

	// A second hugepage pod exceeds the 2Mi hard limit and is denied.
	createPodExpectForbidden(srv.ctx, srv.client, ns.Name, newHugepagePod("pod-2"), t)
}

// TestQuotaTerminatedPodExclusion verifies that when a pod reaches a terminal phase, the controller
// replenishes the compute/pods usage it was charged, while the count/pods object-count usage stays.
// This exercises update-driven replenishment via the
// DefaultUpdateFilter pods branch, which the helper wires into the controller.
func TestQuotaTerminatedPodExclusion(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-terminated", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	countPods := v1.ResourceName("count/pods")
	quota := newResourceQuota(ns.Name, "quota", v1.ResourceList{
		v1.ResourcePods:        resource.MustParse("10"),
		v1.ResourceRequestsCPU: resource.MustParse("10"),
		countPods:              resource.MustParse("10"),
	})
	waitForQuota(t, quota, srv.client)

	// Create a pod with a cpu request (required because the quota tracks requests.cpu).
	pod := newPod("pod-1",
		v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m")},
		nil,
	)
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, pod, t)
	expectedQuotaUsed := v1.ResourceList{
		v1.ResourcePods:        resource.MustParse("1"),
		v1.ResourceRequestsCPU: resource.MustParse("500m"),
		countPods:              resource.MustParse("1"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)

	// Drive the pod to a terminal phase via the status subresource (no kubelet runs here).
	got, err := srv.client.CoreV1().Pods(ns.Name).Get(srv.ctx, pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("getting pod %s: %v", pod.Name, err)
	}
	got.Status.Phase = v1.PodSucceeded
	got.Status.ContainerStatuses = []v1.ContainerStatus{{
		Name: "container",
		State: v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.Now()},
		},
	}}
	if _, err := srv.client.CoreV1().Pods(ns.Name).UpdateStatus(srv.ctx, got, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("updating pod %s status to Succeeded: %v", pod.Name, err)
	}

	// The controller replenishes compute/pods usage for the terminal pod, but count/pods keeps it.
	expectedQuotaUsed = v1.ResourceList{
		v1.ResourcePods:        resource.MustParse("0"),
		v1.ResourceRequestsCPU: resource.MustParse("0"),
		countPods:              resource.MustParse("1"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)
}
