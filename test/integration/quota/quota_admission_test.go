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
	"context"
	"os"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestQuotaDryRun verifies that a dry-run create is enforced by quota (an over-quota dry-run is
// still denied) but is never charged (an in-budget dry-run does not consume the quota).
func TestQuotaDryRun(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-dry-run", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	quota := newResourceQuota(ns.Name, "quota", v1.ResourceList{v1.ResourcePods: resource.MustParse("1")})
	waitForQuota(t, quota, srv.client)
	expectedQuotaUsed := v1.ResourceList{
		v1.ResourcePods: resource.MustParse("0"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)

	dryRun := metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}}

	// An in-budget dry-run create succeeds but must not consume the single pod slot.
	if _, err := srv.client.CoreV1().Pods(ns.Name).Create(srv.ctx, newPod("dry-a", nil, nil), dryRun); err != nil {
		t.Fatalf("in-budget dry-run create should succeed: %v", err)
	}

	// The real create still fits, proving the dry-run above did not charge the quota.
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, newPod("real-b", nil, nil), t)
	expectedQuotaUsed = v1.ResourceList{
		v1.ResourcePods: resource.MustParse("1"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)

	// With the quota now full, an over-quota dry-run create is denied just like a real create.
	err := wait.PollUntilContextTimeout(srv.ctx, quotaPollInterval, quotaPollTimeout, true, func(ctx context.Context) (bool, error) {
		_, err := srv.client.CoreV1().Pods(ns.Name).Create(ctx, newPod("dry-c", nil, nil), dryRun)
		switch {
		case apierrors.IsForbidden(err):
			return true, nil
		case err != nil:
			return false, err
		default:
			// Dry-run success is not persisted, so just keep polling until usage propagates.
			return false, nil
		}
	})
	if err != nil {
		t.Fatalf("over-quota dry-run create should eventually be forbidden but got: %v", err)
	}
}

// TestQuotaMultipleQuotas verifies that when a namespace has multiple ResourceQuotas, the most
// restrictive one binds: both are charged, and the smaller limit denies first.
func TestQuotaMultipleQuotas(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-multiple", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	loose := newResourceQuota(ns.Name, "loose", v1.ResourceList{v1.ResourcePods: resource.MustParse("5")})
	tight := newResourceQuota(ns.Name, "tight", v1.ResourceList{v1.ResourcePods: resource.MustParse("1")})
	waitForQuota(t, loose, srv.client)
	waitForQuota(t, tight, srv.client)

	// The single allowed pod is charged against both quotas.
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, newPod("pod-1", nil, nil), t)
	expectedQuotaUsed := v1.ResourceList{
		v1.ResourcePods: resource.MustParse("1"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, loose.Name, expectedQuotaUsed)
	waitForUsedResourceQuota(t, srv.client, ns.Name, tight.Name, expectedQuotaUsed)

	// The tighter quota (pods: 1) denies the second pod even though the looser one (pods: 5) has room.
	createPodExpectForbidden(srv.ctx, srv.client, ns.Name, newPod("pod-2", nil, nil), t)
}

// TestQuotaLimitedResourceComputeDenial verifies the limitedResources admission config for a
// compute dimension: when requests.cpu is a limited resource, a pod that requests cpu is denied
// unless a quota covering requests.cpu exists. This extends the existing pods/count-based
// limitedResources coverage (TestQuotaLimitedResourceDenial) to a compute resource.
func TestQuotaLimitedResourceComputeDenial(t *testing.T) {
	configFile, err := os.CreateTemp("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(configFile.Name())
	if err := os.WriteFile(configFile.Name(), []byte(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: ResourceQuota
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: ResourceQuotaConfiguration
    limitedResources:
    - resource: pods
      matchContains:
      - requests.cpu
`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{admissionConfigFile: configFile.Name()})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-limited-compute", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	cpuPod := func(name string) *v1.Pod {
		return newPod(name, v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")}, nil)
	}

	// With no covering quota, a pod that consumes requests.cpu is denied by limitedResources.
	createPodExpectForbidden(srv.ctx, srv.client, ns.Name, cpuPod("needs-cpu"), t)

	// A pod that does not consume requests.cpu is unaffected by the limited resource.
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, newPod("no-cpu", nil, nil), t)

	// Once a quota covering requests.cpu exists, the cpu-requesting pod is admitted.
	quota := newResourceQuota(ns.Name, "quota", v1.ResourceList{v1.ResourceRequestsCPU: resource.MustParse("10")})
	waitForQuota(t, quota, srv.client)
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, cpuPod("needs-cpu-allowed"), t)
}
