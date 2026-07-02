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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/integration/framework"
)

// waitForResourceQuotaHard polls until the quota's Status.Hard exactly matches the expected hard
// list (both the set of keys and each value), so it also detects resources being removed.
func waitForResourceQuotaHard(t *testing.T, srv *quotaTestServer, ns, name string, hard v1.ResourceList) {
	t.Helper()
	err := wait.PollUntilContextTimeout(srv.ctx, quotaPollInterval, quotaPollTimeout, true, func(ctx context.Context) (bool, error) {
		q, err := srv.client.CoreV1().ResourceQuotas(ns).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if len(q.Status.Hard) != len(hard) {
			return false, nil
		}
		for k, v := range hard {
			actual, ok := q.Status.Hard[k]
			if !ok || !actual.Equal(v) {
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("quota %s status.hard did not converge to %v: %v", name, hard, err)
	}
}

// updateQuotaHard replaces a quota's Spec.Hard, retrying on conflict because the controller writes
// the status subresource concurrently and bumps the resourceVersion.
func updateQuotaHard(t *testing.T, srv *quotaTestServer, ns, name string, hard v1.ResourceList) {
	t.Helper()
	if err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		q, err := srv.client.CoreV1().ResourceQuotas(ns).Get(srv.ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		q.Spec.Hard = hard
		_, err = srv.client.CoreV1().ResourceQuotas(ns).Update(srv.ctx, q, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatalf("updating quota %s hard: %v", name, err)
	}
}

// TestQuotaStatusInitialized verifies that creating a ResourceQuota leads the controller to
// initialize Status.Hard (mirroring Spec.Hard) and Status.Used (zero for every tracked resource).
func TestQuotaStatusInitialized(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-status-init", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	hard := v1.ResourceList{
		v1.ResourcePods:        resource.MustParse("5"),
		v1.ResourceRequestsCPU: resource.MustParse("1"),
	}
	quota := newResourceQuota(ns.Name, "quota", hard)
	waitForQuota(t, quota, srv.client)

	waitForResourceQuotaHard(t, srv, ns.Name, quota.Name, hard)
	expectedQuotaUsed := v1.ResourceList{
		v1.ResourcePods:        resource.MustParse("0"),
		v1.ResourceRequestsCPU: resource.MustParse("0"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)
}

// TestQuotaUpdateHardRecompute verifies that editing Spec.Hard re-drives the controller: adding a
// resource makes it appear in Status.Hard/Used, and removing one drops it.
func TestQuotaUpdateHardRecompute(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-update-hard", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	quota := newResourceQuota(ns.Name, "quota", v1.ResourceList{v1.ResourcePods: resource.MustParse("5")})
	waitForQuota(t, quota, srv.client)
	waitForResourceQuotaHard(t, srv, ns.Name, quota.Name, v1.ResourceList{v1.ResourcePods: resource.MustParse("5")})

	// Add a resource and tighten pods: the controller recomputes both Hard and Used.
	updateQuotaHard(t, srv, ns.Name, quota.Name, v1.ResourceList{
		v1.ResourcePods:        resource.MustParse("3"),
		v1.ResourceRequestsCPU: resource.MustParse("2"),
	})
	waitForResourceQuotaHard(t, srv, ns.Name, quota.Name, v1.ResourceList{
		v1.ResourcePods:        resource.MustParse("3"),
		v1.ResourceRequestsCPU: resource.MustParse("2"),
	})
	expectedQuotaUsed := v1.ResourceList{
		v1.ResourcePods:        resource.MustParse("0"),
		v1.ResourceRequestsCPU: resource.MustParse("0"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, expectedQuotaUsed)

	// Remove the added resource: it drops out of Status.Hard.
	updateQuotaHard(t, srv, ns.Name, quota.Name, v1.ResourceList{v1.ResourcePods: resource.MustParse("3")})
	waitForResourceQuotaHard(t, srv, ns.Name, quota.Name, v1.ResourceList{v1.ResourcePods: resource.MustParse("3")})
}

// TestQuotaListAndDeleteCollection verifies the ResourceQuota collection endpoints: List returns
// every quota in the namespace and DeleteCollection removes them all.
func TestQuotaListAndDeleteCollection(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-list", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	for _, name := range []string{"quota-a", "quota-b", "quota-c"} {
		q := newResourceQuota(ns.Name, name, v1.ResourceList{v1.ResourcePods: resource.MustParse("1")})
		if _, err := srv.client.CoreV1().ResourceQuotas(ns.Name).Create(srv.ctx, q, metav1.CreateOptions{}); err != nil {
			t.Fatalf("creating quota %s: %v", name, err)
		}
	}

	list, err := srv.client.CoreV1().ResourceQuotas(ns.Name).List(srv.ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("listing quotas: %v", err)
	}
	if len(list.Items) != 3 {
		t.Fatalf("expected 3 quotas, got %d", len(list.Items))
	}

	if err := srv.client.CoreV1().ResourceQuotas(ns.Name).DeleteCollection(srv.ctx, metav1.DeleteOptions{}, metav1.ListOptions{}); err != nil {
		t.Fatalf("delete collection: %v", err)
	}

	// DeleteCollection is observed asynchronously, so poll until the namespace has no quotas.
	err = wait.PollUntilContextTimeout(srv.ctx, quotaPollInterval, quotaPollTimeout, true, func(ctx context.Context) (bool, error) {
		remaining, err := srv.client.CoreV1().ResourceQuotas(ns.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		return len(remaining.Items) == 0, nil
	})
	if err != nil {
		t.Fatalf("quotas were not all deleted: %v", err)
	}
}
