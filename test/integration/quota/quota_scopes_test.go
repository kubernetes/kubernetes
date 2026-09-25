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
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

// mustCreatePriorityClass creates a (cluster-scoped) PriorityClass, tolerating AlreadyExists. The
// Priority admission plugin is on by default and rejects pods referencing a class that does not
// exist, so scope tests must create their classes before creating pods that reference them.
func mustCreatePriorityClass(t *testing.T, srv *quotaTestServer, name string, value int32) {
	t.Helper()
	_, err := srv.client.SchedulingV1().PriorityClasses().Create(srv.ctx, &schedulingv1.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Value:      value,
	}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("creating priority class %s: %v", name, err)
	}
}

// newPriorityClassScopedQuota builds a ResourceQuota whose PriorityClass scope is expressed via a
// single scope-selector requirement (the only supported form for the PriorityClass scope).
func newPriorityClassScopedQuota(ns, name string, hard v1.ResourceList, op v1.ScopeSelectorOperator, values []string) *v1.ResourceQuota {
	return &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec: v1.ResourceQuotaSpec{
			Hard: hard,
			ScopeSelector: &v1.ScopeSelector{
				MatchExpressions: []v1.ScopedResourceSelectorRequirement{
					{ScopeName: v1.ResourceQuotaScopePriorityClass, Operator: op, Values: values},
				},
			},
		},
	}
}

// terminatingPod returns a pod that the quota system treats as Terminating (activeDeadlineSeconds set).
func terminatingPod(name string) *v1.Pod {
	pod := newPod(name, nil, nil)
	pod.Spec.ActiveDeadlineSeconds = ptr.To(int64(100))
	return pod
}

// priorityPod returns a pod that references the given priority class.
func priorityPod(name, pclass string, requests v1.ResourceList) *v1.Pod {
	pod := newPod(name, requests, nil)
	pod.Spec.PriorityClassName = pclass
	return pod
}

// burstablePod returns a Burstable pod (it sets a cpu request but no limit).
func burstablePod(name string) *v1.Pod {
	requests := v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")}
	return newPod(name, requests, nil)
}

// crossNamespaceAffinityPod returns a pod whose required pod affinity targets another namespace,
// which the CrossNamespacePodAffinity scope matches.
func crossNamespaceAffinityPod(name string) *v1.Pod {
	pod := newPod(name, nil, nil)
	pod.Spec.Affinity = &v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{{
				LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				Namespaces:    []string{"other-namespace"},
				TopologyKey:   "kubernetes.io/hostname",
			}},
		},
	}
	return pod
}

// TestQuotaScopes verifies that a scoped pods=1 quota counts only the pods matching its scope: the
// "uncounted" pod does not match the scope and never charges the quota, while a "counted" pod
// matches, fills the single-pod budget, and a second matching pod is denied. Each scope (Terminating
// / BestEffort / PriorityClass / CrossNamespacePodAffinity, and their negations) is a row.
func TestQuotaScopes(t *testing.T) {
	podsHard := v1.ResourceList{v1.ResourcePods: resource.MustParse("1")}

	tests := []struct {
		name string
		// priorityClasses, if set, are created (cluster-scoped) before the quota.
		priorityClasses []string
		quota           func(ns string) *v1.ResourceQuota
		uncounted       func(name string) *v1.Pod
		counted         func(name string) *v1.Pod
	}{
		{
			name: "Terminating",
			quota: func(ns string) *v1.ResourceQuota {
				return newResourceQuota(ns, "quota", podsHard, v1.ResourceQuotaScopeTerminating)
			},
			uncounted: func(name string) *v1.Pod { return newPod(name, nil, nil) },
			counted:   terminatingPod,
		},
		{
			name: "NotTerminating",
			quota: func(ns string) *v1.ResourceQuota {
				return newResourceQuota(ns, "quota", podsHard, v1.ResourceQuotaScopeNotTerminating)
			},
			uncounted: terminatingPod,
			counted:   func(name string) *v1.Pod { return newPod(name, nil, nil) },
		},
		{
			name: "BestEffort",
			quota: func(ns string) *v1.ResourceQuota {
				return newResourceQuota(ns, "quota", podsHard, v1.ResourceQuotaScopeBestEffort)
			},
			uncounted: burstablePod,
			counted:   func(name string) *v1.Pod { return newPod(name, nil, nil) },
		},
		{
			name: "NotBestEffort",
			quota: func(ns string) *v1.ResourceQuota {
				return newResourceQuota(ns, "quota", podsHard, v1.ResourceQuotaScopeNotBestEffort)
			},
			uncounted: func(name string) *v1.Pod { return newPod(name, nil, nil) },
			counted:   burstablePod,
		},
		{
			name:            "PriorityClassIn",
			priorityClasses: []string{"pclass-in"},
			quota: func(ns string) *v1.ResourceQuota {
				return newPriorityClassScopedQuota(ns, "quota", podsHard, v1.ScopeSelectorOpIn, []string{"pclass-in"})
			},
			uncounted: func(name string) *v1.Pod { return newPod(name, nil, nil) },
			counted:   func(name string) *v1.Pod { return priorityPod(name, "pclass-in", nil) },
		},
		{
			name:            "PriorityClassNotIn",
			priorityClasses: []string{"pclass-excluded", "pclass-counted"},
			quota: func(ns string) *v1.ResourceQuota {
				return newPriorityClassScopedQuota(ns, "quota", podsHard, v1.ScopeSelectorOpNotIn, []string{"pclass-excluded"})
			},
			uncounted: func(name string) *v1.Pod { return priorityPod(name, "pclass-excluded", nil) },
			counted:   func(name string) *v1.Pod { return priorityPod(name, "pclass-counted", nil) },
		},
		{
			name:            "PriorityClassExists",
			priorityClasses: []string{"pclass-exists"},
			quota: func(ns string) *v1.ResourceQuota {
				return newPriorityClassScopedQuota(ns, "quota", podsHard, v1.ScopeSelectorOpExists, nil)
			},
			uncounted: func(name string) *v1.Pod { return newPod(name, nil, nil) },
			counted:   func(name string) *v1.Pod { return priorityPod(name, "pclass-exists", nil) },
		},
		{
			name: "CrossNamespacePodAffinity",
			quota: func(ns string) *v1.ResourceQuota {
				return newResourceQuota(ns, "quota", podsHard, v1.ResourceQuotaScopeCrossNamespacePodAffinity)
			},
			uncounted: func(name string) *v1.Pod { return newPod(name, nil, nil) },
			counted:   crossNamespaceAffinityPod,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
			defer tearDown()

			ns := framework.CreateNamespaceOrDie(srv.client, "quota-scope", t)
			defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

			for _, pc := range tc.priorityClasses {
				mustCreatePriorityClass(t, srv, pc, 1000)
			}

			quota := tc.quota(ns.Name)
			waitForQuota(t, quota, srv.client)
			wantUsedZero := v1.ResourceList{v1.ResourcePods: resource.MustParse("0")}
			waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, wantUsedZero)

			// The uncounted pod does not match the scope, so it never charges the quota.
			createPodExpectAllowed(srv.ctx, srv.client, ns.Name, tc.uncounted("uncounted"), t)
			// The first matching pod fills the single-pod scoped quota.
			createPodExpectAllowed(srv.ctx, srv.client, ns.Name, tc.counted("counted-1"), t)
			wantUsedOne := v1.ResourceList{v1.ResourcePods: resource.MustParse("1")}
			waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, wantUsedOne)
			// A second matching pod exceeds it.
			createPodExpectForbidden(srv.ctx, srv.client, ns.Name, tc.counted("counted-2"), t)
		})
	}
}

// TestQuotaScopePriorityClassMultiCount verifies that when a quota's PriorityClass scope uses "In" with several classes,
// pods of any listed class draw from the same pod budget.
func TestQuotaScopePriorityClassMultiCount(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-pc-multi", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	mustCreatePriorityClass(t, srv, "pclass-a", 1000)
	mustCreatePriorityClass(t, srv, "pclass-b", 1000)

	podsHard := v1.ResourceList{v1.ResourcePods: resource.MustParse("2")}
	classes := []string{"pclass-a", "pclass-b"}
	quota := newPriorityClassScopedQuota(ns.Name, "quota", podsHard, v1.ScopeSelectorOpIn, classes)
	waitForQuota(t, quota, srv.client)
	wantUsedZero := v1.ResourceList{v1.ResourcePods: resource.MustParse("0")}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, wantUsedZero)

	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, priorityPod("pod-a", "pclass-a", nil), t)
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, priorityPod("pod-b", "pclass-b", nil), t)
	wantUsedTwo := v1.ResourceList{v1.ResourcePods: resource.MustParse("2")}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, wantUsedTwo)
	createPodExpectForbidden(srv.ctx, srv.client, ns.Name, priorityPod("pod-c", "pclass-a", nil), t)
}

// TestQuotaScopePriorityClassCompute verifies a PriorityClass-scoped quota can limit compute
// resources (not just pod count): a matching pod charges the scoped compute usage and exceeding it
// denies, while a pod without the class is unscoped and not charged.
func TestQuotaScopePriorityClassCompute(t *testing.T) {
	srv, tearDown := startQuotaTestServer(t, quotaServerOptions{})
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(srv.client, "quota-pc-compute", t)
	defer framework.DeleteNamespaceOrDie(srv.client, ns, t)

	mustCreatePriorityClass(t, srv, "pclass-compute", 1000)

	computeHard := v1.ResourceList{
		v1.ResourceRequestsCPU:    resource.MustParse("1"),
		v1.ResourceRequestsMemory: resource.MustParse("1Gi"),
	}
	quota := newPriorityClassScopedQuota(ns.Name, "quota", computeHard, v1.ScopeSelectorOpIn, []string{"pclass-compute"})
	waitForQuota(t, quota, srv.client)

	// A matching pod charges the scoped compute usage.
	matchingRequests := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("500m"),
		v1.ResourceMemory: resource.MustParse("512Mi"),
	}
	matching := priorityPod("matching", "pclass-compute", matchingRequests)
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, matching, t)

	// A pod without the class is unscoped: it is allowed and does not change the scoped usage.
	createPodExpectAllowed(srv.ctx, srv.client, ns.Name, newPod("unscoped", nil, nil), t)
	wantUsed := v1.ResourceList{
		v1.ResourceRequestsCPU:    resource.MustParse("500m"),
		v1.ResourceRequestsMemory: resource.MustParse("512Mi"),
	}
	waitForUsedResourceQuota(t, srv.client, ns.Name, quota.Name, wantUsed)

	// A second matching pod pushing requests.cpu over the hard limit (500m + 600m > 1) is denied.
	overBudgetRequests := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("600m"),
		v1.ResourceMemory: resource.MustParse("256Mi"),
	}
	overBudget := priorityPod("over-budget", "pclass-compute", overBudgetRequests)
	createPodExpectForbidden(srv.ctx, srv.client, ns.Name, overBudget, t)
}
