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

package deployment

import (
	"context"
	"slices"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/ktesting"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

const replicaSetPodGroupFinalizer = "apps.kubernetes.io/replicaset-podgroup-cleanup"

// waitForWorkload waits for a Workload owned by the given Deployment to appear
// (expectAbsent=false) or to not exist (expectAbsent=true).
func waitForWorkload(ctx context.Context, t *testing.T, c clientset.Interface, deployment *apps.Deployment, expectAbsent bool, timeout time.Duration) *schedulingv1beta1.Workload {
	t.Helper()
	var workload *schedulingv1beta1.Workload
	err := wait.PollUntilContextTimeout(ctx, pollInterval, timeout, true, func(ctx context.Context) (bool, error) {
		workloads, listErr := c.SchedulingV1beta1().Workloads(deployment.Namespace).List(ctx, metav1.ListOptions{})
		if listErr != nil {
			return false, listErr
		}
		for i := range workloads.Items {
			for _, ref := range workloads.Items[i].OwnerReferences {
				if ref.UID == deployment.UID && ref.Kind == "Deployment" {
					if expectAbsent {
						return false, nil
					}
					workload = &workloads.Items[i]
					return true, nil
				}
			}
		}
		return expectAbsent, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for Workload owned by Deployment %q (expectAbsent=%v): %v", deployment.Name, expectAbsent, err)
	}
	return workload
}

// waitForPodGroup waits for a non-terminating PodGroup controlled by rs
// (expectAbsent=false), or for no PodGroup controlled by rs (expectAbsent=true).
func waitForPodGroup(ctx context.Context, t *testing.T, c clientset.Interface, rs *apps.ReplicaSet, expectAbsent bool, timeout time.Duration) *schedulingv1beta1.PodGroup {
	t.Helper()
	var podGroup *schedulingv1beta1.PodGroup
	err := wait.PollUntilContextTimeout(ctx, pollInterval, timeout, true, func(ctx context.Context) (bool, error) {
		podGroups, listErr := c.SchedulingV1beta1().PodGroups(rs.Namespace).List(ctx, metav1.ListOptions{})
		if listErr != nil {
			return false, listErr
		}
		for i := range podGroups.Items {
			pg := &podGroups.Items[i]
			owner := metav1.GetControllerOf(pg)
			if owner == nil || owner.Kind != "ReplicaSet" || owner.UID != rs.UID {
				continue
			}
			if expectAbsent {
				return false, nil
			}
			if pg.DeletionTimestamp != nil {
				continue
			}
			podGroup = pg
			return true, nil
		}
		return expectAbsent, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for PodGroup controlled by ReplicaSet %q (expectAbsent=%v): %v", rs.Name, expectAbsent, err)
	}
	return podGroup
}

// waitForNewReplicaSet polls until the Deployment's new ReplicaSet exists.
func waitForNewReplicaSet(ctx context.Context, t *testing.T, tester *deploymentTester) *apps.ReplicaSet {
	t.Helper()
	var rs *apps.ReplicaSet
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(context.Context) (bool, error) {
		got, getErr := tester.getNewReplicaSet()
		if getErr != nil {
			return false, getErr
		}
		if got == nil {
			return false, nil
		}
		rs = got
		return true, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for new ReplicaSet of Deployment %q: %v", tester.deployment.Name, err)
	}
	return rs
}

func hasOwnerRef(refs []metav1.OwnerReference, kind string, uid types.UID) bool {
	for _, ref := range refs {
		if ref.Kind == kind && ref.UID == uid {
			return true
		}
	}
	return false
}

func gangMinCount(wl *schedulingv1beta1.Workload) int32 {
	if len(wl.Spec.PodGroupTemplates) == 0 || wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang == nil {
		return -1
	}
	return wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount
}

// waitForGangMinCount waits until the Workload template and the PodGroup both
// report the wanted gang minCount.
func waitForGangMinCount(ctx context.Context, t *testing.T, c clientset.Interface, ns, workloadName, podGroupName string, want int32) {
	t.Helper()
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		wl, err := c.SchedulingV1beta1().Workloads(ns).Get(ctx, workloadName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		pg, err := c.SchedulingV1beta1().PodGroups(ns).Get(ctx, podGroupName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return gangMinCount(wl) == want && pg.Spec.SchedulingPolicy.Gang != nil && pg.Spec.SchedulingPolicy.Gang.MinCount == want, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for gang minCount %d on Workload %q and PodGroup %q: %v", want, workloadName, podGroupName, err)
	}
}

// waitForPodGroupDeletionRequested waits until the PodGroup owned by the
// ReplicaSet is absent or marked for deletion.
func waitForPodGroupDeletionRequested(ctx context.Context, t *testing.T, c clientset.Interface, rs *apps.ReplicaSet) {
	t.Helper()
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		podGroups, err := c.SchedulingV1beta1().PodGroups(rs.Namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		for i := range podGroups.Items {
			for _, ref := range podGroups.Items[i].OwnerReferences {
				if ref.UID == rs.UID && ref.Kind == "ReplicaSet" {
					return podGroups.Items[i].DeletionTimestamp != nil, nil
				}
			}
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for deletion to be requested for PodGroup owned by ReplicaSet %q: %v", rs.Name, err)
	}
}

func waitForPodsWithPodGroup(ctx context.Context, t *testing.T, c clientset.Interface, ns, podGroupName string, want int) {
	t.Helper()
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		pods, err := c.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{LabelSelector: labels.Set(testLabels()).String()})
		if err != nil {
			return false, err
		}
		if len(pods.Items) < want {
			return false, nil
		}
		for i := range pods.Items {
			sg := pods.Items[i].Spec.SchedulingGroup
			if sg == nil || sg.PodGroupName == nil || *sg.PodGroupName != podGroupName {
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for %d pods wired to PodGroup %q: %v", want, podGroupName, err)
	}
}

func newGangDeployment(name, ns string, replicas int32, minCount *int32) *apps.Deployment {
	d := newDeployment(name, ns, replicas)
	d.Spec.Strategy = apps.DeploymentStrategy{Type: apps.RecreateDeploymentStrategyType}
	d.Spec.Scheduling = &apps.DeploymentSchedulingConfiguration{
		SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: minCount},
		},
	}
	return d
}

func enableWorkloadGates(t *testing.T, topology bool) {
	t.Helper()
	overrides := featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	}
	if topology {
		overrides[features.TopologyAwareWorkloadScheduling] = true
	}
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, overrides)
}

func TestDeploymentGangScheduling(t *testing.T) {
	enableWorkloadGates(t, true)

	cases := map[string]struct {
		replicas      int32
		minCount      *int32
		disruptionAll bool
		topologyKey   string
		wantMinCount  int32
	}{
		"basic gang":                     {replicas: 2, minCount: ptr.To[int32](2), wantMinCount: 2},
		"default minCount from replicas": {replicas: 2, wantMinCount: 2},
		"all-or-nothing disruption":      {replicas: 2, minCount: ptr.To[int32](2), disruptionAll: true, wantMinCount: 2},
		"topology constraint":            {replicas: 2, minCount: ptr.To[int32](2), topologyKey: "topology.kubernetes.io/zone", wantMinCount: 2},
		"disruption and topology":        {replicas: 2, minCount: ptr.To[int32](2), disruptionAll: true, topologyKey: "topology.kubernetes.io/zone", wantMinCount: 2},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			closeFn, rm, dc, informers, c := dcSetup(ctx, t)
			defer closeFn()

			ns := framework.CreateNamespaceOrDie(c, "dep-gang", t)
			defer framework.DeleteNamespaceOrDie(c, ns, t)

			d := newGangDeployment("gang", ns.Name, tc.replicas, tc.minCount)
			if tc.disruptionAll {
				d.Spec.Scheduling.DisruptionMode = &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
					All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
				}
			}
			if tc.topologyKey != "" {
				d.Spec.Scheduling.SchedulingConstraints = &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
					Topology: []schedulingv1alpha3.TopologyConstraint{{Key: tc.topologyKey}},
				}
			}

			tester := &deploymentTester{t: t, c: c, deployment: d}
			var err error
			tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(ctx, d, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create gang Deployment: %v", err)
			}

			stop := runControllersAndInformers(t, rm, dc, informers)
			defer stop()

			workload := waitForWorkload(ctx, t, c, tester.deployment, false, pollTimeout)
			if !hasOwnerRef(workload.OwnerReferences, "Deployment", tester.deployment.UID) {
				t.Errorf("Workload %q is not owned by the Deployment: %v", workload.Name, workload.OwnerReferences)
			}
			if len(workload.Spec.PodGroupTemplates) != 1 {
				t.Fatalf("expected 1 PodGroupTemplate, got %d", len(workload.Spec.PodGroupTemplates))
			}
			pgt := workload.Spec.PodGroupTemplates[0]
			if pgt.SchedulingPolicy.Gang == nil || pgt.SchedulingPolicy.Gang.MinCount != tc.wantMinCount {
				t.Errorf("Workload gang = %#v, want minCount %d", pgt.SchedulingPolicy.Gang, tc.wantMinCount)
			}

			rs := waitForNewReplicaSet(ctx, t, tester)
			podGroup := waitForPodGroup(ctx, t, c, rs, false, pollTimeout)
			if !hasOwnerRef(podGroup.OwnerReferences, "ReplicaSet", rs.UID) {
				t.Errorf("PodGroup %q is not owned by the ReplicaSet: %v", podGroup.Name, podGroup.OwnerReferences)
			}
			if !hasOwnerRef(podGroup.OwnerReferences, "Workload", workload.UID) {
				t.Errorf("PodGroup %q is not owned by the Workload: %v", podGroup.Name, podGroup.OwnerReferences)
			}
			if podGroup.Spec.SchedulingPolicy.Gang == nil || podGroup.Spec.SchedulingPolicy.Gang.MinCount != tc.wantMinCount {
				t.Errorf("PodGroup gang = %#v, want minCount %d", podGroup.Spec.SchedulingPolicy.Gang, tc.wantMinCount)
			}

			if tc.disruptionAll {
				if pgt.DisruptionMode == nil || pgt.DisruptionMode.All == nil {
					t.Errorf("Workload disruptionMode = %#v, want All", pgt.DisruptionMode)
				}
				if podGroup.Spec.DisruptionMode == nil || podGroup.Spec.DisruptionMode.All == nil {
					t.Errorf("PodGroup disruptionMode = %#v, want All", podGroup.Spec.DisruptionMode)
				}
			}
			if tc.topologyKey != "" {
				if pgt.SchedulingConstraints == nil || len(pgt.SchedulingConstraints.Topology) != 1 || pgt.SchedulingConstraints.Topology[0].Key != tc.topologyKey {
					t.Errorf("Workload topology = %#v, want key %q", pgt.SchedulingConstraints, tc.topologyKey)
				}
				if podGroup.Spec.SchedulingConstraints == nil || len(podGroup.Spec.SchedulingConstraints.Topology) != 1 || podGroup.Spec.SchedulingConstraints.Topology[0].Key != tc.topologyKey {
					t.Errorf("PodGroup topology = %#v, want key %q", podGroup.Spec.SchedulingConstraints, tc.topologyKey)
				}
			}

			waitForPodsWithPodGroup(ctx, t, c, ns.Name, podGroup.Name, int(tc.replicas))
		})
	}
}

// TestDeploymentSchedulingExplicitMinCountUpdate verifies that changes to an
// explicit gang minCount propagate to the Workload template and PodGroup.
func TestDeploymentSchedulingExplicitMinCountUpdate(t *testing.T) {
	enableWorkloadGates(t, false)

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	closeFn, rm, dc, informers, c := dcSetup(ctx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(c, "dep-explicit-mincount", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	tester := &deploymentTester{t: t, c: c, deployment: newGangDeployment("gang", ns.Name, 8, ptr.To[int32](4))}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(ctx, tester.deployment, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create gang Deployment: %v", err)
	}

	stop := runControllersAndInformers(t, rm, dc, informers)
	defer stop()

	workload := waitForWorkload(ctx, t, c, tester.deployment, false, pollTimeout)
	rs := waitForNewReplicaSet(ctx, t, tester)
	podGroup := waitForPodGroup(ctx, t, c, rs, false, pollTimeout)
	waitForGangMinCount(ctx, t, c, ns.Name, workload.Name, podGroup.Name, 4)

	if _, err := tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Scheduling.SchedulingPolicy.Gang.MinCount = ptr.To[int32](6)
	}); err != nil {
		t.Fatalf("failed to update minCount: %v", err)
	}
	waitForGangMinCount(ctx, t, c, ns.Name, workload.Name, podGroup.Name, 6)
}

// TestDeploymentSchedulingDerivedMinCountUpdate verifies that, without an
// explicit gang minCount, the Workload template and PodGroup minCount track replicas.
func TestDeploymentSchedulingDerivedMinCountUpdate(t *testing.T) {
	enableWorkloadGates(t, false)

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	closeFn, rm, dc, informers, c := dcSetup(ctx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(c, "dep-derived-mincount", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	tester := &deploymentTester{t: t, c: c, deployment: newGangDeployment("gang", ns.Name, 2, nil)}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(ctx, tester.deployment, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create gang Deployment: %v", err)
	}

	stop := runControllersAndInformers(t, rm, dc, informers)
	defer stop()

	workload := waitForWorkload(ctx, t, c, tester.deployment, false, pollTimeout)
	rs := waitForNewReplicaSet(ctx, t, tester)
	podGroup := waitForPodGroup(ctx, t, c, rs, false, pollTimeout)
	waitForGangMinCount(ctx, t, c, ns.Name, workload.Name, podGroup.Name, 2)

	if _, err := tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Replicas = ptr.To[int32](4)
	}); err != nil {
		t.Fatalf("failed to scale up: %v", err)
	}
	waitForGangMinCount(ctx, t, c, ns.Name, workload.Name, podGroup.Name, 4)
}

// TestDeploymentSchedulingScaleToZero verifies that scaling to zero removes Pods
// and requests PodGroup deletion while retaining the Workload with minimum 1.
func TestDeploymentSchedulingScaleToZero(t *testing.T) {
	enableWorkloadGates(t, false)

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	closeFn, rm, dc, informers, c := dcSetup(ctx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(c, "dep-zero", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	tester := &deploymentTester{t: t, c: c, deployment: newGangDeployment("gang", ns.Name, 2, nil)}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(ctx, tester.deployment, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create gang Deployment: %v", err)
	}

	stop := runControllersAndInformers(t, rm, dc, informers)
	defer stop()

	workload := waitForWorkload(ctx, t, c, tester.deployment, false, pollTimeout)
	rs := waitForNewReplicaSet(ctx, t, tester)
	podGroup := waitForPodGroup(ctx, t, c, rs, false, pollTimeout)
	waitForGangMinCount(ctx, t, c, ns.Name, workload.Name, podGroup.Name, 2)
	waitForPodsWithPodGroup(ctx, t, c, ns.Name, podGroup.Name, 2)

	if _, err := tester.updateDeployment(func(update *apps.Deployment) {
		update.Spec.Replicas = new(int32(0))
	}); err != nil {
		t.Fatalf("failed to scale to zero: %v", err)
	}
	waitForPodGroupDeletionRequested(ctx, t, c, rs)
	err = wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		wl, err := c.SchedulingV1beta1().Workloads(ns.Name).Get(ctx, workload.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return gangMinCount(wl) == 1, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for Workload %q minCount 1: %v", workload.Name, err)
	}

	var scaledRS *apps.ReplicaSet
	err = wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		var err error
		scaledRS, err = c.AppsV1().ReplicaSets(ns.Name).Get(ctx, rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		pods, err := c.CoreV1().Pods(ns.Name).List(ctx, metav1.ListOptions{LabelSelector: labels.Set(testLabels()).String()})
		if err != nil {
			return false, err
		}
		return scaledRS.Spec.Replicas != nil && *scaledRS.Spec.Replicas == 0 && scaledRS.Status.Replicas == 0 && len(pods.Items) == 0, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for ReplicaSet %q to scale to zero with no Pods: %v", rs.Name, err)
	}
	if scaledRS.UID != rs.UID {
		t.Fatalf("ReplicaSet UID changed after scaling to zero: got %q, want %q", scaledRS.UID, rs.UID)
	}
	retainedWorkload, err := c.SchedulingV1beta1().Workloads(ns.Name).Get(ctx, workload.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get retained Workload %q: %v", workload.Name, err)
	}
	if retainedWorkload.UID != workload.UID {
		t.Fatalf("Workload UID changed after scaling to zero: got %q, want %q", retainedWorkload.UID, workload.UID)
	}
}

// TestDeploymentReplicaSetPodGroupCleanupOnDelete verifies that deleting a
// scheduling-enabled ReplicaSet requests PodGroup deletion and removes its
// cleanup finalizer, so the ReplicaSet is not stranded.
func TestDeploymentReplicaSetPodGroupCleanupOnDelete(t *testing.T) {
	enableWorkloadGates(t, false)

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	closeFn, rm, dc, informers, c := dcSetup(ctx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(c, "dep-rs-cleanup", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	tester := &deploymentTester{t: t, c: c, deployment: newGangDeployment("gang", ns.Name, 2, ptr.To[int32](2))}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(ctx, tester.deployment, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create gang Deployment: %v", err)
	}

	stop := runControllersAndInformers(t, rm, dc, informers)
	defer stop()

	waitForWorkload(ctx, t, c, tester.deployment, false, pollTimeout)
	rs := waitForNewReplicaSet(ctx, t, tester)
	waitForPodGroup(ctx, t, c, rs, false, pollTimeout)

	err = wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		got, err := c.AppsV1().ReplicaSets(ns.Name).Get(ctx, rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return slices.Contains(got.Finalizers, replicaSetPodGroupFinalizer), nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for the cleanup finalizer on ReplicaSet %q: %v", rs.Name, err)
	}

	if err := c.AppsV1().ReplicaSets(ns.Name).Delete(ctx, rs.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("failed to delete ReplicaSet: %v", err)
	}

	waitForPodGroupDeletionRequested(ctx, t, c, rs)

	err = wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		got, err := c.AppsV1().ReplicaSets(ns.Name).Get(ctx, rs.Name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		if err != nil {
			return false, err
		}
		return got.UID != rs.UID, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for ReplicaSet %q (UID %q) to be deleted: %v", rs.Name, rs.UID, err)
	}
}

// TestDeploymentSchedulingControllerRestart verifies that restarting controllers
// preserves the existing Workload and PodGroup.
func TestDeploymentSchedulingControllerRestart(t *testing.T) {
	enableWorkloadGates(t, false)

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	teardown, config, c := startDeploymentAPIServer(t)
	defer teardown()

	ns := framework.CreateNamespaceOrDie(c, "dep-restart", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	tester := &deploymentTester{t: t, c: c, deployment: newGangDeployment("gang", ns.Name, 2, ptr.To[int32](2))}
	var err error
	tester.deployment, err = c.AppsV1().Deployments(ns.Name).Create(ctx, tester.deployment, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create gang Deployment: %v", err)
	}

	rm, dc, informers := newDeploymentControllers(ctx, t, config)
	stop := runControllersAndInformers(t, rm, dc, informers)
	defer stop()
	informers.WaitForCacheSync(ctx.Done())

	workload := waitForWorkload(ctx, t, c, tester.deployment, false, pollTimeout)
	rs := waitForNewReplicaSet(ctx, t, tester)
	podGroup := waitForPodGroup(ctx, t, c, rs, false, pollTimeout)

	stop()
	rm, dc, informers = newDeploymentControllers(ctx, t, config)
	stop = runControllersAndInformers(t, rm, dc, informers)
	defer stop()
	informers.WaitForCacheSync(ctx.Done())

	if _, err := tester.updateDeployment(func(update *apps.Deployment) {
		if update.Annotations == nil {
			update.Annotations = map[string]string{}
		}
		update.Annotations["test.k8s.io/restart"] = "1"
	}); err != nil {
		t.Fatalf("failed to force reconcile after restart: %v", err)
	}

	if got := waitForWorkload(ctx, t, c, tester.deployment, false, pollTimeout); got.UID != workload.UID {
		t.Errorf("Workload UID changed after restart: got %s, want %s", got.UID, workload.UID)
	}
	if got := waitForPodGroup(ctx, t, c, rs, false, pollTimeout); got.UID != podGroup.UID {
		t.Errorf("PodGroup UID changed after restart: got %s, want %s", got.UID, podGroup.UID)
	}

	workloads, err := c.SchedulingV1beta1().Workloads(ns.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("failed to list Workloads: %v", err)
	}
	count := 0
	for i := range workloads.Items {
		if hasOwnerRef(workloads.Items[i].OwnerReferences, "Deployment", tester.deployment.UID) {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected exactly 1 Workload owned by Deployment, got %d", count)
	}
}
