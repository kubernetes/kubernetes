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

package podgroup

import (
	"context"
	"fmt"
	"strings"

	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	framework "k8s.io/kube-scheduler/framework"

	"k8s.io/kubernetes/pkg/scheduler/backend/queue"
	testutils "k8s.io/kubernetes/test/integration/util"
)

type WaitForAnyPodsScheduled struct {
	Pods             []*v1.Pod
	NumScheduled     int
	NumUnschedulable int
}

type PodGroupConditionCheck struct {
	PodGroupName    string
	ConditionStatus metav1.ConditionStatus
	Reason          string
}

type VerifyAssignments struct {
	Pods  []string
	Nodes sets.Set[string]
}

type VerifyAssignedInOneDomain struct {
	Pods        []string
	TopologyKey string
}

type VerifyMockPostFilterPluginCalled struct {
	Called int
	Mock *MockPostFilterPlugin
}

type Step struct {
	Name                         string
	CreateNodes                  []*v1.Node
	CreatePodGroup               *schedulingapi.PodGroup
	CreatePodGroupForbiddenError *schedulingapi.PodGroup
	CreatePods                   []*v1.Pod
	CreateWorkloads               []*schedulingapi.Workload
	DeletePods                   []string
	DeleteWorkloads              []*schedulingapi.Workload
	WaitForPodsGatedOnPreEnqueue []string
	WaitForPodsUnschedulable     []string
	WaitForPodsScheduled         []string
	WaitForPodsRemoved           []string
	WaitForPodGroupCreated       string
	WaitForAnyPodsScheduled      *WaitForAnyPodsScheduled
	WaitForPodGroupCondition     *PodGroupConditionCheck
	VerifyAssignments            *VerifyAssignments
	VerifyAssignedInOneDomain    *VerifyAssignedInOneDomain
	VerifyWorkloadAwarePreemption []*v1.Pod
	VerifyMockPostFilterPluginCalled *VerifyMockPostFilterPluginCalled
}

// MockPostFilterPlugin is a custom PostFilter plugin that just counts invocations.
type MockPostFilterPlugin struct {
	count int
}

func (m *MockPostFilterPlugin) Name() string {
	return "MockPostFilter"
}

func (m *MockPostFilterPlugin) PostFilter(ctx context.Context, state framework.CycleState, pod *v1.Pod, filteredNodeStatusMap framework.NodeToStatusReader) (*framework.PostFilterResult, *framework.Status) {
	m.count++
	return nil, framework.NewStatus(framework.Unschedulable)
}

func podInUnschedulablePods(queue queue.SchedulingQueue, podName string) bool {
	unschedPods := queue.UnschedulablePods()
	for _, pod := range unschedPods {
		if pod.Name == podName {
			return true
		}
	}
	return false
}

func podGroupHasScheduledCondition(cs kubernetes.Interface, ns, name string, status metav1.ConditionStatus, reason string) wait.ConditionWithContextFunc {
	return func(ctx context.Context) (bool, error) {
		pg, err := cs.SchedulingV1alpha2().PodGroups(ns).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		for _, c := range pg.Status.Conditions {
			if c.Type == schedulingapi.PodGroupScheduled &&
				c.Status == status && c.Reason == reason {
				return true, nil
			}
		}
		return false, nil
	}
}

func createNodes(testCtx *testutils.TestContext, nodes []*v1.Node) error {
	cs := testCtx.ClientSet
	for _, node := range nodes {
		n := node.DeepCopy()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, n, metav1.CreateOptions{}); err != nil {
			return fmt.Errorf("failed to create node %s: %w", n.Name, err)
		}
	}
	return nil
}

func createPods(testCtx *testutils.TestContext, pods []*v1.Pod, ns string) error {
	cs := testCtx.ClientSet
	for _, pod := range pods {
		p := pod.DeepCopy()
		p.Namespace = ns
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
			return fmt.Errorf("failed to create pod %s: %w", p.Name, err)
		}
	}
	return nil
}

func createPodGroup(testCtx *testutils.TestContext, pg *schedulingapi.PodGroup, ns string) error {
	cs := testCtx.ClientSet
	pgCopy := pg.DeepCopy()
	pgCopy.Namespace = ns
	if _, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{}); err != nil {
		return fmt.Errorf("failed to create pod group %s: %w", pgCopy.Name, err)
	}
	return nil
}

func createPodGroupForbiddenError(testCtx *testutils.TestContext, pg *schedulingapi.PodGroup, ns string) error {
	cs := testCtx.ClientSet
	pgCopy := pg.DeepCopy()
	pgCopy.Namespace = ns
	_, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{})

	if err == nil {
		return fmt.Errorf("Expected PodGroup creation to be rejected, but it succeeded")
	}
	if !apierrors.IsForbidden(err) {
		return fmt.Errorf("Expected Forbidden error, got: %v", err)
	}
	return nil
}

func createWorkloads(testCtx *testutils.TestContext, wls []*schedulingapi.Workload, ns string) error {
	cs := testCtx.ClientSet
	for _, wl := range wls {
		wlCopy := wl.DeepCopy()
		wlCopy.Namespace = ns
		if _, err := cs.SchedulingV1alpha2().Workloads(ns).Create(testCtx.Ctx, wlCopy, metav1.CreateOptions{}); err != nil {
			return fmt.Errorf("failed to create workload %s: %w", wlCopy.Name, err)
		}
	}
	return nil
}

func deletePods(testCtx *testutils.TestContext, podNames []string, ns string) error {
	cs := testCtx.ClientSet
	for _, podName := range podNames {
		if err := cs.CoreV1().Pods(ns).Delete(testCtx.Ctx, podName, metav1.DeleteOptions{}); err != nil {
			return fmt.Errorf("failed to delete pod %s: %w", podName, err)
		}
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
			func(_ context.Context) (bool, error) {
				_, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
				if err != nil {
					if apierrors.IsNotFound(err) {
						return true, nil
					}
					return false, err
				}
				return false, nil
			},
		)
		if err != nil {
			return fmt.Errorf("failed to wait for pod %s to be no longer visible in scheduler: %w", podName, err)
		}
	}
	return nil
}

func deleteWorkloads(testCtx *testutils.TestContext, wls []*schedulingapi.Workload, ns string) error {
	cs := testCtx.ClientSet
	for _, wl := range wls {
		if err := cs.SchedulingV1alpha2().Workloads(ns).Delete(testCtx.Ctx, wl.Name, metav1.DeleteOptions{}); err != nil {
			return fmt.Errorf("failed to delete workload %s: %w", wl.Name, err)
		}
	}
	return nil
}

func waitForPodsGatedOnPreEnqueue(testCtx *testutils.TestContext, podNames []string) error {
	for _, podName := range podNames {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
			func(_ context.Context) (bool, error) {
				return podInUnschedulablePods(testCtx.Scheduler.SchedulingQueue, podName), nil
			},
		)
		if err != nil {
			return fmt.Errorf("failed to wait for pod %s to be in unschedulable pods pool: %w", podName, err)
		}
	}
	return nil
}

func waitForPodsUnschedulable(testCtx *testutils.TestContext, podNames []string, ns string) error {
	cs := testCtx.ClientSet
	for _, podName := range podNames {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
			testutils.PodUnschedulable(cs, ns, podName))
		if err != nil {
			return fmt.Errorf("failed to wait for pod %s to be unschedulable: %w", podName, err)
		}
	}
	return nil
}

func waitForPodsScheduled(testCtx *testutils.TestContext, podNames []string, ns string) error {
	cs := testCtx.ClientSet
	for _, podName := range podNames {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
			testutils.PodScheduled(cs, ns, podName))
		if err != nil {
			return fmt.Errorf("failed to wait for pod %s to be scheduled: %w", podName, err)
		}
	}
	return nil
}

func waitForPodsRemoved(testCtx *testutils.TestContext, podNames []string, ns string) error {
	cs := testCtx.ClientSet
	for _, podName := range podNames {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
			testutils.PodDeleted(testCtx.Ctx, cs, ns, podName))
		if err != nil {
			return fmt.Errorf("failed to wait for pod %s to be removed: %w", podName, err)
		}
	}
	return nil
}

func waitForAnyPodsScheduled(testCtx *testutils.TestContext, waitAny *WaitForAnyPodsScheduled, ns string) error {
	cs := testCtx.ClientSet
	err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
		func(ctx context.Context) (bool, error) {
			scheduledCount := 0
			unschedulableCount := 0
			for _, pod := range waitAny.Pods {
				if ok, err := testutils.PodScheduled(cs, ns, pod.Name)(ctx); err != nil {
					return false, err
				} else if ok {
					scheduledCount++
					continue
				}
				if ok, err := testutils.PodUnschedulable(cs, ns, pod.Name)(ctx); err != nil {
					return false, err
				} else if ok {
					unschedulableCount++
				}
			}
			return scheduledCount == waitAny.NumScheduled && unschedulableCount == waitAny.NumUnschedulable, nil
		},
	)
	if err != nil {
		return fmt.Errorf("failed to wait for pods to be scheduled or unschedulable: %w", err)
	}
	return nil
}

func waitForPodGroupCreated(testCtx *testutils.TestContext, pgName string, ns string) error {
	err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
		func(_ context.Context) (bool, error) {
			_, err := testCtx.InformerFactory.Scheduling().V1alpha2().PodGroups().Lister().PodGroups(ns).Get(pgName)
			if err != nil {
				if apierrors.IsNotFound(err) {
					return false, nil
				}
				return false, err
			}
			return true, nil
		},
	)
	if err != nil {
		return fmt.Errorf("failed to wait for pod group %s to be discoverable by scheduler: %w", pgName, err)
	}
	return nil
}

func waitForPodGroupCondition(testCtx *testutils.TestContext, check *PodGroupConditionCheck, ns string) error {
	cs := testCtx.ClientSet
	err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
		podGroupHasScheduledCondition(cs, ns, check.PodGroupName, check.ConditionStatus, check.Reason))
	if err != nil {
		return fmt.Errorf("failed to wait for PodGroup %s condition (status=%s, reason=%s): %w",
			check.PodGroupName, check.ConditionStatus, check.Reason, err)
	}
	return nil
}

func verifyAssignments(testCtx *testutils.TestContext, verify *VerifyAssignments, ns string) error {
	cs := testCtx.ClientSet
	for _, podName := range verify.Pods {
		assignedPod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to retrieve assigned pod %s: %w", podName, err)
		}
		nodeName := assignedPod.Spec.NodeName
		if nodeName == "" {
			return fmt.Errorf("pod %s is not assigned", podName)
		}
		if !verify.Nodes.Has(nodeName) {
			return fmt.Errorf("wanted pod %s scheduled on node within %v but got assignment to %s", podName, verify.Nodes, nodeName)
		}
	}
	return nil
}

func verifyAssignedInOneDomain(testCtx *testutils.TestContext, verify *VerifyAssignedInOneDomain, ns string) error {
	cs := testCtx.ClientSet
	expectedDomain := ""
	for _, podName := range verify.Pods {
		assignedPod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to retrieve assigned pod %s: %w", podName, err)
		}
		nodeName := assignedPod.Spec.NodeName
		if nodeName == "" {
			return fmt.Errorf("pod %s is not assigned", podName)
		}
		node, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to retrieve node %s: %w", nodeName, err)
		}
		domain := node.Labels[verify.TopologyKey]
		if domain == "" {
			return fmt.Errorf("invalid domain value \"\" in node %s", nodeName)
		}
		if expectedDomain == "" {
			expectedDomain = domain
		} else if expectedDomain != domain {
			return fmt.Errorf("pod %s assigned to a different domain. Expected %s but got %s", podName, expectedDomain, domain)
		}
	}
	return nil
}

func verifyWorkloadAwarePreemption(testCtx *testutils.TestContext, pods []*v1.Pod, ns string) error {
	cs := testCtx.ClientSet
	err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		for _, pod := range pods {
			events, err := cs.CoreV1().Events(ns).List(ctx, metav1.ListOptions{
				FieldSelector: "involvedObject.name=" + pod.Name,
			})
			if err != nil {
				return false, err
			}
			for _, event := range events.Items {
				if event.Reason == "Preempted" && strings.HasPrefix(event.Message, "Preempted by podgroup") {
					return true, nil
				}
			}
		}
		return false, nil
	})
	if err != nil {
		return fmt.Errorf("WorkloadAwarePreemption was not called within timeout")
	}
	return nil
}

func verifyMockPostFilterPluginCalled(testCtx *testutils.TestContext, verify *VerifyMockPostFilterPluginCalled, ns string) error {
	err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		if verify.Mock.count == verify.Called {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return fmt.Errorf("MockPostFilter was called %d times, expected exactly %d", verify.Mock.count, verify.Called)
	}
	return nil
}

func RunSteps(testCtx *testutils.TestContext, steps []Step, ns string) error {
	for i, step := range steps {
		var err error
		switch {
		case step.CreateNodes != nil:
			err = createNodes(testCtx, step.CreateNodes)
		case step.CreatePods != nil:
			err = createPods(testCtx, step.CreatePods, ns)
		case step.CreatePodGroup != nil:
			err = createPodGroup(testCtx, step.CreatePodGroup, ns)
		case step.CreatePodGroupForbiddenError != nil:
			err = createPodGroupForbiddenError(testCtx, step.CreatePodGroupForbiddenError, ns)
		case step.CreateWorkloads != nil:
			err = createWorkloads(testCtx, step.CreateWorkloads, ns)
		case step.DeletePods != nil:
			err = deletePods(testCtx, step.DeletePods, ns)
		case step.DeleteWorkloads != nil:
			err = deleteWorkloads(testCtx, step.DeleteWorkloads, ns)
		case step.WaitForPodsGatedOnPreEnqueue != nil:
			err = waitForPodsGatedOnPreEnqueue(testCtx, step.WaitForPodsGatedOnPreEnqueue)
		case step.WaitForPodsUnschedulable != nil:
			err = waitForPodsUnschedulable(testCtx, step.WaitForPodsUnschedulable, ns)
		case step.WaitForPodsScheduled != nil:
			err = waitForPodsScheduled(testCtx, step.WaitForPodsScheduled, ns)
		case step.WaitForPodsRemoved != nil:
			err = waitForPodsRemoved(testCtx, step.WaitForPodsRemoved, ns)
		case step.WaitForAnyPodsScheduled != nil:
			err = waitForAnyPodsScheduled(testCtx, step.WaitForAnyPodsScheduled, ns)
		case step.WaitForPodGroupCreated != "":
			err = waitForPodGroupCreated(testCtx, step.WaitForPodGroupCreated, ns)
		case step.WaitForPodGroupCondition != nil:
			err = waitForPodGroupCondition(testCtx, step.WaitForPodGroupCondition, ns)
		case step.VerifyAssignments != nil:
			err = verifyAssignments(testCtx, step.VerifyAssignments, ns)
		case step.VerifyAssignedInOneDomain != nil:
			err = verifyAssignedInOneDomain(testCtx, step.VerifyAssignedInOneDomain, ns)
		case step.VerifyWorkloadAwarePreemption != nil:
			err = verifyWorkloadAwarePreemption(testCtx, step.VerifyWorkloadAwarePreemption, ns)
		case step.VerifyMockPostFilterPluginCalled != nil:
			err = verifyMockPostFilterPluginCalled(testCtx, step.VerifyMockPostFilterPluginCalled, ns)
		}
		if err != nil {
			return fmt.Errorf("step %d (%s) failed: %w", i, step.Name, err)
		}
	}
	return nil
}
