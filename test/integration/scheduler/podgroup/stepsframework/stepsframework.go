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
	Mock   *MockPostFilterPlugin
}

/*
	 Steps are allowing us to create a test in a more readable way.

		We can create test as a fllow of steps, each step is an operation that will be performed on the cluster.
		Every Step should have a Name, that is used to identify the step and one operation.
		Step framework will perform only first operation it will enounter in a given step.

		For example, this will only create workload but will not wait for it to be ready:
		Step {
			Name: "Create and wait for workload to be ready",
			CreateWorkloads: []*schedulingapi.Workload{
				st.MakeWorkload().Name("workload").
					PodGroupTemplate(st.MakePodGroupTemplate().Name("t1").MinCount(3).Obj()).
					Obj(),
			},
			WaitForWorkloadReady: "workload",
		}

		When we have all steps defined, we can run the test by iterating over the steps using the RunSteps function.

		For example:

		steps := []Step{
			{
				Name: "Create node",
				CreateNodes: []*v1.Node{
					st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
				},
			},
			{
				Name: "Create a pod group"
				CreatePodGroup: st.MakePodGroup().Name("pg1").TemplateRef("t1", "workload").
					Priority(100).MinCount(3).Obj(),
			},
			{
				Name:                   "Verify pod group is created",
				WaitForPodGroupCreated: "pg1",
			},
		}
		runSteps(t, testCtx, steps)
*/
type Step struct {
	// Name of the step, used to identify the step. Should be in every step.
	// Used to describe the step in the test output.
	Name string
	// CreateNodes is use to create nodes in the cluster, it takes []*v1.Node as input.
	CreateNodes []*v1.Node
	// CreatePodGroup is use to create a pod group, it takes *schedulingapi.PodGroup as input.
	CreatePodGroup *schedulingapi.PodGroup
	// CreatePodGroupForbiddenError is use to create a pod group that should be rejected by the cluster, it takes *schedulingapi.PodGroup as input.
	CreatePodGroupForbiddenError *schedulingapi.PodGroup
	// CreatePods is use to create pods in the cluster, it takes []*v1.Pod as input.
	CreatePods []*v1.Pod
	// CreateWorkloads is use to create workloads in the cluster, it takes []*schedulingapi.Workload as input.
	CreateWorkloads []*schedulingapi.Workload
	// DeletePods is use to delete pods from the cluster, it takes []string as input.
	DeletePods []string
	// DeleteWorkloads is use to delete workloads from the cluster, it takes []*schedulingapi.Workload as input.
	DeleteWorkloads []*schedulingapi.Workload
	// WaitForPodsGatedOnPreEnqueue is use to wait for pods to be gated on pre-enqueue, it takes []string as input.
	WaitForPodsGatedOnPreEnqueue []string
	// WaitForPodsUnschedulable is use to wait for pods to be unschedulable, it takes []string as input.
	WaitForPodsUnschedulable []string
	// WaitForPodsScheduled is use to wait for pods to be scheduled, it takes []string as input.
	WaitForPodsScheduled []string
	// WaitForPodsRemoved is use to wait for pods to be removed, it takes []string as input.
	WaitForPodsRemoved []string
	// WaitForPodGroupCreated is use to wait for a pod group to be created, it takes string as input.
	WaitForPodGroupCreated string
	// WaitForAnyPodsScheduled is use to wait for any pod in the pod group to be scheduled, it takes *WaitForAnyPodsScheduled as input.
	WaitForAnyPodsScheduled *WaitForAnyPodsScheduled
	// WaitForPodGroupCondition is use to wait for a pod group to have a certain condition, it takes *PodGroupConditionCheck as input.
	WaitForPodGroupCondition *PodGroupConditionCheck
	// VerifyAssignments is use to verify that the pods are assigned to the correct nodes, it takes *VerifyAssignments as input.
	VerifyAssignments *VerifyAssignments
	// VerifyAssignedInOneDomain is use to verify that the pods are assigned to nodes in the same domain, it takes *VerifyAssignedInOneDomain as input.
	VerifyAssignedInOneDomain *VerifyAssignedInOneDomain
	// VerifyWorkloadAwarePreemption is use to verify that the workloads are preempted according to the workload aware preemption policy, it takes []*v1.Pod as input.
	VerifyWorkloadAwarePreemption []*v1.Pod
	// VerifyMockPostFilterPluginCalled is use to verify that the mock post filter plugin is called, it takes *VerifyMockPostFilterPluginCalled as input.
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

func createPods(testCtx *testutils.TestContext, ns string, pods []*v1.Pod) error {
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

func createPodGroup(testCtx *testutils.TestContext, ns string, pg *schedulingapi.PodGroup) error {
	cs := testCtx.ClientSet
	pgCopy := pg.DeepCopy()
	pgCopy.Namespace = ns
	if _, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{}); err != nil {
		return fmt.Errorf("failed to create pod group %s: %w", pgCopy.Name, err)
	}
	return nil
}

func createPodGroupForbiddenError(testCtx *testutils.TestContext, ns string, pg *schedulingapi.PodGroup) error {
	cs := testCtx.ClientSet
	pgCopy := pg.DeepCopy()
	pgCopy.Namespace = ns
	_, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{})

	if err == nil {
		return fmt.Errorf("expected PodGroup creation to be rejected, but it succeeded")
	}
	if !apierrors.IsForbidden(err) {
		return fmt.Errorf("expected Forbidden error, got: %w", err)
	}
	return nil
}

func createWorkloads(testCtx *testutils.TestContext, ns string, wls []*schedulingapi.Workload) error {
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

func deletePods(testCtx *testutils.TestContext, ns string, podNames []string) error {
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

func deleteWorkloads(testCtx *testutils.TestContext, ns string, wls []*schedulingapi.Workload) error {
	cs := testCtx.ClientSet
	for _, wl := range wls {
		if err := cs.SchedulingV1alpha2().Workloads(ns).Delete(testCtx.Ctx, wl.Name, metav1.DeleteOptions{}); err != nil {
			return fmt.Errorf("failed to delete workload %s: %w", wl.Name, err)
		}
	}
	return nil
}

func waitForPodsGatedOnPreEnqueue(testCtx *testutils.TestContext, ns string, podNames []string) error {
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

func waitForPodsUnschedulable(testCtx *testutils.TestContext, ns string, podNames []string) error {
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

func waitForPodsScheduled(testCtx *testutils.TestContext, ns string, podNames []string) error {
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

func waitForPodsRemoved(testCtx *testutils.TestContext, ns string, podNames []string) error {
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

func waitForAnyPodsScheduled(testCtx *testutils.TestContext, ns string, waitAny *WaitForAnyPodsScheduled) error {
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

func waitForPodGroupCreated(testCtx *testutils.TestContext, ns string, pgName string) error {
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

func waitForPodGroupCondition(testCtx *testutils.TestContext, ns string, check *PodGroupConditionCheck) error {
	cs := testCtx.ClientSet
	err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
		podGroupHasScheduledCondition(cs, ns, check.PodGroupName, check.ConditionStatus, check.Reason))
	if err != nil {
		return fmt.Errorf("failed to wait for PodGroup %s condition (status=%s, reason=%s): %w",
			check.PodGroupName, check.ConditionStatus, check.Reason, err)
	}
	return nil
}

func verifyAssignments(testCtx *testutils.TestContext, ns string, verify *VerifyAssignments) error {
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

func verifyAssignedInOneDomain(testCtx *testutils.TestContext, ns string, verify *VerifyAssignedInOneDomain) error {
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

func verifyWorkloadAwarePreemption(testCtx *testutils.TestContext, ns string, pods []*v1.Pod) error {
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

func verifyMockPostFilterPluginCalled(testCtx *testutils.TestContext, ns string, verify *VerifyMockPostFilterPluginCalled) error {
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

// RunSteps executes steps in the given order. It executes only first encountered operation in step.
// If there is no operation in the step, it will skip the step.
// If there is an error in any step, it will stop and return the error.
func RunSteps(testCtx *testutils.TestContext, ns string, steps []Step) error {
	for i, step := range steps {
		if step.Name == "" {
			return fmt.Errorf("step name cannot be empty")
		}
		var err error
		switch {
		case step.CreateNodes != nil:
			err = createNodes(testCtx, step.CreateNodes)
		case step.CreatePods != nil:
			err = createPods(testCtx, ns, step.CreatePods)
		case step.CreatePodGroup != nil:
			err = createPodGroup(testCtx, ns, step.CreatePodGroup)
		case step.CreatePodGroupForbiddenError != nil:
			err = createPodGroupForbiddenError(testCtx, ns, step.CreatePodGroupForbiddenError)
		case step.CreateWorkloads != nil:
			err = createWorkloads(testCtx, ns, step.CreateWorkloads)
		case step.DeletePods != nil:
			err = deletePods(testCtx, ns, step.DeletePods)
		case step.DeleteWorkloads != nil:
			err = deleteWorkloads(testCtx, ns, step.DeleteWorkloads)
		case step.WaitForPodsGatedOnPreEnqueue != nil:
			err = waitForPodsGatedOnPreEnqueue(testCtx, ns, step.WaitForPodsGatedOnPreEnqueue)
		case step.WaitForPodsUnschedulable != nil:
			err = waitForPodsUnschedulable(testCtx, ns, step.WaitForPodsUnschedulable)
		case step.WaitForPodsScheduled != nil:
			err = waitForPodsScheduled(testCtx, ns, step.WaitForPodsScheduled)
		case step.WaitForPodsRemoved != nil:
			err = waitForPodsRemoved(testCtx, ns, step.WaitForPodsRemoved)
		case step.WaitForAnyPodsScheduled != nil:
			err = waitForAnyPodsScheduled(testCtx, ns, step.WaitForAnyPodsScheduled)
		case step.WaitForPodGroupCreated != "":
			err = waitForPodGroupCreated(testCtx, ns, step.WaitForPodGroupCreated)
		case step.WaitForPodGroupCondition != nil:
			err = waitForPodGroupCondition(testCtx, ns, step.WaitForPodGroupCondition)
		case step.VerifyAssignments != nil:
			err = verifyAssignments(testCtx, ns, step.VerifyAssignments)
		case step.VerifyAssignedInOneDomain != nil:
			err = verifyAssignedInOneDomain(testCtx, ns, step.VerifyAssignedInOneDomain)
		case step.VerifyWorkloadAwarePreemption != nil:
			err = verifyWorkloadAwarePreemption(testCtx, ns, step.VerifyWorkloadAwarePreemption)
		case step.VerifyMockPostFilterPluginCalled != nil:
			err = verifyMockPostFilterPluginCalled(testCtx, ns, step.VerifyMockPostFilterPluginCalled)
		}
		if err != nil {
			return fmt.Errorf("step %d (%s) failed: %w", i, step.Name, err)
		}
	}
	return nil
}
