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

package preemption

import (
	"context"
	"fmt"
	"sync"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/scheduler/preemption/asyncframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestAsyncPreemption_InterPreemptorCollision tests FM-301: Two competing preemptors
// of different priorities targeting the same node/victims during asynchronous deletion.
// It verifies that:
// 1. Higher-priority preemptor P2 correctly serializes and takes precedence over lower-priority P1.
// 2. Lower-priority preemptor P1's nominated node state is cleared when P2 targets the same node.
// 3. No double-eviction or queue state corruption occurs regardless of which preemptor completes eviction first.
func TestAsyncPreemption_InterPreemptorCollision(t *testing.T) {
	tests := []struct {
		name  string
		steps []asyncframework.Step
	}{
		{
			name: "Higher-priority P2 arrives during P1 async preemption, P2 finishes eviction first",
			steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled victim Pods occupying full node capacity",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create lower-priority preemptor P1",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(500).Obj(),
					},
				},
				{
					Name: "schedule preemptor P1 (triggers async preemption and nominates node)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p1",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check preemptor P1 is gated in queue",
					PodGatedInQueue: "preemptor-p1",
				},
				{
					Name:                 "check preemptor P1 is running preemption",
					PodRunningPreemption: new(2),
				},
				{
					Name: "verify preemptor P1 nominated node",
					VerifyNominatedNodeName: &asyncframework.VerifyNominatedNodeName{
						PodName:          "preemptor-p1",
						ExpectedNodeName: "node",
					},
				},
				{
					Name: "create higher-priority preemptor P2",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(1000).Obj(),
					},
				},
				{
					Name: "schedule higher-priority preemptor P2 (targets same node)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p2",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:                 "check preemptor P2 is running preemption",
					PodRunningPreemption: new(3),
				},
				{
					Name:            "check preemptor P2 is gated in queue",
					PodGatedInQueue: "preemptor-p2",
				},
				{
					Name: "verify lower-priority P1 nomination was cleared when P2 targeted the node",
					VerifyNominatedNodeName: &asyncframework.VerifyNominatedNodeName{
						PodName:          "preemptor-p1",
						ExpectedNodeName: "",
					},
				},
				{
					Name:               "complete preemption API calls for higher-priority P2 first",
					CompletePreemption: "preemptor-p2",
				},
				{
					Name: "schedule higher-priority P2 (expects success)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor-p2",
						ExpectSuccess: true,
					},
				},
				{
					Name:               "complete preemption API calls for lower-priority P1",
					CompletePreemption: "preemptor-p1",
				},
				{
					Name: "wait for lower-priority P1 to finish preemption goroutine",
					VerifyPodRunningPreemption: &asyncframework.VerifyPodRunningPreemption{
						PodIndex: 2,
						Expected: false,
					},
				},
				{
					Name:        "activate lower-priority P1",
					ActivatePod: "preemptor-p1",
				},
				{
					Name: "schedule lower-priority P1 (expects unschedulable since P2 occupies node)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p1",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:                    "verify lower-priority P1 remains in unschedulable queue",
					VerifyPodInUnschedulable: "preemptor-p1",
				},
			},
		},
		{
			name: "Higher-priority P2 arrives during P1 async preemption, P1 finishes eviction first",
			steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled victim Pods occupying full node capacity",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create lower-priority preemptor P1",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(500).Obj(),
					},
				},
				{
					Name: "schedule preemptor P1 (triggers async preemption and nominates node)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p1",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check preemptor P1 is gated in queue",
					PodGatedInQueue: "preemptor-p1",
				},
				{
					Name:                 "check preemptor P1 is running preemption",
					PodRunningPreemption: new(2),
				},
				{
					Name: "create higher-priority preemptor P2",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(1000).Obj(),
					},
				},
				{
					Name: "schedule higher-priority preemptor P2",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p2",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:                 "check preemptor P2 is running preemption",
					PodRunningPreemption: new(3),
				},
				{
					Name:            "check preemptor P2 is gated in queue",
					PodGatedInQueue: "preemptor-p2",
				},
				{
					Name:               "complete preemption API calls for lower-priority P1 first",
					CompletePreemption: "preemptor-p1",
				},
				{
					Name:               "complete preemption API calls for higher-priority P2",
					CompletePreemption: "preemptor-p2",
				},
				{
					Name: "schedule higher-priority P2 (higher priority in activeQ, expects success)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor-p2",
						ExpectSuccess: true,
					},
				},
				{
					Name: "schedule lower-priority P1 (expects unschedulable)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p1",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:                    "verify lower-priority P1 remains in unschedulable queue",
					VerifyPodInUnschedulable: "preemptor-p1",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preemptionDoneChannels := &sync.Map{}
			blockBindingChannel := make(chan struct{})
			defer close(blockBindingChannel)
			preemptionConfig := asyncframework.AsyncPreemptionTestConfig{
				PreemptionDoneChannels: preemptionDoneChannels,
				BlockBindingChannel:    blockBindingChannel,
			}
			testCtx, preemptionPlugin, cs := asyncframework.InitTestForAsyncPreemption(t, preemptionConfig)

			logger, _ := ktesting.NewTestContext(t)
			if testCtx.Scheduler.APIDispatcher != nil {
				testCtx.Scheduler.APIDispatcher.Run(logger)
				defer testCtx.Scheduler.APIDispatcher.Close()
			}
			testCtx.Scheduler.SchedulingQueue.Run(logger)
			defer testCtx.Scheduler.SchedulingQueue.Close()

			createdPods := []*v1.Pod{}
			defer func() {
				testutils.CleanupPods(testCtx.Ctx, cs, t, createdPods)
			}()

			config := asyncframework.AsyncPreemptionStepRunnerConfig{
				CreatedPods:            createdPods,
				ClientSet:              cs,
				PreemptionDoneChannels: preemptionDoneChannels,
				Logger:                 logger,
				PreemptionPlugin:       preemptionPlugin,
				BlockBindingChannel:    blockBindingChannel,
			}

			asyncframework.RunAsyncPreemptionSteps(testCtx, t, tt.steps, config)
		})
	}
}

// TestAsyncPreemption_PartialDeletionFailureRollback tests FM-302: When an intermediate
// victim deletion fails during multi-victim preemption, subsequent deletions are immediately
// aborted, the preemptor is cleanly activated and requeued, and unaffected victims remain alive.
func TestAsyncPreemption_PartialDeletionFailureRollback(t *testing.T) {
	tests := []struct {
		name           string
		preemptPodHook asyncframework.PreemptPodHookFn
		steps          []asyncframework.Step
	}{
		{
			name: "Intermediate victim deletion failure in 3-victim set halts remaining evictions and activates preemptor",
			preemptPodHook: func(ctx context.Context, c fwk.PreemptionCandidate, preemptor preemption.ExecutorPreemptor, victim *v1.Pod, pluginName string) (bool, error, bool) {
				// Inject failure on victim-1
				if victim.Name == "victim-1" {
					return false, apierrors.NewInternalError(fmt.Errorf("simulated 500 internal error on victim deletion")), true
				}
				// Allow default deletion for other victims
				return false, nil, false
			},
			steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create 3 scheduled victim Pods (victim-0, victim-1, victim-2) using 3 CPU total",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().Name("victim-0").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
					},
				},
				{
					Name: "create victim-1",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().Name("victim-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
					},
				},
				{
					Name: "create victim-2",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().Name("victim-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
					},
				},
				{
					Name: "create preemptor Pod requiring 4 CPU (must evict all 3 victims)",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-p").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(500).Obj(),
					},
				},
				{
					Name: "schedule preemptor Pod (triggers async preemption against victim-0, victim-1, victim-2)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p",
						ExpectInQueue:       true,
					},
				},
				{
					Name: "verify preemptor is no longer running preemption after deletion failure",
					VerifyPodRunningPreemption: &asyncframework.VerifyPodRunningPreemption{
						PodIndex: 3,
						Expected: false,
					},
				},
				{
					Name: "verify victim-2 (the last victim) was NOT deleted due to eviction abort",
					VerifyPodsNotDeleted: []int{2},
				},
				{
					Name: "schedule preemptor again to confirm clean requeueing and no scheduler panic",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p",
						ExpectInQueue:       true,
					},
				},
			},
		},
		{
			name: "Single victim deletion failure with admission forbidden error cleans up and activates preemptor",
			preemptPodHook: func(ctx context.Context, c fwk.PreemptionCandidate, preemptor preemption.ExecutorPreemptor, victim *v1.Pod, pluginName string) (bool, error, bool) {
				if victim.Name == "victim-forbidden" {
					return false, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, "victim-forbidden", fmt.Errorf("admission webhook rejected deletion")), true
				}
				return false, nil, false
			},
			steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled victim Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("victim-forbidden").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
					},
				},
				{
					Name: "create preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-p").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(500).Obj(),
					},
				},
				{
					Name: "schedule preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p",
						ExpectInQueue:       true,
					},
				},
				{
					Name: "verify preemptor is no longer running preemption",
					VerifyPodRunningPreemption: &asyncframework.VerifyPodRunningPreemption{
						PodIndex: 1,
						Expected: false,
					},
				},
				{
					Name: "verify victim-forbidden was NOT deleted",
					VerifyPodsNotDeleted: []int{0},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preemptionDoneChannels := &sync.Map{}
			blockBindingChannel := make(chan struct{})
			defer close(blockBindingChannel)
			preemptionConfig := asyncframework.AsyncPreemptionTestConfig{
				PreemptionDoneChannels: preemptionDoneChannels,
				BlockBindingChannel:    blockBindingChannel,
				PreemptPodHook:         tt.preemptPodHook,
			}
			testCtx, preemptionPlugin, cs := asyncframework.InitTestForAsyncPreemption(t, preemptionConfig)

			logger, _ := ktesting.NewTestContext(t)
			if testCtx.Scheduler.APIDispatcher != nil {
				testCtx.Scheduler.APIDispatcher.Run(logger)
				defer testCtx.Scheduler.APIDispatcher.Close()
			}
			testCtx.Scheduler.SchedulingQueue.Run(logger)
			defer testCtx.Scheduler.SchedulingQueue.Close()

			createdPods := []*v1.Pod{}
			defer func() {
				testutils.CleanupPods(testCtx.Ctx, cs, t, createdPods)
			}()

			config := asyncframework.AsyncPreemptionStepRunnerConfig{
				CreatedPods:            createdPods,
				ClientSet:              cs,
				PreemptionDoneChannels: preemptionDoneChannels,
				Logger:                 logger,
				PreemptionPlugin:       preemptionPlugin,
				BlockBindingChannel:    blockBindingChannel,
			}

			asyncframework.RunAsyncPreemptionSteps(testCtx, t, tt.steps, config)
		})
	}
}

// TestAsyncPreemption_PreemptorDeletionDuringExecution tests lifecycle cancellation:
// Deleting or mutating a preemptor Pod while its background async eviction goroutine
// is in-flight cleans up gracefully without goroutine leaks, nil pointer panics, or orphaned state.
func TestAsyncPreemption_PreemptorDeletionDuringExecution(t *testing.T) {
	tests := []struct {
		name  string
		steps []asyncframework.Step
	}{
		{
			name: "Preemptor Pod deleted from API server during active preemption",
			steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled victim Pods",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-p").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(500).Obj(),
					},
				},
				{
					Name: "schedule preemptor Pod (starts async preemption, blocked in background goroutine)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check preemptor is gated in queue",
					PodGatedInQueue: "preemptor-p",
				},
				{
					Name:                 "check preemptor is running preemption",
					PodRunningPreemption: new(2),
				},
				{
					Name:      "delete preemptor Pod from API server while async preemption is in progress",
					DeletePod: "preemptor-p",
				},
				{
					Name:               "complete preemption API calls (background goroutine resumes and cleans up)",
					CompletePreemption: "preemptor-p",
				},
				{
					Name: "verify preemptor is no longer running preemption",
					VerifyPodRunningPreemption: &asyncframework.VerifyPodRunningPreemption{
						PodIndex: 2,
						Expected: false,
					},
				},
				{
					Name: "create a new subsequent Pod to verify scheduler continues healthy operation",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("new-subsequent-pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(500).Obj(),
					},
				},
				{
					Name: "schedule new subsequent Pod (should succeed on newly freed node)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "new-subsequent-pod",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			name: "Preemptor Pod mutated with labels while async preemption is in progress",
			steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled victim Pods",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-p").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(500).Obj(),
					},
				},
				{
					Name: "schedule preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-p",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check preemptor is gated in queue",
					PodGatedInQueue: "preemptor-p",
				},
				{
					Name:                 "check preemptor is running preemption",
					PodRunningPreemption: new(2),
				},
				{
					Name: "mutate preemptor Pod with new labels while async eviction is running",
					MutatePod: func(testCtx *testutils.TestContext, t *testing.T, cs kubernetes.Interface, createdPods []*v1.Pod) {
						pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, "preemptor-p", metav1.GetOptions{})
						if err != nil {
							t.Fatalf("Failed to get preemptor-p: %v", err)
						}
						podCopy := pod.DeepCopy()
						if podCopy.Labels == nil {
							podCopy.Labels = make(map[string]string)
						}
						podCopy.Labels["mutated"] = "true"
						if _, err := cs.CoreV1().Pods(testCtx.NS.Name).Update(testCtx.Ctx, podCopy, metav1.UpdateOptions{}); err != nil {
							t.Fatalf("Failed to update preemptor-p: %v", err)
						}
					},
				},
				{
					Name:               "complete preemption API calls",
					CompletePreemption: "preemptor-p",
				},
				{
					Name: "schedule mutated preemptor Pod (should succeed on freed node)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor-p",
						ExpectSuccess: true,
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preemptionDoneChannels := &sync.Map{}
			blockBindingChannel := make(chan struct{})
			defer close(blockBindingChannel)
			preemptionConfig := asyncframework.AsyncPreemptionTestConfig{
				PreemptionDoneChannels: preemptionDoneChannels,
				BlockBindingChannel:    blockBindingChannel,
			}
			testCtx, preemptionPlugin, cs := asyncframework.InitTestForAsyncPreemption(t, preemptionConfig)

			logger, _ := ktesting.NewTestContext(t)
			if testCtx.Scheduler.APIDispatcher != nil {
				testCtx.Scheduler.APIDispatcher.Run(logger)
				defer testCtx.Scheduler.APIDispatcher.Close()
			}
			testCtx.Scheduler.SchedulingQueue.Run(logger)
			defer testCtx.Scheduler.SchedulingQueue.Close()

			createdPods := []*v1.Pod{}
			defer func() {
				testutils.CleanupPods(testCtx.Ctx, cs, t, createdPods)
			}()

			config := asyncframework.AsyncPreemptionStepRunnerConfig{
				CreatedPods:            createdPods,
				ClientSet:              cs,
				PreemptionDoneChannels: preemptionDoneChannels,
				Logger:                 logger,
				PreemptionPlugin:       preemptionPlugin,
				BlockBindingChannel:    blockBindingChannel,
			}

			asyncframework.RunAsyncPreemptionSteps(testCtx, t, tt.steps, config)
		})
	}
}
