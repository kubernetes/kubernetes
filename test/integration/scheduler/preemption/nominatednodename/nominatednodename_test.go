/*
Copyright 2017 The Kubernetes Authors.

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

package nominatednodename

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// imported from testutils
var (
	initPausePod   = testutils.InitPausePod
	createNode     = testutils.CreateNode
	createPausePod = testutils.CreatePausePod
	runPausePod    = testutils.RunPausePod
	deletePod      = testutils.DeletePod
	initTest       = testutils.InitTestSchedulerWithNS
)

var lowPriority, mediumPriority, highPriority = int32(100), int32(200), int32(300)

const (
	alwaysFailPlugin = "alwaysFailPlugin"
	doNotFailMe      = "do-not-fail-me"
)

// A fake plugin implements PreBind extension point.
// It always fails with an Unschedulable status, unless the pod contains a `doNotFailMe` string.
type alwaysFail struct{}

func (af *alwaysFail) Name() string {
	return alwaysFailPlugin
}

func (af *alwaysFail) PreBindPreFlight(_ context.Context, _ fwk.CycleState, p *v1.Pod, _ string) *fwk.Status {
	return nil
}

func (af *alwaysFail) PreBind(_ context.Context, _ fwk.CycleState, p *v1.Pod, _ string) *fwk.Status {
	if strings.Contains(p.Name, doNotFailMe) {
		return nil
	}
	return fwk.NewStatus(fwk.Unschedulable)
}

func newAlwaysFail(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	return &alwaysFail{}, nil
}

// TestNominatedNode verifies if a pod's nominatedNodeName is set and unset
// properly in different scenarios.
func TestNominatedNode(t *testing.T) {
	tests := []struct {
		name string
		// Initial nodes to simulate special conditions.
		initNodes    []*v1.Node
		nodeCapacity map[v1.ResourceName]string
		// A slice of pods to be created in batch.
		podsToCreate [][]*v1.Pod
		// Each postCheck function is run after each batch of pods' creation.
		postChecks []func(ctx context.Context, cs clientset.Interface, pod *v1.Pod) error
		// Delete the fake node or not. Optional.
		deleteFakeNode bool
		// Pods to be deleted. Optional.
		podNamesToDelete []string
		// Whether NominatedNodeName will be always nil at the end of the test,
		// regardless of the NominatedNodeNameForExpectation feature gate state.
		expectNilNominatedNodeName bool

		// Register dummy plugin to simulate particular scheduling failures. Optional.
		customPlugins     *configv1.Plugins
		outOfTreeRegistry frameworkruntime.Registry
	}{
		{
			name:         "mid-priority pod preempts low-priority pod, followed by a high-priority pod with another preemption",
			nodeCapacity: map[v1.ResourceName]string{v1.ResourceCPU: "5"},
			podsToCreate: [][]*v1.Pod{
				{
					st.MakePod().Name("low-1").Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
					st.MakePod().Name("low-2").Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
					st.MakePod().Name("low-3").Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
					st.MakePod().Name("low-4").Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
				},
				{
					st.MakePod().Name("medium").Priority(mediumPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Obj(),
				},
				{
					st.MakePod().Name("high").Priority(highPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
				},
			},
			postChecks: []func(ctx context.Context, cs clientset.Interface, pod *v1.Pod) error{
				testutils.WaitForPodToSchedule,
				testutils.WaitForNominatedNodeName,
				testutils.WaitForNominatedNodeName,
			},
			podNamesToDelete:           []string{"low-1", "low-2", "low-3", "low-4"},
			expectNilNominatedNodeName: true,
		},
		{
			name:         "mid-priority pod preempts low-priority pod, followed by a high-priority pod without additional preemption",
			nodeCapacity: map[v1.ResourceName]string{v1.ResourceCPU: "2"},
			podsToCreate: [][]*v1.Pod{
				{
					st.MakePod().Name("low").Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
				},
				{
					st.MakePod().Name("medium").Priority(mediumPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
				},
				{
					st.MakePod().Name("high").Priority(highPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
				},
			},
			postChecks: []func(ctx context.Context, cs clientset.Interface, pod *v1.Pod) error{
				testutils.WaitForPodToSchedule,
				testutils.WaitForNominatedNodeName,
				testutils.WaitForPodToSchedule,
			},
			podNamesToDelete: []string{"low"},
		},
		{
			name:         "mid-priority pod preempts low-priority pod, followed by a node deletion",
			nodeCapacity: map[v1.ResourceName]string{v1.ResourceCPU: "1"},
			podsToCreate: [][]*v1.Pod{
				{
					st.MakePod().Name("low").Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
				},
				{
					st.MakePod().Name("medium").Priority(mediumPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
				},
			},
			postChecks: []func(ctx context.Context, cs clientset.Interface, pod *v1.Pod) error{
				testutils.WaitForPodToSchedule,
				testutils.WaitForNominatedNodeName,
			},
			// Delete the fake node to simulate an ErrNoNodesAvailable error.
			deleteFakeNode:   true,
			podNamesToDelete: []string{"low"},
		},
		{
			name: "mid-priority pod preempts low-priority pod at the beginning, but could not find candidates after the nominated node is deleted",
			// Create a taint node to simulate the `UnschedulableAndUnresolvable` condition in `findCandidates` during preemption.
			initNodes: []*v1.Node{
				st.MakeNode().Name("taint-node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).
					Taints([]v1.Taint{
						{
							Key:    "taint-node",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					}).
					Obj(),
			},
			nodeCapacity: map[v1.ResourceName]string{v1.ResourceCPU: "1"},
			podsToCreate: [][]*v1.Pod{
				{
					st.MakePod().Name("low").Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
				},
				{
					st.MakePod().Name("medium").Priority(mediumPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
				},
			},
			postChecks: []func(ctx context.Context, cs clientset.Interface, pod *v1.Pod) error{
				testutils.WaitForPodToSchedule,
				testutils.WaitForNominatedNodeName,
			},
			// Delete the fake node to trigger the `UnschedulableAndUnresolvable` condition in `findCandidates`.
			deleteFakeNode:   true,
			podNamesToDelete: []string{"low"},
		},
		{
			name:         "mid-priority pod preempts low-priority pod, but failed the scheduling unexpectedly",
			nodeCapacity: map[v1.ResourceName]string{v1.ResourceCPU: "1"},
			podsToCreate: [][]*v1.Pod{
				{
					st.MakePod().Name(fmt.Sprintf("low-%v", doNotFailMe)).Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
				},
				{
					st.MakePod().Name("medium").Priority(mediumPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
				},
			},
			postChecks: []func(ctx context.Context, cs clientset.Interface, pod *v1.Pod) error{
				testutils.WaitForPodToSchedule,
				testutils.WaitForNominatedNodeName,
			},
			podNamesToDelete: []string{fmt.Sprintf("low-%v", doNotFailMe)},
			customPlugins: &configv1.Plugins{
				PreBind: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: alwaysFailPlugin},
					},
				},
			},
			outOfTreeRegistry: frameworkruntime.Registry{alwaysFailPlugin: newAlwaysFail},
		},
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, asyncAPICallsEnabled := range []bool{true, false} {
			for _, nominatedNodeNameForExpectationEnabled := range []bool{false} {
				for _, tt := range tests {
					t.Run(fmt.Sprintf("%s (Async preemption: %v, Async API calls: %v, NNN for expectation: %v)", tt.name, asyncPreemptionEnabled, asyncAPICallsEnabled, nominatedNodeNameForExpectationEnabled), func(t *testing.T) {
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncPreemption, asyncPreemptionEnabled)
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncAPICalls, asyncAPICallsEnabled)
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NominatedNodeNameForExpectation, nominatedNodeNameForExpectationEnabled)

						cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
							Profiles: []configv1.KubeSchedulerProfile{{
								SchedulerName: ptr.To(v1.DefaultSchedulerName),
								Plugins:       tt.customPlugins,
							}},
						})
						testCtx := initTest(
							t,
							"preemption",
							scheduler.WithProfiles(cfg.Profiles...),
							scheduler.WithFrameworkOutOfTreeRegistry(tt.outOfTreeRegistry),
						)

						cs, ns := testCtx.ClientSet, testCtx.NS.Name
						for _, node := range tt.initNodes {
							if _, err := createNode(cs, node); err != nil {
								t.Fatalf("Error creating initial node %v: %v", node.Name, err)
							}
						}

						// Create a node with the specified capacity.
						nodeName := "fake-node"
						if _, err := createNode(cs, st.MakeNode().Name(nodeName).Capacity(tt.nodeCapacity).Obj()); err != nil {
							t.Fatalf("Error creating node %v: %v", nodeName, err)
						}

						// Create pods and run post check if necessary.
						for i, pods := range tt.podsToCreate {
							for _, p := range pods {
								p.Namespace = ns
								if _, err := createPausePod(cs, p); err != nil {
									t.Fatalf("Error creating pod %v: %v", p.Name, err)
								}
							}
							// If necessary, run the post check function.
							if len(tt.postChecks) > i && tt.postChecks[i] != nil {
								for _, p := range pods {
									if err := tt.postChecks[i](testCtx.Ctx, cs, p); err != nil {
										t.Fatalf("Pod %v didn't pass the postChecks[%v]: %v", p.Name, i, err)
									}
								}
							}
						}

						// Delete the fake node if necessary.
						if tt.deleteFakeNode {
							if err := cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, *metav1.NewDeleteOptions(0)); err != nil {
								t.Fatalf("Node %v cannot be deleted: %v", nodeName, err)
							}
						}

						// Force deleting the terminating pods if necessary.
						// This is required if we demand to delete terminating Pods physically.
						for _, podName := range tt.podNamesToDelete {
							if err := deletePod(cs, podName, ns); err != nil {
								t.Fatalf("Pod %v cannot be deleted: %v", podName, err)
							}
						}

						if nominatedNodeNameForExpectationEnabled && !tt.expectNilNominatedNodeName {
							// Verify if .status.nominatedNodeName is not cleared when NominatedNodeNameForExpectation is enabled.
							// Wait for 1 second to make sure the pod is re-processed, what would potentially clear the NominatedNodeName (when the feature won't work).
							select {
							case <-time.After(time.Second):
							case <-testCtx.Ctx.Done():
							}
							pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, "medium", metav1.GetOptions{})
							if err != nil {
								t.Errorf("Error getting the medium pod: %v", err)
							} else if len(pod.Status.NominatedNodeName) == 0 {
								t.Errorf(".status.nominatedNodeName of the medium pod was cleared: %v", err)
							}
						} else {
							// Verify if .status.nominatedNodeName is cleared when NominatedNodeNameForExpectation is disabled.
							if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
								pod, err := cs.CoreV1().Pods(ns).Get(ctx, "medium", metav1.GetOptions{})
								if err != nil {
									t.Errorf("Error getting the medium pod: %v", err)
								}
								if len(pod.Status.NominatedNodeName) == 0 {
									return true, nil
								}
								return false, err
							}); err != nil {
								t.Errorf(".status.nominatedNodeName of the medium pod was not cleared: %v", err)
							}
						}
					})
				}
			}
		}
	}
}

func initTestPreferNominatedNode(t *testing.T, nsPrefix string, opts ...scheduler.Option) *testutils.TestContext {
	testCtx := testutils.InitTestSchedulerWithOptions(t, testutils.InitTestAPIServer(t, nsPrefix, nil), 0, opts...)
	testutils.SyncSchedulerInformerFactory(testCtx)
	// wraps the NextPod() method to make it appear the preemption has been done already and the nominated node has been set.
	f := testCtx.Scheduler.NextPod
	testCtx.Scheduler.NextPod = func(logger klog.Logger) (*framework.QueuedPodInfo, error) {
		podInfo, _ := f(klog.FromContext(testCtx.Ctx))
		// Scheduler.Next() may return nil when scheduler is shutting down.
		if podInfo != nil {
			podInfo.Pod.Status.NominatedNodeName = "node-1"
		}
		return podInfo, nil
	}
	go testCtx.Scheduler.Run(testCtx.Ctx)
	return testCtx
}

// TestPreferNominatedNode test that if the nominated node pass all the filters, then preemptor pod will run on the nominated node,
// otherwise, it will be scheduled to another node in the cluster that ables to pass all the filters.
func TestPreferNominatedNode(t *testing.T) {
	defaultNodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
	}
	tests := []struct {
		name         string
		nodeNames    []string
		existingPods []*v1.Pod
		pod          *v1.Pod
		runningNode  string
	}{
		{
			name:      "nominated node released all resource, preemptor is scheduled to the nominated node",
			nodeNames: []string{"node-1", "node-2"},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod1",
					Priority:  &lowPriority,
					NodeName:  "node-2",
					Resources: defaultPodRes,
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			runningNode: "node-1",
		},
		{
			name:      "nominated node cannot pass all the filters, preemptor should find a different node",
			nodeNames: []string{"node-1", "node-2"},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod",
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod1",
				Priority: &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			runningNode: "node-2",
		},
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, test := range tests {
			t.Run(fmt.Sprintf("%s (Async preemption enabled: %v)", test.name, asyncPreemptionEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncPreemption, asyncPreemptionEnabled)

				testCtx := initTestPreferNominatedNode(t, "perfer-nominated-node")
				cs := testCtx.ClientSet
				nsName := testCtx.NS.Name
				var err error
				var preemptor *v1.Pod
				for _, nodeName := range test.nodeNames {
					_, err := createNode(cs, st.MakeNode().Name(nodeName).Capacity(defaultNodeRes).Obj())
					if err != nil {
						t.Fatalf("Error creating node %v: %v", nodeName, err)
					}
				}

				pods := make([]*v1.Pod, len(test.existingPods))
				// Create and run existingPods.
				for i, p := range test.existingPods {
					p.Namespace = nsName
					pods[i], err = runPausePod(cs, p)
					if err != nil {
						t.Fatalf("Error running pause pod: %v", err)
					}
				}
				test.pod.Namespace = nsName
				preemptor, err = createPausePod(cs, test.pod)
				if err != nil {
					t.Errorf("Error while creating high priority pod: %v", err)
				}
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
					preemptor, err = cs.CoreV1().Pods(test.pod.Namespace).Get(ctx, test.pod.Name, metav1.GetOptions{})
					if err != nil {
						t.Errorf("Error getting the preemptor pod info: %v", err)
					}
					if len(preemptor.Spec.NodeName) == 0 {
						return false, err
					}
					return true, nil
				})
				if err != nil {
					t.Errorf("Cannot schedule Pod %v/%v, error: %v", test.pod.Namespace, test.pod.Name, err)
				}
				// Make sure the pod has been scheduled to the right node.
				if preemptor.Spec.NodeName != test.runningNode {
					t.Errorf("Expect pod running on %v, got %v.", test.runningNode, preemptor.Spec.NodeName)
				}
			})
		}
	}
}
