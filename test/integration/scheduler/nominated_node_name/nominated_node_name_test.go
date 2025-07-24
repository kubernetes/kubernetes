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

package nominatednodename

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

const (
	pollInterval  = 100 * time.Millisecond
	shortTimeout  = 5 * time.Second
	normalTimeout = 10 * time.Second
)

var (
	lowPriorityValue  = int32(100)
	highPriorityValue = int32(300)
	lowPriority       = &lowPriorityValue
	highPriority      = &highPriorityValue
)

func getUniqueNodeName(t *testing.T, baseName string) string {
	suffix := string(uuid.NewUUID())[0:8]
	return fmt.Sprintf("%s-%s", baseName, suffix)
}

func waitForPodDeleted(ctx context.Context, cs kubernetes.Interface, namespace, name string) error {
	return wait.PollUntilContextTimeout(ctx, pollInterval, normalTimeout, true, func(ctx context.Context) (bool, error) {
		_, err := cs.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
}

func waitForPodsPending(ctx context.Context, cs kubernetes.Interface, namespace string, podNames []string) error {
	return wait.PollUntilContextTimeout(ctx, pollInterval, shortTimeout, true, func(ctx context.Context) (bool, error) {
		for _, podName := range podNames {
			pod, err := cs.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if pod.Status.Phase != v1.PodPending || pod.Spec.NodeName != "" {
				return false, nil
			}
		}
		return true, nil
	})
}

type testNode struct {
	name     string
	capacity map[v1.ResourceName]string
}

type testPod struct {
	name      string
	priority  *int32
	resources *v1.ResourceRequirements
	nodeName  string
}

func createNodesFromSpecs(t *testing.T, cs kubernetes.Interface, specs []testNode) []string {
	nodeNames := make([]string, len(specs))
	for i, spec := range specs {
		nodeName := getUniqueNodeName(t, spec.name)
		nodeNames[i] = nodeName
		nodeBuilder := st.MakeNode().Name(nodeName)

		if spec.capacity != nil {
			nodeBuilder = nodeBuilder.Capacity(spec.capacity)
		} else {
			nodeBuilder = nodeBuilder.Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:    "1000m",
				v1.ResourceMemory: "100Mi",
			})
		}

		_, err := testutils.CreateNode(cs, nodeBuilder.Obj())
		if err != nil {
			t.Fatalf("create node %s: %v", spec.name, err)
		}
	}
	return nodeNames
}

func createPodsFromSpecs(t *testing.T, ctx *testutils.TestContext, specs []testPod, runPods bool) []*v1.Pod {
	pods := make([]*v1.Pod, len(specs))
	var err error
	for i, spec := range specs {
		pod := testutils.InitPausePod(&testutils.PausePodConfig{
			Name:      spec.name,
			Namespace: ctx.NS.Name,
			Priority:  spec.priority,
			Resources: spec.resources,
		})

		if spec.nodeName != "" {
			pod.Spec.NodeName = spec.nodeName
		}

		if runPods {
			pods[i], err = testutils.RunPausePod(ctx.ClientSet, pod)
		} else {
			pods[i], err = testutils.CreatePausePod(ctx.ClientSet, pod)
		}

		if err != nil {
			t.Fatalf("create pod %s: %v", spec.name, err)
		}
	}
	return pods
}

// TestPreemptionAndNominatedNodeNameScenarios tests preemption scenarios with NominatedNodeName.
func TestPreemptionAndNominatedNodeNameScenarios(t *testing.T) {
	tests := []struct {
		name                        string
		nodeSpecs                   []testNode
		existingPods                []testPod
		preemptor                   testPod
		postNominationAction        func(t *testing.T, ctx *testutils.TestContext, preemptor *v1.Pod, existingPods []*v1.Pod, nodeNames []string)
		expectedNominatedNodePrefix string
		expectedScheduledNodePrefix string
		verifyFunc                  func(t *testing.T, ctx *testutils.TestContext, preemptor *v1.Pod, existingPods []*v1.Pod, nodeNames []string) error
	}{
		{
			name: "basic preemption sets NominatedNodeName",
			nodeSpecs: []testNode{
				{name: "node", capacity: map[v1.ResourceName]string{v1.ResourceCPU: "1000m", v1.ResourceMemory: "100Mi"}},
			},
			existingPods: []testPod{
				{
					name:     "low-priority-pod",
					priority: lowPriority,
					resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("900m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
					},
				},
			},
			preemptor: testPod{
				name:     "high-priority-pod",
				priority: highPriority,
				resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("500m"),
						v1.ResourceMemory: resource.MustParse("30Mi"),
					},
				},
			},
			expectedNominatedNodePrefix: "node",
			expectedScheduledNodePrefix: "",
		},
		{
			name: "prefers nominated node",
			nodeSpecs: []testNode{
				{
					name:     "node1",
					capacity: map[v1.ResourceName]string{v1.ResourceCPU: "1000m", v1.ResourceMemory: "100Mi"},
				},
				{
					name:     "node2",
					capacity: map[v1.ResourceName]string{v1.ResourceCPU: "1000m", v1.ResourceMemory: "100Mi"},
				},
			},
			existingPods: []testPod{
				{
					name:     "low-priority-pod",
					priority: lowPriority,
					resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("900m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
					},
					nodeName: "node2",
				},
				{
					name:     "low-priority-pod-2",
					priority: lowPriority,
					resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("500m"),
							v1.ResourceMemory: resource.MustParse("30Mi"),
						},
					},
					nodeName: "node1",
				},
				{
					name:     "low-priority-pod-3",
					priority: lowPriority,
					resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("450m"),
							v1.ResourceMemory: resource.MustParse("20Mi"),
						},
					},
					nodeName: "node1",
				},
			},
			preemptor: testPod{
				name:     "high-priority-pod",
				priority: highPriority,
				resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("600m"),
						v1.ResourceMemory: resource.MustParse("50Mi"),
					},
				},
			},
			postNominationAction: func(t *testing.T, ctx *testutils.TestContext, preemptor *v1.Pod, existingPods []*v1.Pod, nodeNames []string) {
				err := testutils.DeletePod(ctx.ClientSet, existingPods[0].Name, existingPods[0].Namespace)
				if err != nil {
					t.Fatalf("delete pod %s: %v", existingPods[0].Name, err)
				}

				if err := waitForPodDeleted(ctx.Ctx, ctx.ClientSet, existingPods[0].Namespace, existingPods[0].Name); err != nil {
					t.Fatalf("wait pod %s deletion: %v", existingPods[0].Name, err)
				}
			},
			expectedNominatedNodePrefix: "node2",
			expectedScheduledNodePrefix: "node2",
		},
		{
			name: "keeps NNN when node unavailable",
			nodeSpecs: []testNode{
				{name: "node", capacity: map[v1.ResourceName]string{v1.ResourceCPU: "1000m", v1.ResourceMemory: "100Mi"}},
			},
			existingPods: []testPod{
				{
					name:     "low-priority-pod",
					priority: lowPriority,
					resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("900m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
					},
				},
			},
			preemptor: testPod{
				name:     "high-priority-pod",
				priority: highPriority,
				resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("500m"),
						v1.ResourceMemory: resource.MustParse("30Mi"),
					},
				},
			},
			postNominationAction: func(t *testing.T, ctx *testutils.TestContext, preemptor *v1.Pod, existingPods []*v1.Pod, nodeNames []string) {
				node, err := ctx.ClientSet.CoreV1().Nodes().Get(ctx.Ctx, nodeNames[0], metav1.GetOptions{})
				if err != nil {
					t.Fatalf("get node %s: %v", nodeNames[0], err)
				}
				node.Spec.Unschedulable = true
				_, err = ctx.ClientSet.CoreV1().Nodes().Update(ctx.Ctx, node, metav1.UpdateOptions{})
				if err != nil {
					t.Fatalf("update node %s: %v", nodeNames[0], err)
				}
			},
			expectedNominatedNodePrefix: "node",
			verifyFunc: func(t *testing.T, ctx *testutils.TestContext, preemptor *v1.Pod, existingPods []*v1.Pod, nodeNames []string) error {
				pod, err := ctx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(ctx.Ctx, preemptor.Name, metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("get pod %s: %v", preemptor.Name, err)
				}

				if pod.Status.NominatedNodeName == "" {
					return fmt.Errorf("NominatedNodeName was cleared")
				}

				node, err := ctx.ClientSet.CoreV1().Nodes().Get(ctx.Ctx, nodeNames[0], metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("get node %s: %v", nodeNames[0], err)
				}
				node.Spec.Unschedulable = false
				_, err = ctx.ClientSet.CoreV1().Nodes().Update(ctx.Ctx, node, metav1.UpdateOptions{})
				if err != nil {
					return fmt.Errorf("update node %s: %v", nodeNames[0], err)
				}

				err = testutils.DeletePod(ctx.ClientSet, existingPods[0].Name, existingPods[0].Namespace)
				if err != nil {
					return fmt.Errorf("delete pod %s: %v", existingPods[0].Name, err)
				}
				return nil
			},
			expectedScheduledNodePrefix: "node",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			testCtx := testutils.InitTestSchedulerWithNS(t, "preemption-nnn-test")

			nodeNames := createNodesFromSpecs(t, testCtx.ClientSet, tc.nodeSpecs)

			for i := range tc.existingPods {
				if tc.existingPods[i].nodeName != "" {
					for j, spec := range tc.nodeSpecs {
						if tc.existingPods[i].nodeName == spec.name {
							tc.existingPods[i].nodeName = nodeNames[j]
							break
						}
					}
				}
			}
			existingPods := createPodsFromSpecs(t, testCtx, tc.existingPods, true)

			if err := testutils.WaitCachedPodsStable(testCtx, existingPods); err != nil {
				t.Fatalf("wait pods stable: %v", err)
			}

			preemptorPods := createPodsFromSpecs(t, testCtx, []testPod{tc.preemptor}, false)
			preemptor := preemptorPods[0]

			err := testutils.WaitForNominatedNodeName(testCtx.Ctx, testCtx.ClientSet, preemptor)
			if err != nil {
				t.Fatalf("wait NominatedNodeName: %v", err)
			}

			preemptor, err = testCtx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get pod: %v", err)
			}

			if tc.expectedNominatedNodePrefix != "" && !strings.HasPrefix(preemptor.Status.NominatedNodeName, tc.expectedNominatedNodePrefix) {
				t.Errorf("want NominatedNodeName prefix %q, got %q", tc.expectedNominatedNodePrefix, preemptor.Status.NominatedNodeName)
			}

			if tc.postNominationAction != nil {
				tc.postNominationAction(t, testCtx, preemptor, existingPods, nodeNames)
			}

			if tc.verifyFunc != nil {
				if err := tc.verifyFunc(t, testCtx, preemptor, existingPods, nodeNames); err != nil {
					t.Fatalf("verify: %v", err)
				}
			}

			if tc.expectedScheduledNodePrefix != "" || tc.postNominationAction != nil {
				err = testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, preemptor)
				if err != nil {
					t.Fatalf("schedule pod: %v", err)
				}

				preemptor, err = testCtx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("get scheduled pod: %v", err)
				}

				if tc.expectedScheduledNodePrefix != "" && !strings.HasPrefix(preemptor.Spec.NodeName, tc.expectedScheduledNodePrefix) {
					t.Errorf("want node prefix %q, got %q", tc.expectedScheduledNodePrefix, preemptor.Spec.NodeName)
				}
			}
		})
	}
}

// TestClearingNominatedNodeNameAfterBinding tests NominatedNodeName clearing behavior after binding.
func TestClearingNominatedNodeNameAfterBinding(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("ClearingNominatedNodeNameAfterBinding: %t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ClearingNominatedNodeNameAfterBinding, enabled)

			testCtx := testutils.InitTestSchedulerWithNS(t, "clearing-nnn-test")
			createNodesFromSpecs(t, testCtx.ClientSet, []testNode{
				{
					name: "node",
					capacity: map[v1.ResourceName]string{
						v1.ResourceCPU:    "1000m",
						v1.ResourceMemory: "100Mi",
					},
				},
			})

			victim := testutils.InitPausePod(&testutils.PausePodConfig{
				Name:      "victim-pod",
				Namespace: testCtx.NS.Name,
				Priority:  lowPriority,
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("900m"),
						v1.ResourceMemory: resource.MustParse("50Mi"),
					},
				},
			})
			victim, err := testutils.RunPausePod(testCtx.ClientSet, victim)
			if err != nil {
				t.Fatalf("create victim pod: %v", err)
			}

			if err := testutils.WaitCachedPodsStable(testCtx, []*v1.Pod{victim}); err != nil {
				t.Fatalf("wait victim pod stable: %v", err)
			}

			preemptor := testutils.InitPausePod(&testutils.PausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  highPriority,
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("500m"),
						v1.ResourceMemory: resource.MustParse("30Mi"),
					},
				},
			})
			preemptor, err = testutils.CreatePausePod(testCtx.ClientSet, preemptor)
			if err != nil {
				t.Fatalf("create preemptor pod: %v", err)
			}

			err = testutils.WaitForNominatedNodeName(testCtx.Ctx, testCtx.ClientSet, preemptor)
			if err != nil {
				t.Fatalf("NominatedNodeName was not set: %v", err)
			}

			preemptor, err = testCtx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get pod: %v", err)
			}

			if preemptor.Status.NominatedNodeName == "" {
				t.Fatalf("NominatedNodeName is empty")
			}

			err = testutils.DeletePod(testCtx.ClientSet, victim.Name, victim.Namespace)
			if err != nil {
				t.Fatalf("delete victim pod: %v", err)
			}

			err = testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, preemptor)
			if err != nil {
				t.Fatalf("Preemptor pod failed to schedule: %v", err)
			}

			preemptor, err = testCtx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get scheduled pod: %v", err)
			}

			if enabled {
				if preemptor.Status.NominatedNodeName != "" {
					t.Errorf("NominatedNodeName not cleared: %s", preemptor.Status.NominatedNodeName)
				}
			} else {
				if preemptor.Status.NominatedNodeName == "" {
					t.Errorf("NominatedNodeName was cleared")
				}
			}

		})
	}
}

// TestExternalComponentSetsNominatedNodeName tests scheduler handling of externally set NominatedNodeName.
func TestExternalComponentSetsNominatedNodeName(t *testing.T) {
	testCtx := testutils.InitTestSchedulerWithNS(t, "external-nnn-test")

	nodeNames := createNodesFromSpecs(t, testCtx.ClientSet, []testNode{
		{name: "node1"},
		{name: "node2"},
	})

	targetNodeName := nodeNames[1]
	pod := st.MakePod().Name("external-nnn-pod").Namespace(testCtx.NS.Name).
		Container("c").
		SchedulingGates([]string{"example.com/external-nnn-test"}).
		Obj()

	createdPod, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create pod: %v", err)
	}

	createdPod.Status.NominatedNodeName = targetNodeName
	_, err = testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, createdPod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("update pod status: %v", err)
	}

	updatedPod, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, createdPod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get pod: %v", err)
	}
	if updatedPod.Status.NominatedNodeName != targetNodeName {
		t.Fatalf("NominatedNodeName mismatch: want %s, got %s", targetNodeName, updatedPod.Status.NominatedNodeName)
	}

	updatedPod.Spec.SchedulingGates = nil
	_, err = testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Update(testCtx.Ctx, updatedPod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("remove scheduling gate: %v", err)
	}

	err = testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, updatedPod)
	if err != nil {
		t.Fatalf("schedule pod %s: %v", updatedPod.Name, err)
	}
	scheduledPod, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, updatedPod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get scheduled pod %s: %v", updatedPod.Name, err)
	}

	if scheduledPod.Spec.NodeName != targetNodeName {
		t.Errorf("want node %s, got %s", targetNodeName, scheduledPod.Spec.NodeName)
	}
}

// TestHighPriorityIgnoresNominatedNode tests high priority pod scheduling to available nodes.
func TestHighPriorityIgnoresNominatedNode(t *testing.T) {
	testCtx := testutils.InitTestSchedulerWithNS(t, "high-priority-ignores-nnn")

	nodeNames := createNodesFromSpecs(t, testCtx.ClientSet, []testNode{
		{
			name: "node1",
			capacity: map[v1.ResourceName]string{
				v1.ResourceCPU:    "1000m",
				v1.ResourceMemory: "100Mi",
			},
		},
		{
			name: "node2",
			capacity: map[v1.ResourceName]string{
				v1.ResourceCPU:    "1000m",
				v1.ResourceMemory: "100Mi",
			},
		},
	})

	occupiedNodeName := nodeNames[0]
	emptyNodeName := nodeNames[1]

	existingPod := testutils.InitPausePod(&testutils.PausePodConfig{
		Name:      "existing-pod",
		Namespace: testCtx.NS.Name,
		Priority:  lowPriority,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("900m"),
				v1.ResourceMemory: resource.MustParse("50Mi"),
			},
		},
	})
	existingPod.Spec.NodeName = occupiedNodeName
	existingPod, err := testutils.CreatePausePod(testCtx.ClientSet, existingPod)
	if err != nil {
		t.Fatalf("create existing pod: %v", err)
	}

	highPriorityPod := testutils.InitPausePod(&testutils.PausePodConfig{
		Name:      "high-priority-pod",
		Namespace: testCtx.NS.Name,
		Priority:  highPriority,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("800m"),
				v1.ResourceMemory: resource.MustParse("50Mi"),
			},
		},
	})
	highPriorityPod, err = testutils.CreatePausePod(testCtx.ClientSet, highPriorityPod)
	if err != nil {
		t.Fatalf("create high priority pod: %v", err)
	}

	err = testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, highPriorityPod)
	if err != nil {
		t.Fatalf("High priority pod failed to schedule: %v", err)
	}

	highPriorityPod, err = testCtx.ClientSet.CoreV1().Pods(highPriorityPod.Namespace).Get(testCtx.Ctx, highPriorityPod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get high priority pod: %v", err)
	}

	if highPriorityPod.Spec.NodeName != emptyNodeName {
		t.Errorf("want node %s, got %s", emptyNodeName, highPriorityPod.Spec.NodeName)
	}
}

// TestNominatedNodeNamePriorityWhenNodeAppears tests pod with NNN scheduling when node appears.
func TestNominatedNodeNamePriorityWhenNodeAppears(t *testing.T) {
	testCtx := testutils.InitTestSchedulerWithNS(t, "nnn-priority-test")

	node1Name := getUniqueNodeName(t, "node1")
	_, err := testutils.CreateNode(testCtx.ClientSet,
		st.MakeNode().Name(node1Name).Capacity(map[v1.ResourceName]string{
			v1.ResourceCPU:    "500m",
			v1.ResourceMemory: "50Mi",
		}).Obj())
	if err != nil {
		t.Fatalf("create node %s: %v", node1Name, err)
	}

	nominatedNodeName := getUniqueNodeName(t, "nominated-node")
	podWithNNN := testutils.InitPausePod(&testutils.PausePodConfig{
		Name:      "pod-with-nnn",
		Namespace: testCtx.NS.Name,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("600m"),
				v1.ResourceMemory: resource.MustParse("40Mi"),
			},
		},
	})
	podWithNNN, err = testutils.CreatePausePod(testCtx.ClientSet, podWithNNN)
	if err != nil {
		t.Fatalf("create pod: %v", err)
	}

	patch := []byte(fmt.Sprintf(`{"status":{"nominatedNodeName":"%s"}}`, nominatedNodeName))
	_, err = testCtx.ClientSet.CoreV1().Pods(podWithNNN.Namespace).Patch(
		testCtx.Ctx, podWithNNN.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "status")
	if err != nil {
		t.Fatalf("set NominatedNodeName: %v", err)
	}

	regularPod := testutils.InitPausePod(&testutils.PausePodConfig{
		Name:      "pod-without-nnn",
		Namespace: testCtx.NS.Name,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("600m"),
				v1.ResourceMemory: resource.MustParse("40Mi"),
			},
		},
	})
	regularPod, err = testutils.CreatePausePod(testCtx.ClientSet, regularPod)
	if err != nil {
		t.Fatalf("create pod: %v", err)
	}

	if err := waitForPodsPending(testCtx.Ctx, testCtx.ClientSet, testCtx.NS.Name, []string{podWithNNN.Name, regularPod.Name}); err != nil {
		t.Fatalf("wait pods pending: %v", err)
	}

	podWithNNN, err = testCtx.ClientSet.CoreV1().Pods(podWithNNN.Namespace).Get(testCtx.Ctx, podWithNNN.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get pod: %v", err)
	}
	if podWithNNN.Spec.NodeName != "" {
		t.Fatalf("pod %s scheduled too early to %s", podWithNNN.Name, podWithNNN.Spec.NodeName)
	}

	regularPod, err = testCtx.ClientSet.CoreV1().Pods(regularPod.Namespace).Get(testCtx.Ctx, regularPod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get pod: %v", err)
	}
	if regularPod.Spec.NodeName != "" {
		t.Fatalf("pod %s scheduled unexpectedly to %s", regularPod.Name, regularPod.Spec.NodeName)
	}

	_, err = testutils.CreateNode(testCtx.ClientSet,
		st.MakeNode().Name(nominatedNodeName).Capacity(map[v1.ResourceName]string{
			v1.ResourceCPU:    "700m",
			v1.ResourceMemory: "50Mi",
		}).Obj())
	if err != nil {
		t.Fatalf("create node %s: %v", nominatedNodeName, err)
	}

	err = testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, podWithNNN)
	if err != nil {
		t.Fatalf("Pod with NNN failed to schedule: %v", err)
	}

	podWithNNN, err = testCtx.ClientSet.CoreV1().Pods(podWithNNN.Namespace).Get(testCtx.Ctx, podWithNNN.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get pod: %v", err)
	}

	if podWithNNN.Spec.NodeName != nominatedNodeName {
		t.Errorf("want node %s, got %s", nominatedNodeName, podWithNNN.Spec.NodeName)
	}

	regularPod, err = testCtx.ClientSet.CoreV1().Pods(regularPod.Namespace).Get(testCtx.Ctx, regularPod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get pod: %v", err)
	}
	if regularPod.Spec.NodeName == nominatedNodeName {
		t.Errorf("regular pod scheduled on reserved node")
	}
}

// TestSchedulerOverwritesNominatedNodeName tests scheduler overwriting NominatedNodeName.
func TestSchedulerOverwritesNominatedNodeName(t *testing.T) {
	testCtx := testutils.InitTestSchedulerWithNS(t, "overwrite-nnn-test")

	node1Name := getUniqueNodeName(t, "node1")
	node2Name := getUniqueNodeName(t, "node2")
	node3Name := getUniqueNodeName(t, "node3")

	nodeNames := []string{node1Name, node2Name, node3Name}
	for _, nodeName := range nodeNames {
		_, err := testutils.CreateNode(testCtx.ClientSet,
			st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:    "1000m",
				v1.ResourceMemory: "100Mi",
			}).Obj())
		if err != nil {
			t.Fatalf("create node %s: %v", nodeName, err)
		}
	}

	victim1 := testutils.InitPausePod(&testutils.PausePodConfig{
		Name:      "victim-pod-1",
		Namespace: testCtx.NS.Name,
		Priority:  lowPriority,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("900m"),
				v1.ResourceMemory: resource.MustParse("50Mi"),
			},
		},
	})
	victim1.Spec.NodeName = node1Name
	victim1, err := testutils.CreatePausePod(testCtx.ClientSet, victim1)
	if err != nil {
		t.Fatalf("create victim pod: %v", err)
	}

	victim2 := testutils.InitPausePod(&testutils.PausePodConfig{
		Name:      "victim-pod-2",
		Namespace: testCtx.NS.Name,
		Priority:  lowPriority,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("900m"),
				v1.ResourceMemory: resource.MustParse("50Mi"),
			},
		},
	})
	victim2.Spec.NodeName = node2Name
	victim2, err = testutils.CreatePausePod(testCtx.ClientSet, victim2)
	if err != nil {
		t.Fatalf("create victim pod: %v", err)
	}

	victim3 := testutils.InitPausePod(&testutils.PausePodConfig{
		Name:      "victim-pod-3",
		Namespace: testCtx.NS.Name,
		Priority:  lowPriority,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("300m"),
				v1.ResourceMemory: resource.MustParse("20Mi"),
			},
		},
	})
	victim3.Spec.NodeName = node3Name
	victim3, err = testutils.CreatePausePod(testCtx.ClientSet, victim3)
	if err != nil {
		t.Fatalf("create victim pod: %v", err)
	}

	preemptor := testutils.InitPausePod(&testutils.PausePodConfig{
		Name:      "preemptor-pod",
		Namespace: testCtx.NS.Name,
		Priority:  highPriority,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("800m"),
				v1.ResourceMemory: resource.MustParse("50Mi"),
			},
		},
	})
	preemptor, err = testutils.CreatePausePod(testCtx.ClientSet, preemptor)
	if err != nil {
		t.Fatalf("create preemptor pod: %v", err)
	}

	err = testutils.WaitForNominatedNodeName(testCtx.Ctx, testCtx.ClientSet, preemptor)
	if err != nil {
		t.Fatalf("NominatedNodeName was not set: %v", err)
	}

	preemptor, err = testCtx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get pod: %v", err)
	}
	initialNNN := preemptor.Status.NominatedNodeName

	err = testutils.DeletePod(testCtx.ClientSet, victim2.Name, victim2.Namespace)
	if err != nil {
		t.Fatalf("delete victim pod: %v", err)
	}

	err = testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, preemptor)
	if err != nil {
		t.Fatalf("Preemptor pod failed to schedule: %v", err)
	}

	preemptor, err = testCtx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get scheduled pod: %v", err)
	}

	if preemptor.Spec.NodeName == "" {
		t.Errorf("pod not scheduled")
	}

	if initialNNN != "" && preemptor.Spec.NodeName != initialNNN {
		t.Errorf("want node %s, got %s", initialNNN, preemptor.Spec.NodeName)
	}
}
