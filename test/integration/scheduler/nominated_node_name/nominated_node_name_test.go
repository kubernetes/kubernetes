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
	"k8s.io/apimachinery/pkg/runtime"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	schedulerutils "k8s.io/kubernetes/test/integration/scheduler"
	testutils "k8s.io/kubernetes/test/integration/util"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

type FakePermitPlugin struct {
	code fwk.Code
}

type RunForeverPreBindPlugin struct {
	cancel <-chan struct{}
}

type NoNNNPostBindPlugin struct {
	t      *testing.T
	cancel <-chan struct{}
}

func (bp *NoNNNPostBindPlugin) Name() string {
	return "NoNNNPostBindPlugin"
}

func (bp *NoNNNPostBindPlugin) PostBind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) {
	if p.Status.NominatedNodeName != "" {
		bp.t.Fatalf("PostBind should not set .status.nominatedNodeName for pod %v/%v, but it was set to %v", p.Namespace, p.Name, p.Status.NominatedNodeName)
	}
}

// Name returns name of the plugin.
func (pp *FakePermitPlugin) Name() string {
	return "FakePermitPlugin"
}

// Permit implements the permit test plugin.
func (pp *FakePermitPlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if pp.code == fwk.Wait {
		return fwk.NewStatus(pp.code, ""), 10 * time.Minute
	}
	return fwk.NewStatus(pp.code, ""), 0
}

// Name returns name of the plugin.
func (pp *RunForeverPreBindPlugin) Name() string {
	return "RunForeverPreBindPlugin"
}

// PreBindPreFlight is a test function that returns nil for testing.
func (pp *RunForeverPreBindPlugin) PreBindPreFlight(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	return nil
}

// PreBind is a test function that returns (true, nil) or errors for testing.
func (pp *RunForeverPreBindPlugin) PreBind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	select {
	case <-ctx.Done():
		return fwk.NewStatus(fwk.Error, "context cancelled")
	case <-pp.cancel:
		return fwk.NewStatus(fwk.Error, "pre-bind cancelled")
	}
}

// Test_PutNominatedNodeNameInBindingCycle makes sure that nominatedNodeName is set in the binding cycle
// when the PreBind or Permit plugin (WaitOnPermit) is going to work.
func Test_PutNominatedNodeNameInBindingCycle(t *testing.T) {
	cancel := make(chan struct{})
	tests := []struct {
		name                    string
		plugin                  framework.Plugin
		expectNominatedNodeName bool
		cleanup                 func()
	}{
		{
			name:                    "NominatedNodeName is put if PreBindPlugin will run",
			plugin:                  &RunForeverPreBindPlugin{cancel: cancel},
			expectNominatedNodeName: true,
			cleanup: func() {
				close(cancel)
			},
		},
		{
			name:                    "NominatedNodeName is put if PermitPlugin will run at WaitOnPermit",
			expectNominatedNodeName: true,
			plugin: &FakePermitPlugin{
				code: fwk.Wait,
			},
		},
		{
			name: "NominatedNodeName is not put if PermitPlugin won't run at WaitOnPermit",
			plugin: &FakePermitPlugin{
				code: fwk.Success,
			},
		},
		{
			name:   "NominatedNodeName is not put if PermitPlugin nor PreBindPlugin will run",
			plugin: nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testContext := testutils.InitTestAPIServer(t, "nnn-test", nil)
			if test.cleanup != nil {
				defer test.cleanup()
			}

			pf := func(plugin framework.Plugin) frameworkruntime.PluginFactory {
				return func(_ context.Context, _ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
					return plugin, nil
				}
			}

			plugins := []framework.Plugin{&NoNNNPostBindPlugin{cancel: testContext.Ctx.Done(), t: t}}
			if test.plugin != nil {
				plugins = append(plugins, test.plugin)
			}

			registry, prof := schedulerutils.InitRegistryAndConfig(t, pf, plugins...)

			testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 10, true,
				scheduler.WithProfiles(prof),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer teardown()

			pod, err := testutils.CreatePausePod(testCtx.ClientSet,
				testutils.InitPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Fatalf("Error while creating a test pod: %v", err)
			}

			if test.expectNominatedNodeName {
				if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, testCtx.ClientSet, pod); err != nil {
					t.Errorf(".status.nominatedNodeName was not set for pod %v/%v: %v", pod.Namespace, pod.Name, err)
				}
			} else {
				if err := testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, pod); err != nil {
					t.Errorf("Pod %v/%v was not scheduled: %v", pod.Namespace, pod.Name, err)
				}
			}
		})
	}
}

const (
	pollInterval  = 100 * time.Millisecond
	shortTimeout  = 5 * time.Second
	normalTimeout = 10 * time.Second
)

var (
	lowPriority  = ptr.To[int32](100)
	highPriority = ptr.To[int32](300)
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

// hasPrefix checks if value has any of the prefixes in the set
func hasPrefix(set sets.Set[string], value string) bool {
	for prefix := range set {
		if strings.HasPrefix(value, prefix) {
			return true
		}
	}
	return false
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
	name              string
	priority          *int32
	resources         *v1.ResourceRequirements
	nodeName          string
	nominatedNodeName string
}

func createTestNode(t *testing.T, cs kubernetes.Interface, spec testNode) string {
	nodeName := getUniqueNodeName(t, spec.name)
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
	return nodeName
}

func setupTestPod(t *testing.T, ctx *testutils.TestContext, spec testPod, runPod bool) *v1.Pod {
	pod := testutils.InitPausePod(&testutils.PausePodConfig{
		Name:      spec.name,
		Namespace: ctx.NS.Name,
		Priority:  spec.priority,
		Resources: spec.resources,
	})

	if spec.nodeName != "" {
		pod.Spec.NodeName = spec.nodeName
	}

	if spec.nominatedNodeName != "" {
		pod.Spec.SchedulingGates = []v1.PodSchedulingGate{
			{Name: "test.example.com/nnn-test"},
		}
	}

	var err error
	var createdPod *v1.Pod
	if runPod {
		createdPod, err = testutils.RunPausePod(ctx.ClientSet, pod)
	} else {
		createdPod, err = testutils.CreatePausePod(ctx.ClientSet, pod)
	}

	if err != nil {
		t.Fatalf("create pod %s: %v", spec.name, err)
	}

	// Set NNN if specified
	if spec.nominatedNodeName != "" {
		patch := []byte(fmt.Sprintf(`{"status":{"nominatedNodeName":"%s"}}`, spec.nominatedNodeName))
		createdPod, err = ctx.ClientSet.CoreV1().Pods(ctx.NS.Name).Patch(ctx.Ctx, createdPod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "status")
		if err != nil {
			t.Fatalf("update pod %s status with NNN: %v", spec.name, err)
		}
	}

	// Remove scheduling gate after setting NNN
	if pod.Spec.SchedulingGates != nil {
		patch := []byte(`{"spec":{"schedulingGates":null}}`)
		createdPod, err = ctx.ClientSet.CoreV1().Pods(ctx.NS.Name).Patch(ctx.Ctx, createdPod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
		if err != nil {
			t.Fatalf("remove scheduling gate from pod %s: %v", spec.name, err)
		}
	}
	return createdPod
}

// TestPreemptionAndNominatedNodeNameScenarios tests preemption scenarios with NominatedNodeName.
func TestPreemptionAndNominatedNodeNameScenarios(t *testing.T) {
	tests := []struct {
		name                            string
		nodeSpecs                       []testNode
		existingPods                    []testPod
		preemptor                       testPod
		postNominationAction            func(t *testing.T, ctx *testutils.TestContext, preemptor *v1.Pod, existingPods []*v1.Pod, nodeNames []string)
		expectedNominatedNodeNamePrefix sets.Set[string]
		expectedScheduledNodeNamePrefix sets.Set[string]
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
			postNominationAction: func(t *testing.T, ctx *testutils.TestContext, preemptor *v1.Pod, existingPods []*v1.Pod, nodeNames []string) {
				victim := existingPods[0]
				err := testutils.DeletePod(ctx.ClientSet, victim.Name, victim.Namespace)
				if err != nil {
					t.Fatalf("delete pod %s: %v", victim.Name, err)
				}

				if err := waitForPodDeleted(ctx.Ctx, ctx.ClientSet, victim.Namespace, victim.Name); err != nil {
					t.Fatalf("wait pod %s deletion: %v", victim.Name, err)
				}
			},
			expectedNominatedNodeNamePrefix: sets.New("node"),
			expectedScheduledNodeNamePrefix: sets.New("node"),
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
			expectedNominatedNodeNamePrefix: sets.New("node2"),
			expectedScheduledNodeNamePrefix: sets.New("node2"),
		},
		{
			name: "Overwrite NominatedNodeName with preemption",
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
					name:     "low-priority-pod-1",
					priority: lowPriority,
					resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("500m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
					},
					nodeName: "node1",
				},
				{
					name:     "low-priority-pod-2",
					priority: lowPriority,
					resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("500m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
					},
					nodeName: "node1",
				},
				{
					name:     "low-priority-pod-3",
					priority: lowPriority,
					resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("900m"),
							v1.ResourceMemory: resource.MustParse("20Mi"),
						},
					},
					nodeName: "node2",
				},
			},
			postNominationAction: func(t *testing.T, ctx *testutils.TestContext, preemptor *v1.Pod, existingPods []*v1.Pod, nodeNames []string) {
				victim := existingPods[2]
				err := testutils.DeletePod(ctx.ClientSet, victim.Name, victim.Namespace)
				if err != nil {
					t.Fatalf("delete pod %s: %v", victim.Name, err)
				}

				if err := waitForPodDeleted(ctx.Ctx, ctx.ClientSet, victim.Namespace, victim.Name); err != nil {
					t.Fatalf("wait pod %s deletion: %v", victim.Name, err)
				}
			},
			preemptor: testPod{
				name:     "high-priority-pod",
				priority: highPriority,
				resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("650m"),
						v1.ResourceMemory: resource.MustParse("20Mi"),
					},
				},
				nominatedNodeName: "node1",
			},
			expectedNominatedNodeNamePrefix: sets.New("node2"),
			expectedScheduledNodeNamePrefix: sets.New("node2"),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NominatedNodeNameForExpectation, true)
			testCtx := testutils.InitTestSchedulerWithNS(t, "preemption-nnn-test")

			nodeNames := make([]string, len(tc.nodeSpecs))
			for i, nodeSpec := range tc.nodeSpecs {
				nodeNames[i] = createTestNode(t, testCtx.ClientSet, nodeSpec)
			}

			for i := range tc.existingPods {
				if tc.existingPods[i].nodeName != "" {
					for j, nodeSpec := range tc.nodeSpecs {
						if tc.existingPods[i].nodeName == nodeSpec.name {
							tc.existingPods[i].nodeName = nodeNames[j]
							break
						}
					}
				}
			}
			existingPods := make([]*v1.Pod, len(tc.existingPods))
			for i, podSpec := range tc.existingPods {
				existingPods[i] = setupTestPod(t, testCtx, podSpec, true)
			}
			if err := testutils.WaitCachedPodsStable(testCtx, existingPods); err != nil {
				t.Fatalf("wait pods stable: %v", err)
			}

			preemptor := setupTestPod(t, testCtx, tc.preemptor, false)
			err := testutils.WaitForNominatedNodeName(testCtx.Ctx, testCtx.ClientSet, preemptor)
			if err != nil {
				t.Fatalf("wait NominatedNodeName: %v", err)
			}

			preemptor, err = testCtx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get pod: %v", err)
			}

			if tc.expectedNominatedNodeNamePrefix != nil && !hasPrefix(tc.expectedNominatedNodeNamePrefix, preemptor.Status.NominatedNodeName) {
				t.Errorf("want NominatedNodeName prefix %q, got %q", tc.expectedNominatedNodeNamePrefix, preemptor.Status.NominatedNodeName)
			}

			if tc.postNominationAction != nil {
				tc.postNominationAction(t, testCtx, preemptor, existingPods, nodeNames)
			}

			if tc.expectedScheduledNodeNamePrefix != nil || tc.postNominationAction != nil {
				err = testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, preemptor)
				if err != nil {
					t.Fatalf("schedule pod: %v", err)
				}

				preemptor, err = testCtx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("get scheduled pod: %v", err)
				}

				if tc.expectedScheduledNodeNamePrefix != nil && !hasPrefix(tc.expectedScheduledNodeNamePrefix, preemptor.Spec.NodeName) {
					t.Errorf("want one of %v node(s), got %q", tc.expectedScheduledNodeNamePrefix, preemptor.Spec.NodeName)
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
			createTestNode(t, testCtx.ClientSet, testNode{
				name: "node",
				capacity: map[v1.ResourceName]string{
					v1.ResourceCPU:    "1000m",
					v1.ResourceMemory: "100Mi",
				},
			})

			victim := setupTestPod(t, testCtx, testPod{
				name:     "victim-pod",
				priority: lowPriority,
				resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("900m"),
						v1.ResourceMemory: resource.MustParse("50Mi"),
					},
				},
			}, true)

			if err := testutils.WaitCachedPodsStable(testCtx, []*v1.Pod{victim}); err != nil {
				t.Fatalf("wait victim pod stable: %v", err)
			}

			preemptor := setupTestPod(t, testCtx, testPod{
				name:     "preemptor-pod",
				priority: highPriority,
				resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("500m"),
						v1.ResourceMemory: resource.MustParse("30Mi"),
					},
				},
			}, false)

			err := testutils.WaitForNominatedNodeName(testCtx.Ctx, testCtx.ClientSet, preemptor)
			if err != nil {
				t.Fatalf("NominatedNodeName was not set: %v", err)
			}

			preemptor, err = testCtx.ClientSet.CoreV1().Pods(preemptor.Namespace).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get pod: %v", err)
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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ClearingNominatedNodeNameAfterBinding, true)
	testCtx := testutils.InitTestSchedulerWithNS(t, "external-nnn-test")

	nodeSpecs := []testNode{
		{name: "node1"},
		{name: "node2"},
	}
	nodeNames := make([]string, len(nodeSpecs))
	for i, nodeSpec := range nodeSpecs {
		nodeNames[i] = createTestNode(t, testCtx.ClientSet, nodeSpec)
	}

	targetNodeName := nodeNames[1]

	// Add a dummy pod to node2 to make the test more stable
	// This ensures the scheduler prefers node1 by default but respects NominatedNodeName
	dummyPod := setupTestPod(t, testCtx, testPod{
		name:     "dummy-pod",
		nodeName: targetNodeName,
		resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("10Mi"),
			},
		},
	}, true)
	if err := testutils.WaitCachedPodsStable(testCtx, []*v1.Pod{dummyPod}); err != nil {
		t.Fatalf("wait dummy pod stable: %v", err)
	}

	pod := setupTestPod(t, testCtx, testPod{
		name:              "external-nnn-pod",
		nominatedNodeName: targetNodeName,
	}, false)

	err := testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, pod)
	if err != nil {
		t.Fatalf("schedule pod %s: %v", pod.Name, err)
	}
	scheduledPod, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get scheduled pod %s: %v", pod.Name, err)
	}

	if scheduledPod.Spec.NodeName != targetNodeName {
		t.Errorf("want node %s, got %s", targetNodeName, scheduledPod.Spec.NodeName)
	}
}

// TestHighPriorityIgnoresNominatedNode tests high priority pod ignores NominatedNodeName reservations.
func TestHighPriorityIgnoresNominatedNode(t *testing.T) {
	testCtx := testutils.InitTestSchedulerWithNS(t, "high-priority-ignores-nnn")

	// Create a single node
	nodeName := createTestNode(t, testCtx.ClientSet, testNode{
		name: "node",
		capacity: map[v1.ResourceName]string{
			v1.ResourceCPU:    "1000m",
			v1.ResourceMemory: "100Mi",
		},
	})

	lowPriorityPod := setupTestPod(t, testCtx, testPod{
		name:     "low-priority-pod",
		priority: lowPriority,
		resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1500m"), // Exceeds node capacity
				v1.ResourceMemory: resource.MustParse("50Mi"),
			},
		},
		nominatedNodeName: nodeName,
	}, false)

	highPriorityPod := setupTestPod(t, testCtx, testPod{
		name:     "high-priority-pod",
		priority: highPriority,
		resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("800m"),
				v1.ResourceMemory: resource.MustParse("50Mi"),
			},
		},
	}, true)

	if highPriorityPod.Spec.NodeName != nodeName {
		t.Errorf("want node %s, got %s", nodeName, highPriorityPod.Spec.NodeName)
	}

	// Verify low priority pod is still pending
	lowPriorityPod, err := testCtx.ClientSet.CoreV1().Pods(lowPriorityPod.Namespace).Get(testCtx.Ctx, lowPriorityPod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get low priority pod: %v", err)
	}
	if lowPriorityPod.Status.Phase != v1.PodPending {
		t.Errorf("low priority pod is not pending: %s", lowPriorityPod.Status.Phase)
	}
	if lowPriorityPod.Status.NominatedNodeName != nodeName {
		t.Errorf("low priority pod NominatedNodeName is not set to %s, got %s", nodeName, lowPriorityPod.Status.NominatedNodeName)
	}
}

// TestPodWithNNNIsScheduledBeforePodWithoutNNN tests a pod with NNN set to node N gets scheduled on node N before pods that don't have NNN set.
func TestPodWithNNNIsScheduledBeforePodWithoutNNN(t *testing.T) {
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
	podWithNNN := setupTestPod(t, testCtx, testPod{
		name: "pod-with-nnn",
		resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("600m"),
				v1.ResourceMemory: resource.MustParse("40Mi"),
			},
		},
		nominatedNodeName: nominatedNodeName,
	}, false)

	regularPod := setupTestPod(t, testCtx, testPod{
		name: "pod-without-nnn",
		resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("600m"),
				v1.ResourceMemory: resource.MustParse("40Mi"),
			},
		},
	}, false)

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
