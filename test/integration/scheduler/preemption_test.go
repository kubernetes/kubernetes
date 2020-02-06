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

// This file tests preemption functionality of the scheduler.

package scheduler

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	"k8s.io/kubernetes/plugin/pkg/admission/priority"
	testutils "k8s.io/kubernetes/test/utils"
)

var lowPriority, mediumPriority, highPriority = int32(100), int32(200), int32(300)

func waitForNominatedNodeNameWithTimeout(cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	if err := wait.Poll(100*time.Millisecond, timeout, func() (bool, error) {
		pod, err := cs.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if len(pod.Status.NominatedNodeName) > 0 {
			return true, nil
		}
		return false, err
	}); err != nil {
		return fmt.Errorf("Pod %v/%v annotation did not get set: %v", pod.Namespace, pod.Name, err)
	}
	return nil
}

func waitForNominatedNodeName(cs clientset.Interface, pod *v1.Pod) error {
	return waitForNominatedNodeNameWithTimeout(cs, pod, wait.ForeverTestTimeout)
}

const tokenFilterName = "token-filter"

type tokenFilter struct {
	Tokens       int
	Unresolvable bool
}

// Name returns name of the plugin.
func (fp *tokenFilter) Name() string {
	return tokenFilterName
}

func (fp *tokenFilter) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod,
	nodeInfo *schedulernodeinfo.NodeInfo) *framework.Status {
	if fp.Tokens > 0 {
		fp.Tokens--
		return nil
	}
	status := framework.Unschedulable
	if fp.Unresolvable {
		status = framework.UnschedulableAndUnresolvable
	}
	return framework.NewStatus(status, fmt.Sprintf("can't fit %v", pod.Name))
}

func (fp *tokenFilter) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) *framework.Status {
	return nil
}

func (fp *tokenFilter) AddPod(ctx context.Context, state *framework.CycleState, podToSchedule *v1.Pod,
	podToAdd *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *framework.Status {
	fp.Tokens--
	return nil
}

func (fp *tokenFilter) RemovePod(ctx context.Context, state *framework.CycleState, podToSchedule *v1.Pod,
	podToRemove *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *framework.Status {
	fp.Tokens++
	return nil
}

func (fp *tokenFilter) PreFilterExtensions() framework.PreFilterExtensions {
	return fp
}

var _ framework.FilterPlugin = &tokenFilter{}

// TestPreemption tests a few preemption scenarios.
func TestPreemption(t *testing.T) {
	// Initialize scheduler with a filter plugin.
	var filter tokenFilter
	registry := make(framework.Registry)
	err := registry.Register(filterPluginName, func(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
		return &filter, nil
	})
	if err != nil {
		t.Fatalf("Error registering a filter: %v", err)
	}
	plugins := &schedulerconfig.Plugins{
		Filter: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: filterPluginName,
				},
			},
		},
		PreFilter: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: filterPluginName,
				},
			},
		},
	}
	testCtx := initTestSchedulerWithOptions(t,
		initTestMaster(t, "preemptiom", nil),
		false, nil, time.Second,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))

	defer cleanupTest(t, testCtx)
	cs := testCtx.clientSet

	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
	}

	maxTokens := 1000
	tests := []struct {
		description         string
		existingPods        []*v1.Pod
		pod                 *v1.Pod
		initTokens          int
		unresolvable        bool
		preemptedPodIndexes map[int]struct{}
	}{
		{
			description: "basic pod preemption",
			initTokens:  maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}},
		},
		{
			description: "basic pod preemption with filter",
			initTokens:  1,
			existingPods: []*v1.Pod{
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}},
		},
		{
			// same as the previous test, but the filter is unresolvable.
			description:  "basic pod preemption with unresolvable filter",
			initTokens:   1,
			unresolvable: true,
			existingPods: []*v1.Pod{
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{},
		},
		{
			description: "preemption is performed to satisfy anti-affinity",
			initTokens:  maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(cs, &pausePodConfig{
					Name: "pod-0", Namespace: testCtx.ns.Name,
					Priority:  &mediumPriority,
					Labels:    map[string]string{"pod": "p0"},
					Resources: defaultPodRes,
				}),
				initPausePod(cs, &pausePodConfig{
					Name: "pod-1", Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Labels:    map[string]string{"pod": "p1"},
					Resources: defaultPodRes,
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "pod",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"preemptor"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				}),
			},
			// A higher priority pod with anti-affinity.
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Labels:    map[string]string{"pod": "preemptor"},
				Resources: defaultPodRes,
				Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "pod",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"p0"},
										},
									},
								},
								TopologyKey: "node",
							},
						},
					},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}, 1: {}},
		},
		{
			// This is similar to the previous case only pod-1 is high priority.
			description: "preemption is not performed when anti-affinity is not satisfied",
			initTokens:  maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(cs, &pausePodConfig{
					Name: "pod-0", Namespace: testCtx.ns.Name,
					Priority:  &mediumPriority,
					Labels:    map[string]string{"pod": "p0"},
					Resources: defaultPodRes,
				}),
				initPausePod(cs, &pausePodConfig{
					Name: "pod-1", Namespace: testCtx.ns.Name,
					Priority:  &highPriority,
					Labels:    map[string]string{"pod": "p1"},
					Resources: defaultPodRes,
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "pod",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"preemptor"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				}),
			},
			// A higher priority pod with anti-affinity.
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Labels:    map[string]string{"pod": "preemptor"},
				Resources: defaultPodRes,
				Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "pod",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"p0"},
										},
									},
								},
								TopologyKey: "node",
							},
						},
					},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{},
		},
	}

	// Create a node with some resources and a label.
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
	}
	node, err := createNode(testCtx.clientSet, "node1", nodeRes)
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}
	nodeLabels := map[string]string{"node": node.Name}
	if err = testutils.AddLabelsToNode(testCtx.clientSet, node.Name, nodeLabels); err != nil {
		t.Fatalf("Cannot add labels to node: %v", err)
	}
	if err = waitForNodeLabels(testCtx.clientSet, node.Name, nodeLabels); err != nil {
		t.Fatalf("Adding labels to node didn't succeed: %v", err)
	}

	for _, test := range tests {
		t.Logf("================ Running test: %v\n", test.description)
		filter.Tokens = test.initTokens
		filter.Unresolvable = test.unresolvable
		pods := make([]*v1.Pod, len(test.existingPods))
		// Create and run existingPods.
		for i, p := range test.existingPods {
			pods[i], err = runPausePod(cs, p)
			if err != nil {
				t.Fatalf("Test [%v]: Error running pause pod: %v", test.description, err)
			}
		}
		// Create the "pod".
		preemptor, err := createPausePod(cs, test.pod)
		if err != nil {
			t.Errorf("Error while creating high priority pod: %v", err)
		}
		// Wait for preemption of pods and make sure the other ones are not preempted.
		for i, p := range pods {
			if _, found := test.preemptedPodIndexes[i]; found {
				if err = wait.Poll(time.Second, wait.ForeverTestTimeout, podIsGettingEvicted(cs, p.Namespace, p.Name)); err != nil {
					t.Errorf("Test [%v]: Pod %v/%v is not getting evicted.", test.description, p.Namespace, p.Name)
				}
			} else {
				if p.DeletionTimestamp != nil {
					t.Errorf("Test [%v]: Didn't expect pod %v to get preempted.", test.description, p.Name)
				}
			}
		}
		// Also check that the preemptor pod gets the NominatedNodeName field set.
		if len(test.preemptedPodIndexes) > 0 {
			if err := waitForNominatedNodeName(cs, preemptor); err != nil {
				t.Errorf("Test [%v]: NominatedNodeName field was not set for pod %v: %v", test.description, preemptor.Name, err)
			}
		}

		// Cleanup
		pods = append(pods, preemptor)
		cleanupPods(cs, t, pods)
	}
}

// TestDisablePreemption tests disable pod preemption of scheduler works as expected.
func TestDisablePreemption(t *testing.T) {
	// Initialize scheduler, and disable preemption.
	testCtx := initTestDisablePreemption(t, "disable-preemption")
	defer cleanupTest(t, testCtx)
	cs := testCtx.clientSet

	tests := []struct {
		description  string
		existingPods []*v1.Pod
		pod          *v1.Pod
	}{
		{
			description: "pod preemption will not happen",
			existingPods: []*v1.Pod{
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
		},
	}

	// Create a node with some resources and a label.
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
	}
	_, err := createNode(testCtx.clientSet, "node1", nodeRes)
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}

	for _, test := range tests {
		pods := make([]*v1.Pod, len(test.existingPods))
		// Create and run existingPods.
		for i, p := range test.existingPods {
			pods[i], err = runPausePod(cs, p)
			if err != nil {
				t.Fatalf("Test [%v]: Error running pause pod: %v", test.description, err)
			}
		}
		// Create the "pod".
		preemptor, err := createPausePod(cs, test.pod)
		if err != nil {
			t.Errorf("Error while creating high priority pod: %v", err)
		}
		// Ensure preemptor should keep unschedulable.
		if err := waitForPodUnschedulable(cs, preemptor); err != nil {
			t.Errorf("Test [%v]: Preemptor %v should not become scheduled",
				test.description, preemptor.Name)
		}

		// Ensure preemptor should not be nominated.
		if err := waitForNominatedNodeNameWithTimeout(cs, preemptor, 5*time.Second); err == nil {
			t.Errorf("Test [%v]: Preemptor %v should not be nominated",
				test.description, preemptor.Name)
		}

		// Cleanup
		pods = append(pods, preemptor)
		cleanupPods(cs, t, pods)
	}
}

// This test verifies that system critical priorities are created automatically and resolved properly.
func TestPodPriorityResolution(t *testing.T) {
	admission := priority.NewPlugin()
	testCtx := initTestScheduler(t, initTestMaster(t, "preemption", admission), true, nil)
	defer cleanupTest(t, testCtx)
	cs := testCtx.clientSet

	// Build clientset and informers for controllers.
	externalClientset := kubernetes.NewForConfigOrDie(&restclient.Config{
		QPS:           -1,
		Host:          testCtx.httpServer.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)
	admission.SetExternalKubeClientSet(externalClientset)
	admission.SetExternalKubeInformerFactory(externalInformers)
	externalInformers.Start(testCtx.ctx.Done())
	externalInformers.WaitForCacheSync(testCtx.ctx.Done())

	tests := []struct {
		Name             string
		PriorityClass    string
		Pod              *v1.Pod
		ExpectedPriority int32
		ExpectedError    error
	}{
		{
			Name:             "SystemNodeCritical priority class",
			PriorityClass:    scheduling.SystemNodeCritical,
			ExpectedPriority: scheduling.SystemCriticalPriority + 1000,
			Pod: initPausePod(cs, &pausePodConfig{
				Name:              fmt.Sprintf("pod1-%v", scheduling.SystemNodeCritical),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: scheduling.SystemNodeCritical,
			}),
		},
		{
			Name:             "SystemClusterCritical priority class",
			PriorityClass:    scheduling.SystemClusterCritical,
			ExpectedPriority: scheduling.SystemCriticalPriority,
			Pod: initPausePod(cs, &pausePodConfig{
				Name:              fmt.Sprintf("pod2-%v", scheduling.SystemClusterCritical),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: scheduling.SystemClusterCritical,
			}),
		},
		{
			Name:             "Invalid priority class should result in error",
			PriorityClass:    "foo",
			ExpectedPriority: scheduling.SystemCriticalPriority,
			Pod: initPausePod(cs, &pausePodConfig{
				Name:              fmt.Sprintf("pod3-%v", scheduling.SystemClusterCritical),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: "foo",
			}),
			ExpectedError: fmt.Errorf("Error creating pause pod: pods \"pod3-system-cluster-critical\" is forbidden: no PriorityClass with name foo was found"),
		},
	}

	// Create a node with some resources and a label.
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
	}
	_, err := createNode(testCtx.clientSet, "node1", nodeRes)
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}

	pods := make([]*v1.Pod, 0, len(tests))
	for _, test := range tests {
		t.Logf("================ Running test: %v\n", test.Name)
		t.Run(test.Name, func(t *testing.T) {
			pod, err := runPausePod(cs, test.Pod)
			if err != nil {
				if test.ExpectedError == nil {
					t.Fatalf("Test [PodPriority/%v]: Error running pause pod: %v", test.PriorityClass, err)
				}
				if err.Error() != test.ExpectedError.Error() {
					t.Fatalf("Test [PodPriority/%v]: Expected error %v but got error %v", test.PriorityClass, test.ExpectedError, err)
				}
				return
			}
			pods = append(pods, pod)
			if pod.Spec.Priority != nil {
				if *pod.Spec.Priority != test.ExpectedPriority {
					t.Errorf("Expected pod %v to have priority %v but was %v", pod.Name, test.ExpectedPriority, pod.Spec.Priority)
				}
			} else {
				t.Errorf("Expected pod %v to have priority %v but was nil", pod.Name, test.PriorityClass)
			}
		})
	}
	cleanupPods(cs, t, pods)
	cleanupNodes(cs, t)
}

func mkPriorityPodWithGrace(tc *testContext, name string, priority int32, grace int64) *v1.Pod {
	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
	}
	pod := initPausePod(tc.clientSet, &pausePodConfig{
		Name:      name,
		Namespace: tc.ns.Name,
		Priority:  &priority,
		Labels:    map[string]string{"pod": name},
		Resources: defaultPodRes,
	})
	// Setting grace period to zero. Otherwise, we may never see the actual deletion
	// of the pods in integration tests.
	pod.Spec.TerminationGracePeriodSeconds = &grace
	return pod
}

// This test ensures that while the preempting pod is waiting for the victims to
// terminate, other pending lower priority pods are not scheduled in the room created
// after preemption and while the higher priority pods is not scheduled yet.
func TestPreemptionStarvation(t *testing.T) {
	// Initialize scheduler.
	testCtx := initTest(t, "preemption")
	defer cleanupTest(t, testCtx)
	cs := testCtx.clientSet

	tests := []struct {
		description        string
		numExistingPod     int
		numExpectedPending int
		preemptor          *v1.Pod
	}{
		{
			// This test ensures that while the preempting pod is waiting for the victims
			// terminate, other lower priority pods are not scheduled in the room created
			// after preemption and while the higher priority pods is not scheduled yet.
			description:        "starvation test: higher priority pod is scheduled before the lower priority ones",
			numExistingPod:     10,
			numExpectedPending: 5,
			preemptor: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
		},
	}

	// Create a node with some resources and a label.
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
	}
	_, err := createNode(testCtx.clientSet, "node1", nodeRes)
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}

	for _, test := range tests {
		pendingPods := make([]*v1.Pod, test.numExpectedPending)
		numRunningPods := test.numExistingPod - test.numExpectedPending
		runningPods := make([]*v1.Pod, numRunningPods)
		// Create and run existingPods.
		for i := 0; i < numRunningPods; i++ {
			runningPods[i], err = createPausePod(cs, mkPriorityPodWithGrace(testCtx, fmt.Sprintf("rpod-%v", i), mediumPriority, 0))
			if err != nil {
				t.Fatalf("Test [%v]: Error creating pause pod: %v", test.description, err)
			}
		}
		// make sure that runningPods are all scheduled.
		for _, p := range runningPods {
			if err := waitForPodToSchedule(cs, p); err != nil {
				t.Fatalf("Pod %v/%v didn't get scheduled: %v", p.Namespace, p.Name, err)
			}
		}
		// Create pending pods.
		for i := 0; i < test.numExpectedPending; i++ {
			pendingPods[i], err = createPausePod(cs, mkPriorityPodWithGrace(testCtx, fmt.Sprintf("ppod-%v", i), mediumPriority, 0))
			if err != nil {
				t.Fatalf("Test [%v]: Error creating pending pod: %v", test.description, err)
			}
		}
		// Make sure that all pending pods are being marked unschedulable.
		for _, p := range pendingPods {
			if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout,
				podUnschedulable(cs, p.Namespace, p.Name)); err != nil {
				t.Errorf("Pod %v/%v didn't get marked unschedulable: %v", p.Namespace, p.Name, err)
			}
		}
		// Create the preemptor.
		preemptor, err := createPausePod(cs, test.preemptor)
		if err != nil {
			t.Errorf("Error while creating the preempting pod: %v", err)
		}
		// Check that the preemptor pod gets the annotation for nominated node name.
		if err := waitForNominatedNodeName(cs, preemptor); err != nil {
			t.Errorf("Test [%v]: NominatedNodeName annotation was not set for pod %v/%v: %v", test.description, preemptor.Namespace, preemptor.Name, err)
		}
		// Make sure that preemptor is scheduled after preemptions.
		if err := waitForPodToScheduleWithTimeout(cs, preemptor, 60*time.Second); err != nil {
			t.Errorf("Preemptor pod %v didn't get scheduled: %v", preemptor.Name, err)
		}
		// Cleanup
		klog.Info("Cleaning up all pods...")
		allPods := pendingPods
		allPods = append(allPods, runningPods...)
		allPods = append(allPods, preemptor)
		cleanupPods(cs, t, allPods)
	}
}

// TestPreemptionRaces tests that other scheduling events and operations do not
// race with the preemption process.
func TestPreemptionRaces(t *testing.T) {
	// Initialize scheduler.
	testCtx := initTest(t, "preemption-race")
	defer cleanupTest(t, testCtx)
	cs := testCtx.clientSet

	tests := []struct {
		description       string
		numInitialPods    int // Pods created and executed before running preemptor
		numAdditionalPods int // Pods created after creating the preemptor
		numRepetitions    int // Repeat the tests to check races
		preemptor         *v1.Pod
	}{
		{
			// This test ensures that while the preempting pod is waiting for the victims
			// terminate, other lower priority pods are not scheduled in the room created
			// after preemption and while the higher priority pods is not scheduled yet.
			description:       "ensures that other pods are not scheduled while preemptor is being marked as nominated (issue #72124)",
			numInitialPods:    2,
			numAdditionalPods: 50,
			numRepetitions:    10,
			preemptor: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(4900, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(4900, resource.DecimalSI)},
				},
			}),
		},
	}

	// Create a node with some resources and a label.
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(100, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(5000, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(5000, resource.DecimalSI),
	}
	_, err := createNode(testCtx.clientSet, "node1", nodeRes)
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}

	for _, test := range tests {
		if test.numRepetitions <= 0 {
			test.numRepetitions = 1
		}
		for n := 0; n < test.numRepetitions; n++ {
			initialPods := make([]*v1.Pod, test.numInitialPods)
			additionalPods := make([]*v1.Pod, test.numAdditionalPods)
			// Create and run existingPods.
			for i := 0; i < test.numInitialPods; i++ {
				initialPods[i], err = createPausePod(cs, mkPriorityPodWithGrace(testCtx, fmt.Sprintf("rpod-%v", i), mediumPriority, 0))
				if err != nil {
					t.Fatalf("Test [%v]: Error creating pause pod: %v", test.description, err)
				}
			}
			// make sure that initial Pods are all scheduled.
			for _, p := range initialPods {
				if err := waitForPodToSchedule(cs, p); err != nil {
					t.Fatalf("Pod %v/%v didn't get scheduled: %v", p.Namespace, p.Name, err)
				}
			}
			// Create the preemptor.
			klog.Info("Creating the preemptor pod...")
			preemptor, err := createPausePod(cs, test.preemptor)
			if err != nil {
				t.Errorf("Error while creating the preempting pod: %v", err)
			}

			klog.Info("Creating additional pods...")
			for i := 0; i < test.numAdditionalPods; i++ {
				additionalPods[i], err = createPausePod(cs, mkPriorityPodWithGrace(testCtx, fmt.Sprintf("ppod-%v", i), mediumPriority, 0))
				if err != nil {
					t.Fatalf("Test [%v]: Error creating pending pod: %v", test.description, err)
				}
			}
			// Check that the preemptor pod gets nominated node name.
			if err := waitForNominatedNodeName(cs, preemptor); err != nil {
				t.Errorf("Test [%v]: NominatedNodeName annotation was not set for pod %v/%v: %v", test.description, preemptor.Namespace, preemptor.Name, err)
			}
			// Make sure that preemptor is scheduled after preemptions.
			if err := waitForPodToScheduleWithTimeout(cs, preemptor, 60*time.Second); err != nil {
				t.Errorf("Preemptor pod %v didn't get scheduled: %v", preemptor.Name, err)
			}

			klog.Info("Check unschedulable pods still exists and were never scheduled...")
			for _, p := range additionalPods {
				pod, err := cs.CoreV1().Pods(p.Namespace).Get(p.Name, metav1.GetOptions{})
				if err != nil {
					t.Errorf("Error in getting Pod %v/%v info: %v", p.Namespace, p.Name, err)
				}
				if len(pod.Spec.NodeName) > 0 {
					t.Errorf("Pod %v/%v is already scheduled", p.Namespace, p.Name)
				}
				_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
				if cond != nil && cond.Status != v1.ConditionFalse {
					t.Errorf("Pod %v/%v is no longer unschedulable: %v", p.Namespace, p.Name, err)
				}
			}
			// Cleanup
			klog.Info("Cleaning up all pods...")
			allPods := additionalPods
			allPods = append(allPods, initialPods...)
			allPods = append(allPods, preemptor)
			cleanupPods(cs, t, allPods)
		}
	}
}

// TestNominatedNodeCleanUp checks that when there are nominated pods on a
// node and a higher priority pod is nominated to run on the node, the nominated
// node name of the lower priority pods is cleared.
// Test scenario:
// 1. Create a few low priority pods with long grade period that fill up a node.
// 2. Create a medium priority pod that preempt some of those pods.
// 3. Check that nominated node name of the medium priority pod is set.
// 4. Create a high priority pod that preempts some pods on that node.
// 5. Check that nominated node name of the high priority pod is set and nominated
//    node name of the medium priority pod is cleared.
func TestNominatedNodeCleanUp(t *testing.T) {
	// Initialize scheduler.
	testCtx := initTest(t, "preemption")
	defer cleanupTest(t, testCtx)

	cs := testCtx.clientSet

	defer cleanupPodsInNamespace(cs, t, testCtx.ns.Name)

	// Create a node with some resources and a label.
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
	}
	_, err := createNode(testCtx.clientSet, "node1", nodeRes)
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}

	// Step 1. Create a few low priority pods.
	lowPriPods := make([]*v1.Pod, 4)
	for i := 0; i < len(lowPriPods); i++ {
		lowPriPods[i], err = createPausePod(cs, mkPriorityPodWithGrace(testCtx, fmt.Sprintf("lpod-%v", i), lowPriority, 60))
		if err != nil {
			t.Fatalf("Error creating pause pod: %v", err)
		}
	}
	// make sure that the pods are all scheduled.
	for _, p := range lowPriPods {
		if err := waitForPodToSchedule(cs, p); err != nil {
			t.Fatalf("Pod %v/%v didn't get scheduled: %v", p.Namespace, p.Name, err)
		}
	}
	// Step 2. Create a medium priority pod.
	podConf := initPausePod(cs, &pausePodConfig{
		Name:      "medium-priority",
		Namespace: testCtx.ns.Name,
		Priority:  &mediumPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(400, resource.DecimalSI)},
		},
	})
	medPriPod, err := createPausePod(cs, podConf)
	if err != nil {
		t.Errorf("Error while creating the medium priority pod: %v", err)
	}
	// Step 3. Check that nominated node name of the medium priority pod is set.
	if err := waitForNominatedNodeName(cs, medPriPod); err != nil {
		t.Errorf("NominatedNodeName annotation was not set for pod %v/%v: %v", medPriPod.Namespace, medPriPod.Name, err)
	}
	// Step 4. Create a high priority pod.
	podConf = initPausePod(cs, &pausePodConfig{
		Name:      "high-priority",
		Namespace: testCtx.ns.Name,
		Priority:  &highPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
		},
	})
	highPriPod, err := createPausePod(cs, podConf)
	if err != nil {
		t.Errorf("Error while creating the high priority pod: %v", err)
	}
	// Step 5. Check that nominated node name of the high priority pod is set.
	if err := waitForNominatedNodeName(cs, highPriPod); err != nil {
		t.Errorf("NominatedNodeName annotation was not set for pod %v/%v: %v", medPriPod.Namespace, medPriPod.Name, err)
	}
	// And the nominated node name of the medium priority pod is cleared.
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		pod, err := cs.CoreV1().Pods(medPriPod.Namespace).Get(medPriPod.Name, metav1.GetOptions{})
		if err != nil {
			t.Errorf("Error getting the medium priority pod info: %v", err)
		}
		if len(pod.Status.NominatedNodeName) == 0 {
			return true, nil
		}
		return false, err
	}); err != nil {
		t.Errorf("The nominated node name of the medium priority pod was not cleared: %v", err)
	}
}

func mkMinAvailablePDB(name, namespace string, uid types.UID, minAvailable int, matchLabels map[string]string) *policy.PodDisruptionBudget {
	intMinAvailable := intstr.FromInt(minAvailable)
	return &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable: &intMinAvailable,
			Selector:     &metav1.LabelSelector{MatchLabels: matchLabels},
		},
	}
}

func addPodConditionReady(pod *v1.Pod) {
	pod.Status = v1.PodStatus{
		Phase: v1.PodRunning,
		Conditions: []v1.PodCondition{
			{
				Type:   v1.PodReady,
				Status: v1.ConditionTrue,
			},
		},
	}
}

// TestPDBInPreemption tests PodDisruptionBudget support in preemption.
func TestPDBInPreemption(t *testing.T) {
	// Initialize scheduler.
	testCtx := initTest(t, "preemption-pdb")
	defer cleanupTest(t, testCtx)
	cs := testCtx.clientSet

	initDisruptionController(t, testCtx)

	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
	}
	defaultNodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
	}

	type nodeConfig struct {
		name string
		res  *v1.ResourceList
	}

	tests := []struct {
		description         string
		nodes               []*nodeConfig
		pdbs                []*policy.PodDisruptionBudget
		pdbPodNum           []int32
		existingPods        []*v1.Pod
		pod                 *v1.Pod
		preemptedPodIndexes map[int]struct{}
	}{
		{
			description: "A non-PDB violating pod is preempted despite its higher priority",
			nodes:       []*nodeConfig{{name: "node-1", res: defaultNodeRes}},
			pdbs: []*policy.PodDisruptionBudget{
				mkMinAvailablePDB("pdb-1", testCtx.ns.Name, types.UID("pdb-1-uid"), 2, map[string]string{"foo": "bar"}),
			},
			pdbPodNum: []int32{2},
			existingPods: []*v1.Pod{
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "low-pod1",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					Labels:    map[string]string{"foo": "bar"},
				}),
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "low-pod2",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					Labels:    map[string]string{"foo": "bar"},
				}),
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "mid-pod3",
					Namespace: testCtx.ns.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
				}),
			},
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{2: {}},
		},
		{
			description: "A node without any PDB violating pods is preferred for preemption",
			nodes: []*nodeConfig{
				{name: "node-1", res: defaultNodeRes},
				{name: "node-2", res: defaultNodeRes},
			},
			pdbs: []*policy.PodDisruptionBudget{
				mkMinAvailablePDB("pdb-1", testCtx.ns.Name, types.UID("pdb-1-uid"), 2, map[string]string{"foo": "bar"}),
			},
			pdbPodNum: []int32{1},
			existingPods: []*v1.Pod{
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "low-pod1",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
					Labels:    map[string]string{"foo": "bar"},
				}),
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "mid-pod2",
					Namespace: testCtx.ns.Name,
					Priority:  &mediumPriority,
					NodeName:  "node-2",
					Resources: defaultPodRes,
				}),
			},
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{1: {}},
		},
		{
			description: "A node with fewer PDB violating pods is preferred for preemption",
			nodes: []*nodeConfig{
				{name: "node-1", res: defaultNodeRes},
				{name: "node-2", res: defaultNodeRes},
				{name: "node-3", res: defaultNodeRes},
			},
			pdbs: []*policy.PodDisruptionBudget{
				mkMinAvailablePDB("pdb-1", testCtx.ns.Name, types.UID("pdb-1-uid"), 2, map[string]string{"foo1": "bar"}),
				mkMinAvailablePDB("pdb-2", testCtx.ns.Name, types.UID("pdb-2-uid"), 2, map[string]string{"foo2": "bar"}),
			},
			pdbPodNum: []int32{1, 5},
			existingPods: []*v1.Pod{
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "low-pod1",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
					Labels:    map[string]string{"foo1": "bar"},
				}),
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "mid-pod1",
					Namespace: testCtx.ns.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
				}),
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "low-pod2",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-2",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "mid-pod2",
					Namespace: testCtx.ns.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
					NodeName:  "node-2",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "low-pod4",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-3",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "low-pod5",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-3",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(testCtx.clientSet, &pausePodConfig{
					Name:      "low-pod6",
					Namespace: testCtx.ns.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-3",
					Labels:    map[string]string{"foo2": "bar"},
				}),
			},
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.ns.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(400, resource.DecimalSI)},
				},
			}),
			// The third node is chosen because PDB is not violated for node 3 and the victims have lower priority than node-2.
			preemptedPodIndexes: map[int]struct{}{4: {}, 5: {}, 6: {}},
		},
	}

	for _, test := range tests {
		t.Logf("================ Running test: %v\n", test.description)
		for _, nodeConf := range test.nodes {
			_, err := createNode(cs, nodeConf.name, nodeConf.res)
			if err != nil {
				t.Fatalf("Error creating node %v: %v", nodeConf.name, err)
			}
		}

		pods := make([]*v1.Pod, len(test.existingPods))
		var err error
		// Create and run existingPods.
		for i, p := range test.existingPods {
			if pods[i], err = runPausePod(cs, p); err != nil {
				t.Fatalf("Test [%v]: Error running pause pod: %v", test.description, err)
			}
			// Add pod condition ready so that PDB is updated.
			addPodConditionReady(p)
			if _, err := testCtx.clientSet.CoreV1().Pods(testCtx.ns.Name).UpdateStatus(p); err != nil {
				t.Fatal(err)
			}
		}
		// Wait for Pods to be stable in scheduler cache.
		if err := waitCachedPodsStable(testCtx, test.existingPods); err != nil {
			t.Fatalf("Not all pods are stable in the cache: %v", err)
		}

		// Create PDBs.
		for _, pdb := range test.pdbs {
			_, err := testCtx.clientSet.PolicyV1beta1().PodDisruptionBudgets(testCtx.ns.Name).Create(pdb)
			if err != nil {
				t.Fatalf("Failed to create PDB: %v", err)
			}
		}
		// Wait for PDBs to become stable.
		if err := waitForPDBsStable(testCtx, test.pdbs, test.pdbPodNum); err != nil {
			t.Fatalf("Not all pdbs are stable in the cache: %v", err)
		}

		// Create the "pod".
		preemptor, err := createPausePod(cs, test.pod)
		if err != nil {
			t.Errorf("Error while creating high priority pod: %v", err)
		}
		// Wait for preemption of pods and make sure the other ones are not preempted.
		for i, p := range pods {
			if _, found := test.preemptedPodIndexes[i]; found {
				if err = wait.Poll(time.Second, wait.ForeverTestTimeout, podIsGettingEvicted(cs, p.Namespace, p.Name)); err != nil {
					t.Errorf("Test [%v]: Pod %v/%v is not getting evicted.", test.description, p.Namespace, p.Name)
				}
			} else {
				if p.DeletionTimestamp != nil {
					t.Errorf("Test [%v]: Didn't expect pod %v/%v to get preempted.", test.description, p.Namespace, p.Name)
				}
			}
		}
		// Also check that the preemptor pod gets the annotation for nominated node name.
		if len(test.preemptedPodIndexes) > 0 {
			if err := waitForNominatedNodeName(cs, preemptor); err != nil {
				t.Errorf("Test [%v]: NominatedNodeName annotation was not set for pod %v/%v: %v", test.description, preemptor.Namespace, preemptor.Name, err)
			}
		}

		// Cleanup
		pods = append(pods, preemptor)
		cleanupPods(cs, t, pods)
		cs.PolicyV1beta1().PodDisruptionBudgets(testCtx.ns.Name).DeleteCollection(nil, metav1.ListOptions{})
		cs.CoreV1().Nodes().DeleteCollection(nil, metav1.ListOptions{})
	}
}
