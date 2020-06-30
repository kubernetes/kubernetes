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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/plugin/pkg/admission/priority"
	testutils "k8s.io/kubernetes/test/integration/util"
)

var lowPriority, mediumPriority, highPriority = int32(100), int32(200), int32(300)

func waitForNominatedNodeNameWithTimeout(cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	if err := wait.Poll(100*time.Millisecond, timeout, func() (bool, error) {
		pod, err := cs.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
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
	nodeInfo *framework.NodeInfo) *framework.Status {
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
	podToAdd *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	fp.Tokens--
	return nil
}

func (fp *tokenFilter) RemovePod(ctx context.Context, state *framework.CycleState, podToSchedule *v1.Pod,
	podToRemove *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
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
	registry := make(frameworkruntime.Registry)
	err := registry.Register(filterPluginName, func(_ runtime.Object, fh framework.FrameworkHandle) (framework.Plugin, error) {
		return &filter, nil
	})
	if err != nil {
		t.Fatalf("Error registering a filter: %v", err)
	}
	prof := schedulerconfig.KubeSchedulerProfile{
		SchedulerName: v1.DefaultSchedulerName,
		Plugins: &schedulerconfig.Plugins{
			Filter: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{
					{Name: filterPluginName},
				},
			},
			PreFilter: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{
					{Name: filterPluginName},
				},
			},
		},
	}
	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestMaster(t, "preemptiom", nil),
		false, nil, time.Second,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet

	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
	}

	maxTokens := 1000
	tests := []struct {
		name                string
		existingPods        []*v1.Pod
		pod                 *v1.Pod
		initTokens          int
		unresolvable        bool
		preemptedPodIndexes map[int]struct{}
	}{
		{
			name:       "basic pod preemption",
			initTokens: maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}},
		},
		{
			name:       "basic pod preemption with filter",
			initTokens: 1,
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
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
			name:         "basic pod preemption with unresolvable filter",
			initTokens:   1,
			unresolvable: true,
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{},
		},
		{
			name:       "preemption is performed to satisfy anti-affinity",
			initTokens: maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name: "pod-0", Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Labels:    map[string]string{"pod": "p0"},
					Resources: defaultPodRes,
				}),
				initPausePod(&pausePodConfig{
					Name: "pod-1", Namespace: testCtx.NS.Name,
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
			pod: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
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
			name:       "preemption is not performed when anti-affinity is not satisfied",
			initTokens: maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name: "pod-0", Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Labels:    map[string]string{"pod": "p0"},
					Resources: defaultPodRes,
				}),
				initPausePod(&pausePodConfig{
					Name: "pod-1", Namespace: testCtx.NS.Name,
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
			pod: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
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
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	nodeObject := st.MakeNode().Name("node1").Capacity(nodeRes).Label("node", "node1").Obj()
	if _, err := createNode(testCtx.ClientSet, nodeObject); err != nil {
		t.Fatalf("Error creating node: %v", err)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filter.Tokens = test.initTokens
			filter.Unresolvable = test.unresolvable
			pods := make([]*v1.Pod, len(test.existingPods))
			// Create and run existingPods.
			for i, p := range test.existingPods {
				pods[i], err = runPausePod(cs, p)
				if err != nil {
					t.Fatalf("Error running pause pod: %v", err)
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
						t.Errorf("Pod %v/%v is not getting evicted.", p.Namespace, p.Name)
					}
				} else {
					if p.DeletionTimestamp != nil {
						t.Errorf("Didn't expect pod %v to get preempted.", p.Name)
					}
				}
			}
			// Also check that the preemptor pod gets the NominatedNodeName field set.
			if len(test.preemptedPodIndexes) > 0 {
				if err := waitForNominatedNodeName(cs, preemptor); err != nil {
					t.Errorf("NominatedNodeName field was not set for pod %v: %v", preemptor.Name, err)
				}
			}

			// Cleanup
			pods = append(pods, preemptor)
			testutils.CleanupPods(cs, t, pods)
		})
	}
}

// TestNonPreemption tests NonPreempt option of PriorityClass of scheduler works as expected.
func TestNonPreemption(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NonPreemptingPriority, true)()

	var preemptNever = v1.PreemptNever
	// Initialize scheduler.
	testCtx := initTest(t, "non-preemption")
	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet
	tests := []struct {
		name             string
		PreemptionPolicy *v1.PreemptionPolicy
	}{
		{
			name:             "pod preemption will happen",
			PreemptionPolicy: nil,
		},
		{
			name:             "pod preemption will not happen",
			PreemptionPolicy: &preemptNever,
		},
	}
	victim := initPausePod(&pausePodConfig{
		Name:      "victim-pod",
		Namespace: testCtx.NS.Name,
		Priority:  &lowPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
		},
	})

	preemptor := initPausePod(&pausePodConfig{
		Name:      "preemptor-pod",
		Namespace: testCtx.NS.Name,
		Priority:  &highPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
		},
	})

	// Create a node with some resources
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	_, err := createNode(testCtx.ClientSet, st.MakeNode().Name("node1").Capacity(nodeRes).Obj())
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer testutils.CleanupPods(cs, t, []*v1.Pod{preemptor, victim})
			preemptor.Spec.PreemptionPolicy = test.PreemptionPolicy
			victimPod, err := createPausePod(cs, victim)
			if err != nil {
				t.Fatalf("Error while creating victim: %v", err)
			}
			if err := waitForPodToScheduleWithTimeout(cs, victimPod, 5*time.Second); err != nil {
				t.Fatalf("victim %v should be become scheduled", victimPod.Name)
			}

			preemptorPod, err := createPausePod(cs, preemptor)
			if err != nil {
				t.Fatalf("Error while creating preemptor: %v", err)
			}

			err = waitForNominatedNodeNameWithTimeout(cs, preemptorPod, 5*time.Second)
			// test.PreemptionPolicy == nil means we expect the preemptor to be nominated.
			expect := test.PreemptionPolicy == nil
			// err == nil indicates the preemptor is indeed nominated.
			got := err == nil
			if got != expect {
				t.Errorf("Expect preemptor to be nominated=%v, but got=%v", expect, got)
			}
		})
	}
}

// TestDisablePreemption tests disable pod preemption of scheduler works as expected.
func TestDisablePreemption(t *testing.T) {
	// Initialize scheduler, and disable preemption.
	testCtx := initTestDisablePreemption(t, "disable-preemption")
	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet

	tests := []struct {
		name         string
		existingPods []*v1.Pod
		pod          *v1.Pod
	}{
		{
			name: "pod preemption will not happen",
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
		},
	}

	// Create a node with some resources
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	_, err := createNode(testCtx.ClientSet, st.MakeNode().Name("node1").Capacity(nodeRes).Obj())
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pods := make([]*v1.Pod, len(test.existingPods))
			// Create and run existingPods.
			for i, p := range test.existingPods {
				pods[i], err = runPausePod(cs, p)
				if err != nil {
					t.Fatalf("Test [%v]: Error running pause pod: %v", test.name, err)
				}
			}
			// Create the "pod".
			preemptor, err := createPausePod(cs, test.pod)
			if err != nil {
				t.Errorf("Error while creating high priority pod: %v", err)
			}
			// Ensure preemptor should keep unschedulable.
			if err := waitForPodUnschedulable(cs, preemptor); err != nil {
				t.Errorf("Preemptor %v should not become scheduled", preemptor.Name)
			}

			// Ensure preemptor should not be nominated.
			if err := waitForNominatedNodeNameWithTimeout(cs, preemptor, 5*time.Second); err == nil {
				t.Errorf("Preemptor %v should not be nominated", preemptor.Name)
			}

			// Cleanup
			pods = append(pods, preemptor)
			testutils.CleanupPods(cs, t, pods)
		})
	}
}

// This test verifies that system critical priorities are created automatically and resolved properly.
func TestPodPriorityResolution(t *testing.T) {
	admission := priority.NewPlugin()
	testCtx := testutils.InitTestScheduler(t, testutils.InitTestMaster(t, "preemption", admission), true, nil)
	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet

	// Build clientset and informers for controllers.
	externalClientset := kubernetes.NewForConfigOrDie(&restclient.Config{
		QPS:           -1,
		Host:          testCtx.HTTPServer.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)
	admission.SetExternalKubeClientSet(externalClientset)
	admission.SetExternalKubeInformerFactory(externalInformers)

	// Waiting for all controllers to sync
	testutils.SyncInformerFactory(testCtx)
	externalInformers.Start(testCtx.Ctx.Done())
	externalInformers.WaitForCacheSync(testCtx.Ctx.Done())

	// Run all controllers
	go testCtx.Scheduler.Run(testCtx.Ctx)

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
			Pod: initPausePod(&pausePodConfig{
				Name:              fmt.Sprintf("pod1-%v", scheduling.SystemNodeCritical),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: scheduling.SystemNodeCritical,
			}),
		},
		{
			Name:             "SystemClusterCritical priority class",
			PriorityClass:    scheduling.SystemClusterCritical,
			ExpectedPriority: scheduling.SystemCriticalPriority,
			Pod: initPausePod(&pausePodConfig{
				Name:              fmt.Sprintf("pod2-%v", scheduling.SystemClusterCritical),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: scheduling.SystemClusterCritical,
			}),
		},
		{
			Name:             "Invalid priority class should result in error",
			PriorityClass:    "foo",
			ExpectedPriority: scheduling.SystemCriticalPriority,
			Pod: initPausePod(&pausePodConfig{
				Name:              fmt.Sprintf("pod3-%v", scheduling.SystemClusterCritical),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: "foo",
			}),
			ExpectedError: fmt.Errorf("Error creating pause pod: pods \"pod3-system-cluster-critical\" is forbidden: no PriorityClass with name foo was found"),
		},
	}

	// Create a node with some resources
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	_, err := createNode(testCtx.ClientSet, st.MakeNode().Name("node1").Capacity(nodeRes).Obj())
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}

	pods := make([]*v1.Pod, 0, len(tests))
	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
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
		})
	}
	testutils.CleanupPods(cs, t, pods)
	testutils.CleanupNodes(cs, t)
}

func mkPriorityPodWithGrace(tc *testutils.TestContext, name string, priority int32, grace int64) *v1.Pod {
	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
	}
	pod := initPausePod(&pausePodConfig{
		Name:      name,
		Namespace: tc.NS.Name,
		Priority:  &priority,
		Labels:    map[string]string{"pod": name},
		Resources: defaultPodRes,
	})
	pod.Spec.TerminationGracePeriodSeconds = &grace
	return pod
}

// This test ensures that while the preempting pod is waiting for the victims to
// terminate, other pending lower priority pods are not scheduled in the room created
// after preemption and while the higher priority pods is not scheduled yet.
func TestPreemptionStarvation(t *testing.T) {
	// Initialize scheduler.
	testCtx := initTest(t, "preemption")
	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet

	tests := []struct {
		name               string
		numExistingPod     int
		numExpectedPending int
		preemptor          *v1.Pod
	}{
		{
			// This test ensures that while the preempting pod is waiting for the victims
			// terminate, other lower priority pods are not scheduled in the room created
			// after preemption and while the higher priority pods is not scheduled yet.
			name:               "starvation test: higher priority pod is scheduled before the lower priority ones",
			numExistingPod:     10,
			numExpectedPending: 5,
			preemptor: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
		},
	}

	// Create a node with some resources
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	_, err := createNode(testCtx.ClientSet, st.MakeNode().Name("node1").Capacity(nodeRes).Obj())
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pendingPods := make([]*v1.Pod, test.numExpectedPending)
			numRunningPods := test.numExistingPod - test.numExpectedPending
			runningPods := make([]*v1.Pod, numRunningPods)
			// Create and run existingPods.
			for i := 0; i < numRunningPods; i++ {
				runningPods[i], err = createPausePod(cs, mkPriorityPodWithGrace(testCtx, fmt.Sprintf("rpod-%v", i), mediumPriority, 0))
				if err != nil {
					t.Fatalf("Error creating pause pod: %v", err)
				}
			}
			// make sure that runningPods are all scheduled.
			for _, p := range runningPods {
				if err := testutils.WaitForPodToSchedule(cs, p); err != nil {
					t.Fatalf("Pod %v/%v didn't get scheduled: %v", p.Namespace, p.Name, err)
				}
			}
			// Create pending pods.
			for i := 0; i < test.numExpectedPending; i++ {
				pendingPods[i], err = createPausePod(cs, mkPriorityPodWithGrace(testCtx, fmt.Sprintf("ppod-%v", i), mediumPriority, 0))
				if err != nil {
					t.Fatalf("Error creating pending pod: %v", err)
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
				t.Errorf("NominatedNodeName annotation was not set for pod %v/%v: %v", preemptor.Namespace, preemptor.Name, err)
			}
			// Make sure that preemptor is scheduled after preemptions.
			if err := testutils.WaitForPodToScheduleWithTimeout(cs, preemptor, 60*time.Second); err != nil {
				t.Errorf("Preemptor pod %v didn't get scheduled: %v", preemptor.Name, err)
			}
			// Cleanup
			klog.Info("Cleaning up all pods...")
			allPods := pendingPods
			allPods = append(allPods, runningPods...)
			allPods = append(allPods, preemptor)
			testutils.CleanupPods(cs, t, allPods)
		})
	}
}

// TestPreemptionRaces tests that other scheduling events and operations do not
// race with the preemption process.
func TestPreemptionRaces(t *testing.T) {
	// Initialize scheduler.
	testCtx := initTest(t, "preemption-race")
	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet

	tests := []struct {
		name              string
		numInitialPods    int // Pods created and executed before running preemptor
		numAdditionalPods int // Pods created after creating the preemptor
		numRepetitions    int // Repeat the tests to check races
		preemptor         *v1.Pod
	}{
		{
			// This test ensures that while the preempting pod is waiting for the victims
			// terminate, other lower priority pods are not scheduled in the room created
			// after preemption and while the higher priority pods is not scheduled yet.
			name:              "ensures that other pods are not scheduled while preemptor is being marked as nominated (issue #72124)",
			numInitialPods:    2,
			numAdditionalPods: 50,
			numRepetitions:    10,
			preemptor: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(4900, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(4900, resource.DecimalSI)},
				},
			}),
		},
	}

	// Create a node with some resources
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "100",
		v1.ResourceCPU:    "5000m",
		v1.ResourceMemory: "5000",
	}
	_, err := createNode(testCtx.ClientSet, st.MakeNode().Name("node1").Capacity(nodeRes).Obj())
	if err != nil {
		t.Fatalf("Error creating nodes: %v", err)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
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
						t.Fatalf("Error creating pause pod: %v", err)
					}
				}
				// make sure that initial Pods are all scheduled.
				for _, p := range initialPods {
					if err := testutils.WaitForPodToSchedule(cs, p); err != nil {
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
						t.Fatalf("Error creating pending pod: %v", err)
					}
				}
				// Check that the preemptor pod gets nominated node name.
				if err := waitForNominatedNodeName(cs, preemptor); err != nil {
					t.Errorf("NominatedNodeName annotation was not set for pod %v/%v: %v", preemptor.Namespace, preemptor.Name, err)
				}
				// Make sure that preemptor is scheduled after preemptions.
				if err := testutils.WaitForPodToScheduleWithTimeout(cs, preemptor, 60*time.Second); err != nil {
					t.Errorf("Preemptor pod %v didn't get scheduled: %v", preemptor.Name, err)
				}

				klog.Info("Check unschedulable pods still exists and were never scheduled...")
				for _, p := range additionalPods {
					pod, err := cs.CoreV1().Pods(p.Namespace).Get(context.TODO(), p.Name, metav1.GetOptions{})
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
				testutils.CleanupPods(cs, t, allPods)
			}
		})
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
	defer testutils.CleanupTest(t, testCtx)

	cs := testCtx.ClientSet

	defer cleanupPodsInNamespace(cs, t, testCtx.NS.Name)

	// Create a node with some resources
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	_, err := createNode(testCtx.ClientSet, st.MakeNode().Name("node1").Capacity(nodeRes).Obj())
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
		if err := testutils.WaitForPodToSchedule(cs, p); err != nil {
			t.Fatalf("Pod %v/%v didn't get scheduled: %v", p.Namespace, p.Name, err)
		}
	}
	// Step 2. Create a medium priority pod.
	podConf := initPausePod(&pausePodConfig{
		Name:      "medium-priority",
		Namespace: testCtx.NS.Name,
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
	podConf = initPausePod(&pausePodConfig{
		Name:      "high-priority",
		Namespace: testCtx.NS.Name,
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
		t.Errorf("NominatedNodeName annotation was not set for pod %v/%v: %v", highPriPod.Namespace, highPriPod.Name, err)
	}
	// And the nominated node name of the medium priority pod is cleared.
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		pod, err := cs.CoreV1().Pods(medPriPod.Namespace).Get(context.TODO(), medPriPod.Name, metav1.GetOptions{})
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
	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet

	initDisruptionController(t, testCtx)

	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
	}
	defaultNodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}

	type nodeConfig struct {
		name string
		res  map[v1.ResourceName]string
	}

	tests := []struct {
		name                string
		nodes               []*nodeConfig
		pdbs                []*policy.PodDisruptionBudget
		pdbPodNum           []int32
		existingPods        []*v1.Pod
		pod                 *v1.Pod
		preemptedPodIndexes map[int]struct{}
	}{
		{
			name:  "A non-PDB violating pod is preempted despite its higher priority",
			nodes: []*nodeConfig{{name: "node-1", res: defaultNodeRes}},
			pdbs: []*policy.PodDisruptionBudget{
				mkMinAvailablePDB("pdb-1", testCtx.NS.Name, types.UID("pdb-1-uid"), 2, map[string]string{"foo": "bar"}),
			},
			pdbPodNum: []int32{2},
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name:      "low-pod1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					Labels:    map[string]string{"foo": "bar"},
				}),
				initPausePod(&pausePodConfig{
					Name:      "low-pod2",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					Labels:    map[string]string{"foo": "bar"},
				}),
				initPausePod(&pausePodConfig{
					Name:      "mid-pod3",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
				}),
			},
			pod: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{2: {}},
		},
		{
			name: "A node without any PDB violating pods is preferred for preemption",
			nodes: []*nodeConfig{
				{name: "node-1", res: defaultNodeRes},
				{name: "node-2", res: defaultNodeRes},
			},
			pdbs: []*policy.PodDisruptionBudget{
				mkMinAvailablePDB("pdb-1", testCtx.NS.Name, types.UID("pdb-1-uid"), 2, map[string]string{"foo": "bar"}),
			},
			pdbPodNum: []int32{1},
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name:      "low-pod1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
					Labels:    map[string]string{"foo": "bar"},
				}),
				initPausePod(&pausePodConfig{
					Name:      "mid-pod2",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					NodeName:  "node-2",
					Resources: defaultPodRes,
				}),
			},
			pod: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{1: {}},
		},
		{
			name: "A node with fewer PDB violating pods is preferred for preemption",
			nodes: []*nodeConfig{
				{name: "node-1", res: defaultNodeRes},
				{name: "node-2", res: defaultNodeRes},
				{name: "node-3", res: defaultNodeRes},
			},
			pdbs: []*policy.PodDisruptionBudget{
				mkMinAvailablePDB("pdb-1", testCtx.NS.Name, types.UID("pdb-1-uid"), 2, map[string]string{"foo1": "bar"}),
				mkMinAvailablePDB("pdb-2", testCtx.NS.Name, types.UID("pdb-2-uid"), 2, map[string]string{"foo2": "bar"}),
			},
			pdbPodNum: []int32{1, 5},
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name:      "low-pod1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
					Labels:    map[string]string{"foo1": "bar"},
				}),
				initPausePod(&pausePodConfig{
					Name:      "mid-pod1",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
				}),
				initPausePod(&pausePodConfig{
					Name:      "low-pod2",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-2",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(&pausePodConfig{
					Name:      "mid-pod2",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
					NodeName:  "node-2",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(&pausePodConfig{
					Name:      "low-pod4",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-3",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(&pausePodConfig{
					Name:      "low-pod5",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-3",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(&pausePodConfig{
					Name:      "low-pod6",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-3",
					Labels:    map[string]string{"foo2": "bar"},
				}),
			},
			pod: initPausePod(&pausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
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
		t.Run(test.name, func(t *testing.T) {
			for _, nodeConf := range test.nodes {
				_, err := createNode(cs, st.MakeNode().Name(nodeConf.name).Capacity(nodeConf.res).Obj())
				if err != nil {
					t.Fatalf("Error creating node %v: %v", nodeConf.name, err)
				}
			}

			pods := make([]*v1.Pod, len(test.existingPods))
			var err error
			// Create and run existingPods.
			for i, p := range test.existingPods {
				if pods[i], err = runPausePod(cs, p); err != nil {
					t.Fatalf("Test [%v]: Error running pause pod: %v", test.name, err)
				}
				// Add pod condition ready so that PDB is updated.
				addPodConditionReady(p)
				if _, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(context.TODO(), p, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
			}
			// Wait for Pods to be stable in scheduler cache.
			if err := waitCachedPodsStable(testCtx, test.existingPods); err != nil {
				t.Fatalf("Not all pods are stable in the cache: %v", err)
			}

			// Create PDBs.
			for _, pdb := range test.pdbs {
				_, err := testCtx.ClientSet.PolicyV1beta1().PodDisruptionBudgets(testCtx.NS.Name).Create(context.TODO(), pdb, metav1.CreateOptions{})
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
						t.Errorf("Test [%v]: Pod %v/%v is not getting evicted.", test.name, p.Namespace, p.Name)
					}
				} else {
					if p.DeletionTimestamp != nil {
						t.Errorf("Test [%v]: Didn't expect pod %v/%v to get preempted.", test.name, p.Namespace, p.Name)
					}
				}
			}
			// Also check that the preemptor pod gets the annotation for nominated node name.
			if len(test.preemptedPodIndexes) > 0 {
				if err := waitForNominatedNodeName(cs, preemptor); err != nil {
					t.Errorf("Test [%v]: NominatedNodeName annotation was not set for pod %v/%v: %v", test.name, preemptor.Namespace, preemptor.Name, err)
				}
			}

			// Cleanup
			pods = append(pods, preemptor)
			testutils.CleanupPods(cs, t, pods)
			cs.PolicyV1beta1().PodDisruptionBudgets(testCtx.NS.Name).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
			cs.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
		})
	}
}
