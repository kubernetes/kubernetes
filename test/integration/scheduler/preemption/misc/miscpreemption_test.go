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

package miscpreemption

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	policy "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2"
	configv1 "k8s.io/kube-scheduler/config/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/plugin/pkg/admission/priority"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// imported from testutils
var (
	initPausePod                    = testutils.InitPausePod
	createNode                      = testutils.CreateNode
	createPausePod                  = testutils.CreatePausePod
	runPausePod                     = testutils.RunPausePod
	initTest                        = testutils.InitTestSchedulerWithNS
	initTestDisablePreemption       = testutils.InitTestDisablePreemption
	initDisruptionController        = testutils.InitDisruptionController
	waitCachedPodsStable            = testutils.WaitCachedPodsStable
	podIsGettingEvicted             = testutils.PodIsGettingEvicted
	podUnschedulable                = testutils.PodUnschedulable
	waitForPDBsStable               = testutils.WaitForPDBsStable
	waitForPodToScheduleWithTimeout = testutils.WaitForPodToScheduleWithTimeout
	waitForPodUnschedulable         = testutils.WaitForPodUnschedulable
	waitForPodSchedulingGated       = testutils.WaitForPodSchedulingGated
)

var lowPriority, mediumPriority, highPriority = int32(100), int32(200), int32(300)

// TestNonPreemption tests NonPreempt option of PriorityClass of scheduler works as expected.
func TestNonPreemption(t *testing.T) {
	var preemptNever = v1.PreemptNever
	// Initialize scheduler.
	testCtx := initTest(t, "non-preemption")
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
	victim := initPausePod(&testutils.PausePodConfig{
		Name:      "victim-pod",
		Namespace: testCtx.NS.Name,
		Priority:  &lowPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
		},
	})

	preemptor := initPausePod(&testutils.PausePodConfig{
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

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, test := range tests {
			t.Run(fmt.Sprintf("%s (Async preemption enabled: %v)", test.name, asyncPreemptionEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncPreemption, asyncPreemptionEnabled)

				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{preemptor, victim})
				preemptor.Spec.PreemptionPolicy = test.PreemptionPolicy
				victimPod, err := createPausePod(cs, victim)
				if err != nil {
					t.Fatalf("Error while creating victim: %v", err)
				}
				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, victimPod, 5*time.Second); err != nil {
					t.Fatalf("victim %v should be become scheduled", victimPod.Name)
				}

				preemptorPod, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Error while creating preemptor: %v", err)
				}

				err = testutils.WaitForNominatedNodeNameWithTimeout(testCtx.Ctx, cs, preemptorPod, 5*time.Second)
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
}

// TestDisablePreemption tests disable pod preemption of scheduler works as expected.
func TestDisablePreemption(t *testing.T) {
	// Initialize scheduler, and disable preemption.
	testCtx := initTestDisablePreemption(t, "disable-preemption")
	cs := testCtx.ClientSet

	tests := []struct {
		name         string
		existingPods []*v1.Pod
		pod          *v1.Pod
	}{
		{
			name: "pod preemption will not happen",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
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

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, test := range tests {
			t.Run(fmt.Sprintf("%s (Async preemption enabled: %v)", test.name, asyncPreemptionEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncPreemption, asyncPreemptionEnabled)

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
				if err := waitForPodUnschedulable(testCtx.Ctx, cs, preemptor); err != nil {
					t.Errorf("Preemptor %v should not become scheduled", preemptor.Name)
				}

				// Ensure preemptor should not be nominated.
				if err := testutils.WaitForNominatedNodeNameWithTimeout(testCtx.Ctx, cs, preemptor, 5*time.Second); err == nil {
					t.Errorf("Preemptor %v should not be nominated", preemptor.Name)
				}

				// Cleanup
				pods = append(pods, preemptor)
				testutils.CleanupPods(testCtx.Ctx, cs, t, pods)
			})
		}
	}
}

// This test verifies that system critical priorities are created automatically and resolved properly.
func TestPodPriorityResolution(t *testing.T) {
	admission := priority.NewPlugin()
	testCtx := testutils.InitTestScheduler(t, testutils.InitTestAPIServer(t, "preemption", admission))
	cs := testCtx.ClientSet

	// Build clientset and informers for controllers.
	externalClientConfig := restclient.CopyConfig(testCtx.KubeConfig)
	externalClientConfig.QPS = -1
	externalClientset := clientset.NewForConfigOrDie(externalClientConfig)
	externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)
	admission.SetExternalKubeClientSet(externalClientset)
	admission.SetExternalKubeInformerFactory(externalInformers)

	// Waiting for all controllers to sync
	testutils.SyncSchedulerInformerFactory(testCtx)
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
			Pod: initPausePod(&testutils.PausePodConfig{
				Name:              fmt.Sprintf("pod1-%v", scheduling.SystemNodeCritical),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: scheduling.SystemNodeCritical,
			}),
		},
		{
			Name:             "SystemClusterCritical priority class",
			PriorityClass:    scheduling.SystemClusterCritical,
			ExpectedPriority: scheduling.SystemCriticalPriority,
			Pod: initPausePod(&testutils.PausePodConfig{
				Name:              fmt.Sprintf("pod2-%v", scheduling.SystemClusterCritical),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: scheduling.SystemClusterCritical,
			}),
		},
		{
			Name:             "Invalid priority class should result in error",
			PriorityClass:    "foo",
			ExpectedPriority: scheduling.SystemCriticalPriority,
			Pod: initPausePod(&testutils.PausePodConfig{
				Name:              fmt.Sprintf("pod3-%v", scheduling.SystemClusterCritical),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: "foo",
			}),
			ExpectedError: fmt.Errorf("failed to create pause pod: pods \"pod3-system-cluster-critical\" is forbidden: no PriorityClass with name foo was found"),
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
	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, test := range tests {
			t.Run(fmt.Sprintf("%s (Async preemption enabled: %v)", test.Name, asyncPreemptionEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncPreemption, asyncPreemptionEnabled)

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
				testutils.CleanupPods(testCtx.Ctx, cs, t, pods)
			})
		}
	}
	testutils.CleanupNodes(cs, t)
}

func mkPriorityPodWithGrace(tc *testutils.TestContext, name string, priority int32, grace int64) *v1.Pod {
	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
	}
	pod := initPausePod(&testutils.PausePodConfig{
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
			preemptor: initPausePod(&testutils.PausePodConfig{
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

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, clearingNominatedNodeNameAfterBinding := range []bool{true, false} {
			for _, test := range tests {
				t.Run(fmt.Sprintf("%s (Async preemption enabled: %v, ClearingNominatedNodeNameAfterBinding: %v)", test.name, asyncPreemptionEnabled, clearingNominatedNodeNameAfterBinding), func(t *testing.T) {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.SchedulerAsyncPreemption:              asyncPreemptionEnabled,
						features.ClearingNominatedNodeNameAfterBinding: clearingNominatedNodeNameAfterBinding,
					})

					pendingPods := make([]*v1.Pod, test.numExpectedPending)
					numRunningPods := test.numExistingPod - test.numExpectedPending
					runningPods := make([]*v1.Pod, numRunningPods)
					// Create and run existingPods.
					for i := range numRunningPods {
						runningPods[i], err = createPausePod(cs, mkPriorityPodWithGrace(testCtx, fmt.Sprintf("rpod-%v", i), mediumPriority, 0))
						if err != nil {
							t.Fatalf("Error creating pause pod: %v", err)
						}
					}
					// make sure that runningPods are all scheduled.
					for _, p := range runningPods {
						if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, p); err != nil {
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
						if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
							podUnschedulable(cs, p.Namespace, p.Name)); err != nil {
							t.Errorf("Pod %v/%v didn't get marked unschedulable: %v", p.Namespace, p.Name, err)
						}
					}
					// Create the preemptor.
					preemptor, err := createPausePod(cs, test.preemptor)
					if err != nil {
						t.Errorf("Error while creating the preempting pod: %v", err)
					}

					// Make sure that preemptor is scheduled after preemptions.
					if err := testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, cs, preemptor, 60*time.Second); err != nil {
						t.Errorf("Preemptor pod %v didn't get scheduled: %v", preemptor.Name, err)
					}

					// Check if .status.nominatedNodeName of the preemptor pod gets set when feature gate is disabled.
					// This test always expects preemption to occur since numExistingPod (10) fills the node completely.
					if !clearingNominatedNodeNameAfterBinding {
						if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptor); err != nil {
							t.Errorf(".status.nominatedNodeName was not set for pod %v/%v: %v", preemptor.Namespace, preemptor.Name, err)
						}
					}
					// Cleanup
					klog.Info("Cleaning up all pods...")
					allPods := pendingPods
					allPods = append(allPods, runningPods...)
					allPods = append(allPods, preemptor)
					testutils.CleanupPods(testCtx.Ctx, cs, t, allPods)
				})
			}
		}
	}
}

// TestPreemptionRaces tests that other scheduling events and operations do not
// race with the preemption process.
func TestPreemptionRaces(t *testing.T) {
	// Initialize scheduler.
	testCtx := initTest(t, "preemption-race")
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
			numAdditionalPods: 20,
			numRepetitions:    5,
			preemptor: initPausePod(&testutils.PausePodConfig{
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

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, clearingNominatedNodeNameAfterBinding := range []bool{true, false} {
			for _, test := range tests {
				t.Run(fmt.Sprintf("%s (Async preemption enabled: %v, ClearingNominatedNodeNameAfterBinding: %v)", test.name, asyncPreemptionEnabled, clearingNominatedNodeNameAfterBinding), func(t *testing.T) {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.SchedulerAsyncPreemption:              asyncPreemptionEnabled,
						features.ClearingNominatedNodeNameAfterBinding: clearingNominatedNodeNameAfterBinding,
					})

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
							if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, p); err != nil {
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
						// Make sure that preemptor is scheduled after preemptions.
						if err := testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, cs, preemptor, 60*time.Second); err != nil {
							t.Errorf("Preemptor pod %v didn't get scheduled: %v", preemptor.Name, err)
						}

						// Check that the preemptor pod gets nominated node name when feature gate is disabled.
						if !clearingNominatedNodeNameAfterBinding {
							if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptor); err != nil {
								t.Errorf(".status.nominatedNodeName was not set for pod %v/%v: %v", preemptor.Namespace, preemptor.Name, err)
							}
						}

						klog.Info("Check unschedulable pods still exists and were never scheduled...")
						for _, p := range additionalPods {
							pod, err := cs.CoreV1().Pods(p.Namespace).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
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
						testutils.CleanupPods(testCtx.Ctx, cs, t, allPods)
					}
				})
			}
		}
	}
}

func mkMinAvailablePDB(name, namespace string, uid types.UID, minAvailable int, matchLabels map[string]string) *policy.PodDisruptionBudget {
	intMinAvailable := intstr.FromInt32(int32(minAvailable))
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

	tests := []struct {
		name                string
		nodeCnt             int
		pdbs                []*policy.PodDisruptionBudget
		pdbPodNum           []int32
		existingPods        []*v1.Pod
		pod                 *v1.Pod
		preemptedPodIndexes map[int]struct{}
	}{
		{
			name:    "A non-PDB violating pod is preempted despite its higher priority",
			nodeCnt: 1,
			pdbs: []*policy.PodDisruptionBudget{
				mkMinAvailablePDB("pdb-1", testCtx.NS.Name, types.UID("pdb-1-uid"), 2, map[string]string{"foo": "bar"}),
			},
			pdbPodNum: []int32{2},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					Labels:    map[string]string{"foo": "bar"},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod2",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					Labels:    map[string]string{"foo": "bar"},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "mid-pod3",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
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
			name:    "A node without any PDB violating pods is preferred for preemption",
			nodeCnt: 2,
			pdbs: []*policy.PodDisruptionBudget{
				mkMinAvailablePDB("pdb-1", testCtx.NS.Name, types.UID("pdb-1-uid"), 2, map[string]string{"foo": "bar"}),
			},
			pdbPodNum: []int32{1},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
					Labels:    map[string]string{"foo": "bar"},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "mid-pod2",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					NodeName:  "node-2",
					Resources: defaultPodRes,
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
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
			name:    "A node with fewer PDB violating pods is preferred for preemption",
			nodeCnt: 3,
			pdbs: []*policy.PodDisruptionBudget{
				mkMinAvailablePDB("pdb-1", testCtx.NS.Name, types.UID("pdb-1-uid"), 2, map[string]string{"foo1": "bar"}),
				mkMinAvailablePDB("pdb-2", testCtx.NS.Name, types.UID("pdb-2-uid"), 2, map[string]string{"foo2": "bar"}),
			},
			pdbPodNum: []int32{1, 5},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
					Labels:    map[string]string{"foo1": "bar"},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "mid-pod1",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
					NodeName:  "node-1",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod2",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-2",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "mid-pod2",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
					NodeName:  "node-2",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod4",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-3",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod5",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-3",
					Labels:    map[string]string{"foo2": "bar"},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "low-pod6",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Resources: defaultPodRes,
					NodeName:  "node-3",
					Labels:    map[string]string{"foo2": "bar"},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
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

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, clearingNominatedNodeNameAfterBinding := range []bool{true, false} {
			for _, test := range tests {
				t.Run(fmt.Sprintf("%s (Async preemption enabled: %v, ClearingNominatedNodeNameAfterBinding: %v)", test.name, asyncPreemptionEnabled, clearingNominatedNodeNameAfterBinding), func(t *testing.T) {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.SchedulerAsyncPreemption:              asyncPreemptionEnabled,
						features.ClearingNominatedNodeNameAfterBinding: clearingNominatedNodeNameAfterBinding,
					})

					for i := 1; i <= test.nodeCnt; i++ {
						nodeName := fmt.Sprintf("node-%v", i)
						_, err := createNode(cs, st.MakeNode().Name(nodeName).Capacity(defaultNodeRes).Obj())
						if err != nil {
							t.Fatalf("Error creating node %v: %v", nodeName, err)
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
						if _, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, p, metav1.UpdateOptions{}); err != nil {
							t.Fatal(err)
						}
					}
					// Wait for Pods to be stable in scheduler cache.
					if err := waitCachedPodsStable(testCtx, test.existingPods); err != nil {
						t.Fatalf("Not all pods are stable in the cache: %v", err)
					}

					// Create PDBs.
					for _, pdb := range test.pdbs {
						_, err := testCtx.ClientSet.PolicyV1().PodDisruptionBudgets(testCtx.NS.Name).Create(testCtx.Ctx, pdb, metav1.CreateOptions{})
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
							if err = wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, wait.ForeverTestTimeout, false,
								podIsGettingEvicted(cs, p.Namespace, p.Name)); err != nil {
								t.Errorf("Test [%v]: Pod %v/%v is not getting evicted.", test.name, p.Namespace, p.Name)
							}
						} else {
							if p.DeletionTimestamp != nil {
								t.Errorf("Test [%v]: Didn't expect pod %v/%v to get preempted.", test.name, p.Namespace, p.Name)
							}
						}
					}
					// Also check if .status.nominatedNodeName of the preemptor pod gets set.
					if len(test.preemptedPodIndexes) > 0 && !clearingNominatedNodeNameAfterBinding {
						if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptor); err != nil {
							t.Errorf("Test [%v]: .status.nominatedNodeName was not set for pod %v/%v: %v", test.name, preemptor.Namespace, preemptor.Name, err)
						}
					}

					// Cleanup
					pods = append(pods, preemptor)
					testutils.CleanupPods(testCtx.Ctx, cs, t, pods)
					if err := cs.PolicyV1().PodDisruptionBudgets(testCtx.NS.Name).DeleteCollection(testCtx.Ctx, metav1.DeleteOptions{}, metav1.ListOptions{}); err != nil {
						t.Errorf("error while deleting PDBs, error: %v", err)
					}
					if err := cs.CoreV1().Nodes().DeleteCollection(testCtx.Ctx, metav1.DeleteOptions{}, metav1.ListOptions{}); err != nil {
						t.Errorf("error whiling deleting nodes, error: %v", err)
					}
				})
			}
		}
	}
}

// TestReadWriteOncePodPreemption tests preemption scenarios for pods with
// ReadWriteOncePod PVCs.
func TestReadWriteOncePodPreemption(t *testing.T) {
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				Filter: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: volumerestrictions.Name},
					},
				},
				PreFilter: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: volumerestrictions.Name},
					},
				},
			},
		}},
	})

	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "preemption", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...))
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs := testCtx.ClientSet

	storage := v1.VolumeResourceRequirements{Requests: v1.ResourceList{v1.ResourceStorage: resource.MustParse("1Mi")}}
	volType := v1.HostPathDirectoryOrCreate
	pv1 := st.MakePersistentVolume().
		Name("pv-with-read-write-once-pod-1").
		AccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod}).
		Capacity(storage.Requests).
		HostPathVolumeSource(&v1.HostPathVolumeSource{Path: "/mnt1", Type: &volType}).
		Obj()
	pvc1 := st.MakePersistentVolumeClaim().
		Name("pvc-with-read-write-once-pod-1").
		Namespace(testCtx.NS.Name).
		// Annotation and volume name required for PVC to be considered bound.
		Annotation(volume.AnnBindCompleted, "true").
		VolumeName(pv1.Name).
		AccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod}).
		Resources(storage).
		Obj()
	pv2 := st.MakePersistentVolume().
		Name("pv-with-read-write-once-pod-2").
		AccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod}).
		Capacity(storage.Requests).
		HostPathVolumeSource(&v1.HostPathVolumeSource{Path: "/mnt2", Type: &volType}).
		Obj()
	pvc2 := st.MakePersistentVolumeClaim().
		Name("pvc-with-read-write-once-pod-2").
		Namespace(testCtx.NS.Name).
		// Annotation and volume name required for PVC to be considered bound.
		Annotation(volume.AnnBindCompleted, "true").
		VolumeName(pv2.Name).
		AccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod}).
		Resources(storage).
		Obj()

	tests := []struct {
		name                string
		init                func() error
		existingPods        []*v1.Pod
		pod                 *v1.Pod
		unresolvable        bool
		preemptedPodIndexes map[int]struct{}
		cleanup             func() error
	}{
		{
			name: "preempt single pod",
			init: func() error {
				_, err := testutils.CreatePV(cs, pv1)
				if err != nil {
					return fmt.Errorf("cannot create pv: %v", err)
				}
				_, err = testutils.CreatePVC(cs, pvc1)
				if err != nil {
					return fmt.Errorf("cannot create pvc: %v", err)
				}
				return nil
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Volumes: []v1.Volume{{
						Name: "volume",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvc1.Name,
							},
						},
					}},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Volumes: []v1.Volume{{
					Name: "volume",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: pvc1.Name,
						},
					},
				}},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}},
			cleanup: func() error {
				if err := testutils.DeletePVC(cs, pvc1.Name, pvc1.Namespace); err != nil {
					return fmt.Errorf("cannot delete pvc: %v", err)
				}
				if err := testutils.DeletePV(cs, pv1.Name); err != nil {
					return fmt.Errorf("cannot delete pv: %v", err)
				}
				return nil
			},
		},
		{
			name: "preempt two pods",
			init: func() error {
				for _, pv := range []*v1.PersistentVolume{pv1, pv2} {
					_, err := testutils.CreatePV(cs, pv)
					if err != nil {
						return fmt.Errorf("cannot create pv: %v", err)
					}
				}
				for _, pvc := range []*v1.PersistentVolumeClaim{pvc1, pvc2} {
					_, err := testutils.CreatePVC(cs, pvc)
					if err != nil {
						return fmt.Errorf("cannot create pvc: %v", err)
					}
				}
				return nil
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "victim-pod-1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Volumes: []v1.Volume{{
						Name: "volume",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvc1.Name,
							},
						},
					}},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "victim-pod-2",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Volumes: []v1.Volume{{
						Name: "volume",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvc2.Name,
							},
						},
					}},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Volumes: []v1.Volume{
					{
						Name: "volume-1",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvc1.Name,
							},
						},
					},
					{
						Name: "volume-2",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvc2.Name,
							},
						},
					},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}, 1: {}},
			cleanup: func() error {
				for _, pvc := range []*v1.PersistentVolumeClaim{pvc1, pvc2} {
					if err := testutils.DeletePVC(cs, pvc.Name, pvc.Namespace); err != nil {
						return fmt.Errorf("cannot delete pvc: %v", err)
					}
				}
				for _, pv := range []*v1.PersistentVolume{pv1, pv2} {
					if err := testutils.DeletePV(cs, pv.Name); err != nil {
						return fmt.Errorf("cannot delete pv: %v", err)
					}
				}
				return nil
			},
		},
		{
			name: "preempt single pod with two volumes",
			init: func() error {
				for _, pv := range []*v1.PersistentVolume{pv1, pv2} {
					_, err := testutils.CreatePV(cs, pv)
					if err != nil {
						return fmt.Errorf("cannot create pv: %v", err)
					}
				}
				for _, pvc := range []*v1.PersistentVolumeClaim{pvc1, pvc2} {
					_, err := testutils.CreatePVC(cs, pvc)
					if err != nil {
						return fmt.Errorf("cannot create pvc: %v", err)
					}
				}
				return nil
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "victim-pod",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					Volumes: []v1.Volume{
						{
							Name: "volume-1",
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: pvc1.Name,
								},
							},
						},
						{
							Name: "volume-2",
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: pvc2.Name,
								},
							},
						},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Volumes: []v1.Volume{
					{
						Name: "volume-1",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvc1.Name,
							},
						},
					},
					{
						Name: "volume-2",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvc2.Name,
							},
						},
					},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}},
			cleanup: func() error {
				for _, pvc := range []*v1.PersistentVolumeClaim{pvc1, pvc2} {
					if err := testutils.DeletePVC(cs, pvc.Name, pvc.Namespace); err != nil {
						return fmt.Errorf("cannot delete pvc: %v", err)
					}
				}
				for _, pv := range []*v1.PersistentVolume{pv1, pv2} {
					if err := testutils.DeletePV(cs, pv.Name); err != nil {
						return fmt.Errorf("cannot delete pv: %v", err)
					}
				}
				return nil
			},
		},
	}

	// Create a node with some resources and a label.
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	nodeObject := st.MakeNode().Name("node1").Capacity(nodeRes).Label("node", "node1").Obj()
	if _, err := createNode(cs, nodeObject); err != nil {
		t.Fatalf("Error creating node: %v", err)
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, clearingNominatedNodeNameAfterBinding := range []bool{true, false} {
			for _, test := range tests {
				t.Run(fmt.Sprintf("%s (Async preemption enabled: %v, ClearingNominatedNodeNameAfterBinding: %v)", test.name, asyncPreemptionEnabled, clearingNominatedNodeNameAfterBinding), func(t *testing.T) {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.SchedulerAsyncPreemption:              asyncPreemptionEnabled,
						features.ClearingNominatedNodeNameAfterBinding: clearingNominatedNodeNameAfterBinding,
					})

					if err := test.init(); err != nil {
						t.Fatalf("Error while initializing test: %v", err)
					}

					pods := make([]*v1.Pod, len(test.existingPods))
					t.Cleanup(func() {
						testutils.CleanupPods(testCtx.Ctx, cs, t, pods)
						if err := test.cleanup(); err != nil {
							t.Errorf("Error cleaning up test: %v", err)
						}
					})
					// Create and run existingPods.
					for i, p := range test.existingPods {
						var err error
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
					pods = append(pods, preemptor)
					// Wait for preemption of pods and make sure the other ones are not preempted.
					for i, p := range pods {
						if _, found := test.preemptedPodIndexes[i]; found {
							if err = wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, wait.ForeverTestTimeout, false,
								podIsGettingEvicted(cs, p.Namespace, p.Name)); err != nil {
								t.Errorf("Pod %v/%v is not getting evicted.", p.Namespace, p.Name)
							}
						} else {
							if p.DeletionTimestamp != nil {
								t.Errorf("Didn't expect pod %v to get preempted.", p.Name)
							}
						}
					}
					// Also check that the preemptor pod gets the NominatedNodeName field set.
					if len(test.preemptedPodIndexes) > 0 && !clearingNominatedNodeNameAfterBinding {
						if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptor); err != nil {
							t.Errorf("NominatedNodeName field was not set for pod %v: %v", preemptor.Name, err)
						}
					}
				})
			}
		}
	}
}

// TestDeterministicEqualTimestampVictimSelection verifies that when multiple candidate nodes
// host victim pods with identical StartTime timestamps and identical priorities, the scheduler
// deterministically selects the exact same candidate node and victim UID across repeated scheduling runs.
func TestDeterministicEqualTimestampVictimSelection(t *testing.T) {
	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
	}
	defaultNodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "1000m",
		v1.ResourceMemory: "1000",
	}

	fixedStartTime := metav1.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("AsyncPreemptionEnabled_%v", asyncPreemptionEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.SchedulerAsyncPreemption: asyncPreemptionEnabled,
			})

			// Subtest 1: Repeated scheduling cycles maintain stable nominated node (avoiding cache thrashing)
			t.Run("Multi-node identical victim pods deterministic nomination and cache stability", func(t *testing.T) {
				testCtx := initTest(t, "det-multi-node")
				cs := testCtx.ClientSet

				var nodes []string
				for i := 1; i <= 3; i++ {
					nodeName := fmt.Sprintf("node-multi-%d", i)
					_, err := createNode(cs, st.MakeNode().Name(nodeName).Label("kubernetes.io/hostname", nodeName).Capacity(defaultNodeRes).Obj())
					if err != nil {
						t.Fatalf("Error creating node %v: %v", nodeName, err)
					}
					nodes = append(nodes, nodeName)
				}

				var existingPods []*v1.Pod
				for _, nodeName := range nodes {
					for j := 1; j <= 2; j++ {
						p := initPausePod(&testutils.PausePodConfig{
							Name:      fmt.Sprintf("victim-%s-%d", nodeName, j),
							Namespace: testCtx.NS.Name,
							Priority:  &lowPriority,
							NodeName:  nodeName,
							Resources: defaultPodRes,
						})
						pod, err := runPausePod(cs, p)
						if err != nil {
							t.Fatalf("Error running pause pod: %v", err)
						}
						addPodConditionReady(pod)
						pod.Status.StartTime = &fixedStartTime
						if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{}); err != nil {
							t.Fatalf("Error updating pod status: %v", err)
						}
						existingPods = append(existingPods, pod)
					}
				}

				if err := waitCachedPodsStable(testCtx, existingPods); err != nil {
					t.Fatalf("Not all pods are stable in cache: %v", err)
				}

				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "preemptor-multi",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				})
				preemptorPod, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Error creating preemptor: %v", err)
				}

				if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptorPod); err != nil {
					t.Fatalf("Preemptor .status.nominatedNodeName not set: %v", err)
				}

				updatedPreemptor, err := cs.CoreV1().Pods(preemptorPod.Namespace).Get(testCtx.Ctx, preemptorPod.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get preemptor: %v", err)
				}
				nominatedNode := updatedPreemptor.Status.NominatedNodeName
				if nominatedNode == "" {
					t.Fatalf("Expected nominatedNode to be non-empty")
				}

				// Verify that across consecutive scheduler reconciliations, the nominated node remains deterministically unchanged
				for check := 0; check < 5; check++ {
					time.Sleep(100 * time.Millisecond)
					p, err := cs.CoreV1().Pods(preemptorPod.Namespace).Get(testCtx.Ctx, preemptorPod.Name, metav1.GetOptions{})
					if err != nil {
						t.Fatalf("Failed to get preemptor on check %d: %v", check, err)
					}
					if p.Status.NominatedNodeName != nominatedNode {
						t.Errorf("Check %d: NominatedNodeName changed from %q to %q (cache thrashing detected)", check, nominatedNode, p.Status.NominatedNodeName)
					}
				}
			})

			// Subtest 2: Single node victim tie-breaking by UID determinism
			t.Run("Single-node equal start time victim UID tie-breaking", func(t *testing.T) {
				testCtx := initTest(t, "det-single-node")
				cs := testCtx.ClientSet

				nodeName := "node-single-uid"
				_, err := createNode(cs, st.MakeNode().Name(nodeName).Label("kubernetes.io/hostname", nodeName).Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Failed to create node %v: %v", nodeName, err)
				}

				v1Pod := initPausePod(&testutils.PausePodConfig{
					Name:      "equal-time-victim-1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  nodeName,
					Resources: defaultPodRes,
				})
				v2Pod := initPausePod(&testutils.PausePodConfig{
					Name:      "equal-time-victim-2",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  nodeName,
					Resources: defaultPodRes,
				})

				p1, err := runPausePod(cs, v1Pod)
				if err != nil {
					t.Fatalf("Error running p1: %v", err)
				}
				p2, err := runPausePod(cs, v2Pod)
				if err != nil {
					t.Fatalf("Error running p2: %v", err)
				}
				addPodConditionReady(p1)
				addPodConditionReady(p2)
				p1.Status.StartTime = &fixedStartTime
				p2.Status.StartTime = &fixedStartTime
				if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, p1, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
				if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, p2, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{p1, p2}); err != nil {
					t.Fatalf("Pods not stable in cache: %v", err)
				}

				// The pod with smaller UID is more important (reprieved / spared).
				// The pod with larger UID is less important (evicted).
				expectedEvictedName := p2.Name
				expectedSparedName := p1.Name
				if p2.UID < p1.UID {
					expectedEvictedName = p1.Name
					expectedSparedName = p2.Name
				}

				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "preemptor-single-node",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					NodeSelector: map[string]string{
						"kubernetes.io/hostname": nodeName,
					},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				})
				preemptorPod, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Error creating preemptor: %v", err)
				}

				if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptorPod); err != nil {
					t.Fatalf("Preemptor .status.nominatedNodeName not set: %v", err)
				}

				if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, wait.ForeverTestTimeout, false,
					podIsGettingEvicted(cs, testCtx.NS.Name, expectedEvictedName)); err != nil {
					t.Errorf("Expected pod %v to get evicted: %v", expectedEvictedName, err)
				}

				sparedPod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, expectedSparedName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Error fetching spared pod %v: %v", expectedSparedName, err)
				}
				if sparedPod.DeletionTimestamp != nil {
					t.Errorf("Pod %v was unexpectedly evicted, expected %v to be evicted instead", sparedPod.Name, expectedEvictedName)
				}
			})

			// Subtest 3: Unstarted pods (nil StartTime) tie-breaking by UID determinism
			t.Run("Unstarted pods with nil StartTime tie-break by UID", func(t *testing.T) {
				testCtx := initTest(t, "det-unstarted")
				cs := testCtx.ClientSet

				nodeName := "node-unstarted-uid"
				_, err := createNode(cs, st.MakeNode().Name(nodeName).Label("kubernetes.io/hostname", nodeName).Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Failed to create node %v: %v", nodeName, err)
				}

				u1Pod := initPausePod(&testutils.PausePodConfig{
					Name:      "unstarted-victim-1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  nodeName,
					Resources: defaultPodRes,
				})
				u2Pod := initPausePod(&testutils.PausePodConfig{
					Name:      "unstarted-victim-2",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  nodeName,
					Resources: defaultPodRes,
				})

				p1, err := runPausePod(cs, u1Pod)
				if err != nil {
					t.Fatalf("Error running p1: %v", err)
				}
				p2, err := runPausePod(cs, u2Pod)
				if err != nil {
					t.Fatalf("Error running p2: %v", err)
				}
				p1.Status.StartTime = nil
				p2.Status.StartTime = nil
				if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, p1, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
				if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, p2, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{p1, p2}); err != nil {
					t.Fatalf("Pods not stable in cache: %v", err)
				}

				expectedEvictedName := p2.Name
				expectedSparedName := p1.Name
				if p2.UID < p1.UID {
					expectedEvictedName = p1.Name
					expectedSparedName = p2.Name
				}

				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "preemptor-unstarted",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					NodeSelector: map[string]string{
						"kubernetes.io/hostname": nodeName,
					},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				})
				preemptorPod, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Error creating preemptor: %v", err)
				}

				if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptorPod); err != nil {
					t.Fatalf("Preemptor .status.nominatedNodeName not set: %v", err)
				}

				if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, wait.ForeverTestTimeout, false,
					podIsGettingEvicted(cs, testCtx.NS.Name, expectedEvictedName)); err != nil {
					t.Errorf("Expected unstarted pod %v to get evicted: %v", expectedEvictedName, err)
				}

				sparedPod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, expectedSparedName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Error fetching spared pod %v: %v", expectedSparedName, err)
				}
				if sparedPod.DeletionTimestamp != nil {
					t.Errorf("Unstarted pod %v was unexpectedly evicted, expected %v to be evicted instead", sparedPod.Name, expectedEvictedName)
				}
			})
		})
	}
}

// TestEmptyPDBSelectorMultiNamespacePreemption tests that universal PDBs (selector: {})
// protect unlabeled pods in multi-tenant environments without leaking across namespace boundaries.
func TestEmptyPDBSelectorMultiNamespacePreemption(t *testing.T) {
	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
	}
	defaultNodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("AsyncPreemption_%v", asyncPreemptionEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.SchedulerAsyncPreemption: asyncPreemptionEnabled,
			})

			// Subtest 1: Multi-node cross-namespace isolation with universal PDB
			t.Run("Universal PDB in nsA protects unlabeled pod on node-1, evicts unlabeled pod in nsB on node-2", func(t *testing.T) {
				testCtx := initTest(t, "empty-pdb-multi-node")
				cs := testCtx.ClientSet
				initDisruptionController(t, testCtx)

				nsA := testCtx.NS.Name
				nsBObj, err := cs.CoreV1().Namespaces().Create(testCtx.Ctx, &v1.Namespace{
					ObjectMeta: metav1.ObjectMeta{GenerateName: "tenant-b-"},
				}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create tenant B namespace: %v", err)
				}
				nsB := nsBObj.Name

				node1Name := "node-pdb-1"
				node2Name := "node-pdb-2"
				_, err = createNode(cs, st.MakeNode().Name(node1Name).Label("kubernetes.io/hostname", node1Name).Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Failed to create node-1: %v", err)
				}
				_, err = createNode(cs, st.MakeNode().Name(node2Name).Label("kubernetes.io/hostname", node2Name).Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Failed to create node-2: %v", err)
				}

				podA := initPausePod(&testutils.PausePodConfig{
					Name:      "unlabeled-victim-ns-a",
					Namespace: nsA,
					Priority:  &lowPriority,
					NodeName:  node1Name,
					Resources: defaultPodRes,
				})
				podA, err = runPausePod(cs, podA)
				if err != nil {
					t.Fatalf("Failed to run podA: %v", err)
				}
				addPodConditionReady(podA)
				if _, err := cs.CoreV1().Pods(nsA).UpdateStatus(testCtx.Ctx, podA, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}

				podB := initPausePod(&testutils.PausePodConfig{
					Name:      "unlabeled-victim-ns-b",
					Namespace: nsB,
					Priority:  &lowPriority,
					NodeName:  node2Name,
					Resources: defaultPodRes,
				})
				podB, err = runPausePod(cs, podB)
				if err != nil {
					t.Fatalf("Failed to run podB: %v", err)
				}
				addPodConditionReady(podB)
				if _, err := cs.CoreV1().Pods(nsB).UpdateStatus(testCtx.Ctx, podB, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}

				if err := waitCachedPodsStable(testCtx, []*v1.Pod{podA, podB}); err != nil {
					t.Fatalf("Pods not stable in cache: %v", err)
				}

				minAvail := intstr.FromInt32(1)
				pdbA := &policy.PodDisruptionBudget{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "universal-pdb-ns-a",
						Namespace: nsA,
					},
					Spec: policy.PodDisruptionBudgetSpec{
						MinAvailable: &minAvail,
						Selector:     &metav1.LabelSelector{}, // Empty selector
					},
				}
				createdPDBA, err := cs.PolicyV1().PodDisruptionBudgets(nsA).Create(testCtx.Ctx, pdbA, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create PDB in nsA: %v", err)
				}
				if err := waitForPDBsStable(testCtx, []*policy.PodDisruptionBudget{createdPDBA}, []int32{1}); err != nil {
					t.Fatalf("PDB in nsA not stable: %v", err)
				}

				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "preemptor-pod",
					Namespace: nsA,
					Priority:  &highPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				})
				preemptorPod, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Failed to create preemptor pod: %v", err)
				}

				if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptorPod); err != nil {
					t.Fatalf("Preemptor .status.nominatedNodeName not set: %v", err)
				}

				if err = wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, wait.ForeverTestTimeout, false,
					podIsGettingEvicted(cs, nsB, podB.Name)); err != nil {
					t.Errorf("Expected podB in nsB to get evicted: %v", err)
				}

				gotPodA, err := cs.CoreV1().Pods(nsA).Get(testCtx.Ctx, podA.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get podA: %v", err)
				}
				if gotPodA.DeletionTimestamp != nil {
					t.Errorf("podA in nsA was unexpectedly evicted despite being protected by universal PDB")
				}
			})

			// Subtest 2: Single shared node with unlabeled pods from both namespaces
			t.Run("Universal PDB in nsA protects unlabeled pod on shared node against nsB unlabeled pod", func(t *testing.T) {
				testCtx := initTest(t, "empty-pdb-shared-node")
				cs := testCtx.ClientSet
				initDisruptionController(t, testCtx)

				nsA := testCtx.NS.Name
				nsBObj, err := cs.CoreV1().Namespaces().Create(testCtx.Ctx, &v1.Namespace{
					ObjectMeta: metav1.ObjectMeta{GenerateName: "tenant-b-"},
				}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create tenant B namespace: %v", err)
				}
				nsB := nsBObj.Name

				sharedNodeRes := map[v1.ResourceName]string{
					v1.ResourcePods:   "32",
					v1.ResourceCPU:    "1000m",
					v1.ResourceMemory: "1000",
				}
				sharedNodeName := "shared-node"
				_, err = createNode(cs, st.MakeNode().Name(sharedNodeName).Label("kubernetes.io/hostname", sharedNodeName).Capacity(sharedNodeRes).Obj())
				if err != nil {
					t.Fatalf("Failed to create shared node: %v", err)
				}

				podA := initPausePod(&testutils.PausePodConfig{
					Name:      "shared-victim-ns-a",
					Namespace: nsA,
					Priority:  &lowPriority,
					NodeName:  sharedNodeName,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				})
				podB := initPausePod(&testutils.PausePodConfig{
					Name:      "shared-victim-ns-b",
					Namespace: nsB,
					Priority:  &lowPriority,
					NodeName:  sharedNodeName,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				})
				filler := initPausePod(&testutils.PausePodConfig{
					Name:      "shared-filler",
					Namespace: nsA,
					Priority:  &highPriority,
					NodeName:  sharedNodeName,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				})

				pA, err := runPausePod(cs, podA)
				if err != nil {
					t.Fatalf("Failed to run pA: %v", err)
				}
				pB, err := runPausePod(cs, podB)
				if err != nil {
					t.Fatalf("Failed to run pB: %v", err)
				}
				pF, err := runPausePod(cs, filler)
				if err != nil {
					t.Fatalf("Failed to run pF: %v", err)
				}
				addPodConditionReady(pA)
				addPodConditionReady(pB)
				addPodConditionReady(pF)
				if _, err := cs.CoreV1().Pods(nsA).UpdateStatus(testCtx.Ctx, pA, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
				if _, err := cs.CoreV1().Pods(nsB).UpdateStatus(testCtx.Ctx, pB, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
				if _, err := cs.CoreV1().Pods(nsA).UpdateStatus(testCtx.Ctx, pF, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{pA, pB, pF}); err != nil {
					t.Fatalf("Pods not stable in cache: %v", err)
				}

				minAvail2 := intstr.FromInt32(2)
				pdbA2 := &policy.PodDisruptionBudget{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "universal-pdb-shared-ns-a-2",
						Namespace: nsA,
					},
					Spec: policy.PodDisruptionBudgetSpec{
						MinAvailable: &minAvail2,
						Selector:     &metav1.LabelSelector{},
					},
				}
				createdPDBA2, err := cs.PolicyV1().PodDisruptionBudgets(nsA).Create(testCtx.Ctx, pdbA2, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create PDB: %v", err)
				}
				if err := waitForPDBsStable(testCtx, []*policy.PodDisruptionBudget{createdPDBA2}, []int32{2}); err != nil {
					t.Fatalf("PDB in nsA not stable: %v", err)
				}

				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "shared-preemptor",
					Namespace: nsA,
					Priority:  &mediumPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				})
				preemptorPod, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Failed to create preemptor: %v", err)
				}

				if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptorPod); err != nil {
					t.Fatalf("Preemptor .status.nominatedNodeName not set: %v", err)
				}

				if err = wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, wait.ForeverTestTimeout, false,
					podIsGettingEvicted(cs, nsB, pB.Name)); err != nil {
					t.Errorf("Expected podB in nsB to get evicted: %v", err)
				}

				gotPA, err := cs.CoreV1().Pods(nsA).Get(testCtx.Ctx, pA.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get pA: %v", err)
				}
				if gotPA.DeletionTimestamp != nil {
					t.Errorf("pA in nsA was unexpectedly evicted despite PDB protection")
				}
			})
		})
	}
}

// TestPodOverheadPreemptionFit tests that pod overheads (spec.overhead) are properly
// accounted for in preemption calculations for both preemptors and victims.
func TestPodOverheadPreemptionFit(t *testing.T) {
	testCtx := initTest(t, "pod-overhead-preempt")
	cs := testCtx.ClientSet

	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "1000m",
		v1.ResourceMemory: "1000",
	}

	_, err := createNode(cs, st.MakeNode().Name("node-overhead").Label("kubernetes.io/hostname", "node-overhead").Capacity(nodeRes).Obj())
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Create RuntimeClasses with overheads
	rc300 := &nodev1.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{Name: "rc-overhead-300m"},
		Handler:    "runc",
		Overhead: &nodev1.Overhead{
			PodFixed: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("300m"),
			},
		},
	}
	if _, err := cs.NodeV1().RuntimeClasses().Create(testCtx.Ctx, rc300, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create runtime class rc-overhead-300m: %v", err)
	}

	rc600 := &nodev1.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{Name: "rc-overhead-600m"},
		Handler:    "runc",
		Overhead: &nodev1.Overhead{
			PodFixed: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("600m"),
			},
		},
	}
	if _, err := cs.NodeV1().RuntimeClasses().Create(testCtx.Ctx, rc600, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create runtime class rc-overhead-600m: %v", err)
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("AsyncPreemption_%v", asyncPreemptionEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.SchedulerAsyncPreemption: asyncPreemptionEnabled,
			})

			// Subtest 1: Preemptor with spec.overhead requires evicting multiple victims
			t.Run("Preemptor with spec.overhead requires evicting multiple victims to fit", func(t *testing.T) {
				v1Pod := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("victim-1-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  "node-overhead",
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(300, resource.DecimalSI),
					}},
				})
				v2Pod := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("victim-2-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  "node-overhead",
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(300, resource.DecimalSI),
					}},
				})
				v3Pod := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("victim-3-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  "node-overhead",
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(300, resource.DecimalSI),
					}},
				})

				p1, err := runPausePod(cs, v1Pod)
				if err != nil {
					t.Fatalf("Failed to run victim 1: %v", err)
				}
				p2, err := runPausePod(cs, v2Pod)
				if err != nil {
					t.Fatalf("Failed to run victim 2: %v", err)
				}
				p3, err := runPausePod(cs, v3Pod)
				if err != nil {
					t.Fatalf("Failed to run victim 3: %v", err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{p1, p2, p3}); err != nil {
					t.Fatalf("Pods not stable in cache: %v", err)
				}

				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("preemptor-with-overhead-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(200, resource.DecimalSI),
					}},
				})
				preemptor.Spec.RuntimeClassName = ptr.To("rc-overhead-300m")

				preemptorPod, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Failed to create preemptor with overhead: %v", err)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{p1, p2, p3, preemptorPod})

				var evictedPods []*v1.Pod
				err = wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
					evictedPods = nil
					for _, p := range []*v1.Pod{p1, p2, p3} {
						got, err := cs.CoreV1().Pods(p.Namespace).Get(ctx, p.Name, metav1.GetOptions{})
						if err == nil && got.DeletionTimestamp != nil {
							evictedPods = append(evictedPods, got)
						}
					}
					return len(evictedPods) >= 2, nil
				})
				if err != nil {
					t.Fatalf("Expected at least 2 victims to be evicted: %v", err)
				}

				if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptorPod); err != nil {
					t.Fatalf("Preemptor .status.nominatedNodeName not set: %v", err)
				}

				// Simulate victim termination by force deleting evicted pods
				for _, ep := range evictedPods {
					_ = cs.CoreV1().Pods(ep.Namespace).Delete(testCtx.Ctx, ep.Name, *metav1.NewDeleteOptions(0))
				}

				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, preemptorPod, 30*time.Second); err != nil {
					t.Fatalf("Preemptor with overhead failed to schedule after victims were deleted: %v", err)
				}

				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{p1, p2, p3, preemptorPod})
			})

			// Subtest 2: Victim with spec.overhead frees overhead + request upon preemption
			t.Run("Victim with spec.overhead frees full overhead plus container request", func(t *testing.T) {
				vOverhead := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("victim-has-overhead-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  "node-overhead",
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI),
					}},
				})
				vOverhead.Spec.RuntimeClassName = ptr.To("rc-overhead-300m")

				vNoOverhead := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("victim-no-overhead-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  "node-overhead",
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI),
					}},
				})

				filler := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("filler-high-pri-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					NodeName:  "node-overhead",
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(500, resource.DecimalSI),
					}},
				})

				pOverhead, err := runPausePod(cs, vOverhead)
				if err != nil {
					t.Fatalf("Failed to run pOverhead: %v", err)
				}
				pNoOverhead, err := runPausePod(cs, vNoOverhead)
				if err != nil {
					t.Fatalf("Failed to run pNoOverhead: %v", err)
				}
				pFiller, err := runPausePod(cs, filler)
				if err != nil {
					t.Fatalf("Failed to run pFiller: %v", err)
				}
				t1 := metav1.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
				t2 := metav1.Date(2026, 1, 1, 1, 0, 0, 0, time.UTC)
				pOverhead.Status.StartTime = &t1
				pNoOverhead.Status.StartTime = &t2
				if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pOverhead, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
				if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pNoOverhead, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}

				if err := waitCachedPodsStable(testCtx, []*v1.Pod{pOverhead, pNoOverhead, pFiller}); err != nil {
					t.Fatalf("Pods not stable in cache: %v", err)
				}

				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("preemptor-300m-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(300, resource.DecimalSI),
					}},
				})
				preemptorPod, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Failed to create preemptor: %v", err)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pOverhead, pNoOverhead, pFiller, preemptorPod})

				if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, wait.ForeverTestTimeout, false,
					podIsGettingEvicted(cs, testCtx.NS.Name, pOverhead.Name)); err != nil {
					t.Errorf("Expected victimWithOverhead to get evicted: %v", err)
				}

				gotNoOverhead, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pNoOverhead.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get victimWithoutOverhead: %v", err)
				}
				if gotNoOverhead.DeletionTimestamp != nil {
					t.Errorf("victimWithoutOverhead was unexpectedly evicted when victimWithOverhead was sufficient")
				}

				if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptorPod); err != nil {
					t.Fatalf("Preemptor .status.nominatedNodeName not set: %v", err)
				}

				// Simulate victim termination
				_ = cs.CoreV1().Pods(pOverhead.Namespace).Delete(testCtx.Ctx, pOverhead.Name, *metav1.NewDeleteOptions(0))

				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, preemptorPod, 30*time.Second); err != nil {
					t.Fatalf("Preemptor failed to schedule after victim with overhead was deleted: %v", err)
				}

				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pOverhead, pNoOverhead, pFiller, preemptorPod})
			})

			// Subtest 3: Preemptor whose request + spec.overhead exceeds node allocatable remains unschedulable
			t.Run("Preemptor with overhead exceeding node capacity is unschedulable and triggers no evictions", func(t *testing.T) {
				victim := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("victim-pod-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  "node-overhead",
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(500, resource.DecimalSI),
					}},
				})
				pVictim, err := runPausePod(cs, victim)
				if err != nil {
					t.Fatalf("Failed to run victim: %v", err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{pVictim}); err != nil {
					t.Fatalf("Pod not stable in cache: %v", err)
				}

				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("oversized-preemptor-async-%v", asyncPreemptionEnabled),
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(600, resource.DecimalSI),
					}},
				})
				preemptor.Spec.RuntimeClassName = ptr.To("rc-overhead-600m")

				preemptorPod, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Failed to create oversized preemptor: %v", err)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pVictim, preemptorPod})

				if err := waitForPodUnschedulable(testCtx.Ctx, cs, preemptorPod); err != nil {
					t.Fatalf("Expected preemptor to be unschedulable: %v", err)
				}

				gotVictim, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pVictim.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get victim: %v", err)
				}
				if gotVictim.DeletionTimestamp != nil {
					t.Errorf("Victim was evicted for a preemptor that cannot fit on node")
				}

				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pVictim, preemptorPod})
			})
		})
	}
}

// TestGatedPreemptorEventReevaluation tests FM-101 / PR #139162:
// Verifies that gated preemptor pods residing in the unschedulable queue are correctly
// re-evaluated upon wildcard cluster events (such as StorageClass creation and Node updates/additions)
// and move to activeQ to preempt lower-priority victims as soon as scheduling gates are removed.
func TestGatedPreemptorEventReevaluation(t *testing.T) {
	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
	}
	defaultNodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("AsyncPreemptionEnabled_%v", asyncPreemptionEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.SchedulerAsyncPreemption: asyncPreemptionEnabled,
			})

			// Subtest 1: Gated preemptor in unschedulable queue re-evaluated on wildcard cluster events and preempts upon gate removal
			t.Run("Gated preemptor wildcard event re-evaluation and preemption on ungate", func(t *testing.T) {
				testCtx := initTest(t, "gated-reeval")
				cs := testCtx.ClientSet

				node, err := createNode(cs, st.MakeNode().Name("node-gated-1").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node: %v", err)
				}

				victim := initPausePod(&testutils.PausePodConfig{
					Name:      "victim-low",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  node.Name,
					Resources: defaultPodRes,
				})
				victim.Spec.TerminationGracePeriodSeconds = ptr.To(int64(0))
				pVictim, err := runPausePod(cs, victim)
				if err != nil {
					t.Fatalf("Error running victim: %v", err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{pVictim}); err != nil {
					t.Fatalf("Pod not stable in cache: %v", err)
				}

				gateName := "example.com/preemption-gate"
				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "gated-preemptor",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					Resources: defaultPodRes,
				})
				preemptor.Spec.SchedulingGates = []v1.PodSchedulingGate{{Name: gateName}}

				pGated, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Error creating gated preemptor: %v", err)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pVictim, pGated})

				if err := waitForPodSchedulingGated(testCtx.Ctx, cs, pGated, 30*time.Second); err != nil {
					t.Fatalf("Pod %s did not enter SchedulingGated state: %v", pGated.Name, err)
				}

				// Trigger wildcard cluster events while pod is gated:
				// Event 1: StorageClass creation
				sc := st.MakeStorageClass().Name(fmt.Sprintf("sc-wildcard-%v", asyncPreemptionEnabled)).Provisioner("kubernetes.io/no-provisioner").Obj()
				if _, err := cs.StorageV1().StorageClasses().Create(testCtx.Ctx, sc, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Error creating StorageClass: %v", err)
				}

				// Event 2: Node status/label update
				latestNode, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, node.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get node: %v", err)
				}
				if latestNode.Labels == nil {
					latestNode.Labels = make(map[string]string)
				}
				latestNode.Labels["re-eval-trigger"] = "true"
				if _, err := cs.CoreV1().Nodes().Update(testCtx.Ctx, latestNode, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("Failed to update node: %v", err)
				}

				// Verify victim has not been evicted yet since preemptor is gated
				gotVictim, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pVictim.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get victim pod: %v", err)
				}
				if gotVictim.DeletionTimestamp != nil {
					t.Fatalf("Victim pod was prematurely evicted while preemptor was gated")
				}

				// Remove scheduling gate to trigger activeQ enqueueing and preemption
				patch := []byte(`{"spec": {"schedulingGates": null}}`)
				if _, err := cs.CoreV1().Pods(testCtx.NS.Name).Patch(testCtx.Ctx, pGated.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}); err != nil {
					t.Fatalf("Failed to patch scheduling gates: %v", err)
				}

				// Preemptor must now preempt victim and schedule
				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, pGated, 30*time.Second); err != nil {
					t.Fatalf("Ungated preemptor failed to schedule/preempt: %v", err)
				}

				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pVictim, pGated})
			})

			// Subtest 2: Multiple gated preemptors re-evaluated on node additions and ungate
			t.Run("Multiple gated preemptors scheduled across nodes upon gate removal", func(t *testing.T) {
				testCtx := initTest(t, "gated-multi")
				cs := testCtx.ClientSet

				node1, err := createNode(cs, st.MakeNode().Name("node-multi-gated-1").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node1: %v", err)
				}

				victim1 := initPausePod(&testutils.PausePodConfig{
					Name:      "victim-multi-1",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  node1.Name,
					Resources: defaultPodRes,
				})
				victim1.Spec.TerminationGracePeriodSeconds = ptr.To(int64(0))
				pVictim1, err := runPausePod(cs, victim1)
				if err != nil {
					t.Fatalf("Error running victim1: %v", err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{pVictim1}); err != nil {
					t.Fatalf("Pod not stable in cache: %v", err)
				}

				gateName := "example.com/multi-gate"
				preemptor1 := initPausePod(&testutils.PausePodConfig{
					Name:      "gated-p1",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					Resources: defaultPodRes,
				})
				preemptor1.Spec.SchedulingGates = []v1.PodSchedulingGate{{Name: gateName}}

				pGated1, err := createPausePod(cs, preemptor1)
				if err != nil {
					t.Fatalf("Failed to create gated-p1: %v", err)
				}

				preemptor2 := initPausePod(&testutils.PausePodConfig{
					Name:      "gated-p2",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					Resources: defaultPodRes,
				})
				preemptor2.Spec.SchedulingGates = []v1.PodSchedulingGate{{Name: gateName}}

				pGated2, err := createPausePod(cs, preemptor2)
				if err != nil {
					t.Fatalf("Failed to create gated-p2: %v", err)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pVictim1, pGated1, pGated2})

				if err := waitForPodSchedulingGated(testCtx.Ctx, cs, pGated1, 30*time.Second); err != nil {
					t.Fatalf("pGated1 did not enter SchedulingGated: %v", err)
				}
				if err := waitForPodSchedulingGated(testCtx.Ctx, cs, pGated2, 30*time.Second); err != nil {
					t.Fatalf("pGated2 did not enter SchedulingGated: %v", err)
				}

				// Add second node to trigger node add cluster event
				_, err = createNode(cs, st.MakeNode().Name("node-multi-gated-2").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node2: %v", err)
				}

				// Ungate both pods
				patch := []byte(`{"spec": {"schedulingGates": null}}`)
				for _, p := range []*v1.Pod{pGated1, pGated2} {
					if _, err := cs.CoreV1().Pods(testCtx.NS.Name).Patch(testCtx.Ctx, p.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}); err != nil {
						t.Fatalf("Failed to remove gate from %s: %v", p.Name, err)
					}
				}

				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, pGated1, 30*time.Second); err != nil {
					t.Fatalf("pGated1 failed to schedule: %v", err)
				}
				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, pGated2, 30*time.Second); err != nil {
					t.Fatalf("pGated2 failed to schedule: %v", err)
				}

				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pVictim1, pGated1, pGated2})
			})
		})
	}
}

// TestWasFlushedFromUnschedulablePreemptionLifecycle tests FM-103 / PR #139330:
// Verifies that when a preemptor pod moves from the unschedulable queue through the active queue
// and back into unschedulable upon a failed preemption cycle, the WasFlushedFromUnschedulable flag
// is properly reset so future legitimate flushes and scheduling attempts are not skipped or corrupted.
func TestWasFlushedFromUnschedulablePreemptionLifecycle(t *testing.T) {
	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
	}
	defaultNodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("AsyncPreemptionEnabled_%v", asyncPreemptionEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.SchedulerAsyncPreemption: asyncPreemptionEnabled,
			})

			// Subtest 1: Failed preemption cycle clears WasFlushedFromUnschedulable and allows subsequent legitimate flushes
			t.Run("Failed preemption clears WasFlushedFromUnschedulable and enables future flush scheduling", func(t *testing.T) {
				testCtx := initTest(t, "flush-lifecycle",
					scheduler.WithPodMaxInUnschedulablePodsDuration(2*time.Second),
					scheduler.WithPodInitialBackoffSeconds(1),
					scheduler.WithPodMaxBackoffSeconds(2),
				)
				cs := testCtx.ClientSet

				_, err := createNode(cs, st.MakeNode().Name("node-lifecycle-1").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node: %v", err)
				}

				// Run a protected victim pod with higher priority (highPriority = 300)
				protectedVictim := initPausePod(&testutils.PausePodConfig{
					Name:      "protected-victim",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					NodeName:  "node-lifecycle-1",
					Resources: defaultPodRes,
				})
				protectedVictim.Spec.TerminationGracePeriodSeconds = ptr.To(int64(0))
				pProtected, err := runPausePod(cs, protectedVictim)
				if err != nil {
					t.Fatalf("Error running protected victim: %v", err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{pProtected}); err != nil {
					t.Fatalf("Pod not stable in cache: %v", err)
				}

				// Preemptor with medium priority (mediumPriority = 200) cannot preempt protected-victim
				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "preemptor-retry",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
				})
				pPreemptor, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Error creating preemptor: %v", err)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pProtected, pPreemptor})

				if err := waitForPodUnschedulable(testCtx.Ctx, cs, pPreemptor); err != nil {
					t.Fatalf("Preemptor was expected to be unschedulable initially: %v", err)
				}

				// Allow time for flushUnschedulableEntitiesLeftover to trigger (exceeding 2s duration)
				// The pod moves to activeQ, fails preemption again, and returns to unschedulable queue.
				// On return to unschedulable queue, WasFlushedFromUnschedulable must be reset to false.
				time.Sleep(3 * time.Second)

				// Now add a second node so preemptor-retry can be scheduled upon event/flush
				_, err = createNode(cs, st.MakeNode().Name("node-lifecycle-2").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node 2: %v", err)
				}

				// On the next flush or queue evaluation, preemptor-retry must successfully schedule!
				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, pPreemptor, 30*time.Second); err != nil {
					t.Fatalf("Preemptor failed to schedule after state reset: %v", err)
				}

				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pProtected, pPreemptor})
			})

			// Subtest 2: Gated pod flushed during unschedulable timeout clears WasFlushedFromUnschedulable upon gate removal
			t.Run("Gated pod flushed during unschedulable timeout schedules cleanly upon ungate", func(t *testing.T) {
				testCtx := initTest(t, "flush-gated",
					scheduler.WithPodMaxInUnschedulablePodsDuration(2*time.Second),
					scheduler.WithPodInitialBackoffSeconds(1),
					scheduler.WithPodMaxBackoffSeconds(2),
				)
				cs := testCtx.ClientSet

				_, err := createNode(cs, st.MakeNode().Name("node-lifecycle-2").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node: %v", err)
				}

				victim := initPausePod(&testutils.PausePodConfig{
					Name:      "victim-gated-flush",
					Namespace: testCtx.NS.Name,
					Priority:  &lowPriority,
					NodeName:  "node-lifecycle-2",
					Resources: defaultPodRes,
				})
				victim.Spec.TerminationGracePeriodSeconds = ptr.To(int64(0))
				pVictim, err := runPausePod(cs, victim)
				if err != nil {
					t.Fatalf("Error running victim: %v", err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{pVictim}); err != nil {
					t.Fatalf("Pod not stable in cache: %v", err)
				}

				gateName := "example.com/flush-lifecycle-gate"
				preemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "preemptor-gated-flush",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					Resources: defaultPodRes,
				})
				preemptor.Spec.SchedulingGates = []v1.PodSchedulingGate{{Name: gateName}}

				pGated, err := createPausePod(cs, preemptor)
				if err != nil {
					t.Fatalf("Error creating gated pod: %v", err)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pVictim, pGated})

				if err := waitForPodSchedulingGated(testCtx.Ctx, cs, pGated, 30*time.Second); err != nil {
					t.Fatalf("Pod did not enter SchedulingGated: %v", err)
				}

				// Wait past the 2s flush duration while pod is gated
				time.Sleep(3 * time.Second)

				// Ungate pod
				patch := []byte(`{"spec": {"schedulingGates": null}}`)
				if _, err := cs.CoreV1().Pods(testCtx.NS.Name).Patch(testCtx.Ctx, pGated.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}); err != nil {
					t.Fatalf("Failed to remove scheduling gates: %v", err)
				}

				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, pGated, 30*time.Second); err != nil {
					t.Fatalf("Ungated pod failed to schedule after flush interval: %v", err)
				}

				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pVictim, pGated})
			})
		})
	}
}

// TestEqualFlushFrequencyEnforcement tests FM-102 / PR #139331:
// Verifies that unschedulable pods with high preemption retry counts are not subjected to
// flush frequency starvation and are flushed at uniform intervals determined by FlushTimestamp.
func TestEqualFlushFrequencyEnforcement(t *testing.T) {
	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
	}
	defaultNodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("AsyncPreemptionEnabled_%v", asyncPreemptionEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.SchedulerAsyncPreemption: asyncPreemptionEnabled,
			})

			// Subtest 1: Preemptor with high retry count is flushed at uniform intervals without starvation compared to fresh unschedulable pods
			t.Run("High retry preemptor is flushed uniformly and not starved vs fresh pod", func(t *testing.T) {
				testCtx := initTest(t, "flush-freq",
					scheduler.WithPodMaxInUnschedulablePodsDuration(2*time.Second),
					scheduler.WithPodInitialBackoffSeconds(1),
					scheduler.WithPodMaxBackoffSeconds(2),
				)
				cs := testCtx.ClientSet

				_, err := createNode(cs, st.MakeNode().Name("node-freq-1").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node: %v", err)
				}

				// Run non-preemptible blocking pod (priority 300)
				blocker := initPausePod(&testutils.PausePodConfig{
					Name:      "blocking-pod",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					NodeName:  "node-freq-1",
					Resources: defaultPodRes,
				})
				blocker.Spec.TerminationGracePeriodSeconds = ptr.To(int64(0))
				pBlocker, err := runPausePod(cs, blocker)
				if err != nil {
					t.Fatalf("Error running blocker: %v", err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{pBlocker}); err != nil {
					t.Fatalf("Pod not stable in cache: %v", err)
				}

				// Create high-retry preemptor (mediumPriority = 200)
				highRetryPreemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "high-retry-preemptor",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
				})
				pHighRetry, err := createPausePod(cs, highRetryPreemptor)
				if err != nil {
					t.Fatalf("Error creating high-retry preemptor: %v", err)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pBlocker, pHighRetry})

				if err := waitForPodUnschedulable(testCtx.Ctx, cs, pHighRetry); err != nil {
					t.Fatalf("High retry preemptor was expected to be unschedulable: %v", err)
				}

				// Let highRetryPreemptor accumulate multiple failed scheduling/flush cycles (> 5s)
				time.Sleep(5 * time.Second)

				// Create fresh unschedulable pod (0 retry count)
				freshPreemptor := initPausePod(&testutils.PausePodConfig{
					Name:      "fresh-preemptor",
					Namespace: testCtx.NS.Name,
					Priority:  &mediumPriority,
					Resources: defaultPodRes,
				})
				pFresh, err := createPausePod(cs, freshPreemptor)
				if err != nil {
					t.Fatalf("Error creating fresh preemptor: %v", err)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pFresh})

				if err := waitForPodUnschedulable(testCtx.Ctx, cs, pFresh); err != nil {
					t.Fatalf("Fresh preemptor was expected to be unschedulable: %v", err)
				}

				// Now expand cluster capacity by adding node-freq-2 and node-freq-3 so both can schedule
				_, err = createNode(cs, st.MakeNode().Name("node-freq-2").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node2: %v", err)
				}
				_, err = createNode(cs, st.MakeNode().Name("node-freq-3").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node3: %v", err)
				}

				// Both pods must be scheduled without high-retry pod being starved
				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, pHighRetry, 30*time.Second); err != nil {
					t.Fatalf("High retry preemptor was starved or failed to schedule: %v", err)
				}
				if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, pFresh, 30*time.Second); err != nil {
					t.Fatalf("Fresh preemptor failed to schedule: %v", err)
				}

				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pBlocker, pHighRetry, pFresh})
			})

			// Subtest 2: Multiple unschedulable preemptor pods under churn are flushed uniformly
			t.Run("Multiple saturated unschedulable pods maintain uniform flush and scheduling", func(t *testing.T) {
				testCtx := initTest(t, "flush-churn",
					scheduler.WithPodMaxInUnschedulablePodsDuration(2*time.Second),
					scheduler.WithPodInitialBackoffSeconds(1),
					scheduler.WithPodMaxBackoffSeconds(2),
				)
				cs := testCtx.ClientSet

				node, err := createNode(cs, st.MakeNode().Name("node-churn-1").Capacity(defaultNodeRes).Obj())
				if err != nil {
					t.Fatalf("Error creating node: %v", err)
				}

				blocker := initPausePod(&testutils.PausePodConfig{
					Name:      "blocker-churn",
					Namespace: testCtx.NS.Name,
					Priority:  &highPriority,
					NodeName:  node.Name,
					Resources: defaultPodRes,
				})
				blocker.Spec.TerminationGracePeriodSeconds = ptr.To(int64(0))
				pBlocker, err := runPausePod(cs, blocker)
				if err != nil {
					t.Fatalf("Error running blocker: %v", err)
				}
				if err := waitCachedPodsStable(testCtx, []*v1.Pod{pBlocker}); err != nil {
					t.Fatalf("Pod not stable in cache: %v", err)
				}

				var preemptors []*v1.Pod
				for i := 1; i <= 3; i++ {
					p := initPausePod(&testutils.PausePodConfig{
						Name:      fmt.Sprintf("preemptor-churn-%d", i),
						Namespace: testCtx.NS.Name,
						Priority:  &mediumPriority,
						Resources: defaultPodRes,
					})
					pod, err := createPausePod(cs, p)
					if err != nil {
						t.Fatalf("Error creating churn preemptor %d: %v", i, err)
					}
					preemptors = append(preemptors, pod)
				}
				defer testutils.CleanupPods(testCtx.Ctx, cs, t, append([]*v1.Pod{pBlocker}, preemptors...))

				for _, p := range preemptors {
					if err := waitForPodUnschedulable(testCtx.Ctx, cs, p); err != nil {
						t.Fatalf("Pod %s was expected to be unschedulable: %v", p.Name, err)
					}
				}

				// Allow some flush churn cycles
				time.Sleep(3 * time.Second)

				// Delete blocker pod and add 2 more nodes so all 3 preemptors can schedule
				if err := cs.CoreV1().Pods(testCtx.NS.Name).Delete(testCtx.Ctx, pBlocker.Name, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("Failed to delete blocker: %v", err)
				}
				for i := 2; i <= 3; i++ {
					_, err := createNode(cs, st.MakeNode().Name(fmt.Sprintf("node-churn-%d", i)).Capacity(defaultNodeRes).Obj())
					if err != nil {
						t.Fatalf("Error creating node %d: %v", i, err)
					}
				}

				for _, p := range preemptors {
					if err := waitForPodToScheduleWithTimeout(testCtx.Ctx, cs, p, 30*time.Second); err != nil {
						t.Fatalf("Pod %s failed to schedule during uniform flush: %v", p.Name, err)
					}
				}

				testutils.CleanupPods(testCtx.Ctx, cs, t, append([]*v1.Pod{pBlocker}, preemptors...))
			})
		})
	}
}
