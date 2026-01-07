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
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
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
			v1.ResourceMemory: *resource.NewQuantity(200*1024, resource.DecimalSI)},
		},
	})

	preemptor := initPausePod(&testutils.PausePodConfig{
		Name:      "preemptor-pod",
		Namespace: testCtx.NS.Name,
		Priority:  &highPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(200*1024, resource.DecimalSI)},
		},
	})

	// Create a node with some resources
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500Ki",
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
						v1.ResourceMemory: *resource.NewQuantity(200*1024, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:      "preemptor-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200*1024, resource.DecimalSI)},
				},
			}),
		},
	}

	// Create a node with some resources
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500Ki",
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
		v1.ResourceMemory: "500Ki",
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
		v1.ResourceMemory: *resource.NewQuantity(100*1024, resource.DecimalSI)},
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
					v1.ResourceMemory: *resource.NewQuantity(200*1024, resource.DecimalSI)},
				},
			}),
		},
	}

	// Create a node with some resources
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500Ki",
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
					for i := 0; i < numRunningPods; i++ {
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
		v1.ResourceMemory: *resource.NewQuantity(100*1024, resource.DecimalSI)},
	}
	defaultNodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500Ki",
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
					v1.ResourceMemory: *resource.NewQuantity(200*1024, resource.DecimalSI)},
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
					v1.ResourceMemory: *resource.NewQuantity(200*1024, resource.DecimalSI)},
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
					v1.ResourceMemory: *resource.NewQuantity(400*1024, resource.DecimalSI)},
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
		v1.ResourceMemory: "500Ki",
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
