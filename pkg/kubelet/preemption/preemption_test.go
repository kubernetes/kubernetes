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

package preemption

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	kubeapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	clusterCritical       = "cluster-critical"
	nodeCritical          = "node-critical"
	bestEffort            = "bestEffort"
	burstable             = "burstable"
	highRequestBurstable  = "high-request-burstable"
	highPriorityBurstable = "high-priority-burstable"
	guaranteed            = "guaranteed"
	highRequestGuaranteed = "high-request-guaranteed"
	tinyBurstable         = "tiny"
	maxPods               = 110
)

type fakePodKiller struct {
	killedPods          []*v1.Pod
	errDuringPodKilling bool
}

func newFakePodKiller(errPodKilling bool) *fakePodKiller {
	return &fakePodKiller{killedPods: []*v1.Pod{}, errDuringPodKilling: errPodKilling}
}

func (f *fakePodKiller) clear() {
	f.killedPods = []*v1.Pod{}
}

func (f *fakePodKiller) getKilledPods() []*v1.Pod {
	return f.killedPods
}

func (f *fakePodKiller) killPodNow(pod *v1.Pod, evict bool, gracePeriodOverride *int64, fn func(status *v1.PodStatus)) error {
	if f.errDuringPodKilling {
		f.killedPods = []*v1.Pod{}
		return fmt.Errorf("problem killing pod %v", pod)
	}
	f.killedPods = append(f.killedPods, pod)
	return nil
}

type fakePodProvider struct {
	pods []*v1.Pod
}

func newFakePodProvider() *fakePodProvider {
	return &fakePodProvider{pods: []*v1.Pod{}}
}

func (f *fakePodProvider) setPods(pods []*v1.Pod) {
	f.pods = pods
}

func (f *fakePodProvider) getPods() []*v1.Pod {
	return f.pods
}

func getTestCriticalPodAdmissionHandler(podProvider *fakePodProvider, podKiller *fakePodKiller) *CriticalPodAdmissionHandler {
	return &CriticalPodAdmissionHandler{
		getPodsFunc: podProvider.getPods,
		killPodFunc: podKiller.killPodNow,
		recorder:    &record.FakeRecorder{},
	}
}

func TestHandleAdmissionFailure(t *testing.T) {
	tCtx := ktesting.Init(t)
	type testRun struct {
		testName             string
		isPodKillerWithError bool
		inputPods            []*v1.Pod
		admitPodType         string
		failReasons          []lifecycle.PredicateFailureReason
		expectErr            bool
		expectedOutput       []*v1.Pod
		expectReasons        []lifecycle.PredicateFailureReason
	}
	allPods := getTestPods()
	runs := []testRun{
		{
			testName:             "critical pods cannot be preempted - no other failure reason",
			isPodKillerWithError: false,
			inputPods:            []*v1.Pod{allPods[clusterCritical]},
			admitPodType:         clusterCritical,
			failReasons:          getPredicateFailureReasons(0, 0, 1, false),
			expectErr:            true,
			expectedOutput:       nil,
			expectReasons:        getPredicateFailureReasons(0, 0, 0, false),
		},
		{
			testName:             "non-critical pod should not trigger eviction - no other failure reason",
			isPodKillerWithError: false,
			inputPods:            []*v1.Pod{allPods[burstable]},
			admitPodType:         guaranteed,
			failReasons:          getPredicateFailureReasons(0, 1, 0, false),
			expectErr:            false,
			expectedOutput:       nil,
			expectReasons:        getPredicateFailureReasons(0, 1, 0, false),
		},
		{
			testName:             "best effort pods are not preempted when attempting to free resources - no other failure reason",
			isPodKillerWithError: false,
			inputPods:            []*v1.Pod{allPods[bestEffort]},
			admitPodType:         clusterCritical,
			failReasons:          getPredicateFailureReasons(0, 1, 0, false),
			expectErr:            true,
			expectedOutput:       nil,
			expectReasons:        getPredicateFailureReasons(0, 0, 0, false),
		},
		{
			testName:             "multiple pods evicted - no other failure reason",
			isPodKillerWithError: false,
			inputPods: []*v1.Pod{
				allPods[clusterCritical], allPods[bestEffort], allPods[burstable], allPods[highRequestBurstable],
				allPods[guaranteed], allPods[highRequestGuaranteed]},
			admitPodType:   clusterCritical,
			failReasons:    getPredicateFailureReasons(0, 550, 0, false),
			expectErr:      false,
			expectedOutput: []*v1.Pod{allPods[highRequestBurstable], allPods[highRequestGuaranteed]},
			expectReasons:  getPredicateFailureReasons(0, 0, 0, false),
		},
		{
			testName:             "multiple pods with eviction error - no other failure reason",
			isPodKillerWithError: true,
			inputPods: []*v1.Pod{
				allPods[clusterCritical], allPods[bestEffort], allPods[burstable], allPods[highRequestBurstable],
				allPods[guaranteed], allPods[highRequestGuaranteed]},
			admitPodType:   clusterCritical,
			failReasons:    getPredicateFailureReasons(0, 550, 0, false),
			expectErr:      false,
			expectedOutput: nil,
			expectReasons:  getPredicateFailureReasons(0, 0, 0, false),
		},
		{
			testName:             "non-critical pod should not trigger eviction - with other failure reason",
			isPodKillerWithError: false,
			inputPods:            []*v1.Pod{allPods[burstable]},
			admitPodType:         guaranteed,
			failReasons:          getPredicateFailureReasons(0, 1, 0, true),
			expectErr:            false,
			expectedOutput:       nil,
			expectReasons:        getPredicateFailureReasons(0, 1, 0, true),
		},
		{
			testName:             "critical pods cannot be preempted - with other failure reason",
			isPodKillerWithError: false,
			inputPods:            []*v1.Pod{allPods[clusterCritical]},
			admitPodType:         clusterCritical,
			failReasons:          getPredicateFailureReasons(0, 0, 1, true),
			expectErr:            false,
			expectedOutput:       nil,
			expectReasons:        getPredicateFailureReasons(0, 0, 0, true),
		},
	}
	for _, r := range runs {
		t.Run(r.testName, func(t *testing.T) {
			podProvider := newFakePodProvider()
			podKiller := newFakePodKiller(r.isPodKillerWithError)
			defer podKiller.clear()
			criticalPodAdmissionHandler := getTestCriticalPodAdmissionHandler(podProvider, podKiller)
			podProvider.setPods(r.inputPods)
			admitPodRef := allPods[r.admitPodType]
			filteredReason, outErr := criticalPodAdmissionHandler.HandleAdmissionFailure(tCtx, admitPodRef, r.failReasons)
			outputPods := podKiller.getKilledPods()
			if !r.expectErr && outErr != nil {
				t.Errorf("HandleAdmissionFailure returned an unexpected error during the %s test.  Err: %v", r.testName, outErr)
			} else if r.expectErr && outErr == nil {
				t.Errorf("HandleAdmissionFailure expected an error but returned a successful output=%v during the %s test.", outputPods, r.testName)
			} else if !podListEqual(r.expectedOutput, outputPods) {
				t.Errorf("HandleAdmissionFailure expected %v but got %v during the %s test.", r.expectedOutput, outputPods, r.testName)
			}
			if len(filteredReason) != len(r.expectReasons) {
				t.Fatalf("expect reasons %v, got reasons %v", r.expectReasons, filteredReason)
			}
			for i, reason := range filteredReason {
				if reason.GetReason() != r.expectReasons[i].GetReason() {
					t.Fatalf("expect reasons %v, got reasons %v", r.expectReasons, filteredReason)
				}
			}
		})
	}
}

func BenchmarkGetPodsToPreempt(t *testing.B) {
	allPods := getTestPods()
	inputPods := []*v1.Pod{}
	for i := 0; i < maxPods; i++ {
		inputPods = append(inputPods, allPods[tinyBurstable])
	}
	for n := 0; n < t.N; n++ {
		getPodsToPreempt(allPods[bestEffort], inputPods, admissionRequirementList([]*admissionRequirement{
			{
				resourceName: v1.ResourceCPU,
				quantity:     parseCPUToInt64("110m"),
			}}))
	}
}

func TestGetPodsToPreempt(t *testing.T) {
	type testRun struct {
		testName              string
		preemptor             *v1.Pod
		inputPods             []*v1.Pod
		insufficientResources admissionRequirementList
		expectErr             bool
		expectedOutput        []*v1.Pod
	}
	allPods := getTestPods()
	runs := []testRun{
		{
			testName:              "no requirements",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{},
			insufficientResources: getAdmissionRequirementList(0, 0, 0),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{},
		},
		{
			testName:              "no pods",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{},
			insufficientResources: getAdmissionRequirementList(0, 0, 1),
			expectErr:             true,
			expectedOutput:        nil,
		},
		{
			testName:              "equal pods and resources requirements",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[burstable]},
			insufficientResources: getAdmissionRequirementList(100, 100, 1),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[burstable]},
		},
		{
			testName:              "higher requirements than pod requests",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[burstable]},
			insufficientResources: getAdmissionRequirementList(200, 200, 2),
			expectErr:             true,
			expectedOutput:        nil,
		},
		{
			testName:              "choose between bestEffort and burstable",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[burstable], allPods[bestEffort]},
			insufficientResources: getAdmissionRequirementList(0, 0, 1),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[bestEffort]},
		},
		{
			testName:              "choose between burstable and guaranteed",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[burstable], allPods[guaranteed]},
			insufficientResources: getAdmissionRequirementList(0, 0, 1),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[burstable]},
		},
		{
			testName:              "choose lower request burstable if it meets requirements",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[bestEffort], allPods[highRequestBurstable], allPods[burstable]},
			insufficientResources: getAdmissionRequirementList(100, 100, 0),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[burstable]},
		},
		{
			testName:              "choose higher request burstable if lower does not meet requirements",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[bestEffort], allPods[burstable], allPods[highRequestBurstable]},
			insufficientResources: getAdmissionRequirementList(150, 150, 0),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[highRequestBurstable]},
		},
		{
			testName:              "multiple pods required",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[bestEffort], allPods[burstable], allPods[highRequestBurstable], allPods[guaranteed], allPods[highRequestGuaranteed]},
			insufficientResources: getAdmissionRequirementList(350, 350, 0),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[burstable], allPods[highRequestBurstable]},
		},
		{
			testName:              "evict guaranteed when we have to, and dont evict the extra burstable",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[bestEffort], allPods[burstable], allPods[highRequestBurstable], allPods[guaranteed], allPods[highRequestGuaranteed]},
			insufficientResources: getAdmissionRequirementList(0, 550, 0),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[highRequestBurstable], allPods[highRequestGuaranteed]},
		},
		{
			testName:              "evict guaranteed when we have to, and dont evict the high request guaranteed pod",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[bestEffort], allPods[highRequestBurstable], allPods[guaranteed], allPods[highRequestGuaranteed]},
			insufficientResources: getAdmissionRequirementList(0, 350, 0),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[highRequestBurstable], allPods[guaranteed]},
		},
		{
			testName:              "evict cluster critical pod for node critical pod",
			preemptor:             allPods[nodeCritical],
			inputPods:             []*v1.Pod{allPods[clusterCritical]},
			insufficientResources: getAdmissionRequirementList(100, 0, 0),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[clusterCritical]},
		},
		{
			testName:              "can not evict node critical pod for cluster critical pod",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[nodeCritical]},
			insufficientResources: getAdmissionRequirementList(100, 0, 0),
			expectErr:             true,
			expectedOutput:        nil,
		},
		{
			testName:              "can not evict high priority burstable pod for guaranteed pod",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[guaranteed], allPods[highPriorityBurstable]},
			insufficientResources: getAdmissionRequirementList(100, 0, 0),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[guaranteed]},
		},
		{
			testName:              "evict the high priority pod when we have to, and dont evict the extra guaranteed",
			preemptor:             allPods[clusterCritical],
			inputPods:             []*v1.Pod{allPods[bestEffort], allPods[guaranteed], allPods[highRequestBurstable], allPods[highPriorityBurstable]},
			insufficientResources: getAdmissionRequirementList(0, 500, 0),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{allPods[highRequestBurstable], allPods[highPriorityBurstable]},
		},
	}

	for _, r := range runs {
		t.Run(r.testName, func(t *testing.T) {
			outputPods, outErr := getPodsToPreempt(r.preemptor, r.inputPods, r.insufficientResources)
			if !r.expectErr && outErr != nil {
				t.Errorf("getPodsToPreempt returned an unexpected error during the %s test.  Err: %v", r.testName, outErr)
			} else if r.expectErr && outErr == nil {
				t.Errorf("getPodsToPreempt expected an error but returned a successful output=%v during the %s test.", outputPods, r.testName)
			} else if !podListEqual(r.expectedOutput, outputPods) {
				t.Errorf("getPodsToPreempt expected %v but got %v during the %s test.", r.expectedOutput, outputPods, r.testName)
			}
		})
	}
}

func TestAdmissionRequirementsDistance(t *testing.T) {
	type testRun struct {
		testName       string
		requirements   admissionRequirementList
		inputPod       *v1.Pod
		expectedOutput float64
	}
	allPods := getTestPods()
	runs := []testRun{
		{
			testName:       "no requirements",
			requirements:   getAdmissionRequirementList(0, 0, 0),
			inputPod:       allPods[burstable],
			expectedOutput: 0,
		},
		{
			testName:       "no requests, some requirements",
			requirements:   getAdmissionRequirementList(100, 100, 1),
			inputPod:       allPods[bestEffort],
			expectedOutput: 2,
		},
		{
			testName:       "equal requests and requirements",
			requirements:   getAdmissionRequirementList(100, 100, 1),
			inputPod:       allPods[burstable],
			expectedOutput: 0,
		},
		{
			testName:       "higher requests than requirements",
			requirements:   getAdmissionRequirementList(50, 50, 0),
			inputPod:       allPods[burstable],
			expectedOutput: 0,
		},
	}
	for _, run := range runs {
		t.Run(run.testName, func(t *testing.T) {
			output := run.requirements.distance(run.inputPod)
			if output != run.expectedOutput {
				t.Errorf("expected: %f, got: %f for %s test", run.expectedOutput, output, run.testName)
			}
		})
	}
}

func TestAdmissionRequirementsSubtract(t *testing.T) {
	type testRun struct {
		testName       string
		initial        admissionRequirementList
		inputPod       *v1.Pod
		expectedOutput admissionRequirementList
	}
	allPods := getTestPods()
	runs := []testRun{
		{
			testName:       "subtract a pod from no requirements",
			initial:        getAdmissionRequirementList(0, 0, 0),
			inputPod:       allPods[burstable],
			expectedOutput: getAdmissionRequirementList(0, 0, 0),
		},
		{
			testName:       "subtract no requests from some requirements",
			initial:        getAdmissionRequirementList(100, 100, 1),
			inputPod:       allPods[bestEffort],
			expectedOutput: getAdmissionRequirementList(100, 100, 0),
		},
		{
			testName:       "equal requests and requirements",
			initial:        getAdmissionRequirementList(100, 100, 1),
			inputPod:       allPods[burstable],
			expectedOutput: getAdmissionRequirementList(0, 0, 0),
		},
		{
			testName:       "subtract higher requests than requirements",
			initial:        getAdmissionRequirementList(50, 50, 0),
			inputPod:       allPods[burstable],
			expectedOutput: getAdmissionRequirementList(0, 0, 0),
		},
		{
			testName:       "subtract lower requests than requirements",
			initial:        getAdmissionRequirementList(200, 200, 1),
			inputPod:       allPods[burstable],
			expectedOutput: getAdmissionRequirementList(100, 100, 0),
		},
	}
	for _, run := range runs {
		t.Run(run.testName, func(t *testing.T) {
			output := run.initial.subtract(run.inputPod)
			if !admissionRequirementListEqual(output, run.expectedOutput) {
				t.Errorf("expected: %s, got: %s for %s test", run.expectedOutput.toString(), output.toString(), run.testName)
			}
		})
	}
}

func TestSmallerResourceRequest(t *testing.T) {
	type testRun struct {
		testName       string
		pod1           *v1.Pod
		pod2           *v1.Pod
		expectedResult bool
	}

	podWithNoRequests := getPodWithResources("no-requests", v1.ResourceRequirements{})
	podWithLowMemory := getPodWithResources("low-memory", v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("50Mi"),
			v1.ResourceCPU:    resource.MustParse("100m"),
		},
	})
	podWithHighMemory := getPodWithResources("high-memory", v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("200Mi"),
			v1.ResourceCPU:    resource.MustParse("100m"),
		},
	})
	podWithHighCPU := getPodWithResources("high-cpu", v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("50Mi"),
			v1.ResourceCPU:    resource.MustParse("200m"),
		},
	})
	runs := []testRun{
		{
			testName:       "some requests vs no requests should return false",
			pod1:           podWithLowMemory,
			pod2:           podWithNoRequests,
			expectedResult: false,
		},
		{
			testName:       "lower memory should return true",
			pod1:           podWithLowMemory,
			pod2:           podWithHighMemory,
			expectedResult: true,
		},
		{
			testName:       "memory priority over CPU",
			pod1:           podWithHighMemory,
			pod2:           podWithHighCPU,
			expectedResult: false,
		},
		{
			testName:       "equal resource request should return true",
			pod1:           podWithLowMemory,
			pod2:           podWithLowMemory,
			expectedResult: true,
		},
		{
			testName: "resource type other than CPU and memory are ignored",
			pod1: getPodWithResources("high-storage", v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: resource.MustParse("300Mi"),
				},
			}),
			pod2: getPodWithResources("low-storage", v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: resource.MustParse("200Mi"),
				},
			}),
			expectedResult: true,
		},
	}
	for _, run := range runs {
		t.Run(run.testName, func(t *testing.T) {
			result := smallerResourceRequest(run.pod1, run.pod2)
			if result != run.expectedResult {
				t.Fatalf("smallerResourceRequest(%s, %s) = %v, expected %v",
					run.pod1.Name, run.pod2.Name, result, run.expectedResult)
			}
		})
	}
}

func getTestPods() map[string]*v1.Pod {
	allPods := map[string]*v1.Pod{
		tinyBurstable: getPodWithResources(tinyBurstable, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1m"),
				v1.ResourceMemory: resource.MustParse("1Mi"),
			},
		}),
		bestEffort: getPodWithResources(bestEffort, v1.ResourceRequirements{}),
		clusterCritical: getPodWithResources(clusterCritical, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
		}),
		nodeCritical: getPodWithResources(nodeCritical, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
		}),
		burstable: getPodWithResources(burstable, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
		}),
		guaranteed: getPodWithResources(guaranteed, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
			Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			},
		}),
		highRequestBurstable: getPodWithResources(highRequestBurstable, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("300m"),
				v1.ResourceMemory: resource.MustParse("300Mi"),
			},
		}),
		highRequestGuaranteed: getPodWithResources(highRequestGuaranteed, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("300m"),
				v1.ResourceMemory: resource.MustParse("300Mi"),
			},
			Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("300m"),
				v1.ResourceMemory: resource.MustParse("300Mi"),
			},
		}),
		highPriorityBurstable: getPodWithResources(highPriorityBurstable, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("200Mi"),
			},
		}),
	}
	allPods[clusterCritical].Namespace = kubeapi.NamespaceSystem
	allPods[clusterCritical].Spec.PriorityClassName = scheduling.SystemClusterCritical
	clusterPriority := scheduling.SystemCriticalPriority
	allPods[clusterCritical].Spec.Priority = &clusterPriority

	allPods[nodeCritical].Namespace = kubeapi.NamespaceSystem
	allPods[nodeCritical].Spec.PriorityClassName = scheduling.SystemNodeCritical
	nodePriority := scheduling.SystemCriticalPriority + 100
	allPods[nodeCritical].Spec.Priority = &nodePriority

	priorityBurstable := int32(100)
	allPods[highPriorityBurstable].Spec.Priority = &priorityBurstable

	return allPods
}

func getPodWithResources(name string, requests v1.ResourceRequirements) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: name,
			Annotations:  map[string]string{},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      fmt.Sprintf("%s-container", name),
					Resources: requests,
				},
			},
		},
	}
}

func parseCPUToInt64(res string) int64 {
	r := resource.MustParse(res)
	return (&r).MilliValue()
}

func parseNonCPUResourceToInt64(res string) int64 {
	r := resource.MustParse(res)
	return (&r).Value()
}

func getAdmissionRequirementList(cpu, memory, pods int) admissionRequirementList {
	reqs := []*admissionRequirement{}
	if cpu > 0 {
		reqs = append(reqs, &admissionRequirement{
			resourceName: v1.ResourceCPU,
			quantity:     parseCPUToInt64(fmt.Sprintf("%dm", cpu)),
		})
	}
	if memory > 0 {
		reqs = append(reqs, &admissionRequirement{
			resourceName: v1.ResourceMemory,
			quantity:     parseNonCPUResourceToInt64(fmt.Sprintf("%dMi", memory)),
		})
	}
	if pods > 0 {
		reqs = append(reqs, &admissionRequirement{
			resourceName: v1.ResourcePods,
			quantity:     int64(pods),
		})
	}
	return admissionRequirementList(reqs)
}

func getPredicateFailureReasons(insufficientCPU, insufficientMemory, insufficientPods int, otherReasonExist bool) (reasonByPredicate []lifecycle.PredicateFailureReason) {
	if insufficientCPU > 0 {
		parsedN := parseCPUToInt64(fmt.Sprintf("%dm", insufficientCPU))
		reasonByPredicate = append(reasonByPredicate, &lifecycle.InsufficientResourceError{
			ResourceName: v1.ResourceCPU,
			Requested:    parsedN,
			Capacity:     parsedN * 5 / 4,
			Used:         parsedN * 5 / 4,
		})
	}
	if insufficientMemory > 0 {
		parsedN := parseNonCPUResourceToInt64(fmt.Sprintf("%dMi", insufficientMemory))
		reasonByPredicate = append(reasonByPredicate, &lifecycle.InsufficientResourceError{
			ResourceName: v1.ResourceMemory,
			Requested:    parsedN,
			Capacity:     parsedN * 5 / 4,
			Used:         parsedN * 5 / 4,
		})
	}
	if insufficientPods > 0 {
		parsedN := int64(insufficientPods)
		reasonByPredicate = append(reasonByPredicate, &lifecycle.InsufficientResourceError{
			ResourceName: v1.ResourcePods,
			Requested:    parsedN,
			Capacity:     parsedN + 1,
			Used:         parsedN + 1,
		})
	}
	if otherReasonExist {
		reasonByPredicate = append(reasonByPredicate, &lifecycle.PredicateFailureError{
			PredicateName: "mock predicate error name",
			PredicateDesc: "mock predicate error reason",
		})
	}
	return
}

// this checks if the lists contents contain all of the same elements.
// this is not correct if there are duplicate pods in the list.
// for example: podListEqual([a, a, b], [a, b, b]) will return true
func admissionRequirementListEqual(list1 admissionRequirementList, list2 admissionRequirementList) bool {
	if len(list1) != len(list2) {
		return false
	}
	for _, a := range list1 {
		contains := false
		for _, b := range list2 {
			if a.resourceName == b.resourceName && a.quantity == b.quantity {
				contains = true
			}
		}
		if !contains {
			return false
		}
	}
	return true
}

// podListEqual checks if the lists contents contain all of the same elements.
func podListEqual(list1 []*v1.Pod, list2 []*v1.Pod) bool {
	if len(list1) != len(list2) {
		return false
	}

	m := map[*v1.Pod]int{}
	for _, val := range list1 {
		m[val] = m[val] + 1
	}
	for _, val := range list2 {
		m[val] = m[val] - 1
	}
	for _, v := range m {
		if v != 0 {
			return false
		}
	}
	return true
}
