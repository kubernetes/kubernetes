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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	kubeapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	critical              = "critical"
	bestEffort            = "bestEffort"
	burstable             = "burstable"
	highRequestBurstable  = "high-request-burstable"
	guaranteed            = "guaranteed"
	highRequestGuaranteed = "high-request-guaranteed"
)

type fakePodKiller struct {
	killedPods []*v1.Pod
}

func newFakePodKiller() *fakePodKiller {
	return &fakePodKiller{killedPods: []*v1.Pod{}}
}

func (f *fakePodKiller) clear() {
	f.killedPods = []*v1.Pod{}
}

func (f *fakePodKiller) getKilledPods() []*v1.Pod {
	return f.killedPods
}

func (f *fakePodKiller) killPodNow(pod *v1.Pod, status v1.PodStatus, gracePeriodOverride *int64) error {
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

func TestEvictPodsToFreeRequests(t *testing.T) {
	type testRun struct {
		testName              string
		inputPods             []*v1.Pod
		insufficientResources admissionRequirementList
		expectErr             bool
		expectedOutput        []*v1.Pod
	}
	podProvider := newFakePodProvider()
	podKiller := newFakePodKiller()
	criticalPodAdmissionHandler := getTestCriticalPodAdmissionHandler(podProvider, podKiller)
	allPods := getTestPods(t)
	runs := []testRun{
		{
			testName:  "critical pods cannot be preempted",
			inputPods: []*v1.Pod{allPods[critical]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourcePods,
					quantity:     int64(1),
				},
			}),
			expectErr:      true,
			expectedOutput: nil,
		},
		{
			testName:  "best effort pods are not preempted",
			inputPods: []*v1.Pod{allPods[bestEffort]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourceMemory,
					quantity:     parseNonCpuResourceToInt64("1Mi"),
				},
			}),
			expectErr:      true,
			expectedOutput: nil,
		},
		{
			testName: "multiple pods evicted",
			inputPods: []*v1.Pod{
				allPods[critical], allPods[bestEffort], allPods[burstable], allPods[highRequestBurstable],
				allPods[guaranteed], allPods[highRequestGuaranteed]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourceMemory,
					quantity:     parseNonCpuResourceToInt64("550Mi"),
				},
			}),
			expectErr:      false,
			expectedOutput: []*v1.Pod{allPods[highRequestBurstable], allPods[highRequestGuaranteed]},
		},
	}
	for _, r := range runs {
		podProvider.setPods(r.inputPods)
		outErr := criticalPodAdmissionHandler.evictPodsToFreeRequests(r.insufficientResources)
		outputPods := podKiller.getKilledPods()
		if !r.expectErr && outErr != nil {
			t.Errorf("evictPodsToFreeRequests returned an unexpected error during the %s test.  Err: %v", r.testName, outErr)
		} else if r.expectErr && outErr == nil {
			t.Errorf("evictPodsToFreeRequests expected an error but returned a successful output=%v during the %s test.", outputPods, r.testName)
		} else if !podListEqual(r.expectedOutput, outputPods) {
			t.Errorf("evictPodsToFreeRequests expected %v but got %v during the %s test.", r.expectedOutput, outputPods, r.testName)
		}
		podKiller.clear()
	}
}

func TestGetPodsToPreempt(t *testing.T) {
	type testRun struct {
		testName              string
		inputPods             []*v1.Pod
		insufficientResources admissionRequirementList
		expectErr             bool
		expectedOutput        []*v1.Pod
	}
	allPods := getTestPods(t)
	runs := []testRun{
		{
			testName:              "no requirements",
			inputPods:             []*v1.Pod{},
			insufficientResources: admissionRequirementList([]*admissionRequirement{}),
			expectErr:             false,
			expectedOutput:        []*v1.Pod{},
		},
		{
			testName:  "no pods",
			inputPods: []*v1.Pod{},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourcePods,
					quantity:     int64(1),
				},
			}),
			expectErr:      true,
			expectedOutput: nil,
		},
		{
			testName:  "equal pods and resources requirements",
			inputPods: []*v1.Pod{allPods[burstable]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourcePods,
					quantity:     int64(1),
				},
				{
					resourceName: v1.ResourceMemory,
					quantity:     parseNonCpuResourceToInt64("100Mi"),
				},
				{
					resourceName: v1.ResourceCPU,
					quantity:     parseCPUToInt64("100m"),
				},
			}),
			expectErr:      false,
			expectedOutput: []*v1.Pod{allPods[burstable]},
		},
		{
			testName:  "higer requirements than pod requests",
			inputPods: []*v1.Pod{allPods[burstable]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourcePods,
					quantity:     int64(2),
				},
				{
					resourceName: v1.ResourceMemory,
					quantity:     parseNonCpuResourceToInt64("200Mi"),
				},
				{
					resourceName: v1.ResourceCPU,
					quantity:     parseCPUToInt64("200m"),
				},
			}),
			expectErr:      true,
			expectedOutput: nil,
		},
		{
			testName:  "choose between burstable and guaranteed",
			inputPods: []*v1.Pod{allPods[burstable], allPods[guaranteed]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourcePods,
					quantity:     int64(1),
				},
			}),
			expectErr:      false,
			expectedOutput: []*v1.Pod{allPods[burstable]},
		},
		{
			testName:  "choose lower request burstable if it meets requirements",
			inputPods: []*v1.Pod{allPods[burstable], allPods[highRequestBurstable]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourceMemory,
					quantity:     parseNonCpuResourceToInt64("100Mi"),
				},
				{
					resourceName: v1.ResourceCPU,
					quantity:     parseCPUToInt64("100m"),
				},
			}),
			expectErr:      false,
			expectedOutput: []*v1.Pod{allPods[burstable]},
		},
		{
			testName:  "choose higher request burstable if lower does not meet requirements",
			inputPods: []*v1.Pod{allPods[burstable], allPods[highRequestBurstable]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourceMemory,
					quantity:     parseNonCpuResourceToInt64("150Mi"),
				},
				{
					resourceName: v1.ResourceCPU,
					quantity:     parseCPUToInt64("150m"),
				},
			}),
			expectErr:      false,
			expectedOutput: []*v1.Pod{allPods[highRequestBurstable]},
		},
		{
			testName:  "multiple pods required",
			inputPods: []*v1.Pod{allPods[burstable], allPods[highRequestBurstable], allPods[guaranteed], allPods[highRequestGuaranteed]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourceMemory,
					quantity:     parseNonCpuResourceToInt64("350Mi"),
				},
				{
					resourceName: v1.ResourceCPU,
					quantity:     parseCPUToInt64("350m"),
				},
			}),
			expectErr:      false,
			expectedOutput: []*v1.Pod{allPods[burstable], allPods[highRequestBurstable]},
		},
		{
			testName:  "evict guaranteed only when we have to, and dont evict the extra burstable",
			inputPods: []*v1.Pod{allPods[burstable], allPods[highRequestBurstable], allPods[guaranteed], allPods[highRequestGuaranteed]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourceMemory,
					quantity:     parseNonCpuResourceToInt64("550Mi"),
				},
			}),
			expectErr:      false,
			expectedOutput: []*v1.Pod{allPods[highRequestBurstable], allPods[highRequestGuaranteed]},
		},
		{
			testName:  "evict one guaranteed instead of one guaranteed and one burstable",
			inputPods: []*v1.Pod{allPods[burstable], allPods[guaranteed], allPods[highRequestGuaranteed]},
			insufficientResources: admissionRequirementList([]*admissionRequirement{
				{
					resourceName: v1.ResourceCPU,
					quantity:     parseCPUToInt64("150m"),
				},
			}),
			expectErr:      false,
			expectedOutput: []*v1.Pod{allPods[highRequestGuaranteed]},
		},
	}
	for _, r := range runs {
		outputPods, outErr := getPodsToPreempt(r.inputPods, r.insufficientResources)
		if !r.expectErr && outErr != nil {
			t.Errorf("getPodsToPreempt returned an unexpected error during the %s test.  Err: %v", r.testName, outErr)
		} else if r.expectErr && outErr == nil {
			t.Errorf("getPodsToPreempt expected an error but returned a successful output=%v during the %s test.", outputPods, r.testName)
		} else if !podListEqual(r.expectedOutput, outputPods) {
			t.Errorf("getPodsToPreempt expected %v but got %v during the %s test.", r.expectedOutput, outputPods, r.testName)
		}
	}
}

func TestMinCostPodList(t *testing.T) {
	type testRun struct {
		higherCostName string
		higherCostList []*v1.Pod
		lowerCostName  string
		lowerCostList  []*v1.Pod
	}
	runs := []testRun{
		{
			higherCostName: "one pod",
			higherCostList: []*v1.Pod{
				getPodWithResources("pod", v1.ResourceRequirements{
					Requests: v1.ResourceList{
						"cpu":    resource.MustParse("100m"),
						"memory": resource.MustParse("100Mi"),
					},
					Limits: v1.ResourceList{
						"cpu":    resource.MustParse("100m"),
						"memory": resource.MustParse("100Mi"),
					},
				})},
			lowerCostName: "no pods",
			lowerCostList: []*v1.Pod{},
		},
		{
			higherCostName: "one guaranteed pod",
			higherCostList: []*v1.Pod{
				getPodWithResources("guaranteed", v1.ResourceRequirements{
					Requests: v1.ResourceList{
						"cpu":    resource.MustParse("100m"),
						"memory": resource.MustParse("100Mi"),
					},
					Limits: v1.ResourceList{
						"cpu":    resource.MustParse("100m"),
						"memory": resource.MustParse("100Mi"),
					},
				})},
			lowerCostName: "one burstable pod",
			lowerCostList: []*v1.Pod{
				getPodWithResources("burstable", v1.ResourceRequirements{
					Requests: v1.ResourceList{
						"cpu":    resource.MustParse("100m"),
						"memory": resource.MustParse("100Mi"),
					},
				})},
		},
		{
			higherCostName: "two pods",
			higherCostList: []*v1.Pod{getPodWithResources("first", v1.ResourceRequirements{
				Requests: v1.ResourceList{
					"cpu":    resource.MustParse("100m"),
					"memory": resource.MustParse("100Mi"),
				},
			}), getPodWithResources("second", v1.ResourceRequirements{
				Requests: v1.ResourceList{
					"cpu":    resource.MustParse("100m"),
					"memory": resource.MustParse("100Mi"),
				},
			})},
			lowerCostName: "one pod",
			lowerCostList: []*v1.Pod{getPodWithResources("only", v1.ResourceRequirements{
				Requests: v1.ResourceList{
					"cpu":    resource.MustParse("100m"),
					"memory": resource.MustParse("100Mi"),
				},
			})},
		},
		{
			higherCostName: "high memory pod",
			higherCostList: []*v1.Pod{getPodWithResources("high-memory", v1.ResourceRequirements{
				Requests: v1.ResourceList{
					"memory": resource.MustParse("200Mi"),
				},
			})},
			lowerCostName: "low memory pod",
			lowerCostList: []*v1.Pod{getPodWithResources("low-memory", v1.ResourceRequirements{
				Requests: v1.ResourceList{
					"memory": resource.MustParse("100Mi"),
				},
			})},
		},
		{
			higherCostName: "high cpu pod",
			higherCostList: []*v1.Pod{getPodWithResources("high-cpu", v1.ResourceRequirements{
				Requests: v1.ResourceList{
					"cpu": resource.MustParse("200m"),
				},
			})},
			lowerCostName: "low cpu pod",
			lowerCostList: []*v1.Pod{getPodWithResources("low-cpu", v1.ResourceRequirements{
				Requests: v1.ResourceList{
					"cpu": resource.MustParse("100m"),
				},
			})},
		},
	}
	for _, r := range runs {
		minCostList := minCostPodList(r.higherCostList, r.lowerCostList)
		if !podListEqual(minCostList, r.lowerCostList) {
			t.Errorf("minCostPodList returned the %s list before the %s list", r.higherCostName, r.lowerCostName)
		}
	}
}

func getTestPods(t *testing.T) map[string]*v1.Pod {
	allPods := map[string]*v1.Pod{
		bestEffort: getPodWithResources(bestEffort, v1.ResourceRequirements{}),
		critical: getPodWithResources(critical, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu":    resource.MustParse("100m"),
				"memory": resource.MustParse("100Mi"),
			},
		}),
		burstable: getPodWithResources(burstable, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu":    resource.MustParse("100m"),
				"memory": resource.MustParse("100Mi"),
			},
		}),
		guaranteed: getPodWithResources(guaranteed, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu":    resource.MustParse("100m"),
				"memory": resource.MustParse("100Mi"),
			},
			Limits: v1.ResourceList{
				"cpu":    resource.MustParse("100m"),
				"memory": resource.MustParse("100Mi"),
			},
		}),
		highRequestBurstable: getPodWithResources(highRequestBurstable, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu":    resource.MustParse("300m"),
				"memory": resource.MustParse("300Mi"),
			},
		}),
		highRequestGuaranteed: getPodWithResources(highRequestGuaranteed, v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu":    resource.MustParse("300m"),
				"memory": resource.MustParse("300Mi"),
			},
			Limits: v1.ResourceList{
				"cpu":    resource.MustParse("300m"),
				"memory": resource.MustParse("300Mi"),
			},
		}),
	}
	allPods[critical].Namespace = kubeapi.NamespaceSystem
	allPods[critical].Annotations[kubetypes.CriticalPodAnnotationKey] = ""
	if !kubetypes.IsCriticalPod(allPods[critical]) {
		t.Errorf("error setting critical pod as critical")
	}
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

func parseNonCpuResourceToInt64(res string) int64 {
	r := resource.MustParse(res)
	return (&r).Value()
}

// this checks if the lists contents contain all of the same elements.
// this is not correct if there are duplicate pods in the list.
// for example: podListEqual([a, a, b], [a, b, b]) will return true
func podListEqual(list1 []*v1.Pod, list2 []*v1.Pod) bool {
	if len(list1) != len(list2) {
		return false
	}
	for _, a := range list1 {
		contains := false
		for _, b := range list2 {
			if a == b {
				contains = true
			}
		}
		if !contains {
			return false
		}
	}
	return true
}
