/*
Copyright 2015 The Kubernetes Authors.

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

package resource

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestResourceHelpers(t *testing.T) {
	cpuLimit := resource.MustParse("10")
	memoryLimit := resource.MustParse("10G")
	resourceSpec := v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceCPU:    cpuLimit,
			v1.ResourceMemory: memoryLimit,
		},
	}
	if res := resourceSpec.Limits.Cpu(); res.Cmp(cpuLimit) != 0 {
		t.Errorf("expected cpulimit %v, got %v", cpuLimit, res)
	}
	if res := resourceSpec.Limits.Memory(); res.Cmp(memoryLimit) != 0 {
		t.Errorf("expected memorylimit %v, got %v", memoryLimit, res)
	}
	resourceSpec = v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceMemory: memoryLimit,
		},
	}
	if res := resourceSpec.Limits.Cpu(); res.Value() != 0 {
		t.Errorf("expected cpulimit %v, got %v", 0, res)
	}
	if res := resourceSpec.Limits.Memory(); res.Cmp(memoryLimit) != 0 {
		t.Errorf("expected memorylimit %v, got %v", memoryLimit, res)
	}
}

func TestDefaultResourceHelpers(t *testing.T) {
	resourceList := v1.ResourceList{}
	if resourceList.Cpu().Format != resource.DecimalSI {
		t.Errorf("expected %v, actual %v", resource.DecimalSI, resourceList.Cpu().Format)
	}
	if resourceList.Memory().Format != resource.BinarySI {
		t.Errorf("expected %v, actual %v", resource.BinarySI, resourceList.Memory().Format)
	}
}

func TestGetResourceRequest(t *testing.T) {
	cases := []struct {
		pod           *v1.Pod
		cName         string
		resourceName  v1.ResourceName
		expectedValue int64
	}{
		{
			pod:           getPod("foo", podResources{cpuRequest: "9"}),
			resourceName:  v1.ResourceCPU,
			expectedValue: 9000,
		},
		{
			pod:           getPod("foo", podResources{memoryRequest: "90Mi"}),
			resourceName:  v1.ResourceMemory,
			expectedValue: 94371840,
		},
		{
			cName:         "just-overhead for cpu",
			pod:           getPod("foo", podResources{cpuOverhead: "5", memoryOverhead: "5"}),
			resourceName:  v1.ResourceCPU,
			expectedValue: 0,
		},
		{
			cName:         "just-overhead for memory",
			pod:           getPod("foo", podResources{memoryOverhead: "5"}),
			resourceName:  v1.ResourceMemory,
			expectedValue: 0,
		},
		{
			cName:         "cpu overhead and req",
			pod:           getPod("foo", podResources{cpuRequest: "2", cpuOverhead: "5", memoryOverhead: "5"}),
			resourceName:  v1.ResourceCPU,
			expectedValue: 7000,
		},
		{
			cName:         "mem overhead and req",
			pod:           getPod("foo", podResources{cpuRequest: "2", memoryRequest: "1024", cpuOverhead: "5", memoryOverhead: "5"}),
			resourceName:  v1.ResourceMemory,
			expectedValue: 1029,
		},
	}
	as := assert.New(t)
	for idx, tc := range cases {
		actual := GetResourceRequest(tc.pod, tc.resourceName)
		as.Equal(tc.expectedValue, actual, "expected test case [%d] %v: to return %q; got %q instead", idx, tc.cName, tc.expectedValue, actual)
	}
}

func TestExtractResourceValue(t *testing.T) {
	cases := []struct {
		fs            *v1.ResourceFieldSelector
		pod           *v1.Pod
		cName         string
		expectedValue string
		expectedError error
	}{
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{cpuLimit: "9"}),
			expectedValue: "9",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{}),
			expectedValue: "0",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{cpuRequest: "8"}),
			expectedValue: "8",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{cpuRequest: "100m"}),
			expectedValue: "1",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("100m"),
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{cpuRequest: "1200m"}),
			expectedValue: "12",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{memoryRequest: "100Mi"}),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
				Divisor:  resource.MustParse("1Mi"),
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{memoryRequest: "100Mi", memoryLimit: "1Gi"}),
			expectedValue: "100",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.memory",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{memoryRequest: "10Mi", memoryLimit: "100Mi"}),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.cpu",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{cpuLimit: "9"}),
			expectedValue: "9",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{}),
			expectedValue: "0",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{cpuRequest: "8"}),
			expectedValue: "8",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{cpuRequest: "100m"}),
			expectedValue: "1",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("100m"),
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{cpuRequest: "1200m"}),
			expectedValue: "12",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{memoryRequest: "100Mi"}),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
				Divisor:  resource.MustParse("1Mi"),
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{memoryRequest: "100Mi", memoryLimit: "1Gi"}),
			expectedValue: "100",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.memory",
			},
			cName: "init-foo",
			pod:   getPod("foo", podResources{memoryRequest: "10Mi", memoryLimit: "100Mi"}),

			expectedValue: "104857600",
		},
	}
	as := assert.New(t)
	for idx, tc := range cases {
		actual, err := ExtractResourceValueByContainerName(tc.fs, tc.pod, tc.cName)
		if tc.expectedError != nil {
			require.EqualError(t, err, tc.expectedError.Error(), "expected test case [%d] to fail with error %v; got %v", idx, tc.expectedError, err)
		} else {
			require.NoError(t, err, "expected test case [%d] to not return an error; got %v", idx, err)
			as.Equal(tc.expectedValue, actual, "expected test case [%d] to return %q; got %q instead", idx, tc.expectedValue, actual)
		}
	}
}

func TestPodRequestsAndLimits(t *testing.T) {
	cases := []struct {
		pod              *v1.Pod
		cName            string
		expectedRequests v1.ResourceList
		expectedLimits   v1.ResourceList
	}{
		{
			cName:            "just-limit-no-overhead",
			pod:              getPod("foo", podResources{cpuLimit: "9"}),
			expectedRequests: v1.ResourceList{},
			expectedLimits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU): resource.MustParse("9"),
			},
		},
		{
			cName: "just-overhead",
			pod:   getPod("foo", podResources{cpuOverhead: "5", memoryOverhead: "5"}),
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("5"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("5"),
			},
			expectedLimits: v1.ResourceList{},
		},
		{
			cName: "req-and-overhead",
			pod:   getPod("foo", podResources{cpuRequest: "1", memoryRequest: "10", cpuOverhead: "5", memoryOverhead: "5"}),
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("6"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("15"),
			},
			expectedLimits: v1.ResourceList{},
		},
		{
			cName: "all-req-lim-and-overhead",
			pod:   getPod("foo", podResources{cpuRequest: "1", cpuLimit: "2", memoryRequest: "10", memoryLimit: "12", cpuOverhead: "5", memoryOverhead: "5"}),
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("6"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("15"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("7"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("17"),
			},
		},
		{
			cName: "req-some-lim-and-overhead",
			pod:   getPod("foo", podResources{cpuRequest: "1", cpuLimit: "2", memoryRequest: "10", cpuOverhead: "5", memoryOverhead: "5"}),
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("6"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("15"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU): resource.MustParse("7"),
			},
		},
	}
	for idx, tc := range cases {
		resRequests := PodRequests(tc.pod, PodResourcesOptions{})
		resLimits := PodLimits(tc.pod, PodResourcesOptions{})

		if !equality.Semantic.DeepEqual(tc.expectedRequests, resRequests) {
			t.Errorf("test case failure[%d]: %v, requests:\n expected:\t%v\ngot\t\t%v", idx, tc.cName, tc.expectedRequests, resRequests)
		}

		if !equality.Semantic.DeepEqual(tc.expectedLimits, resLimits) {
			t.Errorf("test case failure[%d]: %v, limits:\n expected:\t%v\ngot\t\t%v", idx, tc.cName, tc.expectedLimits, resLimits)
		}
	}
}

func TestPodRequestsAndLimitsWithoutOverhead(t *testing.T) {
	cases := []struct {
		pod              *v1.Pod
		name             string
		expectedRequests v1.ResourceList
		expectedLimits   v1.ResourceList
	}{
		{
			name: "two container no overhead - should just be sum of containers",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("2"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("10"),
								},
							},
						},
						{
							Name: "foobar2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("4"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("12"),
								},
								Limits: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("8"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("24"),
								},
							},
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("5"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("17"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("34"),
			},
		},
		{
			name: "two container with overhead - shouldn't consider overhead",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceName(v1.ResourceCPU):    resource.MustParse("3"),
						v1.ResourceName(v1.ResourceMemory): resource.MustParse("8"),
					},
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("2"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("10"),
								},
							},
						},
						{
							Name: "foobar2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("4"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("12"),
								},
								Limits: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("8"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("24"),
								},
							},
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("5"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("17"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("34"),
			},
		},
		{
			name: "two container with overhead, massive init - should just be the largest init",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceName(v1.ResourceCPU):    resource.MustParse("3"),
						v1.ResourceName(v1.ResourceMemory): resource.MustParse("8"),
					},
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("2"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("10"),
								},
							},
						},
						{
							Name: "foobar2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("4"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("12"),
								},
								Limits: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("8"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("24"),
								},
							},
						},
					},
					InitContainers: []v1.Container{
						{
							Name: "small-init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("5"),
								},
							},
						},
						{
							Name: "big-init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("40"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("120"),
								},
								Limits: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("80"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("240"),
								},
							},
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("40"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("120"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("80"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("240"),
			},
		},
	}
	for idx, tc := range cases {
		resRequests := PodRequests(tc.pod, PodResourcesOptions{ExcludeOverhead: true})
		resLimits := PodLimits(tc.pod, PodResourcesOptions{ExcludeOverhead: true})

		if !equality.Semantic.DeepEqual(tc.expectedRequests, resRequests) {
			t.Errorf("test case failure[%d]: %v, requests:\n expected:\t%v\ngot\t\t%v", idx, tc.name, tc.expectedRequests, resRequests)
		}

		if !equality.Semantic.DeepEqual(tc.expectedLimits, resLimits) {
			t.Errorf("test case failure[%d]: %v, limits:\n expected:\t%v\ngot\t\t%v", idx, tc.name, tc.expectedLimits, resLimits)
		}
	}
}

type podResources struct {
	cpuRequest, cpuLimit, memoryRequest, memoryLimit, cpuOverhead, memoryOverhead string
}

func getPod(cname string, resources podResources) *v1.Pod {
	r := v1.ResourceRequirements{
		Limits:   make(v1.ResourceList),
		Requests: make(v1.ResourceList),
	}

	overhead := make(v1.ResourceList)

	if resources.cpuLimit != "" {
		r.Limits[v1.ResourceCPU] = resource.MustParse(resources.cpuLimit)
	}
	if resources.memoryLimit != "" {
		r.Limits[v1.ResourceMemory] = resource.MustParse(resources.memoryLimit)
	}
	if resources.cpuRequest != "" {
		r.Requests[v1.ResourceCPU] = resource.MustParse(resources.cpuRequest)
	}
	if resources.memoryRequest != "" {
		r.Requests[v1.ResourceMemory] = resource.MustParse(resources.memoryRequest)
	}
	if resources.cpuOverhead != "" {
		overhead[v1.ResourceCPU] = resource.MustParse(resources.cpuOverhead)
	}
	if resources.memoryOverhead != "" {
		overhead[v1.ResourceMemory] = resource.MustParse(resources.memoryOverhead)
	}

	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      cname,
					Resources: r,
				},
			},
			InitContainers: []v1.Container{
				{
					Name:      "init-" + cname,
					Resources: r,
				},
			},
			Overhead: overhead,
		},
	}
}

func TestPodResourceRequests(t *testing.T) {
	restartAlways := v1.ContainerRestartPolicyAlways
	testCases := []struct {
		description      string
		options          PodResourcesOptions
		overhead         v1.ResourceList
		podResizeStatus  v1.PodResizeStatus
		initContainers   []v1.Container
		containers       []v1.Container
		containerStatus  []v1.ContainerStatus
		expectedRequests v1.ResourceList
	}{
		{
			description: "nil options, larger init container",
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("4"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
		},
		{
			description: "nil options, larger containers",
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("5"),
			},
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
		},
		{
			description: "pod overhead excluded",
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("5"),
			},
			options: PodResourcesOptions{
				ExcludeOverhead: true,
			},
			overhead: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("1"),
			},
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
		},
		{
			description: "pod overhead included",
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("6"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			overhead: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
		},
		{
			description: "resized, infeasible",
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("2"),
			},
			podResizeStatus: v1.PodResizeStatusInfeasible,
			options:         PodResourcesOptions{InPlacePodVerticalScalingEnabled: true},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("4"),
						},
					},
				},
			},
			containerStatus: []v1.ContainerStatus{
				{
					Name: "container-1",
					AllocatedResources: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("2"),
					},
				},
			},
		},
		{
			description: "resized, no resize status",
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
			options: PodResourcesOptions{InPlacePodVerticalScalingEnabled: true},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("4"),
						},
					},
				},
			},
			containerStatus: []v1.ContainerStatus{
				{
					Name: "container-1",
					AllocatedResources: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("2"),
					},
				},
			},
		},
		{
			description: "resized, infeasible, feature gate disabled",
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
			podResizeStatus: v1.PodResizeStatusInfeasible,
			options:         PodResourcesOptions{InPlacePodVerticalScalingEnabled: false},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("4"),
						},
					},
				},
			},
			containerStatus: []v1.ContainerStatus{
				{
					Name: "container-1",
					AllocatedResources: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("2"),
					},
				},
			},
		},
		{
			description: "restartable init container",
			expectedRequests: v1.ResourceList{
				// restartable init + regular container
				v1.ResourceCPU: resource.MustParse("2"),
			},
			initContainers: []v1.Container{
				{
					Name:          "restartable-init-1",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
		},
		{
			description: "multiple restartable init containers",
			expectedRequests: v1.ResourceList{
				// max(5, restartable init containers(3+2+1) + regular(1)) = 7
				v1.ResourceCPU: resource.MustParse("7"),
			},
			initContainers: []v1.Container{
				{
					Name: "init-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("5"),
						},
					},
				},
				{
					Name:          "restartable-init-1",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
				{
					Name:          "restartable-init-2",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Name:          "restartable-init-3",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
		},
		{
			description: "multiple restartable and regular init containers",
			expectedRequests: v1.ResourceList{
				// init-2 requires 5 + the previously running restartable init
				// containers(1+2) = 8, the restartable init container that starts
				// after it doesn't count
				v1.ResourceCPU: resource.MustParse("8"),
			},
			initContainers: []v1.Container{
				{
					Name: "init-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("5"),
						},
					},
				},
				{
					Name:          "restartable-init-1",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
				{
					Name:          "restartable-init-2",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Name: "init-2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("5"),
						},
					},
				},
				{
					Name:          "restartable-init-3",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
		},
		{
			description: "restartable-init, init and regular",
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("210"),
			},
			initContainers: []v1.Container{
				{
					Name:          "restartable-init-1",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("10"),
						},
					},
				},
				{
					Name: "init-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("200"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100"),
						},
					},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			p := &v1.Pod{
				Spec: v1.PodSpec{
					Containers:     tc.containers,
					InitContainers: tc.initContainers,
					Overhead:       tc.overhead,
				},
				Status: v1.PodStatus{
					ContainerStatuses: tc.containerStatus,
					Resize:            tc.podResizeStatus,
				},
			}
			request := PodRequests(p, tc.options)
			if !resourcesEqual(tc.expectedRequests, request) {
				t.Errorf("[%s] expected requests = %v, got %v", tc.description, tc.expectedRequests, request)
			}
		})
	}
}

func TestPodResourceRequestsReuse(t *testing.T) {
	expectedRequests := v1.ResourceList{
		v1.ResourceCPU: resource.MustParse("1"),
	}
	p := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: expectedRequests,
					},
				},
			},
		},
	}

	opts := PodResourcesOptions{
		Reuse: v1.ResourceList{
			v1.ResourceCPU: resource.MustParse("25"),
		},
	}
	requests := PodRequests(p, opts)

	if !resourcesEqual(expectedRequests, requests) {
		t.Errorf("expected requests = %v, got %v", expectedRequests, requests)
	}

	// should re-use the maps we passed in
	if !resourcesEqual(expectedRequests, opts.Reuse) {
		t.Errorf("expected to re-use the requests")
	}
}

func TestPodResourceLimits(t *testing.T) {
	restartAlways := v1.ContainerRestartPolicyAlways
	testCases := []struct {
		description    string
		options        PodResourcesOptions
		overhead       v1.ResourceList
		initContainers []v1.Container
		containers     []v1.Container
		expectedLimits v1.ResourceList
	}{
		{
			description: "nil options, larger init container",
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("4"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
		},
		{
			description: "nil options, larger containers",
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("5"),
			},
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
		},
		{
			description: "pod overhead excluded",
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("5"),
			},
			options: PodResourcesOptions{
				ExcludeOverhead: true,
			},
			overhead: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("1"),
			},
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
		},
		{
			description: "pod overhead included",
			overhead: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("6"),
				// overhead is only added to non-zero limits, so there will be no expected memory limit
			},
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
		},
		{
			description:    "no limited containers should result in no limits for the pod",
			expectedLimits: v1.ResourceList{},
			initContainers: []v1.Container{},
			containers: []v1.Container{
				{
					// Unlimited container
				},
			},
		},
		{
			description: "one limited and one unlimited container should result in the limited container's limits for the pod",
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("2"),
				v1.ResourceMemory: resource.MustParse("2Gi"),
			},
			initContainers: []v1.Container{},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("2"),
							v1.ResourceMemory: resource.MustParse("2Gi"),
						},
					},
				},
				{
					// Unlimited container
				},
			},
		},
		{
			description: "one limited and one unlimited init container should result in the limited init container's limits for the pod",
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("2"),
				v1.ResourceMemory: resource.MustParse("2Gi"),
			},
			initContainers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("2"),
							v1.ResourceMemory: resource.MustParse("2Gi"),
						},
					},
				},
				{
					// Unlimited init container
				},
			},
			containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("1"),
							v1.ResourceMemory: resource.MustParse("1Gi"),
						},
					},
				},
			},
		},
		{
			description: "restartable init container",
			expectedLimits: v1.ResourceList{
				// restartable init + regular container
				v1.ResourceCPU: resource.MustParse("2"),
			},
			initContainers: []v1.Container{
				{
					Name:          "restartable-init-1",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
		},
		{
			description: "multiple restartable init containers",
			expectedLimits: v1.ResourceList{
				// max(5, restartable init containers(3+2+1) + regular(1)) = 7
				v1.ResourceCPU: resource.MustParse("7"),
			},
			initContainers: []v1.Container{
				{
					Name: "init-1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("5"),
						},
					},
				},
				{
					Name:          "restartable-init-1",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
				{
					Name:          "restartable-init-2",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Name:          "restartable-init-3",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
		},
		{
			description: "multiple restartable and regular init containers",
			expectedLimits: v1.ResourceList{
				// init-2 requires 5 + the previously running restartable init
				// containers(1+2) = 8, the restartable init container that starts
				// after it doesn't count
				v1.ResourceCPU: resource.MustParse("8"),
			},
			initContainers: []v1.Container{
				{
					Name: "init-1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("5"),
						},
					},
				},
				{
					Name:          "restartable-init-1",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
				{
					Name:          "restartable-init-2",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
				},
				{
					Name: "init-2",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("5"),
						},
					},
				},
				{
					Name:          "restartable-init-3",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("3"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
		},
		{
			description: "restartable-init, init and regular",
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("210"),
			},
			initContainers: []v1.Container{
				{
					Name:          "restartable-init-1",
					RestartPolicy: &restartAlways,
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("10"),
						},
					},
				},
				{
					Name: "init-1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("200"),
						},
					},
				},
			},
			containers: []v1.Container{
				{
					Name: "container-1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100"),
						},
					},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			p := &v1.Pod{
				Spec: v1.PodSpec{
					Containers:     tc.containers,
					InitContainers: tc.initContainers,
					Overhead:       tc.overhead,
				},
			}
			limits := PodLimits(p, tc.options)
			if !resourcesEqual(tc.expectedLimits, limits) {
				t.Errorf("[%s] expected limits = %v, got %v", tc.description, tc.expectedLimits, limits)
			}
		})
	}
}

func resourcesEqual(lhs, rhs v1.ResourceList) bool {
	if len(lhs) != len(rhs) {
		return false
	}
	for name, lhsv := range lhs {
		rhsv, ok := rhs[name]
		if !ok {
			return false
		}
		if !lhsv.Equal(rhsv) {
			return false
		}
	}
	return true
}
