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
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestResourceHelpers(t *testing.T) {
	cpuLimit := resource.MustParse("10")
	memoryLimit := resource.MustParse("10G")
	resourceSpec := api.ResourceRequirements{
		Limits: api.ResourceList{
			api.ResourceCPU:    cpuLimit,
			api.ResourceMemory: memoryLimit,
		},
	}
	if res := resourceSpec.Limits.Cpu(); res.Cmp(cpuLimit) != 0 {
		t.Errorf("expected cpulimit %v, got %v", cpuLimit, res)
	}
	if res := resourceSpec.Limits.Memory(); res.Cmp(memoryLimit) != 0 {
		t.Errorf("expected memorylimit %v, got %v", memoryLimit, res)
	}
	resourceSpec = api.ResourceRequirements{
		Limits: api.ResourceList{
			api.ResourceMemory: memoryLimit,
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
	resourceList := api.ResourceList{}
	if resourceList.Cpu().Format != resource.DecimalSI {
		t.Errorf("expected %v, actual %v", resource.DecimalSI, resourceList.Cpu().Format)
	}
	if resourceList.Memory().Format != resource.BinarySI {
		t.Errorf("expected %v, actual %v", resource.BinarySI, resourceList.Memory().Format)
	}
}

func TestPodRequestsAndLimits(t *testing.T) {
	tests := []struct {
		name         string
		pod          *api.Pod
		expectreqs   map[api.ResourceName]resource.Quantity
		expectlimits map[api.ResourceName]resource.Quantity
	}{
		{
			name: "InitContainers resource < Containers resource",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					UID: uuid.NewUUID(),
				},
				Spec: api.PodSpec{
					InitContainers: []api.Container{
						makeContainer("100m", "200m", "100Mi", "200Mi"),
					},
					Containers: []api.Container{
						makeContainer("100m", "400m", "100Mi", "400Mi"),
						makeContainer("100m", "200m", "100Mi", "200Mi"),
					},
				},
			},
			expectreqs: map[api.ResourceName]resource.Quantity{
				api.ResourceCPU:    resource.MustParse("200m"),
				api.ResourceMemory: resource.MustParse("200Mi"),
			},
			expectlimits: map[api.ResourceName]resource.Quantity{
				api.ResourceCPU:    resource.MustParse("600m"),
				api.ResourceMemory: resource.MustParse("600Mi"),
			},
		},
		{
			name: "InitContainers resource > Containers resource",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					UID: uuid.NewUUID(),
				},
				Spec: api.PodSpec{
					InitContainers: []api.Container{
						makeContainer("500m", "500m", "500Mi", "500Mi"),
						makeContainer("100m", "200m", "100Mi", "200Mi"),
					},
					Containers: []api.Container{
						makeContainer("100m", "200m", "100Mi", "200Mi"),
						makeContainer("100m", "200m", "100Mi", "200Mi"),
					},
				},
			},
			expectreqs: map[api.ResourceName]resource.Quantity{
				api.ResourceCPU:    resource.MustParse("500m"),
				api.ResourceMemory: resource.MustParse("500Mi"),
			},
			expectlimits: map[api.ResourceName]resource.Quantity{
				api.ResourceCPU:    resource.MustParse("500m"),
				api.ResourceMemory: resource.MustParse("500Mi"),
			},
		},
	}
	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requests, limits := PodRequestsAndLimits(test.pod)
			for name, quantity := range test.expectreqs {
				if value, ok := requests[name]; !ok {
					t.Errorf("case[%d]: Error to get value of key=%s from requests result", i, name)
				} else if quantity.Cmp(value) != 0 {
					t.Errorf("case[%d]: Expected value of key=%s from requests result:%v, Got:%v", i, name, quantity, value)
				}
			}
			for name, quantity := range test.expectlimits {
				if value, ok := limits[name]; !ok {
					t.Errorf("case[%d]: Error to get value of key=%s from limits result", i, name)
				} else if quantity.Cmp(value) != 0 {
					t.Errorf("case[%d]: Expected value of key=%s from limits result:%v, Got:%v", i, name, quantity, value)
				}
			}
		})
	}
}

func TestExtractContainerResourceValue(t *testing.T) {
	testcontainer := &api.Container{
		Name: string(uuid.NewUUID()),
		Resources: api.ResourceRequirements{
			Limits: api.ResourceList{
				api.ResourceCPU:              resource.MustParse("200m"),
				api.ResourceMemory:           resource.MustParse("200Mi"),
				api.ResourceStorage:          resource.MustParse("2G"),
				api.ResourceEphemeralStorage: resource.MustParse("2G"),
			},
			Requests: api.ResourceList{
				api.ResourceCPU:              resource.MustParse("100m"),
				api.ResourceMemory:           resource.MustParse("100Mi"),
				api.ResourceStorage:          resource.MustParse("1G"),
				api.ResourceEphemeralStorage: resource.MustParse("300Mi"),
			},
		},
	}
	tests := []struct {
		name         string
		filed        *api.ResourceFieldSelector
		expectresult string
		expecterr    string
	}{
		{
			name:         "get limits cpu",
			filed:        &api.ResourceFieldSelector{Resource: "limits.cpu", Divisor: resource.MustParse("1m")},
			expectresult: "200",
			expecterr:    "",
		},
		{
			name:         "get limits memory",
			filed:        &api.ResourceFieldSelector{Resource: "limits.memory"},
			expectresult: "209715200",
			expecterr:    "",
		},
		{
			name:         "get limits ephemeral-storage",
			filed:        &api.ResourceFieldSelector{Resource: "limits.ephemeral-storage", Divisor: resource.MustParse("1G")},
			expectresult: "2",
			expecterr:    "",
		},
		{
			name:         "get requests cpu",
			filed:        &api.ResourceFieldSelector{Resource: "requests.cpu", Divisor: resource.MustParse("1m")},
			expectresult: "100",
			expecterr:    "",
		},
		{
			name:         "get requests memory",
			filed:        &api.ResourceFieldSelector{Resource: "requests.memory", Divisor: resource.MustParse("1G")},
			expectresult: "1",
			expecterr:    "",
		},
		{
			name:         "get requests ephemeral-storage",
			filed:        &api.ResourceFieldSelector{Resource: "requests.ephemeral-storage"},
			expectresult: "314572800",
			expecterr:    "",
		},
		{
			name:         "get limits storage",
			filed:        &api.ResourceFieldSelector{Resource: "limits.storage", Divisor: resource.MustParse("1G")},
			expectresult: "",
			expecterr:    "unsupported container resource",
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result, err := ExtractContainerResourceValue(test.filed, testcontainer)
			if test.expecterr != "" {
				if !strings.Contains(err.Error(), test.expecterr) {
					t.Errorf("case[%d]:%s Expected err:%s, Got err:%s", i, test.name, test.expecterr, err.Error())
				}
			} else if err != nil {
				t.Errorf("case[%d]:%s Expected no err, Got err:%s", i, test.name, err.Error())
			}
			if result != test.expectresult {
				t.Errorf("case[%d]:%s Expected result:%s, Got result:%s", i, test.name, test.expectresult, result)
			}
		})
	}
}

func makeContainer(cpuReq, cpuLim, memReq, memLim string) api.Container {
	return api.Container{
		Name: string(uuid.NewUUID()),
		Resources: api.ResourceRequirements{
			Limits: api.ResourceList{
				api.ResourceCPU:    resource.MustParse(cpuLim),
				api.ResourceMemory: resource.MustParse(memLim),
			},
			Requests: api.ResourceList{
				api.ResourceCPU:    resource.MustParse(cpuReq),
				api.ResourceMemory: resource.MustParse(memReq),
			},
		},
	}
}
