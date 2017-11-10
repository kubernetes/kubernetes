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

package validation

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateResourceRequirements(t *testing.T) {
	successCase := []struct {
		Name         string
		requirements v1.ResourceRequirements
	}{
		{
			Name: "GPU only setting Limits",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "GPU setting Limits equals Requests",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "Resources with GPU with Requests",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):       resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory):    resource.MustParse("10G"),
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("1"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):       resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory):    resource.MustParse("10G"),
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("1"),
				},
			},
		},
		{
			Name: "Resources with only Limits",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "Resources with only Requests",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "Resources with Requests Less Than Limits",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("9"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("9G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("9"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("9"),
				},
			},
		},
	}
	for _, tc := range successCase {
		if errs := ValidateResourceRequirements(&tc.requirements, field.NewPath("resources")); len(errs) != 0 {
			t.Errorf("%q unexpected error: %v", tc.Name, errs)
		}
	}

	errorCase := []struct {
		Name         string
		requirements v1.ResourceRequirements
	}{
		{
			Name: "GPU only setting Requests",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "GPU setting Limits less than Requests",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("11"),
				},
			},
		},
		{
			Name: "GPU setting Limits larger than Requests",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("9"),
				},
			},
		},
		{
			Name: "Resources with Requests Larger Than Limits",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("10m"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("9"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("9G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("9m"),
				},
			},
		},
		{
			Name: "Invalid Resources with Requests",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName("my.org"): resource.MustParse("10m"),
				},
			},
		},
		{
			Name: "Invalid Resources with Limits",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName("my.org"): resource.MustParse("9m"),
				},
			},
		},
	}
	for _, tc := range errorCase {
		if errs := ValidateResourceRequirements(&tc.requirements, field.NewPath("resources")); len(errs) == 0 {
			t.Errorf("%q expected error", tc.Name)
		}
	}
}

func TestValidatePodLogOptions(t *testing.T) {
	getInt64Point := func(value int64) *int64 { return &value }
	successCase := []struct {
		Name          string
		podLogOptions v1.PodLogOptions
	}{
		{
			Name:          "All pointer members are nil",
			podLogOptions: v1.PodLogOptions{},
		},
		{
			Name: "Valid TailLines",
			podLogOptions: v1.PodLogOptions{
				TailLines: getInt64Point(0),
			},
		},
		{
			Name: "Valid LimitBytes",
			podLogOptions: v1.PodLogOptions{
				LimitBytes: getInt64Point(1),
			},
		},
		{
			Name: "Valid SinceSeconds",
			podLogOptions: v1.PodLogOptions{
				SinceSeconds: getInt64Point(1),
			},
		},
		{
			Name: "Valid SinceTime",
			podLogOptions: v1.PodLogOptions{
				SinceTime: &metav1.Time{},
			},
		},
	}
	for _, tc := range successCase {
		if errs := ValidatePodLogOptions(&tc.podLogOptions); len(errs) != 0 {
			t.Errorf("%q unexpected error", tc.Name)
		}
	}

	errorCase := []struct {
		Name          string
		podLogOptions v1.PodLogOptions
	}{

		{
			Name: "Invalid TailLines",
			podLogOptions: v1.PodLogOptions{
				TailLines: getInt64Point(-1),
			},
		},
		{
			Name: "Invalid LimitBytes",
			podLogOptions: v1.PodLogOptions{
				LimitBytes: getInt64Point(0),
			},
		},
		{
			Name: "Invalid SinceSeconds",
			podLogOptions: v1.PodLogOptions{
				SinceSeconds: getInt64Point(0),
			},
		},
		{
			Name: "Set SinceSeconds and SinceTime the same time",
			podLogOptions: v1.PodLogOptions{
				SinceSeconds: getInt64Point(1),
				SinceTime:    &metav1.Time{},
			},
		},
	}
	for _, tc := range errorCase {
		if errs := ValidatePodLogOptions(&tc.podLogOptions); len(errs) == 0 {
			t.Errorf("%q expected error", tc.Name)
		}
	}
}

func TestAccumulateUniqueHostPorts(t *testing.T) {
	containers := []v1.Container{
		{
			Name: "container-1",
			Ports: []v1.ContainerPort{
				{HostPort: 111},
			},
		},
		{
			Name: "container-2",
			Ports: []v1.ContainerPort{
				{HostPort: 333},
			},
		},
		{
			Name: "container-3",
			Ports: []v1.ContainerPort{
				{HostPort: 111},
			},
		},
	}
	accumulator := sets.NewString()
	errs := AccumulateUniqueHostPorts(containers, &accumulator, field.NewPath("spec", "containers"))
	assert.Len(t, errs, 1)
	assert.Contains(t, errs[0].Error(), "111", "Expect contains 111")
}
