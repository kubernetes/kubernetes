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
			Name: "Resources with Requests equal to Limits",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
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

	var (
		positiveLine             = int64(8)
		negativeLine             = int64(-8)
		limitBytesGreaterThan1   = int64(12)
		limitBytesLessThan1      = int64(0)
		sinceSecondsGreaterThan1 = int64(10)
		sinceSecondsLessThan1    = int64(0)
		timestamp                = metav1.Now()
	)

	successCase := []struct {
		Name          string
		podLogOptions v1.PodLogOptions
	}{
		{
			Name:          "Empty PodLogOptions",
			podLogOptions: v1.PodLogOptions{},
		},
		{
			Name: "PodLogOptions with TailLines",
			podLogOptions: v1.PodLogOptions{
				TailLines: &positiveLine,
			},
		},
		{
			Name: "PodLogOptions with LimitBytes",
			podLogOptions: v1.PodLogOptions{
				LimitBytes: &limitBytesGreaterThan1,
			},
		},
		{
			Name: "PodLogOptions with only sinceSeconds",
			podLogOptions: v1.PodLogOptions{
				SinceSeconds: &sinceSecondsGreaterThan1,
			},
		},
		{
			Name: "PodLogOptions with LimitBytes with TailLines",
			podLogOptions: v1.PodLogOptions{
				LimitBytes: &limitBytesGreaterThan1,
				TailLines:  &positiveLine,
			},
		},
		{
			Name: "PodLogOptions with LimitBytes with TailLines with SinceSeconds",
			podLogOptions: v1.PodLogOptions{
				LimitBytes:   &limitBytesGreaterThan1,
				TailLines:    &positiveLine,
				SinceSeconds: &sinceSecondsGreaterThan1,
			},
		},
	}
	for _, tc := range successCase {
		if errs := ValidatePodLogOptions(&tc.podLogOptions); len(errs) != 0 {
			t.Errorf("%q unexpected error: %v", tc.Name, errs)
		}
	}

	errorCase := []struct {
		Name          string
		podLogOptions v1.PodLogOptions
	}{
		{
			Name: "Invalid podLogOptions with Negative TailLines",
			podLogOptions: v1.PodLogOptions{
				TailLines:    &negativeLine,
				LimitBytes:   &limitBytesGreaterThan1,
				SinceSeconds: &sinceSecondsGreaterThan1,
			},
		},
		{
			Name: "Invalid podLogOptions with zero or negative LimitBytes",
			podLogOptions: v1.PodLogOptions{
				TailLines:    &positiveLine,
				LimitBytes:   &limitBytesLessThan1,
				SinceSeconds: &sinceSecondsGreaterThan1,
			},
		},
		{
			Name: "Invalid podLogOptions with zero or negative SinceSeconds",
			podLogOptions: v1.PodLogOptions{
				TailLines:    &negativeLine,
				LimitBytes:   &limitBytesGreaterThan1,
				SinceSeconds: &sinceSecondsLessThan1,
			},
		}, {
			Name: "Invalid podLogOptions with both SinceSeconds and SinceTime set",
			podLogOptions: v1.PodLogOptions{
				TailLines:    &negativeLine,
				LimitBytes:   &limitBytesGreaterThan1,
				SinceSeconds: &sinceSecondsGreaterThan1,
				SinceTime:    &timestamp,
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
	successCase := []struct {
		containers  []v1.Container
		accumulator *sets.String
		fldPath     *field.Path
		result      string
	}{
		{
			containers: []v1.Container{
				{
					Ports: []v1.ContainerPort{
						{
							HostPort: 8080,
							Protocol: v1.ProtocolUDP,
						},
					},
				},
				{
					Ports: []v1.ContainerPort{
						{
							HostPort: 8080,
							Protocol: v1.ProtocolTCP,
						},
					},
				},
			},
			accumulator: &sets.String{},
			fldPath:     field.NewPath("spec", "containers"),
			result:      "HostPort is not allocated",
		},
		{
			containers: []v1.Container{
				{
					Ports: []v1.ContainerPort{
						{
							HostPort: 8080,
							Protocol: v1.ProtocolUDP,
						},
					},
				},
				{
					Ports: []v1.ContainerPort{
						{
							HostPort: 8081,
							Protocol: v1.ProtocolUDP,
						},
					},
				},
			},
			accumulator: &sets.String{},
			fldPath:     field.NewPath("spec", "containers"),
			result:      "HostPort is not allocated",
		},
	}
	for index, tc := range successCase {
		if errs := AccumulateUniqueHostPorts(tc.containers, tc.accumulator, tc.fldPath); len(errs) != 0 {
			t.Errorf("unexpected error for test case %v: %v", index, errs)
		}
	}
	errorCase := []struct {
		containers  []v1.Container
		accumulator *sets.String
		fldPath     *field.Path
		result      string
	}{
		{
			containers: []v1.Container{
				{
					Ports: []v1.ContainerPort{
						{
							HostPort: 8080,
							Protocol: v1.ProtocolUDP,
						},
					},
				},
				{
					Ports: []v1.ContainerPort{
						{
							HostPort: 8080,
							Protocol: v1.ProtocolUDP,
						},
					},
				},
			},
			accumulator: &sets.String{},
			fldPath:     field.NewPath("spec", "containers"),
			result:      "HostPort is already allocated",
		},
		{
			containers: []v1.Container{
				{
					Ports: []v1.ContainerPort{
						{
							HostPort: 8080,
							Protocol: v1.ProtocolUDP,
						},
					},
				},
				{
					Ports: []v1.ContainerPort{
						{
							HostPort: 8081,
							Protocol: v1.ProtocolUDP,
						},
					},
				},
			},
			accumulator: &sets.String{"8080/UDP": sets.Empty{}},
			fldPath:     field.NewPath("spec", "containers"),
			result:      "HostPort is already allocated",
		},
	}
	for index, tc := range errorCase {
		if errs := AccumulateUniqueHostPorts(tc.containers, tc.accumulator, tc.fldPath); len(errs) == 0 {
			t.Errorf("test case %v: expected error %v, but get nil", index, tc.result)
		}
	}
}
