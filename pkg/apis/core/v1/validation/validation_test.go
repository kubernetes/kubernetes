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
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateResourceRequirements(t *testing.T) {
	successCase := []struct {
		name         string
		requirements v1.ResourceRequirements
	}{{
		name: "Resources with Requests equal to Limits",
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
	}, {
		name: "Resources with only Limits",
		requirements: v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
				v1.ResourceName("my.org/resource"): resource.MustParse("10"),
			},
		},
	}, {
		name: "Resources with only Requests",
		requirements: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
				v1.ResourceName("my.org/resource"): resource.MustParse("10"),
			},
		},
	}, {
		name: "Resources with Requests Less Than Limits",
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
	}}
	for _, tc := range successCase {
		t.Run(tc.name, func(t *testing.T) {
			if errs := ValidateResourceRequirements(&tc.requirements, field.NewPath("resources")); len(errs) != 0 {
				t.Errorf("unexpected error: %v", errs)
			}
		})
	}

	errorCase := []struct {
		name                  string
		requirements          v1.ResourceRequirements
		skipLimitValueCheck   bool
		skipRequestValueCheck bool
	}{{
		name: "Resources with Requests Larger Than Limits",
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
	}, {
		name: "Invalid Resources with Requests",
		requirements: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName("my.org"): resource.MustParse("10m"),
			},
		},
		skipRequestValueCheck: true,
	}, {
		name: "Invalid Resources with Limits",
		requirements: v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceName("my.org"): resource.MustParse("9m"),
			},
		},
		skipLimitValueCheck: true,
	}}
	for _, tc := range errorCase {
		t.Run(tc.name, func(t *testing.T) {
			errs := ValidateResourceRequirements(&tc.requirements, field.NewPath("resources"))
			if len(errs) == 0 {
				t.Errorf("expected error")
			}
			validateNamesAndValuesInDescription(t, tc.requirements.Limits, errs, tc.skipLimitValueCheck, "limit")
			validateNamesAndValuesInDescription(t, tc.requirements.Requests, errs, tc.skipRequestValueCheck, "request")
		})
	}
}

func validateNamesAndValuesInDescription(t *testing.T, r v1.ResourceList, errs field.ErrorList, skipValueTest bool, rl string) {
	for name, value := range r {
		containsName := false
		containsValue := false

		for _, e := range errs {
			if strings.Contains(e.Error(), name.String()) {
				containsName = true
			}

			if strings.Contains(e.Error(), value.String()) {
				containsValue = true
			}
		}
		if !containsName {
			t.Errorf("error must contain %s name", rl)
		}
		if !containsValue && !skipValueTest {
			t.Errorf("error must contain %s value", rl)
		}
	}
}

func TestValidateContainerResourceName(t *testing.T) {
	successCase := []struct {
		name         string
		ResourceName core.ResourceName
	}{{
		name:         "CPU resource",
		ResourceName: "cpu",
	}, {
		name:         "Memory resource",
		ResourceName: "memory",
	}, {
		name:         "Hugepages resource",
		ResourceName: "hugepages-2Mi",
	}, {
		name:         "Namespaced resource",
		ResourceName: "kubernetes.io/resource-foo",
	}, {
		name:         "Extended Resource",
		ResourceName: "my.org/resource-bar",
	}}
	for _, tc := range successCase {
		t.Run(tc.name, func(t *testing.T) {
			if errs := ValidateContainerResourceName(tc.ResourceName, field.NewPath(string(tc.ResourceName))); len(errs) != 0 {
				t.Errorf("unexpected error: %v", errs)
			}
		})
	}

	errorCase := []struct {
		name         string
		ResourceName core.ResourceName
	}{{
		name:         "Invalid standard resource",
		ResourceName: "cpu-core",
	}, {
		name:         "Invalid namespaced resource",
		ResourceName: "kubernetes.io/",
	}, {
		name:         "Invalid extended resource",
		ResourceName: "my.org-foo-resource",
	}}
	for _, tc := range errorCase {
		t.Run(tc.name, func(t *testing.T) {
			if errs := ValidateContainerResourceName(tc.ResourceName, field.NewPath(string(tc.ResourceName))); len(errs) == 0 {
				t.Errorf("expected error")
			}
		})
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
		name          string
		podLogOptions v1.PodLogOptions
	}{{
		name:          "Empty PodLogOptions",
		podLogOptions: v1.PodLogOptions{},
	}, {
		name: "PodLogOptions with TailLines",
		podLogOptions: v1.PodLogOptions{
			TailLines: &positiveLine,
		},
	}, {
		name: "PodLogOptions with LimitBytes",
		podLogOptions: v1.PodLogOptions{
			LimitBytes: &limitBytesGreaterThan1,
		},
	}, {
		name: "PodLogOptions with only sinceSeconds",
		podLogOptions: v1.PodLogOptions{
			SinceSeconds: &sinceSecondsGreaterThan1,
		},
	}, {
		name: "PodLogOptions with LimitBytes with TailLines",
		podLogOptions: v1.PodLogOptions{
			LimitBytes: &limitBytesGreaterThan1,
			TailLines:  &positiveLine,
		},
	}, {
		name: "PodLogOptions with LimitBytes with TailLines with SinceSeconds",
		podLogOptions: v1.PodLogOptions{
			LimitBytes:   &limitBytesGreaterThan1,
			TailLines:    &positiveLine,
			SinceSeconds: &sinceSecondsGreaterThan1,
		},
	}}
	for _, tc := range successCase {
		t.Run(tc.name, func(t *testing.T) {
			if errs := ValidatePodLogOptions(&tc.podLogOptions); len(errs) != 0 {
				t.Errorf("unexpected error: %v", errs)
			}
		})
	}

	errorCase := []struct {
		name          string
		podLogOptions v1.PodLogOptions
	}{{
		name: "Invalid podLogOptions with Negative TailLines",
		podLogOptions: v1.PodLogOptions{
			TailLines:    &negativeLine,
			LimitBytes:   &limitBytesGreaterThan1,
			SinceSeconds: &sinceSecondsGreaterThan1,
		},
	}, {
		name: "Invalid podLogOptions with zero or negative LimitBytes",
		podLogOptions: v1.PodLogOptions{
			TailLines:    &positiveLine,
			LimitBytes:   &limitBytesLessThan1,
			SinceSeconds: &sinceSecondsGreaterThan1,
		},
	}, {
		name: "Invalid podLogOptions with zero or negative SinceSeconds",
		podLogOptions: v1.PodLogOptions{
			TailLines:    &negativeLine,
			LimitBytes:   &limitBytesGreaterThan1,
			SinceSeconds: &sinceSecondsLessThan1,
		},
	}, {
		name: "Invalid podLogOptions with both SinceSeconds and SinceTime set",
		podLogOptions: v1.PodLogOptions{
			TailLines:    &negativeLine,
			LimitBytes:   &limitBytesGreaterThan1,
			SinceSeconds: &sinceSecondsGreaterThan1,
			SinceTime:    &timestamp,
		},
	}}
	for _, tc := range errorCase {
		t.Run(tc.name, func(t *testing.T) {
			if errs := ValidatePodLogOptions(&tc.podLogOptions); len(errs) == 0 {
				t.Errorf("expected error")
			}
		})
	}
}

func TestAccumulateUniqueHostPorts(t *testing.T) {
	successCase := []struct {
		name        string
		containers  []v1.Container
		accumulator *sets.String
		fldPath     *field.Path
	}{{
		name: "HostPort is not allocated while containers use the same port with different protocol",
		containers: []v1.Container{{
			Ports: []v1.ContainerPort{{
				HostPort: 8080,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Ports: []v1.ContainerPort{{
				HostPort: 8080,
				Protocol: v1.ProtocolTCP,
			}},
		}},
		accumulator: &sets.String{},
		fldPath:     field.NewPath("spec", "containers"),
	}, {
		name: "HostPort is not allocated while containers use different ports",
		containers: []v1.Container{{
			Ports: []v1.ContainerPort{{
				HostPort: 8080,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Ports: []v1.ContainerPort{{
				HostPort: 8081,
				Protocol: v1.ProtocolUDP,
			}},
		}},
		accumulator: &sets.String{},
		fldPath:     field.NewPath("spec", "containers"),
	}}
	for _, tc := range successCase {
		t.Run(tc.name, func(t *testing.T) {
			if errs := AccumulateUniqueHostPorts(tc.containers, tc.accumulator, tc.fldPath); len(errs) != 0 {
				t.Errorf("unexpected error: %v", errs)
			}
		})
	}
	errorCase := []struct {
		name        string
		containers  []v1.Container
		accumulator *sets.String
		fldPath     *field.Path
	}{{
		name: "HostPort is already allocated while containers use the same port with UDP",
		containers: []v1.Container{{
			Ports: []v1.ContainerPort{{
				HostPort: 8080,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Ports: []v1.ContainerPort{{
				HostPort: 8080,
				Protocol: v1.ProtocolUDP,
			}},
		}},
		accumulator: &sets.String{},
		fldPath:     field.NewPath("spec", "containers"),
	}, {
		name: "HostPort is already allocated",
		containers: []v1.Container{{
			Ports: []v1.ContainerPort{{
				HostPort: 8080,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Ports: []v1.ContainerPort{{
				HostPort: 8081,
				Protocol: v1.ProtocolUDP,
			}},
		}},
		accumulator: &sets.String{"8080/UDP": sets.Empty{}},
		fldPath:     field.NewPath("spec", "containers"),
	}}
	for _, tc := range errorCase {
		t.Run(tc.name, func(t *testing.T) {
			if errs := AccumulateUniqueHostPorts(tc.containers, tc.accumulator, tc.fldPath); len(errs) == 0 {
				t.Errorf("expected error, but get nil")
			}
		})
	}
}
