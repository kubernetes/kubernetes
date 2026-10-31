/*
Copyright 2025 The Kubernetes Authors.

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

package cgroups

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestGetCPULimitCgroupExpectations(t *testing.T) {
	testCases := []struct {
		name              string
		cpuLimit          *resource.Quantity
		podOnCgroupv2Node bool
		expected          []string
	}{
		{
			name:              "rounding required, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(15, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"1500 100000", "2000 100000"},
		},
		{
			name:              "rounding not required, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(20, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"2000 100000"},
		},
		{
			name:              "rounding required, podOnCGroupv2Node=false",
			cpuLimit:          resource.NewMilliQuantity(15, resource.DecimalSI),
			podOnCgroupv2Node: false,
			expected:          []string{"1500", "2000"},
		},
		{
			name:              "rounding not required, podOnCGroupv2Node=false",
			cpuLimit:          resource.NewMilliQuantity(20, resource.DecimalSI),
			podOnCgroupv2Node: false,
			expected:          []string{"2000"},
		},
		{
			name:              "cpuQuota=0, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(0, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"max 100000"},
		},
		{
			name:              "cpuQuota=0, podOnCGroupv2Node=false",
			cpuLimit:          resource.NewMilliQuantity(0, resource.DecimalSI),
			podOnCgroupv2Node: false,
			expected:          []string{"-1"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := getCPULimitCgroupExpectations(tc.cpuLimit, tc.podOnCgroupv2Node)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestExpectedContainerLimits(t *testing.T) {
	testCases := []struct {
		name       string
		pod        *v1.Pod
		declared   *v1.ResourceRequirements
		wantCPU    string // "" means no CPU limit expected
		wantMemory string // "" means no memory limit expected
	}{
		{
			name: "container declares its own limits",
			pod: &v1.Pod{Spec: v1.PodSpec{Resources: &v1.ResourceRequirements{Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("200Mi"),
			}}}},
			declared: &v1.ResourceRequirements{Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			}},
			wantCPU:    "100m",
			wantMemory: "100Mi",
		},
		{
			name: "container declares none, pod-level limits inherited",
			pod: &v1.Pod{Spec: v1.PodSpec{Resources: &v1.ResourceRequirements{Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("200Mi"),
			}}}},
			declared:   &v1.ResourceRequirements{},
			wantCPU:    "200m",
			wantMemory: "200Mi",
		},
		{
			name: "container CPU only, memory inherited from pod",
			pod: &v1.Pod{Spec: v1.PodSpec{Resources: &v1.ResourceRequirements{Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("200Mi"),
			}}}},
			declared: &v1.ResourceRequirements{Limits: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("100m"),
			}},
			wantCPU:    "100m",
			wantMemory: "200Mi",
		},
		{
			name:       "no pod-level resources, no container limits",
			pod:        &v1.Pod{},
			declared:   &v1.ResourceRequirements{},
			wantCPU:    "",
			wantMemory: "",
		},
		{
			name:       "pod-level resources set but limits empty, no container limits",
			pod:        &v1.Pod{Spec: v1.PodSpec{Resources: &v1.ResourceRequirements{Limits: v1.ResourceList{}}}},
			declared:   &v1.ResourceRequirements{},
			wantCPU:    "",
			wantMemory: "",
		},
		{
			name: "nil declared resources",
			pod: &v1.Pod{Spec: v1.PodSpec{Resources: &v1.ResourceRequirements{Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("200Mi"),
			}}}},
			declared:   nil,
			wantCPU:    "",
			wantMemory: "",
		},
		{
			name: "nil pod, container limits kept without fallback",
			pod:  nil,
			declared: &v1.ResourceRequirements{Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("100Mi"),
			}},
			wantCPU:    "100m",
			wantMemory: "100Mi",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ExpectedContainerLimits(tc.pod, tc.declared)

			cpu, haveCPU := got.Limits[v1.ResourceCPU]
			assert.Equal(t, tc.wantCPU != "", haveCPU, "CPU limit presence")
			if tc.wantCPU != "" {
				assert.Truef(t, cpu.Equal(resource.MustParse(tc.wantCPU)), "CPU limit: got %s, want %s", cpu.String(), tc.wantCPU)
			}

			mem, haveMem := got.Limits[v1.ResourceMemory]
			assert.Equal(t, tc.wantMemory != "", haveMem, "memory limit presence")
			if tc.wantMemory != "" {
				assert.Truef(t, mem.Equal(resource.MustParse(tc.wantMemory)), "memory limit: got %s, want %s", mem.String(), tc.wantMemory)
			}
		})
	}
}
