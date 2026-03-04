//go:build linux

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
