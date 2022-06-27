/*
Copyright 2022 The Kubernetes Authors.

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

package metricsutil

import (
	"bytes"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"testing"
)

func Test_printSingleResourceUsage(t *testing.T) {
	tests := []struct {
		name         string
		resourceType v1.ResourceName
		quantity     resource.Quantity
		unit         bool
		wantOut      string
	}{
		// TODO: Add test cases.
		{
			name:         "print cpu usage of unit mcore when format-unit equal true",
			resourceType: v1.ResourceCPU,
			quantity:     *resource.NewScaledQuantity(500, -3),
			unit:         true,
			wantOut:      "500m",
		},
		{
			name:         "print cpu usage of unit core when format-unit equal true",
			resourceType: v1.ResourceCPU,
			quantity:     *resource.NewScaledQuantity(500, 0),
			unit:         true,
			wantOut:      "500",
		},
		{
			name:         "print memory usage of unit byte when format-unit equal true",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(500, 0),
			unit:         true,
			wantOut:      "500",
		},
		{
			name:         "print memory usage of unit Kbyte when format-unit equal true",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(512, 3),
			unit:         true,
			wantOut:      "500Ki",
		},
		{
			name:         "print memory usage of unit Mbyte when format-unit equal true",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(500, 6),
			unit:         true,
			wantOut:      "477Mi",
		},
		{
			name:         "print memory usage of unit Gbyte when format-unit equal true",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(3984588800, 0),
			unit:         true,
			wantOut:      "4Gi",
		},
		{
			name:         "print memory usage of unit Tbyte when format-unit equal true",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(500, 12),
			unit:         true,
			wantOut:      "455Ti",
		},
		{
			name:         "print storage usage of unit byte when format-unit equal true",
			resourceType: v1.ResourceStorage,
			quantity:     *resource.NewScaledQuantity(500, 3),
			unit:         true,
			wantOut:      "500000",
		},
		{
			name:         "print cpu usage of unit mcore when format-unit equal false",
			resourceType: v1.ResourceCPU,
			quantity:     *resource.NewScaledQuantity(500, -3),
			unit:         false,
			wantOut:      "500m",
		},
		{
			name:         "print cpu usage of unit core when format-unit equal false",
			resourceType: v1.ResourceCPU,
			quantity:     *resource.NewScaledQuantity(500, 0),
			unit:         false,
			wantOut:      "500000m",
		},
		{
			name:         "print memory usage of unit byte when format-unit equal false",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(500, 0),
			unit:         false,
			wantOut:      "0Mi",
		},
		{
			name:         "print memory usage of unit Kbyte when format-unit equal false",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(512, 3),
			unit:         false,
			wantOut:      "0Mi",
		},
		{
			name:         "print memory usage of unit Mbyte when format-unit equal false",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(500, 6),
			unit:         false,
			wantOut:      "476Mi",
		},
		{
			name:         "print memory usage of unit Gbyte when format-unit equal false",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(3984588800, 0),
			unit:         false,
			wantOut:      "3800Mi",
		},
		{
			name:         "print memory usage of unit Tbyte when format-unit equal false",
			resourceType: v1.ResourceMemory,
			quantity:     *resource.NewScaledQuantity(500, 12),
			unit:         false,
			wantOut:      "476837158Mi",
		},
		{
			name:         "print storage usage of unit byte when format-unit equal false",
			resourceType: v1.ResourceStorage,
			quantity:     *resource.NewScaledQuantity(500, 3),
			unit:         false,
			wantOut:      "500000",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &bytes.Buffer{}
			printSingleResourceUsage(out, tt.resourceType, tt.quantity, tt.unit)
			if gotOut := out.String(); gotOut != tt.wantOut {
				t.Errorf("printSingleResourceUsage() = %v, want %v", gotOut, tt.wantOut)
			}
		})
	}
}
