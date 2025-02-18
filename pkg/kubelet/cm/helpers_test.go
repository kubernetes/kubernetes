/*
Copyright 2024 The Kubernetes Authors.

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

package cm

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"testing"
)

func TestGetAllocatableMemory(t *testing.T) {
	testCases := []struct {
		capacity v1.ResourceList
		quantity resource.Quantity
	}{
		{
			capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
				v1.ResourceName("hugepages-1Gi"):   resource.MustParse("1G"),
			},
			quantity: resource.MustParse("9G"),
		},
		{
			capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
				v1.ResourceName("hugepages-2Mi"):   resource.MustParse("200M"),
			},
			quantity: resource.MustParse("800M"),
		},
	}
	for _, testCase := range testCases {
		realQuantity := GetAllocatableMemory(testCase.capacity)
		if testCase.quantity.Cmp(realQuantity) != 0 {
			t.Errorf("resource: %v expected result: %v", testCase.quantity, GetAllocatableMemory(testCase.capacity))
		}
	}
}
