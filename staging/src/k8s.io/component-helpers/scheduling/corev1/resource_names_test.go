/*
Copyright 2020 The Kubernetes Authors.

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

package corev1

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestIsPrefixedNativeResourceName(t *testing.T) {
	testCases := []struct {
		name         string
		resourceName v1.ResourceName
		expectVal    bool
	}{
		{
			name:         "explicit native resource name with kubernetes.io is explicitly mentioned",
			resourceName: "pod.alpha.kubernetes.io/opaque-int-resource-foo",
			expectVal:    true,
		},
		{
			name:         "explicit native resource name with kubernetes.io is explicitly mentioned as prefix",
			resourceName: "kubernetes.io/resource-foo",
			expectVal:    true,
		},
		{
			name:         "implicit native resource name",
			resourceName: "foo",
			expectVal:    false,
		},
		{
			name:         "native resource name outside of kubernetes.io domain",
			resourceName: "a/b",
			expectVal:    false,
		},
		{
			name:         "empty native resource name",
			resourceName: "",
			expectVal:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			v := IsPrefixedNativeResourceName(tc.resourceName)
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}

func TestIsNativeResourceName(t *testing.T) {
	testCases := []struct {
		name         string
		resourceName v1.ResourceName
		expectVal    bool
	}{
		{
			name:         "explicit native resource name with kubernetes.io is explicitly mentioned",
			resourceName: "pod.alpha.kubernetes.io/opaque-int-resource-foo",
			expectVal:    true,
		},
		{
			name:         "explicit native resource name with kubernetes.io is explicitly mentioned as prefix",
			resourceName: "kubernetes.io/resource-foo",
			expectVal:    true,
		},
		{
			name:         "implicit native resource name",
			resourceName: "foo",
			expectVal:    true,
		},
		{
			name:         "empty native resource name",
			resourceName: "",
			expectVal:    true,
		},
		{
			name:         "native resource name outside of kubernetes.io domain",
			resourceName: "a/b",
			expectVal:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			v := IsNativeResourceName(tc.resourceName)
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}

func TestIsExtendedResourceName(t *testing.T) {
	testCases := []struct {
		name         string
		resourceName v1.ResourceName
		expectVal    bool
	}{
		{
			name:         "explicit native resource name with kubernetes.io is explicitly mentioned",
			resourceName: "pod.alpha.kubernetes.io/opaque-int-resource-foo",
			expectVal:    false,
		},
		{
			name:         "explicit native resource name with kubernetes.io is explicitly mentioned as prefix",
			resourceName: "kubernetes.io/resource-foo",
			expectVal:    false,
		},
		{
			name:         "implicit native resource name",
			resourceName: "foo",
			expectVal:    false,
		},
		{
			name:         "native resource name outside of kubernetes.io domain",
			resourceName: "a/b",
			expectVal:    false,
		},
		{
			name:         "empty native resource name",
			resourceName: "",
			expectVal:    false,
		},
		{
			name:         "resource name outside of kubernetes.io domain without requests. prefix",
			resourceName: "example.com/foo",
			expectVal:    true,
		},
		{
			name:         "resource name outside of kubernetes.io domain with requests. prefix",
			resourceName: "requests.nvidia.com/gpu",
			expectVal:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			v := IsExtendedResourceName(tc.resourceName)
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}

func TestIsHugePageResourceName(t *testing.T) {
	testCases := []struct {
		name         string
		resourceName v1.ResourceName
		expectVal    bool
	}{
		{
			name:         "valid hugepages resource name",
			resourceName: "hugepages-2Mi",
			expectVal:    true,
		},
		{
			name:         "invalid hugepages resource name",
			resourceName: "memory",
			expectVal:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			v := IsHugePageResourceName(tc.resourceName)
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}

func TestHugePageResourceName(t *testing.T) {
	testCases := []struct {
		name      string
		pagesize  resource.Quantity
		expectVal v1.ResourceName
	}{
		{
			name:      "hugepages-2Mi resource name",
			pagesize:  *resource.NewQuantity(2*1024*1024, resource.BinarySI),
			expectVal: "hugepages-2Mi",
		},
		{
			name:      "hugepages-1Gi resource name",
			pagesize:  *resource.NewQuantity(1024*1024*1024, resource.BinarySI),
			expectVal: "hugepages-1Gi",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			v := HugePageResourceName(tc.pagesize)
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}

func TestIsAttachableVolumeResourceName(t *testing.T) {
	testCases := []struct {
		name         string
		resourceName v1.ResourceName
		expectVal    bool
	}{
		{
			name:         "valid attachable volumes resource name",
			resourceName: "attachable-volumes-aws-ebs",
			expectVal:    true,
		},
		{
			name:         "invalid attachable volumes resource name",
			resourceName: "memory",
			expectVal:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			v := IsAttachableVolumeResourceName(tc.resourceName)
			if v != tc.expectVal {
				t.Errorf("Got %v but expected %v", v, tc.expectVal)
			}
		})
	}
}
