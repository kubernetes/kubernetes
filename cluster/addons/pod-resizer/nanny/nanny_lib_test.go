// Copyright 2016 The Kubernetes Authors. All rights reserved.
package nanny

import (
	"testing"

	resource "k8s.io/kubernetes/pkg/api/resource"
	api "k8s.io/kubernetes/pkg/api/v1"
)

var (
	// ResourcesLists to compose test cases.
	standard = api.ResourceList{
		"cpu":     resource.MustParse("0.3"),
		"memory":  resource.MustParse("200Mi"),
		"storage": resource.MustParse("10Gi"),
	}
	siStandard = api.ResourceList{
		"cpu":     resource.MustParse("0.3"),
		"memory":  resource.MustParse("200M"),
		"storage": resource.MustParse("10G"),
	}
	noStorage = api.ResourceList{
		"cpu":    resource.MustParse("0.3"),
		"memory": resource.MustParse("200Mi"),
	}
	siNoStorage = api.ResourceList{
		"cpu":    resource.MustParse("0.3"),
		"memory": resource.MustParse("200M"),
	}
	smallMemoryNoStorage = api.ResourceList{
		"cpu":    resource.MustParse("0.3"),
		"memory": resource.MustParse("100Mi"),
	}
	noMemory = api.ResourceList{
		"cpu":     resource.MustParse("0.3"),
		"storage": resource.MustParse("10Gi"),
	}
	noCpu = api.ResourceList{
		"memory":  resource.MustParse("200Mi"),
		"storage": resource.MustParse("10Gi"),
	}
	smallStorage = api.ResourceList{
		"cpu":     resource.MustParse("0.3"),
		"memory":  resource.MustParse("200Mi"),
		"storage": resource.MustParse("1Gi"),
	}
	smallMemory = api.ResourceList{
		"cpu":     resource.MustParse("0.3"),
		"memory":  resource.MustParse("100Mi"),
		"storage": resource.MustParse("10Gi"),
	}
	smallCpu = api.ResourceList{
		"cpu":     resource.MustParse("0.1"),
		"memory":  resource.MustParse("200Mi"),
		"storage": resource.MustParse("10Gi"),
	}
)

func TestCheckResources(t *testing.T) {
	testCases := []struct {
		th   int64
		x, y api.ResourceList
		res  api.ResourceName
		want bool
	}{
		// Test no threshold for the CPU resource type.
		{0, standard, standard, "cpu", false},
		{0, standard, siStandard, "cpu", false},
		{0, standard, noStorage, "cpu", false},
		{0, standard, noMemory, "cpu", false},
		{0, standard, noCpu, "cpu", true},
		{0, standard, smallStorage, "cpu", false},
		{0, standard, smallMemory, "cpu", false},
		{0, standard, smallCpu, "cpu", true},

		// Test no threshold for the memory resource type.
		{0, standard, standard, "memory", false},
		{0, standard, siStandard, "memory", true},
		{0, standard, noStorage, "memory", false},
		{0, standard, noMemory, "memory", true},
		{0, standard, noCpu, "memory", false},
		{0, standard, smallStorage, "memory", false},
		{0, standard, smallMemory, "memory", true},
		{0, standard, smallCpu, "memory", false},

		// Test no threshold for the storage resource type.
		{0, standard, standard, "storage", false},
		{0, standard, siStandard, "storage", true},
		{0, standard, noStorage, "storage", true},
		{0, standard, noMemory, "storage", false},
		{0, standard, noCpu, "storage", false},
		{0, standard, smallStorage, "storage", true},
		{0, standard, smallMemory, "storage", false},
		{0, standard, smallCpu, "storage", false},

		// Test large threshold for the CPU resource type.
		{10, standard, standard, "cpu", false},
		{10, standard, siStandard, "cpu", false},
		{10, standard, noStorage, "cpu", false},
		{10, standard, noMemory, "cpu", false},
		{10, standard, noCpu, "cpu", true},
		{10, standard, smallStorage, "cpu", false},
		{10, standard, smallMemory, "cpu", false},
		{10, standard, smallCpu, "cpu", true},

		// Test large threshold for the memory resource type.
		{10, standard, standard, "memory", false},
		{10, standard, siStandard, "memory", false},
		{10, standard, noStorage, "memory", false},
		{10, standard, noMemory, "memory", true},
		{10, standard, noCpu, "memory", false},
		{10, standard, smallStorage, "memory", false},
		{10, standard, smallMemory, "memory", true},
		{10, standard, smallCpu, "memory", false},

		// Test large threshold for the storage resource type.
		{10, standard, standard, "storage", false},
		{10, standard, siStandard, "storage", false},
		{10, standard, noStorage, "storage", true},
		{10, standard, noMemory, "storage", false},
		{10, standard, noCpu, "storage", false},
		{10, standard, smallStorage, "storage", true},
		{10, standard, smallMemory, "storage", false},
		{10, standard, smallCpu, "storage", false},

		// Test successful comparison when not all ResourceNames are present.
		{0, noStorage, siNoStorage, "cpu", false},
		{0, noStorage, siNoStorage, "memory", true},
		{10, noStorage, siNoStorage, "cpu", false},
		{10, noStorage, siNoStorage, "memory", false},
		{10, noStorage, smallMemoryNoStorage, "memory", true},
	}

	for i, tc := range testCases {
		if tc.want != checkResource(tc.th, tc.x, tc.y, tc.res) {
			t.Errorf("checkResource got %t, want %t for test case %d.", !tc.want, tc.want, i)
		}
	}
}

func TestShouldOverwriteResources(t *testing.T) {
	testCases := []struct {
		th   int64
		x, y api.ResourceList
		want bool
	}{
		// Test no threshold.
		{0, standard, standard, false}, // A threshold of 0 should be exact.
		{0, standard, siStandard, true},
		{0, standard, noStorage, true}, // Overwrite on qualitative differences.
		{0, standard, noMemory, true},
		{0, standard, noCpu, true},
		{0, standard, smallStorage, true}, // Overwrite past the threshold.
		{0, standard, smallMemory, true},
		{0, standard, smallCpu, true},

		// Test a large threshold.
		{10, standard, standard, false},
		{10, standard, siStandard, false}, // A threshold of 10 gives leeway.
		{10, standard, noStorage, true},
		{10, standard, noMemory, true},
		{10, standard, noCpu, true},
		{10, standard, smallStorage, true}, // The differences are larger than the threshold.
		{10, standard, smallMemory, true},
		{10, standard, smallCpu, true},

		// Test successful comparison when not all ResourceNames are present.
		{10, noStorage, siNoStorage, false},
	}
	for i, tc := range testCases {
		if tc.want != shouldOverwriteResources(tc.th, tc.x, tc.y, tc.x, tc.x) {
			t.Errorf("shouldOverwriteResources got %t, want %t for test case %d.", !tc.want, tc.want, i)
		}
	}
}
