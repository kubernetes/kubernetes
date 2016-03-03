// Copyright 2016 The Kubernetes Authors. All rights reserved.
package nanny

import (
	"reflect"
	"testing"

	resource "k8s.io/kubernetes/pkg/api/resource"
	api "k8s.io/kubernetes/pkg/api/v1"
)

var (
	fullEstimator = LinearEstimator{
		Resources: []Resource{
			Resource{
				Base:         resource.MustParse("0.3"),
				ExtraPerNode: resource.MustParse("1"),
				Name:         "cpu",
			},
			Resource{
				Base:         resource.MustParse("30Mi"),
				ExtraPerNode: resource.MustParse("1Mi"),
				Name:         "memory",
			},
			Resource{
				Base:         resource.MustParse("30Gi"),
				ExtraPerNode: resource.MustParse("1Gi"),
				Name:         "storage",
			},
		},
	}
	noCpuEstimator = LinearEstimator{
		Resources: []Resource{
			Resource{
				Base:         resource.MustParse("30Mi"),
				ExtraPerNode: resource.MustParse("1Mi"),
				Name:         "memory",
			},
			Resource{
				Base:         resource.MustParse("30Gi"),
				ExtraPerNode: resource.MustParse("1Gi"),
				Name:         "storage",
			},
		},
	}
	noMemoryEstimator = LinearEstimator{
		Resources: []Resource{
			Resource{
				Base:         resource.MustParse("0.3"),
				ExtraPerNode: resource.MustParse("1"),
				Name:         "cpu",
			},
			Resource{
				Base:         resource.MustParse("30Gi"),
				ExtraPerNode: resource.MustParse("1Gi"),
				Name:         "storage",
			},
		},
	}
	noStorageEstimator = LinearEstimator{
		Resources: []Resource{
			Resource{
				Base:         resource.MustParse("0.3"),
				ExtraPerNode: resource.MustParse("1"),
				Name:         "cpu",
			},
			Resource{
				Base:         resource.MustParse("30Mi"),
				ExtraPerNode: resource.MustParse("1Mi"),
				Name:         "memory",
			},
		},
	}
	emptyEstimator = LinearEstimator{
		Resources: []Resource{},
	}

	baseResources = api.ResourceList{
		"cpu":     resource.MustParse("0.3"),
		"memory":  resource.MustParse("30Mi"),
		"storage": resource.MustParse("30Gi"),
	}

	noCpuBaseResources = api.ResourceList{
		"memory":  resource.MustParse("30Mi"),
		"storage": resource.MustParse("30Gi"),
	}
	noMemoryBaseResources = api.ResourceList{
		"cpu":     resource.MustParse("0.3"),
		"storage": resource.MustParse("30Gi"),
	}
	noStorageBaseResources = api.ResourceList{
		"cpu":    resource.MustParse("0.3"),
		"memory": resource.MustParse("30Mi"),
	}
	threeNodeResources = api.ResourceList{
		"cpu":     resource.MustParse("3.3"),
		"memory":  resource.MustParse("33Mi"),
		"storage": resource.MustParse("33Gi"),
	}
	threeNodeNoCpuResources = api.ResourceList{
		"memory":  resource.MustParse("33Mi"),
		"storage": resource.MustParse("33Gi"),
	}
	threeNodeNoMemoryResources = api.ResourceList{
		"cpu":     resource.MustParse("3.3"),
		"storage": resource.MustParse("33Gi"),
	}
	threeNodeNoStorageResources = api.ResourceList{
		"cpu":    resource.MustParse("3.3"),
		"memory": resource.MustParse("33Mi"),
	}
	noResources = api.ResourceList{}
)

func TestEstimateResources(t *testing.T) {
	testCases := []struct {
		e        ResourceEstimator
		numNodes uint64
		limits   api.ResourceList
		requests api.ResourceList
	}{
		{fullEstimator, 0, baseResources, baseResources},
		{fullEstimator, 3, threeNodeResources, threeNodeResources},
		{noCpuEstimator, 0, noCpuBaseResources, noCpuBaseResources},
		{noCpuEstimator, 3, threeNodeNoCpuResources, threeNodeNoCpuResources},
		{noMemoryEstimator, 0, noMemoryBaseResources, noMemoryBaseResources},
		{noMemoryEstimator, 3, threeNodeNoMemoryResources, threeNodeNoMemoryResources},
		{noStorageEstimator, 0, noStorageBaseResources, noStorageBaseResources},
		{noStorageEstimator, 3, threeNodeNoStorageResources, threeNodeNoStorageResources},
		{emptyEstimator, 0, noResources, noResources},
		{emptyEstimator, 3, noResources, noResources},
	}

	for i, tc := range testCases {
		got := tc.e.scaleWithNodes(tc.numNodes)
		want := &api.ResourceRequirements{
			Limits:   tc.limits,
			Requests: tc.requests,
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("scaleWithNodes got %v, want %v in test case %d", got, want, i)
		}
	}
}
