/*
Copyright 2019 The Kubernetes Authors.

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

package metrics

import (
	"testing"

	discovery "k8s.io/api/discovery/v1beta1"
	"k8s.io/apimachinery/pkg/types"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
)

func TestNumEndpointsAndSlices(t *testing.T) {
	c := NewCache(int32(100))

	p80 := int32(80)
	p443 := int32(443)

	pmKey80443 := endpointutil.NewPortMapKey([]discovery.EndpointPort{{Port: &p80}, {Port: &p443}})
	pmKey80 := endpointutil.NewPortMapKey([]discovery.EndpointPort{{Port: &p80}})

	spCacheEfficient := NewServicePortCache()
	spCacheEfficient.Set(pmKey80, EfficiencyInfo{Endpoints: 45, Slices: 1})
	spCacheEfficient.Set(pmKey80443, EfficiencyInfo{Endpoints: 35, Slices: 1})

	spCacheInefficient := NewServicePortCache()
	spCacheInefficient.Set(pmKey80, EfficiencyInfo{Endpoints: 12, Slices: 5})
	spCacheInefficient.Set(pmKey80443, EfficiencyInfo{Endpoints: 18, Slices: 8})

	c.UpdateServicePortCache(types.NamespacedName{Namespace: "ns1", Name: "svc1"}, spCacheInefficient)
	expectNumEndpointsAndSlices(t, c, 2, 13, 30)

	c.UpdateServicePortCache(types.NamespacedName{Namespace: "ns1", Name: "svc2"}, spCacheEfficient)
	expectNumEndpointsAndSlices(t, c, 4, 15, 110)

	c.UpdateServicePortCache(types.NamespacedName{Namespace: "ns1", Name: "svc3"}, spCacheInefficient)
	expectNumEndpointsAndSlices(t, c, 6, 28, 140)

	c.UpdateServicePortCache(types.NamespacedName{Namespace: "ns1", Name: "svc1"}, spCacheEfficient)
	expectNumEndpointsAndSlices(t, c, 6, 17, 190)

	c.DeleteService(types.NamespacedName{Namespace: "ns1", Name: "svc3"})
	expectNumEndpointsAndSlices(t, c, 4, 4, 160)
}

func expectNumEndpointsAndSlices(t *testing.T, c *Cache, desired int, actual int, numEndpoints int) {
	t.Helper()
	mUpdate := c.desiredAndActualSlices()
	if mUpdate.desired != desired {
		t.Errorf("Expected numEndpointSlices to be %d, got %d", desired, mUpdate.desired)
	}
	if mUpdate.actual != actual {
		t.Errorf("Expected desiredEndpointSlices to be %d, got %d", actual, mUpdate.actual)
	}
	if c.numEndpoints != numEndpoints {
		t.Errorf("Expected numEndpoints to be %d, got %d", numEndpoints, c.numEndpoints)
	}
}
