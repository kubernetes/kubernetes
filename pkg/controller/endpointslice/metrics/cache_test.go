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
	"fmt"
	"testing"

	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/controller/util/endpoint"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
	utilpointer "k8s.io/utils/pointer"
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

func TestPlaceHolderSlice(t *testing.T) {
	c := NewCache(int32(100))

	p80 := int32(80)
	p443 := int32(443)

	pmKey80443 := endpointutil.NewPortMapKey([]discovery.EndpointPort{{Port: &p80}, {Port: &p443}})
	pmKey80 := endpointutil.NewPortMapKey([]discovery.EndpointPort{{Port: &p80}})

	sp := NewServicePortCache()
	sp.Set(pmKey80, EfficiencyInfo{Endpoints: 0, Slices: 1})
	sp.Set(pmKey80443, EfficiencyInfo{Endpoints: 0, Slices: 1})

	c.UpdateServicePortCache(types.NamespacedName{Namespace: "ns1", Name: "svc1"}, sp)
	expectNumEndpointsAndSlices(t, c, 1, 2, 0)
}

func expectNumEndpointsAndSlices(t *testing.T, c *Cache, desired int, actual int, numEndpoints int) {
	t.Helper()
	if c.numSlicesDesired != desired {
		t.Errorf("Expected numSlicesDesired to be %d, got %d", desired, c.numSlicesDesired)
	}
	if c.numSlicesActual != actual {
		t.Errorf("Expected numSlicesActual to be %d, got %d", actual, c.numSlicesActual)
	}
	if c.numEndpoints != numEndpoints {
		t.Errorf("Expected numEndpoints to be %d, got %d", numEndpoints, c.numEndpoints)
	}
}

func benchmarkUpdateServicePortCache(b *testing.B, num int) {
	c := NewCache(int32(100))
	ns := "benchmark"
	httpKey := endpoint.NewPortMapKey([]discovery.EndpointPort{{Port: utilpointer.Int32Ptr(80)}})
	httpsKey := endpoint.NewPortMapKey([]discovery.EndpointPort{{Port: utilpointer.Int32Ptr(443)}})
	spCache := &ServicePortCache{items: map[endpointutil.PortMapKey]EfficiencyInfo{
		httpKey: {
			Endpoints: 182,
			Slices:    2,
		},
		httpsKey: {
			Endpoints: 356,
			Slices:    4,
		},
	}}

	for i := 0; i < num; i++ {
		nName := types.NamespacedName{Namespace: ns, Name: fmt.Sprintf("service-%d", i)}
		c.UpdateServicePortCache(nName, spCache)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nName := types.NamespacedName{Namespace: ns, Name: fmt.Sprintf("bench-%d", i)}
		c.UpdateServicePortCache(nName, spCache)
	}
}

func BenchmarkUpdateServicePortCache100(b *testing.B) {
	benchmarkUpdateServicePortCache(b, 100)
}

func BenchmarkUpdateServicePortCache1000(b *testing.B) {
	benchmarkUpdateServicePortCache(b, 1000)
}

func BenchmarkUpdateServicePortCache10000(b *testing.B) {
	benchmarkUpdateServicePortCache(b, 10000)
}

func BenchmarkUpdateServicePortCache100000(b *testing.B) {
	benchmarkUpdateServicePortCache(b, 100000)
}
