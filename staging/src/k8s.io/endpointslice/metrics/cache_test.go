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

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/utils/pointer"
)

func TestNumEndpointsAndSlices(t *testing.T) {
	c := NewCache(int32(100))

	p80 := int32(80)
	p443 := int32(443)

	pmKey80443 := endpointsliceutil.NewPortMapKey([]discovery.EndpointPort{{Port: &p80}, {Port: &p443}})
	pmKey80 := endpointsliceutil.NewPortMapKey([]discovery.EndpointPort{{Port: &p80}})

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

	pmKey80443 := endpointsliceutil.NewPortMapKey([]discovery.EndpointPort{{Port: &p80}, {Port: &p443}})
	pmKey80 := endpointsliceutil.NewPortMapKey([]discovery.EndpointPort{{Port: &p80}})

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

// Tests the mutations to servicesByTrafficDistribution field within Cache
// object.
func TestCache_ServicesByTrafficDistribution(t *testing.T) {
	cache := NewCache(0)

	service1 := types.NamespacedName{Namespace: "ns1", Name: "service1"}
	service2 := types.NamespacedName{Namespace: "ns1", Name: "service2"}
	service3 := types.NamespacedName{Namespace: "ns2", Name: "service3"}
	service4 := types.NamespacedName{Namespace: "ns3", Name: "service4"}

	// Define helper function for assertion
	mustHaveServicesByTrafficDistribution := func(wantServicesByTrafficDistribution map[string]map[types.NamespacedName]bool, desc string) {
		t.Helper()
		gotServicesByTrafficDistribution := cache.servicesByTrafficDistribution
		if diff := cmp.Diff(wantServicesByTrafficDistribution, gotServicesByTrafficDistribution); diff != "" {
			t.Fatalf("UpdateTrafficDistributionForService(%v) resulted in unexpected diff for cache.servicesByTrafficDistribution; (-want, +got)\n%v", desc, diff)
		}
	}

	// Mutate and make assertions

	desc := "service1 starts using trafficDistribution=PreferClose"
	cache.UpdateTrafficDistributionForService(service1, ptrTo(corev1.ServiceTrafficDistributionPreferClose))
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{
		corev1.ServiceTrafficDistributionPreferClose: {service1: true},
	}, desc)

	desc = "service1 starts using trafficDistribution=PreferClose, retries of similar mutation should be idempotent"
	cache.UpdateTrafficDistributionForService(service1, ptrTo(corev1.ServiceTrafficDistributionPreferClose))
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{ // No delta
		corev1.ServiceTrafficDistributionPreferClose: {service1: true},
	}, desc)

	desc = "service2 starts using trafficDistribution=PreferClose"
	cache.UpdateTrafficDistributionForService(service2, ptrTo(corev1.ServiceTrafficDistributionPreferClose))
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{
		corev1.ServiceTrafficDistributionPreferClose: {service1: true, service2: true}, // Delta
	}, desc)

	desc = "service3 starts using trafficDistribution=InvalidValue"
	cache.UpdateTrafficDistributionForService(service3, ptrTo("InvalidValue"))
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{
		corev1.ServiceTrafficDistributionPreferClose: {service1: true, service2: true},
		trafficDistributionImplementationSpecific:    {service3: true}, // Delta
	}, desc)

	desc = "service4 starts using trafficDistribution=nil"
	cache.UpdateTrafficDistributionForService(service4, nil)
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{ // No delta
		corev1.ServiceTrafficDistributionPreferClose: {service1: true, service2: true},
		trafficDistributionImplementationSpecific:    {service3: true},
	}, desc)

	desc = "service2 transitions trafficDistribution: PreferClose -> InvalidValue"
	cache.UpdateTrafficDistributionForService(service2, ptrTo("InvalidValue"))
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{
		corev1.ServiceTrafficDistributionPreferClose: {service1: true},                 // Delta
		trafficDistributionImplementationSpecific:    {service3: true, service2: true}, // Delta
	}, desc)

	desc = "service3 gets deleted"
	cache.DeleteService(service3)
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{
		corev1.ServiceTrafficDistributionPreferClose: {service1: true},
		trafficDistributionImplementationSpecific:    {service2: true}, // Delta
	}, desc)

	desc = "service1 transitions trafficDistribution: PreferClose -> empty"
	cache.UpdateTrafficDistributionForService(service1, ptrTo(""))
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{
		corev1.ServiceTrafficDistributionPreferClose: {},                               // Delta
		trafficDistributionImplementationSpecific:    {service1: true, service2: true}, // Delta
	}, desc)

	desc = "service1 transitions trafficDistribution: InvalidValue -> nil"
	cache.UpdateTrafficDistributionForService(service1, nil)
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{
		corev1.ServiceTrafficDistributionPreferClose: {},
		trafficDistributionImplementationSpecific:    {service2: true}, // Delta
	}, desc)

	desc = "service2 transitions trafficDistribution: InvalidValue -> nil"
	cache.UpdateTrafficDistributionForService(service2, nil)
	mustHaveServicesByTrafficDistribution(map[string]map[types.NamespacedName]bool{
		corev1.ServiceTrafficDistributionPreferClose: {},
		trafficDistributionImplementationSpecific:    {}, // Delta
	}, desc)

}

func benchmarkUpdateServicePortCache(b *testing.B, num int) {
	c := NewCache(int32(100))
	ns := "benchmark"
	httpKey := endpointsliceutil.NewPortMapKey([]discovery.EndpointPort{{Port: pointer.Int32(80)}})
	httpsKey := endpointsliceutil.NewPortMapKey([]discovery.EndpointPort{{Port: pointer.Int32(443)}})
	spCache := &ServicePortCache{items: map[endpointsliceutil.PortMapKey]EfficiencyInfo{
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

func ptrTo[T any](obj T) *T {
	return &obj
}
