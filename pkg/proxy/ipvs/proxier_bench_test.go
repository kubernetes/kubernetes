//go:build linux

/*
Copyright 2026 The Kubernetes Authors.

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

package ipvs

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/ktesting"
	ipsettest "k8s.io/kubernetes/pkg/proxy/ipvs/ipset/testing"
	utilipvs "k8s.io/kubernetes/pkg/proxy/ipvs/util"
	ipvstest "k8s.io/kubernetes/pkg/proxy/ipvs/util/testing"
	iptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
	"k8s.io/utils/ptr"
)

// netlinkReceiveBufferSize mirrors nl.RECEIVE_BUFFER_SIZE. The netlink package
// allocates a buffer of this size on every Receive, so every IPVS netlink
// operation costs at least this much garbage. The fake below charges each
// emulated netlink call that allocation so that the benchmark reflects the
// cost of talking to the kernel instead of the cost of the fake itself.
const netlinkReceiveBufferSize = 64 << 10

var netlinkReceiveBufferSink []byte

// netlinkCostIPVS counts the IPVS kernel operations issued during a sync and
// models the allocation that each of them causes.
type netlinkCostIPVS struct {
	*ipvstest.FakeIPVS

	getVirtualServerCalls  int
	getVirtualServersCalls int
	getRealServersCalls    int
}

func (f *netlinkCostIPVS) simulateNetlinkReceive() {
	netlinkReceiveBufferSink = make([]byte, netlinkReceiveBufferSize)
}

func (f *netlinkCostIPVS) GetVirtualServer(vs *utilipvs.VirtualServer) (*utilipvs.VirtualServer, error) {
	f.getVirtualServerCalls++
	f.simulateNetlinkReceive()
	return f.FakeIPVS.GetVirtualServer(vs)
}

func (f *netlinkCostIPVS) GetVirtualServers() ([]*utilipvs.VirtualServer, error) {
	f.getVirtualServersCalls++
	f.simulateNetlinkReceive()
	return f.FakeIPVS.GetVirtualServers()
}

func (f *netlinkCostIPVS) GetRealServers(vs *utilipvs.VirtualServer) ([]*utilipvs.RealServer, error) {
	f.getRealServersCalls++
	f.simulateNetlinkReceive()
	return f.FakeIPVS.GetRealServers(vs)
}

func (f *netlinkCostIPVS) resetCounters() {
	f.getVirtualServerCalls = 0
	f.getVirtualServersCalls = 0
	f.getRealServersCalls = 0
}

func (f *netlinkCostIPVS) netlinkCalls() int {
	return f.getVirtualServerCalls + f.getVirtualServersCalls + f.getRealServersCalls
}

func makeTestIPv4ClusterIPServices(numServices int) ([]*v1.Service, []*discovery.EndpointSlice) {
	services := make([]*v1.Service, 0, numServices)
	endpointSlices := make([]*discovery.EndpointSlice, 0, numServices)
	for i := 0; i < numServices; i++ {
		name := fmt.Sprintf("svc-%d", i)
		services = append(services, makeTestService("ns", name, func(svc *v1.Service) {
			svc.Spec.ClusterIP = fmt.Sprintf("10.96.%d.%d", i/256, i%256)
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "http",
				Port:     80,
				Protocol: v1.ProtocolTCP,
			}}
		}))
		endpointSlices = append(endpointSlices, makeTestEndpointSlice("ns", name, 0, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{fmt.Sprintf("10.244.%d.%d", i/256, i%256)},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("http"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}))
	}
	return services, endpointSlices
}

// TestSyncProxyRulesReadsServicesOnce ensures that a sync reads the IPVS
// service table once instead of once per service. Destinations are still read
// per service, since the kernel API has no bulk equivalent for them.
func TestSyncProxyRulesReadsServicesOnce(t *testing.T) {
	const numServices = 10

	_, ctx := ktesting.NewTestContext(t)
	ipvs := &netlinkCostIPVS{FakeIPVS: ipvstest.NewFake()}
	proxier := NewFakeProxier(ctx, iptablestest.NewFake(), ipvs, ipsettest.NewFake(testIPSetVersion), []string{"192.168.0.1"}, nil, v1.IPv4Protocol)

	services, endpointSlices := makeTestIPv4ClusterIPServices(numServices)
	makeServiceMap(proxier, services...)
	makeEndpointSliceMap(proxier, endpointSlices...)
	proxier.syncProxyRules()

	ipvs.resetCounters()
	proxier.syncProxyRules()

	if got := ipvs.getVirtualServersCalls; got != 1 {
		t.Errorf("expected 1 GetVirtualServers call during a sync, got %d", got)
	}
	if got := ipvs.getVirtualServerCalls; got != 0 {
		t.Errorf("expected no GetVirtualServer calls during a sync, got %d", got)
	}
	if got := ipvs.getRealServersCalls; got != numServices {
		t.Errorf("expected %d GetRealServers calls during a sync, got %d", numServices, got)
	}
}

// BenchmarkSyncProxyRulesSteadyState measures a sync where nothing changed
// since the previous one. That is the common case when syncProxyRules is
// triggered by endpoint churn: the services themselves are stable, but
// kube-proxy still walks every service and reads its state back from the
// kernel.
func BenchmarkSyncProxyRulesSteadyState(b *testing.B) {
	for _, numServices := range []int{100, 1000} {
		b.Run(fmt.Sprintf("%dServices", numServices), func(b *testing.B) {
			_, ctx := ktesting.NewTestContext(b)
			ipvs := &netlinkCostIPVS{FakeIPVS: ipvstest.NewFake()}
			proxier := NewFakeProxier(ctx, iptablestest.NewFake(), ipvs, ipsettest.NewFake(testIPSetVersion), []string{"192.168.0.1"}, nil, v1.IPv4Protocol)

			services, endpointSlices := makeTestIPv4ClusterIPServices(numServices)
			makeServiceMap(proxier, services...)
			makeEndpointSliceMap(proxier, endpointSlices...)
			// Apply the initial state so that the timed loop only re-syncs an
			// already converged proxier.
			proxier.syncProxyRules()

			ipvs.resetCounters()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				proxier.syncProxyRules()
			}
			b.StopTimer()

			b.ReportMetric(float64(ipvs.netlinkCalls())/float64(b.N), "netlink/op")
			b.ReportMetric(float64(ipvs.getVirtualServerCalls)/float64(b.N), "svc-get/op")
			b.ReportMetric(float64(ipvs.getVirtualServersCalls)/float64(b.N), "svcs-dump/op")
			b.ReportMetric(float64(ipvs.getRealServersCalls)/float64(b.N), "dest-get/op")
		})
	}
}
