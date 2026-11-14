//go:build linux

/*
Copyright The Kubernetes Authors.

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

package nftables

import (
	"context"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/knftables"
)

// benchNFTables lets the initial sync populate the fake, then turns Run() into a
// no-op. knftables.Fake copies the whole table on every Run(), which is an artifact
// of the fake and would hide the proxier's own per-sync cost.
type benchNFTables struct {
	*knftables.Fake
	skipRun bool
	lastOps int
}

func (b *benchNFTables) Run(ctx context.Context, tx *knftables.Transaction) error {
	b.lastOps = tx.NumOperations()
	if b.skipRun {
		return nil
	}
	return b.Fake.Run(ctx, tx)
}

func makeScaleEndpointSlice(i int, alt bool) *discovery.EndpointSlice {
	return makeTestEndpointSlice("ns", fmt.Sprintf("svc%d", i), 1, func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		third := 3
		if alt {
			third = 4
		}
		for j := 1; j <= 3; j++ {
			n := j
			if j == 3 {
				n = third
			}
			ep := discovery.Endpoint{
				Addresses: []string{fmt.Sprintf("10.%d.%d.%d", i/250, i%250, n)},
				NodeName:  ptr.To(fmt.Sprintf("node-%d", n)),
			}
			// Every 10th service has a local endpoint.
			if j == 1 && i%10 == 0 {
				ep.NodeName = ptr.To(testNodeName)
			}
			eps.Endpoints = append(eps.Endpoints, ep)
		}
		eps.Ports = []discovery.EndpointPort{{
			Name:     ptr.To("p80"),
			Port:     ptr.To[int32](8080),
			Protocol: ptr.To(v1.ProtocolTCP),
		}}
	})
}

func populateScaleProxier(proxier *Proxier, numServices int) {
	services := make([]*v1.Service, 0, numServices)
	slices := make([]*discovery.EndpointSlice, 0, numServices)
	for i := range numServices {
		services = append(services, makeTestService("ns", fmt.Sprintf("svc%d", i), func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = fmt.Sprintf("172.30.%d.%d", i/250, i%250+1)
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
			}}
			// Every 10th service is a NodePort.
			if i%10 == 0 {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.Ports[0].NodePort = int32(30000 + i)
			}
		}))
		slices = append(slices, makeScaleEndpointSlice(i, false))
	}
	makeServiceMap(proxier, services...)
	populateEndpointSlices(proxier, slices...)
}

// BenchmarkSyncProxyRulesOneEndpointChange measures a sync after a single
// EndpointSlice change, as a function of the total number of Services.
func BenchmarkSyncProxyRulesOneEndpointChange(b *testing.B) {
	for _, numServices := range []int{1000, 4000, 16000} {
		b.Run(fmt.Sprintf("services=%d", numServices), func(b *testing.B) {
			fake, proxier := NewFakeProxier(v1.IPv4Protocol)
			nft := &benchNFTables{Fake: fake}
			proxier.nftables = nft
			populateScaleProxier(proxier, numServices)
			if err := proxier.syncProxyRules(); err != nil {
				b.Fatalf("initial sync failed: %v", err)
			}
			nft.skipRun = true

			// Change a non-local endpoint so the service's set of local endpoints
			// (and thus the healthcheck data) doesn't change.
			changed := numServices / 2
			if changed%10 == 0 {
				changed++
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				proxier.OnEndpointSliceUpdate(nil, makeScaleEndpointSlice(changed, i%2 == 0))
				if err := proxier.syncProxyRules(); err != nil {
					b.Fatalf("sync failed: %v", err)
				}
			}
			b.ReportMetric(float64(nft.lastOps), "nft-ops/sync")
		})
	}
}
