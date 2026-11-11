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
	"maps"
	"math/rand"
	"slices"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/knftables"
)

// failingNFTables fails the next Run() when failNext is set.
type failingNFTables struct {
	*knftables.Fake
	failNext bool
}

func (f *failingNFTables) Run(ctx context.Context, tx *knftables.Transaction) error {
	if f.failNext {
		f.failNext = false
		return fmt.Errorf("injected failure")
	}
	return f.Fake.Run(ctx, tx)
}

const numRandomServices = 6

var randomServicePorts = []struct {
	name       string
	protocol   v1.Protocol
	port       int32
	targetPort int32
}{
	{"p80", v1.ProtocolTCP, 80, 8080},
	{"p53", v1.ProtocolUDP, 53, 5353},
	{"p443", v1.ProtocolTCP, 443, 8443},
}

// randomService returns a random variant of Service "ns/svc<i>". The IPs and ports of
// different Services never collide, so that a full sync's result doesn't depend on the
// order it visits Services in.
func randomService(rng *rand.Rand, i int) *v1.Service {
	return makeTestService("ns", fmt.Sprintf("svc%d", i), func(svc *v1.Service) {
		svc.Spec.ClusterIP = fmt.Sprintf("172.30.0.%d", 10+i)
		svc.Spec.Type = []v1.ServiceType{v1.ServiceTypeClusterIP, v1.ServiceTypeNodePort, v1.ServiceTypeLoadBalancer}[rng.Intn(3)]
		for j, p := range randomServicePorts {
			if j > 0 && rng.Intn(2) == 0 {
				continue
			}
			port := v1.ServicePort{
				Name:       p.name,
				Protocol:   p.protocol,
				Port:       p.port,
				TargetPort: intstr.FromInt32(p.targetPort),
			}
			if svc.Spec.Type != v1.ServiceTypeClusterIP {
				port.NodePort = int32(30000 + 10*i + j)
			}
			svc.Spec.Ports = append(svc.Spec.Ports, port)
		}
		if svc.Spec.Type != v1.ServiceTypeClusterIP && rng.Intn(2) == 0 {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
			svc.Spec.HealthCheckNodePort = int32(32000 + i)
		}
		if svc.Spec.Type == v1.ServiceTypeLoadBalancer {
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: fmt.Sprintf("5.6.7.%d", 10+i)}}
			if rng.Intn(2) == 0 {
				svc.Spec.LoadBalancerSourceRanges = []string{"203.0.113.0/25"}
			}
		}
		if rng.Intn(3) == 0 {
			svc.Spec.ExternalIPs = []string{fmt.Sprintf("192.168.99.%d", 10+i)}
		}
		if rng.Intn(3) == 0 {
			svc.Spec.InternalTrafficPolicy = ptr.To(v1.ServiceInternalTrafficPolicyLocal)
		}
		if rng.Intn(3) == 0 {
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{TimeoutSeconds: ptr.To[int32](10800)},
			}
		}
	})
}

// randomEndpointSlice returns a random variant of EndpointSlice k of Service
// "ns/svc<i>". Slice k of every Service picks its endpoints from the same pool of pods,
// so that different Services claim the same hairpin-connections elements. Different
// slices of a Service never share a pod: if they did, which copy the EndpointSliceCache
// keeps would depend on map iteration order, and so would the result of a full sync.
func randomEndpointSlice(rng *rand.Rand, i, k int) *discovery.EndpointSlice {
	return makeTestEndpointSlice("ns", fmt.Sprintf("svc%d", i), k, func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		for n := 1; n <= 5; n++ {
			if rng.Intn(2) == 0 {
				continue
			}
			ep := discovery.Endpoint{
				Addresses: []string{fmt.Sprintf("10.180.%d.%d", k, n)},
				NodeName:  ptr.To("other-node"),
			}
			// Pods with odd addresses run on this node.
			if n%2 == 1 {
				ep.NodeName = ptr.To(testNodeName)
			}
			if rng.Intn(4) == 0 {
				ep.Conditions = discovery.EndpointConditions{
					Ready:       ptr.To(false),
					Serving:     ptr.To(rng.Intn(2) == 0),
					Terminating: ptr.To(true),
				}
			}
			eps.Endpoints = append(eps.Endpoints, ep)
		}
		for _, p := range randomServicePorts {
			if rng.Intn(4) == 0 {
				continue
			}
			eps.Ports = append(eps.Ports, discovery.EndpointPort{
				Name:     ptr.To(p.name),
				Port:     ptr.To(p.targetPort),
				Protocol: ptr.To(p.protocol),
			})
		}
	})
}

// fullSyncDump returns what a full sync of the given objects in a new proxier writes.
func fullSyncDump(t *testing.T, localhostNodePorts bool, services map[string]*v1.Service, endpointSlices map[string]*discovery.EndpointSlice) string {
	t.Helper()
	nft, fp := NewFakeProxier(v1.IPv4Protocol)
	fp.localhostNodePortsEnabled = localhostNodePorts
	makeServiceMap(fp, mapValues(services)...)
	populateEndpointSlices(fp, mapValues(endpointSlices)...)
	if err := fp.syncProxyRules(); err != nil {
		t.Fatalf("full sync failed: %v", err)
	}
	return nft.Dump()
}

func mapValues[T any](m map[string]*T) []*T {
	return slices.Collect(maps.Values(m))
}

// TestPartialSyncMatchesFullSync applies random sequences of Service and EndpointSlice
// changes, and checks that after each partial sync, nftables contains exactly what a
// full sync of the same objects would have written.
func TestPartialSyncMatchesFullSync(t *testing.T) {
	for seed := int64(1); seed <= 40; seed++ {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))
			fake, fp := NewFakeProxier(v1.IPv4Protocol)
			nft := &failingNFTables{Fake: fake}
			fp.nftables = nft
			localhostNodePorts := seed%2 == 0
			fp.localhostNodePortsEnabled = localhostNodePorts
			if err := fp.syncProxyRules(); err != nil {
				t.Fatalf("initial sync failed: %v", err)
			}

			services := map[string]*v1.Service{}
			endpointSlices := map[string]*discovery.EndpointSlice{}
			for step := range 60 {
				// Apply a few changes, so that some syncs handle several at once.
				for range 1 + rng.Intn(3) {
					i := rng.Intn(numRandomServices)
					name := fmt.Sprintf("svc%d", i)
					if rng.Intn(2) == 0 {
						old := services[name]
						if old != nil && rng.Intn(4) == 0 {
							fp.OnServiceDelete(old)
							delete(services, name)
						} else {
							svc := randomService(rng, i)
							fp.OnServiceUpdate(old, svc)
							services[name] = svc
						}
					} else {
						k := rng.Intn(2)
						sliceName := fmt.Sprintf("%s-%d", name, k)
						old := endpointSlices[sliceName]
						if old != nil && rng.Intn(4) == 0 {
							fp.OnEndpointSliceDelete(old)
							delete(endpointSlices, sliceName)
						} else {
							eps := randomEndpointSlice(rng, i, k)
							fp.OnEndpointSliceUpdate(old, eps)
							endpointSlices[sliceName] = eps
						}
					}
				}

				lastFullSync := fp.lastFullSync
				injectFailure := rng.Intn(10) == 0
				nft.failNext = injectFailure
				_ = fp.syncProxyRules()
				if injectFailure {
					if !fp.needFullSync {
						t.Fatalf("step %d: expected a full sync to be needed after a failed sync", step)
					}
					if err := fp.syncProxyRules(); err != nil {
						t.Fatalf("step %d: full sync after failure failed: %v", step, err)
					}
				} else if !fp.lastFullSync.Equal(lastFullSync) || fp.needFullSync {
					t.Fatalf("step %d: expected a successful partial sync", step)
				}

				// Let the stale chains be deleted, as a full sync from scratch
				// won't have them.
				for chain, modtime := range fp.staleChains {
					fp.staleChains[chain] = modtime.Add(-2 * time.Second)
				}
				lastFullSync = fp.lastFullSync
				if err := fp.syncProxyRules(); err != nil {
					t.Fatalf("step %d: sync failed: %v", step, err)
				}
				if !fp.lastFullSync.Equal(lastFullSync) || len(fp.staleChains) != 0 {
					t.Fatalf("step %d: stale chains were not cleaned up by a partial sync: %v", step, fp.staleChains)
				}

				expected := fullSyncDump(t, localhostNodePorts, services, endpointSlices)
				if diff := diffNFTablesTransaction(expected, nft.Dump()); diff != "" {
					t.Fatalf("step %d: partial sync differs from full sync (-full +partial):\n%s", step, diff)
				}

				if got, want := fp.healthCheckNodePorts, fp.svcPortMap.HealthCheckNodePorts(); !maps.Equal(got, want) {
					t.Fatalf("step %d: healthCheckNodePorts is %v, expected %v", step, got, want)
				}
				if got, want := fp.localReadyEndpoints, fp.endpointsMap.LocalReadyEndpoints(); !maps.Equal(got, want) {
					t.Fatalf("step %d: localReadyEndpoints is %v, expected %v", step, got, want)
				}
				checkServicePortTotals(t, step, fp)
			}
		})
	}
}

// checkServicePortTotals checks the totals that partial syncs update incrementally
// against values computed from scratch.
func checkServicePortTotals(t *testing.T, step int, fp *Proxier) {
	t.Helper()
	totalEndpoints, noLocalInternal, noLocalExternal := 0, 0, 0
	for svcName := range fp.svcPortMap {
		totalEndpoints += len(fp.endpointsMap[svcName])
	}
	for _, state := range fp.servicePortStates {
		if state.noLocalEndpointsInternal {
			noLocalInternal++
		}
		if state.noLocalEndpointsExternal {
			noLocalExternal++
		}
	}
	if len(fp.servicePortStates) != len(fp.svcPortMap) {
		t.Fatalf("step %d: have state for %d service ports, expected %d", step, len(fp.servicePortStates), len(fp.svcPortMap))
	}
	if fp.totalEndpoints != totalEndpoints ||
		fp.serviceNoLocalEndpointsTotalInternal != noLocalInternal ||
		fp.serviceNoLocalEndpointsTotalExternal != noLocalExternal {
		t.Fatalf("step %d: totals are (%d, %d, %d), expected (%d, %d, %d)", step,
			fp.totalEndpoints, fp.serviceNoLocalEndpointsTotalInternal, fp.serviceNoLocalEndpointsTotalExternal,
			totalEndpoints, noLocalInternal, noLocalExternal)
	}
}

func TestNFTElementStorageClaims(t *testing.T) {
	ownerA := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "ns", Name: "a"}, Port: "p80", Protocol: v1.ProtocolTCP}
	ownerB := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "ns", Name: "b"}, Port: "p80", Protocol: v1.ProtocolTCP}
	elem := func(value string) *knftables.Element {
		return &knftables.Element{
			Map:   serviceIPsMap,
			Key:   []string{"192.168.99.11", "tcp", "80"},
			Value: []string{value},
		}
	}

	nft, _ := NewFakeProxier(v1.IPv4Protocol)
	s := newNFTElementStorage("map", serviceIPsMap)
	reconcile := func() string {
		tx := nft.NewTransaction()
		s.reconcile(tx)
		return tx.String()
	}
	expectOps := func(desc, ops string) {
		t.Helper()
		if got := reconcile(); got != ops {
			t.Errorf("%s: expected transaction %q, got %q", desc, ops, got)
		}
	}

	key := s.claim(ownerA, elem("goto a"))
	expectOps("first claim", "add element ip kube-proxy service-ips { 192.168.99.11 . tcp . 80 : goto a }\n")

	// The first claim wins a conflict, and the element doesn't change when the
	// winner claims it again.
	s.claim(ownerB, elem("goto b"))
	s.claim(ownerA, elem("goto a"))
	expectOps("conflicting claim", "")

	// The winner can change the value...
	s.claim(ownerA, elem("goto a2"))
	expectOps("winner updates value",
		"delete element ip kube-proxy service-ips { 192.168.99.11 . tcp . 80 }\n"+
			"add element ip kube-proxy service-ips { 192.168.99.11 . tcp . 80 : goto a2 }\n")

	// ...and when it lets go, the next claim takes over.
	s.release(ownerA, key)
	expectOps("winner releases",
		"delete element ip kube-proxy service-ips { 192.168.99.11 . tcp . 80 }\n"+
			"add element ip kube-proxy service-ips { 192.168.99.11 . tcp . 80 : goto b }\n")

	// Releasing something not claimed is a no-op.
	s.release(ownerA, key)
	expectOps("double release", "")

	s.release(ownerB, key)
	expectOps("last claim released", "delete element ip kube-proxy service-ips { 192.168.99.11 . tcp . 80 }\n")
	if len(s.elements) != 0 {
		t.Errorf("expected no elements to be tracked, got %v", s.elements)
	}
}
