/*
Copyright 2017 The Kubernetes Authors.

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

package proxy

import (
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"

	"k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilpointer "k8s.io/utils/pointer"
)

func (proxier *FakeProxier) addEndpoints(endpoints *v1.Endpoints) {
	proxier.endpointsChanges.Update(nil, endpoints)
}

func (proxier *FakeProxier) updateEndpoints(oldEndpoints, endpoints *v1.Endpoints) {
	proxier.endpointsChanges.Update(oldEndpoints, endpoints)
}

func (proxier *FakeProxier) deleteEndpoints(endpoints *v1.Endpoints) {
	proxier.endpointsChanges.Update(endpoints, nil)
}

func TestGetLocalEndpointIPs(t *testing.T) {
	testCases := []struct {
		endpointsMap EndpointsMap
		expected     map[types.NamespacedName]sets.String
	}{{
		// Case[0]: nothing
		endpointsMap: EndpointsMap{},
		expected:     map[types.NamespacedName]sets.String{},
	}, {
		// Case[1]: unnamed port
		endpointsMap: EndpointsMap{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolTCP): []Endpoint{
				&BaseEndpointInfo{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expected: map[types.NamespacedName]sets.String{},
	}, {
		// Case[2]: unnamed port local
		endpointsMap: EndpointsMap{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolTCP): []Endpoint{
				&BaseEndpointInfo{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns1", Name: "ep1"}: sets.NewString("1.1.1.1"),
		},
	}, {
		// Case[3]: named local and non-local ports for the same IP.
		endpointsMap: EndpointsMap{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolTCP): []Endpoint{
				&BaseEndpointInfo{Endpoint: "1.1.1.1:11", IsLocal: false},
				&BaseEndpointInfo{Endpoint: "1.1.1.2:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolTCP): []Endpoint{
				&BaseEndpointInfo{Endpoint: "1.1.1.1:12", IsLocal: false},
				&BaseEndpointInfo{Endpoint: "1.1.1.2:12", IsLocal: true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns1", Name: "ep1"}: sets.NewString("1.1.1.2"),
		},
	}, {
		// Case[4]: named local and non-local ports for different IPs.
		endpointsMap: EndpointsMap{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolTCP): []Endpoint{
				&BaseEndpointInfo{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolTCP): []Endpoint{
				&BaseEndpointInfo{Endpoint: "2.2.2.2:22", IsLocal: true},
				&BaseEndpointInfo{Endpoint: "2.2.2.22:22", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p23", v1.ProtocolTCP): []Endpoint{
				&BaseEndpointInfo{Endpoint: "2.2.2.3:23", IsLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolTCP): []Endpoint{
				&BaseEndpointInfo{Endpoint: "4.4.4.4:44", IsLocal: true},
				&BaseEndpointInfo{Endpoint: "4.4.4.5:44", IsLocal: false},
			},
			makeServicePortName("ns4", "ep4", "p45", v1.ProtocolTCP): []Endpoint{
				&BaseEndpointInfo{Endpoint: "4.4.4.6:45", IsLocal: true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns2", Name: "ep2"}: sets.NewString("2.2.2.2", "2.2.2.22", "2.2.2.3"),
			{Namespace: "ns4", Name: "ep4"}: sets.NewString("4.4.4.4", "4.4.4.6"),
		},
	}}

	for tci, tc := range testCases {
		// outputs
		localIPs := tc.endpointsMap.getLocalEndpointIPs()

		if !reflect.DeepEqual(localIPs, tc.expected) {
			t.Errorf("[%d] expected %#v, got %#v", tci, tc.expected, localIPs)
		}
	}
}

func makeTestEndpoints(namespace, name string, eptFunc func(*v1.Endpoints)) *v1.Endpoints {
	ept := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: make(map[string]string),
		},
	}
	eptFunc(ept)
	return ept
}

// This is a coarse test, but it offers some modicum of confidence as the code is evolved.
func TestEndpointsToEndpointsMap(t *testing.T) {
	testCases := []struct {
		desc         string
		newEndpoints *v1.Endpoints
		expected     map[ServicePortName][]*BaseEndpointInfo
		isIPv6Mode   *bool
		ipFamily     v1.IPFamily
	}{
		{
			desc:     "nothing",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {}),
			expected:     map[ServicePortName][]*BaseEndpointInfo{},
		},
		{
			desc:     "no changes, unnamed port",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{{
							IP: "1.1.1.1",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "",
							Port:     11,
							Protocol: v1.ProtocolTCP,
						}},
					},
				}
			}),
			expected: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "ep1", "", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:11", IsLocal: false},
				},
			},
		},
		{
			desc:     "no changes, named port",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{{
							IP: "1.1.1.1",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "port",
							Port:     11,
							Protocol: v1.ProtocolTCP,
						}},
					},
				}
			}),
			expected: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "ep1", "port", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:11", IsLocal: false},
				},
			},
		},
		{
			desc:     "new port",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{{
							IP: "1.1.1.1",
						}},
						Ports: []v1.EndpointPort{{
							Port:     11,
							Protocol: v1.ProtocolTCP,
						}},
					},
				}
			}),
			expected: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "ep1", "", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:11", IsLocal: false},
				},
			},
		},
		{
			desc:     "remove port",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {}),
			expected:     map[ServicePortName][]*BaseEndpointInfo{},
		},
		{
			desc:     "new IP and port",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{{
							IP: "1.1.1.1",
						}, {
							IP: "2.2.2.2",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p1",
							Port:     11,
							Protocol: v1.ProtocolTCP,
						}, {
							Name:     "p2",
							Port:     22,
							Protocol: v1.ProtocolTCP,
						}},
					},
				}
			}),
			expected: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "ep1", "p1", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:11", IsLocal: false},
					{Endpoint: "2.2.2.2:11", IsLocal: false},
				},
				makeServicePortName("ns1", "ep1", "p2", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:22", IsLocal: false},
					{Endpoint: "2.2.2.2:22", IsLocal: false},
				},
			},
		},
		{
			desc:     "remove IP and port",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{{
							IP: "1.1.1.1",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p1",
							Port:     11,
							Protocol: v1.ProtocolTCP,
						}},
					},
				}
			}),
			expected: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "ep1", "p1", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:11", IsLocal: false},
				},
			},
		},
		{
			desc:     "rename port",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{{
							IP: "1.1.1.1",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p2",
							Port:     11,
							Protocol: v1.ProtocolTCP,
						}},
					},
				}
			}),
			expected: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "ep1", "p2", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:11", IsLocal: false},
				},
			},
		},
		{
			desc:     "renumber port",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{{
							IP: "1.1.1.1",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p1",
							Port:     22,
							Protocol: v1.ProtocolTCP,
						}},
					},
				}
			}),
			expected: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "ep1", "p1", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:22", IsLocal: false},
				},
			},
		},
		{
			desc:     "should omit IPv6 address in IPv4 mode",
			ipFamily: v1.IPv4Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{{
							IP: "1.1.1.1",
						}, {
							IP: "2001:db8:85a3:0:0:8a2e:370:7334",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p1",
							Port:     11,
							Protocol: v1.ProtocolTCP,
						}, {
							Name:     "p2",
							Port:     22,
							Protocol: v1.ProtocolTCP,
						}},
					},
				}
			}),
			expected: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "ep1", "p1", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:11", IsLocal: false},
				},
				makeServicePortName("ns1", "ep1", "p2", v1.ProtocolTCP): {
					{Endpoint: "1.1.1.1:22", IsLocal: false},
				},
			},
		},
		{
			desc:     "should omit IPv4 address in IPv6 mode",
			ipFamily: v1.IPv6Protocol,

			newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{{
							IP: "1.1.1.1",
						}, {
							IP: "2001:db8:85a3:0:0:8a2e:370:7334",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p1",
							Port:     11,
							Protocol: v1.ProtocolTCP,
						}, {
							Name:     "p2",
							Port:     22,
							Protocol: v1.ProtocolTCP,
						}},
					},
				}
			}),
			expected: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "ep1", "p1", v1.ProtocolTCP): {
					{Endpoint: "[2001:db8:85a3:0:0:8a2e:370:7334]:11", IsLocal: false},
				},
				makeServicePortName("ns1", "ep1", "p2", v1.ProtocolTCP): {
					{Endpoint: "[2001:db8:85a3:0:0:8a2e:370:7334]:22", IsLocal: false},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {

			epTracker := NewEndpointChangeTracker("test-hostname", nil, tc.ipFamily, nil, false, nil)

			// outputs
			newEndpoints := epTracker.endpointsToEndpointsMap(tc.newEndpoints)

			if len(newEndpoints) != len(tc.expected) {
				t.Fatalf("[%s] expected %d new, got %d: %v", tc.desc, len(tc.expected), len(newEndpoints), spew.Sdump(newEndpoints))
			}
			for x := range tc.expected {
				if len(newEndpoints[x]) != len(tc.expected[x]) {
					t.Fatalf("[%s] expected %d endpoints for %v, got %d", tc.desc, len(tc.expected[x]), x, len(newEndpoints[x]))
				} else {
					for i := range newEndpoints[x] {
						ep := newEndpoints[x][i].(*BaseEndpointInfo)
						if !(reflect.DeepEqual(*ep, *(tc.expected[x][i]))) {
							t.Fatalf("[%s] expected new[%v][%d] to be %v, got %v", tc.desc, x, i, tc.expected[x][i], *ep)
						}
					}
				}
			}
		})
	}
}

func TestUpdateEndpointsMap(t *testing.T) {
	var nodeName = testHostname

	emptyEndpoint := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{}
	}
	unnamedPort := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	unnamedPortLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPortLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPort := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPortRenamed := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11-2",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPortRenumbered := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     22,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPortsLocalNoLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsets := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.2",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsetsWithLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsetsMultiplePortsLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.3",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p13",
				Port:     13,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsetsIPsPorts1 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.3",
			}, {
				IP:       "1.1.1.4",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p13",
				Port:     13,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p14",
				Port:     14,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsetsIPsPorts2 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "2.2.2.1",
			}, {
				IP:       "2.2.2.2",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p21",
				Port:     21,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p22",
				Port:     22,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexBefore1 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexBefore2 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "2.2.2.2",
				NodeName: &nodeName,
			}, {
				IP:       "2.2.2.22",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p22",
				Port:     22,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP:       "2.2.2.3",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p23",
				Port:     23,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexBefore4 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "4.4.4.4",
				NodeName: &nodeName,
			}, {
				IP:       "4.4.4.5",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p44",
				Port:     44,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP:       "4.4.4.6",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p45",
				Port:     45,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexAfter1 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP: "1.1.1.11",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.2",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p122",
				Port:     122,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexAfter3 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "3.3.3.3",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p33",
				Port:     33,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexAfter4 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "4.4.4.4",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p44",
				Port:     44,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}

	testCases := []struct {
		// previousEndpoints and currentEndpoints are used to call appropriate
		// handlers OnEndpoints* (based on whether corresponding values are nil
		// or non-nil) and must be of equal length.
		name                      string
		previousEndpoints         []*v1.Endpoints
		currentEndpoints          []*v1.Endpoints
		oldEndpoints              map[ServicePortName][]*BaseEndpointInfo
		expectedResult            map[ServicePortName][]*BaseEndpointInfo
		expectedStaleEndpoints    []ServiceEndpoint
		expectedStaleServiceNames map[ServicePortName]bool
		expectedHealthchecks      map[types.NamespacedName]int
	}{{
		name:                      "empty",
		oldEndpoints:              map[ServicePortName][]*BaseEndpointInfo{},
		expectedResult:            map[ServicePortName][]*BaseEndpointInfo{},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		name: "no change, unnamed port",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		name: "no change, named port, local",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		name: "no change, multiple subsets",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		name: "no change, multiple subsets, multiple ports, local",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		name: "no change, multiple endpoints, subsets, IPs, and ports",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.2:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false},
				{Endpoint: "1.1.1.2:12", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false},
				{Endpoint: "1.1.1.4:13", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:14", IsLocal: false},
				{Endpoint: "1.1.1.4:14", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:21", IsLocal: false},
				{Endpoint: "2.2.2.2:21", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:22", IsLocal: false},
				{Endpoint: "2.2.2.2:22", IsLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.2:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false},
				{Endpoint: "1.1.1.2:12", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false},
				{Endpoint: "1.1.1.4:13", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:14", IsLocal: false},
				{Endpoint: "1.1.1.4:14", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:21", IsLocal: false},
				{Endpoint: "2.2.2.2:21", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:22", IsLocal: false},
				{Endpoint: "2.2.2.2:22", IsLocal: true},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 2,
			makeNSN("ns2", "ep2"): 1,
		},
	}, {
		name: "add an Endpoints",
		previousEndpoints: []*v1.Endpoints{
			nil,
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPortLocal),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		name: "remove an Endpoints",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPortLocal),
		},
		currentEndpoints: []*v1.Endpoints{
			nil,
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		name: "add an IP and port",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortsLocalNoLocal),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.2:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false},
				{Endpoint: "1.1.1.2:12", IsLocal: true},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		name: "remove an IP and port",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortsLocalNoLocal),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.2:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false},
				{Endpoint: "1.1.1.2:12", IsLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.2:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}, {
			Endpoint:        "1.1.1.1:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}, {
			Endpoint:        "1.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		name: "add a subset",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsWithLocal),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: true},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		name: "remove a subset",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		name: "rename a port",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenamed),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11-2", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p11-2", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		name: "renumber a port",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenumbered),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:22", IsLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		name: "complex add and remove",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", complexBefore1),
			makeTestEndpoints("ns2", "ep2", complexBefore2),
			nil,
			makeTestEndpoints("ns4", "ep4", complexBefore4),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", complexAfter1),
			nil,
			makeTestEndpoints("ns3", "ep3", complexAfter3),
			makeTestEndpoints("ns4", "ep4", complexAfter4),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.2:22", IsLocal: true},
				{Endpoint: "2.2.2.22:22", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p23", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.3:23", IsLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{Endpoint: "4.4.4.4:44", IsLocal: true},
				{Endpoint: "4.4.4.5:44", IsLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p45", v1.ProtocolUDP): {
				{Endpoint: "4.4.4.6:45", IsLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.11:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p122", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:122", IsLocal: false},
			},
			makeServicePortName("ns3", "ep3", "p33", v1.ProtocolUDP): {
				{Endpoint: "3.3.3.3:33", IsLocal: false},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{Endpoint: "4.4.4.4:44", IsLocal: true},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "2.2.2.2:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP),
		}, {
			Endpoint:        "2.2.2.22:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP),
		}, {
			Endpoint:        "2.2.2.3:23",
			ServicePortName: makeServicePortName("ns2", "ep2", "p23", v1.ProtocolUDP),
		}, {
			Endpoint:        "4.4.4.5:44",
			ServicePortName: makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP),
		}, {
			Endpoint:        "4.4.4.6:45",
			ServicePortName: makeServicePortName("ns4", "ep4", "p45", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP):  true,
			makeServicePortName("ns1", "ep1", "p122", v1.ProtocolUDP): true,
			makeServicePortName("ns3", "ep3", "p33", v1.ProtocolUDP):  true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns4", "ep4"): 1,
		},
	}, {
		name: "change from 0 endpoint address to 1 unnamed port",
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", emptyEndpoint),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		oldEndpoints: map[ServicePortName][]*BaseEndpointInfo{},
		expectedResult: map[ServicePortName][]*BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	},
	}

	for tci, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fp := newFakeProxier(v1.IPv4Protocol)
			fp.hostname = nodeName

			// First check that after adding all previous versions of endpoints,
			// the fp.oldEndpoints is as we expect.
			for i := range tc.previousEndpoints {
				if tc.previousEndpoints[i] != nil {
					fp.addEndpoints(tc.previousEndpoints[i])
				}
			}
			fp.endpointsMap.Update(fp.endpointsChanges)
			compareEndpointsMapsStr(t, fp.endpointsMap, tc.oldEndpoints)

			// Now let's call appropriate handlers to get to state we want to be.
			if len(tc.previousEndpoints) != len(tc.currentEndpoints) {
				t.Fatalf("[%d] different lengths of previous and current endpoints", tci)
				return
			}

			for i := range tc.previousEndpoints {
				prev, curr := tc.previousEndpoints[i], tc.currentEndpoints[i]
				switch {
				case prev == nil:
					fp.addEndpoints(curr)
				case curr == nil:
					fp.deleteEndpoints(prev)
				default:
					fp.updateEndpoints(prev, curr)
				}
			}
			result := fp.endpointsMap.Update(fp.endpointsChanges)
			newMap := fp.endpointsMap
			compareEndpointsMapsStr(t, newMap, tc.expectedResult)
			if len(result.StaleEndpoints) != len(tc.expectedStaleEndpoints) {
				t.Errorf("[%d] expected %d staleEndpoints, got %d: %v", tci, len(tc.expectedStaleEndpoints), len(result.StaleEndpoints), result.StaleEndpoints)
			}
			for _, x := range tc.expectedStaleEndpoints {
				found := false
				for _, stale := range result.StaleEndpoints {
					if stale == x {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("[%d] expected staleEndpoints[%v], but didn't find it: %v", tci, x, result.StaleEndpoints)
				}
			}
			if len(result.StaleServiceNames) != len(tc.expectedStaleServiceNames) {
				t.Errorf("[%d] expected %d staleServiceNames, got %d: %v", tci, len(tc.expectedStaleServiceNames), len(result.StaleServiceNames), result.StaleServiceNames)
			}
			for svcName := range tc.expectedStaleServiceNames {
				found := false
				for _, stale := range result.StaleServiceNames {
					if stale == svcName {
						found = true
					}
				}
				if !found {
					t.Errorf("[%d] expected staleServiceNames[%v], but didn't find it: %v", tci, svcName, result.StaleServiceNames)
				}
			}
			if !reflect.DeepEqual(result.HCEndpointsLocalIPSize, tc.expectedHealthchecks) {
				t.Errorf("[%d] expected healthchecks %v, got %v", tci, tc.expectedHealthchecks, result.HCEndpointsLocalIPSize)
			}
		})
	}
}

func TestLastChangeTriggerTime(t *testing.T) {
	t0 := time.Date(2018, 01, 01, 0, 0, 0, 0, time.UTC)
	t1 := t0.Add(time.Second)
	t2 := t1.Add(time.Second)
	t3 := t2.Add(time.Second)

	createEndpoints := func(namespace, name string, triggerTime time.Time) *v1.Endpoints {
		e := makeTestEndpoints(namespace, name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.1.1.1"}},
				Ports:     []v1.EndpointPort{{Port: 11}},
			}}
		})
		e.Annotations[v1.EndpointsLastChangeTriggerTime] = triggerTime.Format(time.RFC3339Nano)
		return e
	}

	createName := func(namespace, name string) types.NamespacedName {
		return types.NamespacedName{Namespace: namespace, Name: name}
	}

	modifyEndpoints := func(endpoints *v1.Endpoints, triggerTime time.Time) *v1.Endpoints {
		e := endpoints.DeepCopy()
		e.Subsets[0].Ports[0].Port++
		e.Annotations[v1.EndpointsLastChangeTriggerTime] = triggerTime.Format(time.RFC3339Nano)
		return e
	}

	testCases := []struct {
		name     string
		scenario func(fp *FakeProxier)
		expected map[types.NamespacedName][]time.Time
	}{
		{
			name: "Single addEndpoints",
			scenario: func(fp *FakeProxier) {
				e := createEndpoints("ns", "ep1", t0)
				fp.addEndpoints(e)
			},
			expected: map[types.NamespacedName][]time.Time{createName("ns", "ep1"): {t0}},
		},
		{
			name: "addEndpoints then updatedEndpoints",
			scenario: func(fp *FakeProxier) {
				e := createEndpoints("ns", "ep1", t0)
				fp.addEndpoints(e)

				e1 := modifyEndpoints(e, t1)
				fp.updateEndpoints(e, e1)
			},
			expected: map[types.NamespacedName][]time.Time{createName("ns", "ep1"): {t0, t1}},
		},
		{
			name: "Add two endpoints then modify one",
			scenario: func(fp *FakeProxier) {
				e1 := createEndpoints("ns", "ep1", t1)
				fp.addEndpoints(e1)

				e2 := createEndpoints("ns", "ep2", t2)
				fp.addEndpoints(e2)

				e11 := modifyEndpoints(e1, t3)
				fp.updateEndpoints(e1, e11)
			},
			expected: map[types.NamespacedName][]time.Time{createName("ns", "ep1"): {t1, t3}, createName("ns", "ep2"): {t2}},
		},
		{
			name: "Endpoints without annotation set",
			scenario: func(fp *FakeProxier) {
				e := createEndpoints("ns", "ep1", t1)
				delete(e.Annotations, v1.EndpointsLastChangeTriggerTime)
				fp.addEndpoints(e)
			},
			expected: map[types.NamespacedName][]time.Time{},
		},
		{
			name: "addEndpoints then deleteEndpoints",
			scenario: func(fp *FakeProxier) {
				e := createEndpoints("ns", "ep1", t1)
				fp.addEndpoints(e)
				fp.deleteEndpoints(e)
			},
			expected: map[types.NamespacedName][]time.Time{},
		},
		{
			name: "add then delete then add again",
			scenario: func(fp *FakeProxier) {
				e := createEndpoints("ns", "ep1", t1)
				fp.addEndpoints(e)
				fp.deleteEndpoints(e)
				e = modifyEndpoints(e, t2)
				fp.addEndpoints(e)
			},
			expected: map[types.NamespacedName][]time.Time{createName("ns", "ep1"): {t2}},
		},
		{
			name: "delete",
			scenario: func(fp *FakeProxier) {
				e := createEndpoints("ns", "ep1", t1)
				fp.deleteEndpoints(e)
			},
			expected: map[types.NamespacedName][]time.Time{},
		},
	}

	for _, tc := range testCases {
		fp := newFakeProxier(v1.IPv4Protocol)

		tc.scenario(fp)

		result := fp.endpointsMap.Update(fp.endpointsChanges)
		got := result.LastChangeTriggerTimes

		if !reflect.DeepEqual(got, tc.expected) {
			t.Errorf("%s: Invalid LastChangeTriggerTimes, expected: %v, got: %v",
				tc.name, tc.expected, result.LastChangeTriggerTimes)
		}
	}
}

func TestEndpointSliceUpdate(t *testing.T) {
	fqdnSlice := generateEndpointSlice("svc1", "ns1", 2, 5, 999, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)})
	fqdnSlice.AddressType = discovery.AddressTypeFQDN

	testCases := map[string]struct {
		startingSlices        []*discovery.EndpointSlice
		endpointChangeTracker *EndpointChangeTracker
		namespacedName        types.NamespacedName
		paramEndpointSlice    *discovery.EndpointSlice
		paramRemoveSlice      bool
		expectedReturnVal     bool
		expectedCurrentChange map[ServicePortName][]*BaseEndpointInfo
	}{
		// test starting from an empty state
		"add a simple slice that doesn't already exist": {
			startingSlices:        []*discovery.EndpointSlice{},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			paramRemoveSlice:      false,
			expectedReturnVal:     true,
			expectedCurrentChange: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:80"},
				},
				makeServicePortName("ns1", "svc1", "port-1", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:443"},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:443"},
				},
			},
		},
		// test no modification to state - current change should be nil as nothing changes
		"add the same slice that already exists": {
			startingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			paramRemoveSlice:      false,
			expectedReturnVal:     false,
			expectedCurrentChange: nil,
		},
		// ensure that only valide address types are processed
		"add an FQDN slice (invalid address type)": {
			startingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    fqdnSlice,
			paramRemoveSlice:      false,
			expectedReturnVal:     false,
			expectedCurrentChange: nil,
		},
		// test additions to existing state
		"add a slice that overlaps with existing state": {
			startingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
				generateEndpointSlice("svc1", "ns1", 2, 2, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    generateEndpointSlice("svc1", "ns1", 1, 5, 999, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			paramRemoveSlice:      false,
			expectedReturnVal:     true,
			expectedCurrentChange: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.4:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.5:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.2.1:80"},
					&BaseEndpointInfo{Endpoint: "10.0.2.2:80", IsLocal: true},
				},
				makeServicePortName("ns1", "svc1", "port-1", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.4:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.5:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.2.1:443"},
					&BaseEndpointInfo{Endpoint: "10.0.2.2:443", IsLocal: true},
				},
			},
		},
		// test additions to existing state with partially overlapping slices and ports
		"add a slice that overlaps with existing state and partial ports": {
			startingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
				generateEndpointSlice("svc1", "ns1", 2, 2, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    generateEndpointSliceWithOffset("svc1", "ns1", 3, 1, 5, 999, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80)}),
			paramRemoveSlice:      false,
			expectedReturnVal:     true,
			expectedCurrentChange: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.4:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.5:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.2.1:80"},
					&BaseEndpointInfo{Endpoint: "10.0.2.2:80", IsLocal: true},
				},
				makeServicePortName("ns1", "svc1", "port-1", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:443"},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:443"},
					&BaseEndpointInfo{Endpoint: "10.0.2.1:443"},
					&BaseEndpointInfo{Endpoint: "10.0.2.2:443", IsLocal: true},
				},
			},
		},
		// test deletions from existing state with partially overlapping slices and ports
		"remove a slice that overlaps with existing state": {
			startingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
				generateEndpointSlice("svc1", "ns1", 2, 2, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    generateEndpointSlice("svc1", "ns1", 1, 5, 999, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			paramRemoveSlice:      true,
			expectedReturnVal:     true,
			expectedCurrentChange: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.2.1:80"},
					&BaseEndpointInfo{Endpoint: "10.0.2.2:80", IsLocal: true},
				},
				makeServicePortName("ns1", "svc1", "port-1", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.2.1:443"},
					&BaseEndpointInfo{Endpoint: "10.0.2.2:443", IsLocal: true},
				},
			},
		},
		// ensure a removal that has no effect turns into a no-op
		"remove a slice that doesn't even exist in current state": {
			startingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 5, 999, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
				generateEndpointSlice("svc1", "ns1", 2, 2, 999, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    generateEndpointSlice("svc1", "ns1", 3, 5, 999, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			paramRemoveSlice:      true,
			expectedReturnVal:     false,
			expectedCurrentChange: nil,
		},
		// start with all endpoints ready, transition to no endpoints ready
		"transition all endpoints to unready state": {
			startingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    generateEndpointSlice("svc1", "ns1", 1, 3, 1, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			paramRemoveSlice:      false,
			expectedReturnVal:     true,
			expectedCurrentChange: map[ServicePortName][]*BaseEndpointInfo{},
		},
		// start with no endpoints ready, transition to all endpoints ready
		"transition all endpoints to ready state": {
			startingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 2, 1, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    generateEndpointSlice("svc1", "ns1", 1, 2, 999, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			paramRemoveSlice:      false,
			expectedReturnVal:     true,
			expectedCurrentChange: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80", IsLocal: true},
				},
				makeServicePortName("ns1", "svc1", "port-1", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:443", IsLocal: true},
				},
			},
		},
		// start with some endpoints ready, transition to more endpoints ready
		"transition some endpoints to ready state": {
			startingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 2, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
				generateEndpointSlice("svc1", "ns1", 2, 2, 2, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			endpointChangeTracker: NewEndpointChangeTracker("host1", nil, v1.IPv4Protocol, nil, true, nil),
			namespacedName:        types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			paramEndpointSlice:    generateEndpointSlice("svc1", "ns1", 1, 3, 3, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			paramRemoveSlice:      false,
			expectedReturnVal:     true,
			expectedCurrentChange: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.2.1:80", IsLocal: true},
				},
				makeServicePortName("ns1", "svc1", "port-1", v1.ProtocolTCP): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.2.1:443", IsLocal: true},
				},
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			initializeCache(tc.endpointChangeTracker.endpointSliceCache, tc.startingSlices)

			got := tc.endpointChangeTracker.EndpointSliceUpdate(tc.paramEndpointSlice, tc.paramRemoveSlice)
			if !reflect.DeepEqual(got, tc.expectedReturnVal) {
				t.Errorf("EndpointSliceUpdate return value got: %v, want %v", got, tc.expectedReturnVal)
			}
			if tc.endpointChangeTracker.items == nil {
				t.Errorf("Expected ect.items to not be nil")
			}
			changes := tc.endpointChangeTracker.checkoutChanges()
			if tc.expectedCurrentChange == nil {
				if len(changes) != 0 {
					t.Errorf("Expected %s to have no changes", tc.namespacedName)
				}
			} else {
				if len(changes) == 0 || changes[0] == nil {
					t.Fatalf("Expected %s to have changes", tc.namespacedName)
				}
				compareEndpointsMapsStr(t, changes[0].current, tc.expectedCurrentChange)
			}
		})
	}
}

func TestCheckoutChanges(t *testing.T) {
	svcPortName0 := ServicePortName{types.NamespacedName{Namespace: "ns1", Name: "svc1"}, "port-0", v1.ProtocolTCP}
	svcPortName1 := ServicePortName{types.NamespacedName{Namespace: "ns1", Name: "svc1"}, "port-1", v1.ProtocolTCP}

	testCases := map[string]struct {
		endpointChangeTracker *EndpointChangeTracker
		expectedChanges       []*endpointsChange
		useEndpointSlices     bool
		items                 map[types.NamespacedName]*endpointsChange
		appliedSlices         []*discovery.EndpointSlice
		pendingSlices         []*discovery.EndpointSlice
	}{
		"empty slices": {
			endpointChangeTracker: NewEndpointChangeTracker("", nil, v1.IPv4Protocol, nil, true, nil),
			expectedChanges:       []*endpointsChange{},
			useEndpointSlices:     true,
			appliedSlices:         []*discovery.EndpointSlice{},
			pendingSlices:         []*discovery.EndpointSlice{},
		},
		"without slices, empty items": {
			endpointChangeTracker: NewEndpointChangeTracker("", nil, v1.IPv4Protocol, nil, false, nil),
			expectedChanges:       []*endpointsChange{},
			items:                 map[types.NamespacedName]*endpointsChange{},
			useEndpointSlices:     false,
		},
		"without slices, simple items": {
			endpointChangeTracker: NewEndpointChangeTracker("", nil, v1.IPv4Protocol, nil, false, nil),
			expectedChanges: []*endpointsChange{{
				previous: EndpointsMap{
					svcPortName0: []Endpoint{newTestEp("10.0.1.1:80", ""), newTestEp("10.0.1.2:80", "")},
					svcPortName1: []Endpoint{newTestEp("10.0.1.1:443", ""), newTestEp("10.0.1.2:443", "")},
				},
				current: EndpointsMap{
					svcPortName0: []Endpoint{newTestEp("10.0.1.1:80", ""), newTestEp("10.0.1.2:80", "")},
				},
			}},
			items: map[types.NamespacedName]*endpointsChange{
				{Namespace: "ns1", Name: "svc1"}: {
					previous: EndpointsMap{
						svcPortName0: []Endpoint{newTestEp("10.0.1.1:80", ""), newTestEp("10.0.1.2:80", "")},
						svcPortName1: []Endpoint{newTestEp("10.0.1.1:443", ""), newTestEp("10.0.1.2:443", "")},
					},
					current: EndpointsMap{
						svcPortName0: []Endpoint{newTestEp("10.0.1.1:80", ""), newTestEp("10.0.1.2:80", "")},
					},
				},
			},
			useEndpointSlices: false,
		},
		"adding initial slice": {
			endpointChangeTracker: NewEndpointChangeTracker("", nil, v1.IPv4Protocol, nil, true, nil),
			expectedChanges: []*endpointsChange{{
				previous: EndpointsMap{},
				current: EndpointsMap{
					svcPortName0: []Endpoint{newTestEp("10.0.1.1:80", "host1"), newTestEp("10.0.1.2:80", "host1")},
				},
			}},
			useEndpointSlices: true,
			appliedSlices:     []*discovery.EndpointSlice{},
			pendingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 3, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80)}),
			},
		},
		"removing port in update": {
			endpointChangeTracker: NewEndpointChangeTracker("", nil, v1.IPv4Protocol, nil, true, nil),
			expectedChanges: []*endpointsChange{{
				previous: EndpointsMap{
					svcPortName0: []Endpoint{newTestEp("10.0.1.1:80", "host1"), newTestEp("10.0.1.2:80", "host1")},
					svcPortName1: []Endpoint{newTestEp("10.0.1.1:443", "host1"), newTestEp("10.0.1.2:443", "host1")},
				},
				current: EndpointsMap{
					svcPortName0: []Endpoint{newTestEp("10.0.1.1:80", "host1"), newTestEp("10.0.1.2:80", "host1")},
				},
			}},
			useEndpointSlices: true,
			appliedSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 3, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			pendingSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 3, []string{"host1"}, []*int32{utilpointer.Int32Ptr(80)}),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			if tc.useEndpointSlices {
				for _, slice := range tc.appliedSlices {
					tc.endpointChangeTracker.EndpointSliceUpdate(slice, false)
				}
				tc.endpointChangeTracker.checkoutChanges()
				for _, slice := range tc.pendingSlices {
					tc.endpointChangeTracker.EndpointSliceUpdate(slice, false)
				}
			} else {
				tc.endpointChangeTracker.items = tc.items
			}

			changes := tc.endpointChangeTracker.checkoutChanges()

			if len(tc.expectedChanges) != len(changes) {
				t.Fatalf("Expected %d changes, got %d", len(tc.expectedChanges), len(changes))
			}

			for i, change := range changes {
				expectedChange := tc.expectedChanges[i]

				if !reflect.DeepEqual(change.previous, expectedChange.previous) {
					t.Errorf("[%d] Expected change.previous: %+v, got: %+v", i, expectedChange.previous, change.previous)
				}

				if !reflect.DeepEqual(change.current, expectedChange.current) {
					t.Errorf("[%d] Expected change.current: %+v, got: %+v", i, expectedChange.current, change.current)
				}
			}
		})
	}
}

// Test helpers

func compareEndpointsMapsStr(t *testing.T, newMap EndpointsMap, expected map[ServicePortName][]*BaseEndpointInfo) {
	t.Helper()
	if len(newMap) != len(expected) {
		t.Fatalf("expected %d results, got %d: %v", len(expected), len(newMap), newMap)
	}
	endpointEqual := func(a, b *BaseEndpointInfo) bool {
		return a.Endpoint == b.Endpoint && a.IsLocal == b.IsLocal
	}
	for x := range expected {
		if len(newMap[x]) != len(expected[x]) {
			t.Logf("Endpoints %+v", newMap[x])
			t.Fatalf("expected %d endpoints for %v, got %d", len(expected[x]), x, len(newMap[x]))
		} else {
			for i := range expected[x] {
				newEp, ok := newMap[x][i].(*BaseEndpointInfo)
				if !ok {
					t.Fatalf("Failed to cast endpointsInfo")
					continue
				}
				if !endpointEqual(newEp, expected[x][i]) {
					t.Fatalf("expected new[%v][%d] to be %v, got %v (IsLocal expected %v, got %v)", x, i, expected[x][i], newEp, expected[x][i].IsLocal, newEp.IsLocal)
				}
			}
		}
	}
}

func newTestEp(ep, host string) *BaseEndpointInfo {
	endpointInfo := &BaseEndpointInfo{Endpoint: ep}
	if host != "" {
		endpointInfo.Topology = map[string]string{
			"kubernetes.io/hostname": host,
		}
	}
	return endpointInfo
}

func initializeCache(endpointSliceCache *EndpointSliceCache, endpointSlices []*discovery.EndpointSlice) {
	for _, endpointSlice := range endpointSlices {
		endpointSliceCache.updatePending(endpointSlice, false)
	}

	for _, tracker := range endpointSliceCache.trackerByServiceMap {
		tracker.applied = tracker.pending
		tracker.pending = endpointSliceInfoByName{}
	}
}
