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
	"net"
	"reflect"
	"strconv"
	"testing"

	"github.com/davecgh/go-spew/spew"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
)

type fakeEndpointsInfo struct {
	endpoint string
	isLocal  bool
}

func newFakeEndpointsInfo(IP string, port int, isLocal bool) Endpoint {
	return &fakeEndpointsInfo{
		endpoint: net.JoinHostPort(IP, strconv.Itoa(port)),
		isLocal:  isLocal,
	}
}

func (f *fakeEndpointsInfo) String() string {
	return f.endpoint
}

func (f *fakeEndpointsInfo) IsLocal() bool {
	return f.isLocal
}

func (f *fakeEndpointsInfo) IP() string {
	// Must be IP:port
	host, _, _ := net.SplitHostPort(f.endpoint)
	return host
}

func (f *fakeEndpointsInfo) Equal(other Endpoint) bool {
	return f.String() == other.String() &&
		f.IsLocal() == other.IsLocal() &&
		f.IP() == other.IP()
}

func (proxier *FakeProxier) addEndpoints(endpoints *api.Endpoints) {
	proxier.endpointsChanges.Update(nil, endpoints, newFakeEndpointsInfo)
}

func (proxier *FakeProxier) updateEndpoints(oldEndpoints, endpoints *api.Endpoints) {
	proxier.endpointsChanges.Update(oldEndpoints, endpoints, newFakeEndpointsInfo)
}

func (proxier *FakeProxier) deleteEndpoints(endpoints *api.Endpoints) {
	proxier.endpointsChanges.Update(endpoints, nil, newFakeEndpointsInfo)
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
			makeServicePortName("ns1", "ep1", ""): []Endpoint{
				&fakeEndpointsInfo{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expected: map[types.NamespacedName]sets.String{},
	}, {
		// Case[2]: unnamed port local
		endpointsMap: EndpointsMap{
			makeServicePortName("ns1", "ep1", ""): []Endpoint{
				&fakeEndpointsInfo{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns1", Name: "ep1"}: sets.NewString("1.1.1.1"),
		},
	}, {
		// Case[3]: named local and non-local ports for the same IP.
		endpointsMap: EndpointsMap{
			makeServicePortName("ns1", "ep1", "p11"): []Endpoint{
				&fakeEndpointsInfo{endpoint: "1.1.1.1:11", isLocal: false},
				&fakeEndpointsInfo{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): []Endpoint{
				&fakeEndpointsInfo{endpoint: "1.1.1.1:12", isLocal: false},
				&fakeEndpointsInfo{endpoint: "1.1.1.2:12", isLocal: true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns1", Name: "ep1"}: sets.NewString("1.1.1.2"),
		},
	}, {
		// Case[4]: named local and non-local ports for different IPs.
		endpointsMap: EndpointsMap{
			makeServicePortName("ns1", "ep1", "p11"): []Endpoint{
				&fakeEndpointsInfo{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns2", "ep2", "p22"): []Endpoint{
				&fakeEndpointsInfo{endpoint: "2.2.2.2:22", isLocal: true},
				&fakeEndpointsInfo{endpoint: "2.2.2.22:22", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p23"): []Endpoint{
				&fakeEndpointsInfo{endpoint: "2.2.2.3:23", isLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p44"): []Endpoint{
				&fakeEndpointsInfo{endpoint: "4.4.4.4:44", isLocal: true},
				&fakeEndpointsInfo{endpoint: "4.4.4.5:44", isLocal: false},
			},
			makeServicePortName("ns4", "ep4", "p45"): []Endpoint{
				&fakeEndpointsInfo{endpoint: "4.4.4.6:45", isLocal: true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns2", Name: "ep2"}: sets.NewString("2.2.2.2", "2.2.2.22", "2.2.2.3"),
			{Namespace: "ns4", Name: "ep4"}: sets.NewString("4.4.4.4", "4.4.4.6"),
		},
	}}

	for tci, tc := range testCases {
		// outputs
		localIPs := GetLocalEndpointIPs(tc.endpointsMap)

		if !reflect.DeepEqual(localIPs, tc.expected) {
			t.Errorf("[%d] expected %#v, got %#v", tci, tc.expected, localIPs)
		}
	}
}

func makeTestEndpoints(namespace, name string, eptFunc func(*api.Endpoints)) *api.Endpoints {
	ept := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
	eptFunc(ept)
	return ept
}

// This is a coarse test, but it offers some modicum of confidence as the code is evolved.
func Test_endpointsToEndpointsMap(t *testing.T) {
	testCases := []struct {
		newEndpoints *api.Endpoints
		expected     map[ServicePortName][]*fakeEndpointsInfo
	}{{
		// Case[0]: nothing
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {}),
		expected:     map[ServicePortName][]*fakeEndpointsInfo{},
	}, {
		// Case[1]: no changes, unnamed port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "",
						Port: 11,
					}},
				},
			}
		}),
		expected: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[2]: no changes, named port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "port",
						Port: 11,
					}},
				},
			}
		}),
		expected: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "port"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[3]: new port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Port: 11,
					}},
				},
			}
		}),
		expected: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[4]: remove port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {}),
		expected:     map[ServicePortName][]*fakeEndpointsInfo{},
	}, {
		// Case[5]: new IP and port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}, {
						IP: "2.2.2.2",
					}},
					Ports: []api.EndpointPort{{
						Name: "p1",
						Port: 11,
					}, {
						Name: "p2",
						Port: 22,
					}},
				},
			}
		}),
		expected: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p1"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "2.2.2.2:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p2"): {
				{endpoint: "1.1.1.1:22", isLocal: false},
				{endpoint: "2.2.2.2:22", isLocal: false},
			},
		},
	}, {
		// Case[6]: remove IP and port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "p1",
						Port: 11,
					}},
				},
			}
		}),
		expected: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p1"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[7]: rename port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "p2",
						Port: 11,
					}},
				},
			}
		}),
		expected: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p2"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[8]: renumber port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "p1",
						Port: 22,
					}},
				},
			}
		}),
		expected: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p1"): {
				{endpoint: "1.1.1.1:22", isLocal: false},
			},
		},
	}}

	for tci, tc := range testCases {
		// outputs
		newEndpoints := endpointsToEndpointsMap(tc.newEndpoints, "host", newFakeEndpointsInfo)

		if len(newEndpoints) != len(tc.expected) {
			t.Errorf("[%d] expected %d new, got %d: %v", tci, len(tc.expected), len(newEndpoints), spew.Sdump(newEndpoints))
		}
		for x := range tc.expected {
			if len(newEndpoints[x]) != len(tc.expected[x]) {
				t.Errorf("[%d] expected %d endpoints for %v, got %d", tci, len(tc.expected[x]), x, len(newEndpoints[x]))
			} else {
				for i := range newEndpoints[x] {
					ep := newEndpoints[x][i].(*fakeEndpointsInfo)
					if *ep != *(tc.expected[x][i]) {
						t.Errorf("[%d] expected new[%v][%d] to be %v, got %v", tci, x, i, tc.expected[x][i], *ep)
					}
				}
			}
		}
	}
}

func TestUpdateEndpointsMap(t *testing.T) {
	var nodeName = testHostname

	emptyEndpoint := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{}
	}
	unnamedPort := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Port: 11,
			}},
		}}
	}
	unnamedPortLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Port: 11,
			}},
		}}
	}
	namedPortLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}}
	}
	namedPort := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}}
	}
	namedPortRenamed := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11-2",
				Port: 11,
			}},
		}}
	}
	namedPortRenumbered := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 22,
			}},
		}}
	}
	namedPortsLocalNoLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}, {
				Name: "p12",
				Port: 12,
			}},
		}}
	}
	multipleSubsets := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.2",
			}},
			Ports: []api.EndpointPort{{
				Name: "p12",
				Port: 12,
			}},
		}}
	}
	multipleSubsetsWithLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p12",
				Port: 12,
			}},
		}}
	}
	multipleSubsetsMultiplePortsLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}, {
				Name: "p12",
				Port: 12,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.3",
			}},
			Ports: []api.EndpointPort{{
				Name: "p13",
				Port: 13,
			}},
		}}
	}
	multipleSubsetsIPsPorts1 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}, {
				Name: "p12",
				Port: 12,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.3",
			}, {
				IP:       "1.1.1.4",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p13",
				Port: 13,
			}, {
				Name: "p14",
				Port: 14,
			}},
		}}
	}
	multipleSubsetsIPsPorts2 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "2.2.2.1",
			}, {
				IP:       "2.2.2.2",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p21",
				Port: 21,
			}, {
				Name: "p22",
				Port: 22,
			}},
		}}
	}
	complexBefore1 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}}
	}
	complexBefore2 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "2.2.2.2",
				NodeName: &nodeName,
			}, {
				IP:       "2.2.2.22",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p22",
				Port: 22,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP:       "2.2.2.3",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p23",
				Port: 23,
			}},
		}}
	}
	complexBefore4 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "4.4.4.4",
				NodeName: &nodeName,
			}, {
				IP:       "4.4.4.5",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p44",
				Port: 44,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP:       "4.4.4.6",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p45",
				Port: 45,
			}},
		}}
	}
	complexAfter1 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP: "1.1.1.11",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.2",
			}},
			Ports: []api.EndpointPort{{
				Name: "p12",
				Port: 12,
			}, {
				Name: "p122",
				Port: 122,
			}},
		}}
	}
	complexAfter3 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "3.3.3.3",
			}},
			Ports: []api.EndpointPort{{
				Name: "p33",
				Port: 33,
			}},
		}}
	}
	complexAfter4 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "4.4.4.4",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p44",
				Port: 44,
			}},
		}}
	}

	testCases := []struct {
		// previousEndpoints and currentEndpoints are used to call appropriate
		// handlers OnEndpoints* (based on whether corresponding values are nil
		// or non-nil) and must be of equal length.
		previousEndpoints         []*api.Endpoints
		currentEndpoints          []*api.Endpoints
		oldEndpoints              map[ServicePortName][]*fakeEndpointsInfo
		expectedResult            map[ServicePortName][]*fakeEndpointsInfo
		expectedStaleEndpoints    []ServiceEndpoint
		expectedStaleServiceNames map[ServicePortName]bool
		expectedHealthchecks      map[types.NamespacedName]int
	}{{
		// Case[0]: nothing
		oldEndpoints:              map[ServicePortName][]*fakeEndpointsInfo{},
		expectedResult:            map[ServicePortName][]*fakeEndpointsInfo{},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[1]: no change, unnamed port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[2]: no change, named port, local
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[3]: no change, multiple subsets
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: false},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[4]: no change, multiple subsets, multiple ports, local
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{endpoint: "1.1.1.3:13", isLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{endpoint: "1.1.1.3:13", isLocal: false},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[5]: no change, multiple endpoints, subsets, IPs, and ports
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: false},
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{endpoint: "1.1.1.3:13", isLocal: false},
				{endpoint: "1.1.1.4:13", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14"): {
				{endpoint: "1.1.1.3:14", isLocal: false},
				{endpoint: "1.1.1.4:14", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21"): {
				{endpoint: "2.2.2.1:21", isLocal: false},
				{endpoint: "2.2.2.2:21", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{endpoint: "2.2.2.1:22", isLocal: false},
				{endpoint: "2.2.2.2:22", isLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: false},
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{endpoint: "1.1.1.3:13", isLocal: false},
				{endpoint: "1.1.1.4:13", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14"): {
				{endpoint: "1.1.1.3:14", isLocal: false},
				{endpoint: "1.1.1.4:14", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21"): {
				{endpoint: "2.2.2.1:21", isLocal: false},
				{endpoint: "2.2.2.2:21", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{endpoint: "2.2.2.1:22", isLocal: false},
				{endpoint: "2.2.2.2:22", isLocal: true},
			},
		},
		expectedStaleEndpoints:    []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 2,
			makeNSN("ns2", "ep2"): 1,
		},
	}, {
		// Case[6]: add an Endpoints
		previousEndpoints: []*api.Endpoints{
			nil,
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPortLocal),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", ""): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[7]: remove an Endpoints
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPortLocal),
		},
		currentEndpoints: []*api.Endpoints{
			nil,
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", ""),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[8]: add an IP and port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortsLocalNoLocal),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: false},
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12"): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[9]: remove an IP and port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortsLocalNoLocal),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: false},
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.2:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11"),
		}, {
			Endpoint:        "1.1.1.1:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12"),
		}, {
			Endpoint:        "1.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12"),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[10]: add a subset
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsWithLocal),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12"): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[11]: remove a subset
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12"),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[12]: rename a port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenamed),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11-2"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11"),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p11-2"): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[13]: renumber a port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenumbered),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:22", isLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11"),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[14]: complex add and remove
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", complexBefore1),
			makeTestEndpoints("ns2", "ep2", complexBefore2),
			nil,
			makeTestEndpoints("ns4", "ep4", complexBefore4),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", complexAfter1),
			nil,
			makeTestEndpoints("ns3", "ep3", complexAfter3),
			makeTestEndpoints("ns4", "ep4", complexAfter4),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{endpoint: "2.2.2.2:22", isLocal: true},
				{endpoint: "2.2.2.22:22", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p23"): {
				{endpoint: "2.2.2.3:23", isLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p44"): {
				{endpoint: "4.4.4.4:44", isLocal: true},
				{endpoint: "4.4.4.5:44", isLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p45"): {
				{endpoint: "4.4.4.6:45", isLocal: true},
			},
		},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.11:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p122"): {
				{endpoint: "1.1.1.2:122", isLocal: false},
			},
			makeServicePortName("ns3", "ep3", "p33"): {
				{endpoint: "3.3.3.3:33", isLocal: false},
			},
			makeServicePortName("ns4", "ep4", "p44"): {
				{endpoint: "4.4.4.4:44", isLocal: true},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{{
			Endpoint:        "2.2.2.2:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22"),
		}, {
			Endpoint:        "2.2.2.22:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22"),
		}, {
			Endpoint:        "2.2.2.3:23",
			ServicePortName: makeServicePortName("ns2", "ep2", "p23"),
		}, {
			Endpoint:        "4.4.4.5:44",
			ServicePortName: makeServicePortName("ns4", "ep4", "p44"),
		}, {
			Endpoint:        "4.4.4.6:45",
			ServicePortName: makeServicePortName("ns4", "ep4", "p45"),
		}},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12"):  true,
			makeServicePortName("ns1", "ep1", "p122"): true,
			makeServicePortName("ns3", "ep3", "p33"):  true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns4", "ep4"): 1,
		},
	}, {
		// Case[15]: change from 0 endpoint address to 1 unnamed port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", emptyEndpoint),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		oldEndpoints: map[ServicePortName][]*fakeEndpointsInfo{},
		expectedResult: map[ServicePortName][]*fakeEndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints: []ServiceEndpoint{},
		expectedStaleServiceNames: map[ServicePortName]bool{
			makeServicePortName("ns1", "ep1", ""): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	},
	}

	for tci, tc := range testCases {
		fp := newFakeProxier()
		fp.hostname = nodeName

		// First check that after adding all previous versions of endpoints,
		// the fp.oldEndpoints is as we expect.
		for i := range tc.previousEndpoints {
			if tc.previousEndpoints[i] != nil {
				fp.addEndpoints(tc.previousEndpoints[i])
			}
		}
		UpdateEndpointsMap(fp.endpointsMap, fp.endpointsChanges)
		compareEndpointsMaps(t, tci, fp.endpointsMap, tc.oldEndpoints)

		// Now let's call appropriate handlers to get to state we want to be.
		if len(tc.previousEndpoints) != len(tc.currentEndpoints) {
			t.Fatalf("[%d] different lengths of previous and current endpoints", tci)
			continue
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
		result := UpdateEndpointsMap(fp.endpointsMap, fp.endpointsChanges)
		newMap := fp.endpointsMap
		compareEndpointsMaps(t, tci, newMap, tc.expectedResult)
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
	}
}

func compareEndpointsMaps(t *testing.T, tci int, newMap EndpointsMap, expected map[ServicePortName][]*fakeEndpointsInfo) {
	if len(newMap) != len(expected) {
		t.Errorf("[%d] expected %d results, got %d: %v", tci, len(expected), len(newMap), newMap)
	}
	for x := range expected {
		if len(newMap[x]) != len(expected[x]) {
			t.Errorf("[%d] expected %d endpoints for %v, got %d", tci, len(expected[x]), x, len(newMap[x]))
		} else {
			for i := range expected[x] {
				newEp, ok := newMap[x][i].(*fakeEndpointsInfo)
				if !ok {
					t.Errorf("Failed to cast endpointsInfo")
					continue
				}
				if *newEp != *(expected[x][i]) {
					t.Errorf("[%d] expected new[%v][%d] to be %v, got %v", tci, x, i, expected[x][i], newEp)
				}
			}
		}
	}
}
