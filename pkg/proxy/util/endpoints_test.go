package util

import (
	"testing"

	"github.com/davecgh/go-spew/spew"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
)

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
		expected     map[proxy.ServicePortName][]*EndpointsInfo
	}{{
		// Case[0]: nothing
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {}),
		expected:     map[proxy.ServicePortName][]*EndpointsInfo{},
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
		expected: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{"1.1.1.1:11", false},
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
		expected: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", "port"): {
				{"1.1.1.1:11", false},
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
		expected: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{"1.1.1.1:11", false},
			},
		},
	}, {
		// Case[4]: remove port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {}),
		expected:     map[proxy.ServicePortName][]*EndpointsInfo{},
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
		expected: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p1"): {
				{"1.1.1.1:11", false},
				{"2.2.2.2:11", false},
			},
			makeServicePortName("ns1", "ep1", "p2"): {
				{"1.1.1.1:22", false},
				{"2.2.2.2:22", false},
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
		expected: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p1"): {
				{"1.1.1.1:11", false},
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
		expected: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p2"): {
				{"1.1.1.1:11", false},
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
		expected: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p1"): {
				{"1.1.1.1:22", false},
			},
		},
	}}

	for tci, tc := range testCases {
		// outputs
		newEndpoints := endpointsToEndpointsMap(tc.newEndpoints, "host")

		if len(newEndpoints) != len(tc.expected) {
			t.Errorf("[%d] expected %d new, got %d: %v", tci, len(tc.expected), len(newEndpoints), spew.Sdump(newEndpoints))
		}
		for x := range tc.expected {
			if len(newEndpoints[x]) != len(tc.expected[x]) {
				t.Errorf("[%d] expected %d endpoints for %v, got %d", tci, len(tc.expected[x]), x, len(newEndpoints[x]))
			} else {
				for i := range newEndpoints[x] {
					if *(newEndpoints[x][i]) != *(tc.expected[x][i]) {
						t.Errorf("[%d] expected new[%v][%d] to be %v, got %v", tci, x, i, tc.expected[x][i], *(newEndpoints[x][i]))
					}
				}
			}
		}
	}
}
