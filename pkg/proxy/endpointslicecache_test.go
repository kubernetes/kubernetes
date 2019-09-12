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

package proxy

import (
	"fmt"
	"reflect"
	"testing"

	discovery "k8s.io/api/discovery/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilpointer "k8s.io/utils/pointer"
)

func TestEndpointsMapFromESC(t *testing.T) {
	testCases := map[string]struct {
		endpointSlices []*discovery.EndpointSlice
		hostname       string
		namespacedName types.NamespacedName
		expectedMap    map[ServicePortName][]*BaseEndpointInfo
	}{
		"1 slice, 2 hosts, ports 80,443": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			hostname:       "host1",
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80), utilpointer.Int32Ptr(443)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0"): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80", IsLocal: false},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:80", IsLocal: false},
				},
				makeServicePortName("ns1", "svc1", "port-1"): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:443", IsLocal: false},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:443", IsLocal: true},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:443", IsLocal: false},
				},
			},
		},
		"2 slices, same port": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
				generateEndpointSlice("svc1", "ns1", 2, 3, 999, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0"): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:80"},
					&BaseEndpointInfo{Endpoint: "10.0.2.1:80"},
					&BaseEndpointInfo{Endpoint: "10.0.2.2:80"},
					&BaseEndpointInfo{Endpoint: "10.0.2.3:80"},
				},
			},
		},
		// 2 slices, with some overlapping endpoints, result should be a union
		// of the 2.
		"2 overlapping slices, same port": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
				generateEndpointSlice("svc1", "ns1", 1, 4, 999, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0"): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.4:80"},
				},
			},
		},
		// 2 slices with all endpoints overlapping, more unready in first than
		// second. If an endpoint is marked ready, we add it to the
		// EndpointsMap, even if conditions.Ready isn't true for another
		// matching endpoint
		"2 slices, overlapping endpoints, some endpoints unready in 1 or both": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 10, 3, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
				generateEndpointSlice("svc1", "ns1", 1, 10, 6, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0"): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.10:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.4:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.5:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.7:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.8:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.9:80"},
				},
			},
		},
		"2 slices, overlapping endpoints, all unready": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 10, 1, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
				generateEndpointSlice("svc1", "ns1", 1, 10, 1, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{},
		},
		"3 slices with different services and namespaces": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
				generateEndpointSlice("svc2", "ns1", 2, 3, 999, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
				generateEndpointSlice("svc1", "ns2", 3, 3, 999, []string{}, []*int32{utilpointer.Int32Ptr(80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0"): {
					&BaseEndpointInfo{Endpoint: "10.0.1.1:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.2:80"},
					&BaseEndpointInfo{Endpoint: "10.0.1.3:80"},
				},
			},
		},
		// Ensuring that nil port value will not break things. This will
		// represent all ports in the future, but that has not been implemented
		// yet.
		"Nil port should not break anything": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			hostname:       "host1",
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{nil}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{},
		},
	}

	for name, tc := range testCases {
		esCache := NewEndpointSliceCache(tc.hostname, nil, nil, nil)

		for _, endpointSlice := range tc.endpointSlices {
			esCache.Update(endpointSlice)
		}

		compareEndpointsMapsStr(t, name, esCache.EndpointsMap(tc.namespacedName), tc.expectedMap)
	}
}

func TestEndpointInfoByServicePort(t *testing.T) {
	testCases := map[string]struct {
		namespacedName types.NamespacedName
		endpointSlices []*discovery.EndpointSlice
		hostname       string
		expectedMap    spToEndpointMap
	}{
		"simple use case with 3 endpoints": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			hostname:       "host1",
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, []string{"host1", "host2"}, []*int32{utilpointer.Int32Ptr(80)}),
			},
			expectedMap: spToEndpointMap{
				{NamespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"}, Port: "port-0"}: {
					"10.0.1.1": &BaseEndpointInfo{Endpoint: "10.0.1.1:80", IsLocal: false},
					"10.0.1.2": &BaseEndpointInfo{Endpoint: "10.0.1.2:80", IsLocal: true},
					"10.0.1.3": &BaseEndpointInfo{Endpoint: "10.0.1.3:80", IsLocal: false},
				},
			},
		},
	}

	for name, tc := range testCases {
		esCache := NewEndpointSliceCache(tc.hostname, nil, nil, nil)

		for _, endpointSlice := range tc.endpointSlices {
			esCache.Update(endpointSlice)
		}

		got := esCache.endpointInfoByServicePort(tc.namespacedName)
		if !reflect.DeepEqual(got, tc.expectedMap) {
			t.Errorf("[%s] endpointInfoByServicePort does not match. Want: %+v, Got: %+v", name, tc.expectedMap, got)
		}

	}
}

func generateEndpointSliceWithOffset(serviceName, namespace string, sliceNum, offset, numEndpoints, unreadyMod int, hosts []string, portNums []*int32) *discovery.EndpointSlice {
	ipAddressType := discovery.AddressTypeIP
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d", serviceName, sliceNum),
			Namespace: namespace,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports:       []discovery.EndpointPort{},
		AddressType: &ipAddressType,
		Endpoints:   []discovery.Endpoint{},
	}

	for i, portNum := range portNums {
		endpointSlice.Ports = append(endpointSlice.Ports, discovery.EndpointPort{
			Name: utilpointer.StringPtr(fmt.Sprintf("port-%d", i)),
			Port: portNum,
		})
	}

	for i := 1; i <= numEndpoints; i++ {
		endpoint := discovery.Endpoint{
			Addresses:  []string{fmt.Sprintf("10.0.%d.%d", offset, i)},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(i%unreadyMod != 0)},
		}

		if len(hosts) > 0 {
			endpoint.Topology = map[string]string{
				"kubernetes.io/hostname": hosts[i%len(hosts)],
			}
		}

		endpointSlice.Endpoints = append(endpointSlice.Endpoints, endpoint)
	}

	return endpointSlice
}

func generateEndpointSlice(serviceName, namespace string, sliceNum, numEndpoints, unreadyMod int, hosts []string, portNums []*int32) *discovery.EndpointSlice {
	return generateEndpointSliceWithOffset(serviceName, namespace, sliceNum, sliceNum, numEndpoints, unreadyMod, hosts, portNums)
}
