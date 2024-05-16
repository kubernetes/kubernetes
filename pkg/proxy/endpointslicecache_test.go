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

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
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
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, 999, []string{"host1", "host2"}, []*int32{ptr.To[int32](80), ptr.To[int32](443)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{ip: "10.0.1.1", port: 80, endpoint: "10.0.1.1:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 80, endpoint: "10.0.1.2:80", isLocal: true, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.3", port: 80, endpoint: "10.0.1.3:80", isLocal: false, ready: true, serving: true, terminating: false},
				},
				makeServicePortName("ns1", "svc1", "port-1", v1.ProtocolTCP): {
					&BaseEndpointInfo{ip: "10.0.1.1", port: 443, endpoint: "10.0.1.1:443", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 443, endpoint: "10.0.1.2:443", isLocal: true, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.3", port: 443, endpoint: "10.0.1.3:443", isLocal: false, ready: true, serving: true, terminating: false},
				},
			},
		},
		"2 slices, same port": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, 999, []string{}, []*int32{ptr.To[int32](80)}),
				generateEndpointSlice("svc1", "ns1", 2, 3, 999, 999, []string{}, []*int32{ptr.To[int32](80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{ip: "10.0.1.1", port: 80, endpoint: "10.0.1.1:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 80, endpoint: "10.0.1.2:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.3", port: 80, endpoint: "10.0.1.3:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.2.1", port: 80, endpoint: "10.0.2.1:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.2.2", port: 80, endpoint: "10.0.2.2:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.2.3", port: 80, endpoint: "10.0.2.3:80", isLocal: false, ready: true, serving: true, terminating: false},
				},
			},
		},
		// 2 slices, with some overlapping endpoints, result should be a union
		// of the 2.
		"2 overlapping slices, same port": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, 999, []string{}, []*int32{ptr.To[int32](80)}),
				generateEndpointSlice("svc1", "ns1", 1, 4, 999, 999, []string{}, []*int32{ptr.To[int32](80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{ip: "10.0.1.1", port: 80, endpoint: "10.0.1.1:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 80, endpoint: "10.0.1.2:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.3", port: 80, endpoint: "10.0.1.3:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.4", port: 80, endpoint: "10.0.1.4:80", isLocal: false, ready: true, serving: true, terminating: false},
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
				generateEndpointSlice("svc1", "ns1", 1, 10, 3, 999, []string{}, []*int32{ptr.To[int32](80)}),
				generateEndpointSlice("svc1", "ns1", 1, 10, 6, 999, []string{}, []*int32{ptr.To[int32](80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{ip: "10.0.1.10", port: 80, endpoint: "10.0.1.10:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.1", port: 80, endpoint: "10.0.1.1:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 80, endpoint: "10.0.1.2:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.3", port: 80, endpoint: "10.0.1.3:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.4", port: 80, endpoint: "10.0.1.4:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.5", port: 80, endpoint: "10.0.1.5:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.6", port: 80, endpoint: "10.0.1.6:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.7", port: 80, endpoint: "10.0.1.7:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.8", port: 80, endpoint: "10.0.1.8:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.9", port: 80, endpoint: "10.0.1.9:80", isLocal: false, ready: true, serving: true, terminating: false},
				},
			},
		},
		// 2 slices with all endpoints overlapping, some unready and terminating.
		"2 slices, overlapping endpoints, some endpoints unready and some endpoints terminating": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 10, 3, 5, []string{}, []*int32{ptr.To[int32](80)}),
				generateEndpointSlice("svc1", "ns1", 1, 10, 6, 5, []string{}, []*int32{ptr.To[int32](80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{ip: "10.0.1.10", port: 80, endpoint: "10.0.1.10:80", isLocal: false, ready: false, serving: true, terminating: true},
					&BaseEndpointInfo{ip: "10.0.1.1", port: 80, endpoint: "10.0.1.1:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 80, endpoint: "10.0.1.2:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.3", port: 80, endpoint: "10.0.1.3:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.4", port: 80, endpoint: "10.0.1.4:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.5", port: 80, endpoint: "10.0.1.5:80", isLocal: false, ready: false, serving: true, terminating: true},
					&BaseEndpointInfo{ip: "10.0.1.6", port: 80, endpoint: "10.0.1.6:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.7", port: 80, endpoint: "10.0.1.7:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.8", port: 80, endpoint: "10.0.1.8:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.9", port: 80, endpoint: "10.0.1.9:80", isLocal: false, ready: true, serving: true, terminating: false},
				},
			},
		},
		"2 slices, overlapping endpoints, all unready": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 10, 1, 999, []string{}, []*int32{ptr.To[int32](80)}),
				generateEndpointSlice("svc1", "ns1", 1, 10, 1, 999, []string{}, []*int32{ptr.To[int32](80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{ip: "10.0.1.10", port: 80, endpoint: "10.0.1.10:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.1", port: 80, endpoint: "10.0.1.1:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 80, endpoint: "10.0.1.2:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.3", port: 80, endpoint: "10.0.1.3:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.4", port: 80, endpoint: "10.0.1.4:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.5", port: 80, endpoint: "10.0.1.5:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.6", port: 80, endpoint: "10.0.1.6:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.7", port: 80, endpoint: "10.0.1.7:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.8", port: 80, endpoint: "10.0.1.8:80", isLocal: false, ready: false, serving: false, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.9", port: 80, endpoint: "10.0.1.9:80", isLocal: false, ready: false, serving: false, terminating: false},
				},
			},
		},
		"3 slices with different services and namespaces": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, 999, []string{}, []*int32{ptr.To[int32](80)}),
				generateEndpointSlice("svc2", "ns1", 2, 3, 999, 999, []string{}, []*int32{ptr.To[int32](80)}),
				generateEndpointSlice("svc1", "ns2", 3, 3, 999, 999, []string{}, []*int32{ptr.To[int32](80)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{ip: "10.0.1.1", port: 80, endpoint: "10.0.1.1:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 80, endpoint: "10.0.1.2:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.3", port: 80, endpoint: "10.0.1.3:80", isLocal: false, ready: true, serving: true, terminating: false},
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
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, 999, []string{"host1", "host2"}, []*int32{nil}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{},
		},
		// Make sure that different endpoints with duplicate IPs are returned correctly.
		"Different endpoints with duplicate IPs should not be filtered": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			hostname:       "host1",
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSliceWithOffset("svc1", "ns1", 1, 1, 2, 999, 999, []string{"host1", "host2"}, []*int32{ptr.To[int32](80)}),
				generateEndpointSliceWithOffset("svc1", "ns1", 2, 1, 2, 999, 999, []string{"host1", "host2"}, []*int32{ptr.To[int32](8080)}),
			},
			expectedMap: map[ServicePortName][]*BaseEndpointInfo{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					&BaseEndpointInfo{ip: "10.0.1.1", port: 80, endpoint: "10.0.1.1:80", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.1", port: 8080, endpoint: "10.0.1.1:8080", isLocal: false, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 80, endpoint: "10.0.1.2:80", isLocal: true, ready: true, serving: true, terminating: false},
					&BaseEndpointInfo{ip: "10.0.1.2", port: 8080, endpoint: "10.0.1.2:8080", isLocal: true, ready: true, serving: true, terminating: false},
				},
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			esCache := NewEndpointSliceCache(tc.hostname, v1.IPv4Protocol, nil, nil)

			cmc := newCacheMutationCheck(tc.endpointSlices)
			for _, endpointSlice := range tc.endpointSlices {
				esCache.updatePending(endpointSlice, false)
			}

			compareEndpointsMapsStr(t, esCache.getEndpointsMap(tc.namespacedName, esCache.trackerByServiceMap[tc.namespacedName].pending), tc.expectedMap)
			cmc.Check(t)
		})
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
				generateEndpointSlice("svc1", "ns1", 1, 3, 999, 999, []string{"host1", "host2"}, []*int32{ptr.To[int32](80)}),
			},
			expectedMap: spToEndpointMap{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					"10.0.1.1:80": &BaseEndpointInfo{
						ip:          "10.0.1.1",
						port:        80,
						endpoint:    "10.0.1.1:80",
						isLocal:     false,
						ready:       true,
						serving:     true,
						terminating: false,
					},
					"10.0.1.2:80": &BaseEndpointInfo{
						ip:          "10.0.1.2",
						port:        80,
						endpoint:    "10.0.1.2:80",
						isLocal:     true,
						ready:       true,
						serving:     true,
						terminating: false,
					},
					"10.0.1.3:80": &BaseEndpointInfo{
						ip:          "10.0.1.3",
						port:        80,
						endpoint:    "10.0.1.3:80",
						isLocal:     false,
						ready:       true,
						serving:     true,
						terminating: false,
					},
				},
			},
		},
		"4 different slices with duplicate IPs": {
			namespacedName: types.NamespacedName{Name: "svc1", Namespace: "ns1"},
			hostname:       "host1",
			endpointSlices: []*discovery.EndpointSlice{
				generateEndpointSliceWithOffset("svc1", "ns1", 1, 1, 2, 999, 999, []string{"host1", "host2"}, []*int32{ptr.To[int32](80)}),
				generateEndpointSliceWithOffset("svc1", "ns1", 2, 1, 2, 999, 999, []string{"host1", "host2"}, []*int32{ptr.To[int32](8080)}),
			},
			expectedMap: spToEndpointMap{
				makeServicePortName("ns1", "svc1", "port-0", v1.ProtocolTCP): {
					"10.0.1.1:80": &BaseEndpointInfo{
						ip:          "10.0.1.1",
						port:        80,
						endpoint:    "10.0.1.1:80",
						isLocal:     false,
						ready:       true,
						serving:     true,
						terminating: false,
					},
					"10.0.1.2:80": &BaseEndpointInfo{
						ip:          "10.0.1.2",
						port:        80,
						endpoint:    "10.0.1.2:80",
						isLocal:     true,
						ready:       true,
						serving:     true,
						terminating: false,
					},
					"10.0.1.1:8080": &BaseEndpointInfo{
						ip:          "10.0.1.1",
						port:        8080,
						endpoint:    "10.0.1.1:8080",
						isLocal:     false,
						ready:       true,
						serving:     true,
						terminating: false,
					},
					"10.0.1.2:8080": &BaseEndpointInfo{
						ip:          "10.0.1.2",
						port:        8080,
						endpoint:    "10.0.1.2:8080",
						isLocal:     true,
						ready:       true,
						serving:     true,
						terminating: false,
					},
				},
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			esCache := NewEndpointSliceCache(tc.hostname, v1.IPv4Protocol, nil, nil)

			for _, endpointSlice := range tc.endpointSlices {
				esCache.updatePending(endpointSlice, false)
			}

			got := esCache.endpointInfoByServicePort(tc.namespacedName, esCache.trackerByServiceMap[tc.namespacedName].pending)
			if !reflect.DeepEqual(got, tc.expectedMap) {
				t.Errorf("endpointInfoByServicePort does not match. Want: %+v, Got: %+v", tc.expectedMap, got)
			}
		})
	}
}

func TestEsDataChanged(t *testing.T) {
	p80 := int32(80)
	p443 := int32(443)
	port80 := discovery.EndpointPort{Port: &p80, Name: ptr.To("http"), Protocol: ptr.To(v1.ProtocolTCP)}
	port443 := discovery.EndpointPort{Port: &p443, Name: ptr.To("https"), Protocol: ptr.To(v1.ProtocolTCP)}
	endpoint1 := discovery.Endpoint{Addresses: []string{"10.0.1.0"}}
	endpoint2 := discovery.Endpoint{Addresses: []string{"10.0.1.1"}}

	objMeta := metav1.ObjectMeta{
		Name:      "foo",
		Namespace: "bar",
		Labels:    map[string]string{discovery.LabelServiceName: "svc1"},
	}

	testCases := map[string]struct {
		cache         *EndpointSliceCache
		initialSlice  *discovery.EndpointSlice
		updatedSlice  *discovery.EndpointSlice
		expectChanged bool
	}{
		"identical slices, ports only": {
			cache: NewEndpointSliceCache("", v1.IPv4Protocol, nil, nil),
			initialSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port80},
			},
			updatedSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port80},
			},
			expectChanged: false,
		},
		"identical slices, ports out of order": {
			cache: NewEndpointSliceCache("", v1.IPv4Protocol, nil, nil),
			initialSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443, port80},
			},
			updatedSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port80, port443},
			},
			expectChanged: true,
		},
		"port removed": {
			cache: NewEndpointSliceCache("", v1.IPv4Protocol, nil, nil),
			initialSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443, port80},
			},
			updatedSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443},
			},
			expectChanged: true,
		},
		"port added": {
			cache: NewEndpointSliceCache("", v1.IPv4Protocol, nil, nil),
			initialSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443},
			},
			updatedSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443, port80},
			},
			expectChanged: true,
		},
		"identical with endpoints": {
			cache: NewEndpointSliceCache("", v1.IPv4Protocol, nil, nil),
			initialSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443},
				Endpoints:  []discovery.Endpoint{endpoint1, endpoint2},
			},
			updatedSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443},
				Endpoints:  []discovery.Endpoint{endpoint1, endpoint2},
			},
			expectChanged: false,
		},
		"identical with endpoints out of order": {
			cache: NewEndpointSliceCache("", v1.IPv4Protocol, nil, nil),
			initialSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443},
				Endpoints:  []discovery.Endpoint{endpoint1, endpoint2},
			},
			updatedSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443},
				Endpoints:  []discovery.Endpoint{endpoint2, endpoint1},
			},
			expectChanged: true,
		},
		"identical with endpoint added": {
			cache: NewEndpointSliceCache("", v1.IPv4Protocol, nil, nil),
			initialSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443},
				Endpoints:  []discovery.Endpoint{endpoint1},
			},
			updatedSlice: &discovery.EndpointSlice{
				ObjectMeta: objMeta,
				Ports:      []discovery.EndpointPort{port443},
				Endpoints:  []discovery.Endpoint{endpoint2, endpoint1},
			},
			expectChanged: true,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			cmc := newCacheMutationCheck([]*discovery.EndpointSlice{tc.initialSlice})

			if tc.initialSlice != nil {
				tc.cache.updatePending(tc.initialSlice, false)
				tc.cache.checkoutChanges()
			}

			serviceKey, sliceKey, err := endpointSliceCacheKeys(tc.updatedSlice)
			if err != nil {
				t.Fatalf("Expected no error calling endpointSliceCacheKeys(): %v", err)
			}

			esData := &endpointSliceData{tc.updatedSlice, false}
			changed := tc.cache.esDataChanged(serviceKey, sliceKey, esData)

			if tc.expectChanged != changed {
				t.Errorf("Expected esDataChanged() to return %t, got %t", tc.expectChanged, changed)
			}

			cmc.Check(t)
		})
	}
}

func generateEndpointSliceWithOffset(serviceName, namespace string, sliceNum, offset, numEndpoints, unreadyMod int, terminatingMod int, hosts []string, portNums []*int32) *discovery.EndpointSlice {
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d", serviceName, sliceNum),
			Namespace: namespace,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports:       []discovery.EndpointPort{},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints:   []discovery.Endpoint{},
	}

	for i, portNum := range portNums {
		endpointSlice.Ports = append(endpointSlice.Ports, discovery.EndpointPort{
			Name:     ptr.To(fmt.Sprintf("port-%d", i)),
			Port:     portNum,
			Protocol: ptr.To(v1.ProtocolTCP),
		})
	}

	for i := 1; i <= numEndpoints; i++ {
		readyCondition := i%unreadyMod != 0
		terminatingCondition := i%terminatingMod == 0

		ready := ptr.To(readyCondition && !terminatingCondition)
		serving := ptr.To(readyCondition)
		terminating := ptr.To(terminatingCondition)

		endpoint := discovery.Endpoint{
			Addresses: []string{fmt.Sprintf("10.0.%d.%d", offset, i)},
			Conditions: discovery.EndpointConditions{
				Ready:       ready,
				Serving:     serving,
				Terminating: terminating,
			},
		}

		if len(hosts) > 0 {
			hostname := hosts[i%len(hosts)]
			endpoint.NodeName = &hostname
		}

		endpointSlice.Endpoints = append(endpointSlice.Endpoints, endpoint)
	}

	return endpointSlice
}

func generateEndpointSlice(serviceName, namespace string, sliceNum, numEndpoints, unreadyMod int, terminatingMod int, hosts []string, portNums []*int32) *discovery.EndpointSlice {
	return generateEndpointSliceWithOffset(serviceName, namespace, sliceNum, sliceNum, numEndpoints, unreadyMod, terminatingMod, hosts, portNums)
}

// cacheMutationCheck helps ensure that cached objects have not been changed
// in any way throughout a test run.
type cacheMutationCheck struct {
	objects []cacheObject
}

// cacheObject stores a reference to an original object as well as a deep copy
// of that object to track any mutations in the original object.
type cacheObject struct {
	original runtime.Object
	deepCopy runtime.Object
}

// newCacheMutationCheck initializes a cacheMutationCheck with EndpointSlices.
func newCacheMutationCheck(endpointSlices []*discovery.EndpointSlice) cacheMutationCheck {
	cmc := cacheMutationCheck{}
	for _, endpointSlice := range endpointSlices {
		cmc.Add(endpointSlice)
	}
	return cmc
}

// Add appends a runtime.Object and a deep copy of that object into the
// cacheMutationCheck.
func (cmc *cacheMutationCheck) Add(o runtime.Object) {
	cmc.objects = append(cmc.objects, cacheObject{
		original: o,
		deepCopy: o.DeepCopyObject(),
	})
}

// Check verifies that no objects in the cacheMutationCheck have been mutated.
func (cmc *cacheMutationCheck) Check(t *testing.T) {
	for _, o := range cmc.objects {
		if !reflect.DeepEqual(o.original, o.deepCopy) {
			// Cached objects can't be safely mutated and instead should be deep
			// copied before changed in any way.
			t.Errorf("Cached object was unexpectedly mutated. Original: %+v, Mutated: %+v", o.deepCopy, o.original)
		}
	}
}
