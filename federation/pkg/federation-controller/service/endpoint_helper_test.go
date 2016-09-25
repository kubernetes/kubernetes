/*
Copyright 2016 The Kubernetes Authors.

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

package service

import (
	"testing"

	"k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns" // Only for unit testing purposes.
	v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/sets"
)

var fakeDns, _ = clouddns.NewFakeInterface() // No need to check for unsupported interfaces, as the fake interface supports everything that's required.
var fakeDnsZones, _ = fakeDns.Zones()

var fakeServiceController = ServiceController{
	dns:            fakeDns,
	dnsZones:       fakeDnsZones,
	federationName: "fed1",
	zoneName:       "example.com",
	serviceCache:   &serviceCache{fedServiceMap: make(map[string]*cachedService)},
	clusterCache: &clusterClientCache{
		clientMap: make(map[string]*clusterCache),
	},
	knownClusterSet: make(sets.String),
}

func buildEndpoint(subsets [][]string) *v1.Endpoints {
	endpoint := &v1.Endpoints{
		Subsets: []v1.EndpointSubset{
			{Addresses: []v1.EndpointAddress{}},
		},
	}
	for _, element := range subsets {
		address := v1.EndpointAddress{IP: element[0], Hostname: element[1], TargetRef: nil}
		endpoint.Subsets[0].Addresses = append(endpoint.Subsets[0].Addresses, address)
	}
	return endpoint
}

func TestProcessEndpointUpdate(t *testing.T) {
	clusterName := "foo"
	cc := clusterClientCache{
		clientMap: map[string]*clusterCache{
			clusterName: {
				cluster: &v1beta1.Cluster{
					Status: v1beta1.ClusterStatus{
						Zones:  []string{"foozone"},
						Region: "fooregion",
					},
				},
			},
		},
	}
	tests := []struct {
		name          string
		cachedService *cachedService
		endpoint      *v1.Endpoints
		clusterName   string
		expectResult  int
	}{
		{
			"no-cache",
			&cachedService{
				lastState:   &v1.Service{},
				endpointMap: make(map[string]int),
			},
			buildEndpoint([][]string{{"ip1", ""}}),
			clusterName,
			1,
		},
		{
			"has-cache",
			&cachedService{
				lastState: &v1.Service{},
				endpointMap: map[string]int{
					"foo": 1,
				},
			},
			buildEndpoint([][]string{{"ip1", ""}}),
			clusterName,
			1,
		},
	}
	fakeServiceController.clusterCache = &cc
	for _, test := range tests {
		cc.processEndpointUpdate(test.cachedService, test.endpoint, test.clusterName, &fakeServiceController)
		if test.expectResult != test.cachedService.endpointMap[test.clusterName] {
			t.Errorf("Test failed for %s, expected %v, saw %v", test.name, test.expectResult, test.cachedService.endpointMap[test.clusterName])
		}
	}
}

func TestProcessEndpointDeletion(t *testing.T) {
	clusterName := "foo"
	cc := clusterClientCache{
		clientMap: map[string]*clusterCache{
			clusterName: {
				cluster: &v1beta1.Cluster{
					Status: v1beta1.ClusterStatus{
						Zones:  []string{"foozone"},
						Region: "fooregion",
					},
				},
			},
		},
	}
	tests := []struct {
		name          string
		cachedService *cachedService
		endpoint      *v1.Endpoints
		clusterName   string
		expectResult  int
	}{
		{
			"no-cache",
			&cachedService{
				lastState:   &v1.Service{},
				endpointMap: make(map[string]int),
			},
			buildEndpoint([][]string{{"ip1", ""}}),
			clusterName,
			0,
		},
		{
			"has-cache",
			&cachedService{
				lastState: &v1.Service{},
				endpointMap: map[string]int{
					clusterName: 1,
				},
			},
			buildEndpoint([][]string{{"ip1", ""}}),
			clusterName,
			0,
		},
	}
	fakeServiceController.clusterCache = &cc
	for _, test := range tests {
		cc.processEndpointDeletion(test.cachedService, test.clusterName, &fakeServiceController)
		if test.expectResult != test.cachedService.endpointMap[test.clusterName] {
			t.Errorf("Test failed for %s, expected %v, saw %v", test.name, test.expectResult, test.cachedService.endpointMap[test.clusterName])
		}
	}
}
