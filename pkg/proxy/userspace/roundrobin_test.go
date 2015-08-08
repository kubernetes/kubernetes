/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package userspace

import (
	"net"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/types"
)

func TestValidateWorks(t *testing.T) {
	if isValidEndpoint(&hostPortPair{}) {
		t.Errorf("Didn't fail for empty set")
	}
	if isValidEndpoint(&hostPortPair{host: "foobar"}) {
		t.Errorf("Didn't fail with invalid port")
	}
	if isValidEndpoint(&hostPortPair{host: "foobar", port: -1}) {
		t.Errorf("Didn't fail with a negative port")
	}
	if !isValidEndpoint(&hostPortPair{host: "foobar", port: 8080}) {
		t.Errorf("Failed a valid config.")
	}
}

func TestFilterWorks(t *testing.T) {
	endpoints := []hostPortPair{
		{host: "foobar", port: 1},
		{host: "foobar", port: 2},
		{host: "foobar", port: -1},
		{host: "foobar", port: 3},
		{host: "foobar", port: -2},
	}
	filtered := flattenValidEndpoints(endpoints)

	if len(filtered) != 3 {
		t.Errorf("Failed to filter to the correct size")
	}
	if filtered[0] != "foobar:1" {
		t.Errorf("Index zero is not foobar:1")
	}
	if filtered[1] != "foobar:2" {
		t.Errorf("Index one is not foobar:2")
	}
	if filtered[2] != "foobar:3" {
		t.Errorf("Index two is not foobar:3")
	}
}

func TestLoadBalanceFailsWithNoEndpoints(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	var endpoints []api.Endpoints
	loadBalancer.OnEndpointsUpdate(endpoints)
	service := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, "does-not-exist"}
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil {
		t.Errorf("Didn't fail with non-existent service")
	}
	if len(endpoint) != 0 {
		t.Errorf("Got an endpoint")
	}
}

func expectEndpoint(t *testing.T, loadBalancer *LoadBalancerRR, service proxy.ServicePortName, expected string, netaddr net.Addr) {
	endpoint, err := loadBalancer.NextEndpoint(service, netaddr)
	if err != nil {
		t.Errorf("Didn't find a service for %s, expected %s, failed with: %v", service, expected, err)
	}
	if endpoint != expected {
		t.Errorf("Didn't get expected endpoint for service %s client %v, expected %s, got: %s", service, netaddr, expected, endpoint)
	}
}

func TestLoadBalanceWorksWithSingleEndpoint(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	service := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, "p"}
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "endpoint1"}},
			Ports:     []api.EndpointPort{{Name: "p", Port: 40}},
		}},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)
	expectEndpoint(t, loadBalancer, service, "endpoint1:40", nil)
	expectEndpoint(t, loadBalancer, service, "endpoint1:40", nil)
	expectEndpoint(t, loadBalancer, service, "endpoint1:40", nil)
	expectEndpoint(t, loadBalancer, service, "endpoint1:40", nil)
}

func stringsInSlice(haystack []string, needles ...string) bool {
	for _, needle := range needles {
		found := false
		for i := range haystack {
			if haystack[i] == needle {
				found = true
				break
			}
		}
		if found == false {
			return false
		}
	}
	return true
}

func TestLoadBalanceWorksWithMultipleEndpoints(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	service := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, "p"}
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "endpoint"}},
			Ports:     []api.EndpointPort{{Name: "p", Port: 1}, {Name: "p", Port: 2}, {Name: "p", Port: 3}},
		}},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)

	shuffledEndpoints := loadBalancer.services[service].endpoints
	if !stringsInSlice(shuffledEndpoints, "endpoint:1", "endpoint:2", "endpoint:3") {
		t.Errorf("did not find expected endpoints: %v", shuffledEndpoints)
	}
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], nil)
}

func TestLoadBalanceWorksWithMultipleEndpointsMultiplePorts(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	serviceP := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, "p"}
	serviceQ := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, "q"}
	endpoint, err := loadBalancer.NextEndpoint(serviceP, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: serviceP.Name, Namespace: serviceP.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint1"}, {IP: "endpoint2"}},
				Ports:     []api.EndpointPort{{Name: "p", Port: 1}, {Name: "q", Port: 2}},
			},
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint3"}},
				Ports:     []api.EndpointPort{{Name: "p", Port: 3}, {Name: "q", Port: 4}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)

	shuffledEndpoints := loadBalancer.services[serviceP].endpoints
	if !stringsInSlice(shuffledEndpoints, "endpoint1:1", "endpoint2:1", "endpoint3:3") {
		t.Errorf("did not find expected endpoints: %v", shuffledEndpoints)
	}
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[0], nil)

	shuffledEndpoints = loadBalancer.services[serviceQ].endpoints
	if !stringsInSlice(shuffledEndpoints, "endpoint1:2", "endpoint2:2", "endpoint3:4") {
		t.Errorf("did not find expected endpoints: %v", shuffledEndpoints)
	}
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[0], nil)
}

func TestLoadBalanceWorksWithMultipleEndpointsAndUpdates(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	serviceP := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, "p"}
	serviceQ := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, "q"}
	endpoint, err := loadBalancer.NextEndpoint(serviceP, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: serviceP.Name, Namespace: serviceP.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint1"}},
				Ports:     []api.EndpointPort{{Name: "p", Port: 1}, {Name: "q", Port: 10}},
			},
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint2"}},
				Ports:     []api.EndpointPort{{Name: "p", Port: 2}, {Name: "q", Port: 20}},
			},
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint3"}},
				Ports:     []api.EndpointPort{{Name: "p", Port: 3}, {Name: "q", Port: 30}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)

	shuffledEndpoints := loadBalancer.services[serviceP].endpoints
	if !stringsInSlice(shuffledEndpoints, "endpoint1:1", "endpoint2:2", "endpoint3:3") {
		t.Errorf("did not find expected endpoints: %v", shuffledEndpoints)
	}
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[0], nil)

	shuffledEndpoints = loadBalancer.services[serviceQ].endpoints
	if !stringsInSlice(shuffledEndpoints, "endpoint1:10", "endpoint2:20", "endpoint3:30") {
		t.Errorf("did not find expected endpoints: %v", shuffledEndpoints)
	}
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[0], nil)

	// Then update the configuration with one fewer endpoints, make sure
	// we start in the beginning again
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: serviceP.Name, Namespace: serviceP.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint4"}},
				Ports:     []api.EndpointPort{{Name: "p", Port: 4}, {Name: "q", Port: 40}},
			},
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint5"}},
				Ports:     []api.EndpointPort{{Name: "p", Port: 5}, {Name: "q", Port: 50}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)

	shuffledEndpoints = loadBalancer.services[serviceP].endpoints
	if !stringsInSlice(shuffledEndpoints, "endpoint4:4", "endpoint5:5") {
		t.Errorf("did not find expected endpoints: %v", shuffledEndpoints)
	}
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, serviceP, shuffledEndpoints[1], nil)

	shuffledEndpoints = loadBalancer.services[serviceQ].endpoints
	if !stringsInSlice(shuffledEndpoints, "endpoint4:40", "endpoint5:50") {
		t.Errorf("did not find expected endpoints: %v", shuffledEndpoints)
	}
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, serviceQ, shuffledEndpoints[1], nil)

	// Clear endpoints
	endpoints[0] = api.Endpoints{ObjectMeta: api.ObjectMeta{Name: serviceP.Name, Namespace: serviceP.Namespace}, Subsets: nil}
	loadBalancer.OnEndpointsUpdate(endpoints)

	endpoint, err = loadBalancer.NextEndpoint(serviceP, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
}

func TestLoadBalanceWorksWithServiceRemoval(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	fooServiceP := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, "p"}
	barServiceP := proxy.ServicePortName{types.NamespacedName{"testnamespace", "bar"}, "p"}
	endpoint, err := loadBalancer.NextEndpoint(fooServiceP, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 2)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: fooServiceP.Name, Namespace: fooServiceP.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint1"}, {IP: "endpoint2"}, {IP: "endpoint3"}},
				Ports:     []api.EndpointPort{{Name: "p", Port: 123}},
			},
		},
	}
	endpoints[1] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: barServiceP.Name, Namespace: barServiceP.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint4"}, {IP: "endpoint5"}, {IP: "endpoint6"}},
				Ports:     []api.EndpointPort{{Name: "p", Port: 456}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)
	shuffledFooEndpoints := loadBalancer.services[fooServiceP].endpoints
	expectEndpoint(t, loadBalancer, fooServiceP, shuffledFooEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, fooServiceP, shuffledFooEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, fooServiceP, shuffledFooEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, fooServiceP, shuffledFooEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, fooServiceP, shuffledFooEndpoints[1], nil)

	shuffledBarEndpoints := loadBalancer.services[barServiceP].endpoints
	expectEndpoint(t, loadBalancer, barServiceP, shuffledBarEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, barServiceP, shuffledBarEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, barServiceP, shuffledBarEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, barServiceP, shuffledBarEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, barServiceP, shuffledBarEndpoints[1], nil)

	// Then update the configuration by removing foo
	loadBalancer.OnEndpointsUpdate(endpoints[1:])
	endpoint, err = loadBalancer.NextEndpoint(fooServiceP, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	// but bar is still there, and we continue RR from where we left off.
	expectEndpoint(t, loadBalancer, barServiceP, shuffledBarEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, barServiceP, shuffledBarEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, barServiceP, shuffledBarEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, barServiceP, shuffledBarEndpoints[2], nil)
}

func TestStickyLoadBalanceWorksWithNewServiceCalledFirst(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	service := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, ""}
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	// Call NewService() before OnEndpointsUpdate()
	loadBalancer.NewService(service, api.ServiceAffinityClientIP, 0)
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []api.EndpointSubset{
			{Addresses: []api.EndpointAddress{{IP: "endpoint1"}}, Ports: []api.EndpointPort{{Port: 1}}},
			{Addresses: []api.EndpointAddress{{IP: "endpoint2"}}, Ports: []api.EndpointPort{{Port: 2}}},
			{Addresses: []api.EndpointAddress{{IP: "endpoint3"}}, Ports: []api.EndpointPort{{Port: 3}}},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)

	client1 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	client2 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 2), Port: 0}
	client3 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 3), Port: 0}

	ep1, err := loadBalancer.NextEndpoint(service, client1)
	if err != nil {
		t.Errorf("Didn't find a service for %s: %v", service, err)
	}
	expectEndpoint(t, loadBalancer, service, ep1, client1)
	expectEndpoint(t, loadBalancer, service, ep1, client1)
	expectEndpoint(t, loadBalancer, service, ep1, client1)

	ep2, err := loadBalancer.NextEndpoint(service, client2)
	if err != nil {
		t.Errorf("Didn't find a service for %s: %v", service, err)
	}
	expectEndpoint(t, loadBalancer, service, ep2, client2)
	expectEndpoint(t, loadBalancer, service, ep2, client2)
	expectEndpoint(t, loadBalancer, service, ep2, client2)

	ep3, err := loadBalancer.NextEndpoint(service, client3)
	if err != nil {
		t.Errorf("Didn't find a service for %s: %v", service, err)
	}
	expectEndpoint(t, loadBalancer, service, ep3, client3)
	expectEndpoint(t, loadBalancer, service, ep3, client3)
	expectEndpoint(t, loadBalancer, service, ep3, client3)

	expectEndpoint(t, loadBalancer, service, ep1, client1)
	expectEndpoint(t, loadBalancer, service, ep2, client2)
	expectEndpoint(t, loadBalancer, service, ep3, client3)
	expectEndpoint(t, loadBalancer, service, ep1, client1)
	expectEndpoint(t, loadBalancer, service, ep2, client2)
	expectEndpoint(t, loadBalancer, service, ep3, client3)
}

func TestStickyLoadBalanceWorksWithNewServiceCalledSecond(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	service := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, ""}
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	// Call OnEndpointsUpdate() before NewService()
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []api.EndpointSubset{
			{Addresses: []api.EndpointAddress{{IP: "endpoint1"}}, Ports: []api.EndpointPort{{Port: 1}}},
			{Addresses: []api.EndpointAddress{{IP: "endpoint2"}}, Ports: []api.EndpointPort{{Port: 2}}},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)
	loadBalancer.NewService(service, api.ServiceAffinityClientIP, 0)

	client1 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	client2 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 2), Port: 0}
	client3 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 3), Port: 0}

	ep1, err := loadBalancer.NextEndpoint(service, client1)
	if err != nil {
		t.Errorf("Didn't find a service for %s: %v", service, err)
	}
	expectEndpoint(t, loadBalancer, service, ep1, client1)
	expectEndpoint(t, loadBalancer, service, ep1, client1)
	expectEndpoint(t, loadBalancer, service, ep1, client1)

	ep2, err := loadBalancer.NextEndpoint(service, client2)
	if err != nil {
		t.Errorf("Didn't find a service for %s: %v", service, err)
	}
	expectEndpoint(t, loadBalancer, service, ep2, client2)
	expectEndpoint(t, loadBalancer, service, ep2, client2)
	expectEndpoint(t, loadBalancer, service, ep2, client2)

	ep3, err := loadBalancer.NextEndpoint(service, client3)
	if err != nil {
		t.Errorf("Didn't find a service for %s: %v", service, err)
	}
	expectEndpoint(t, loadBalancer, service, ep3, client3)
	expectEndpoint(t, loadBalancer, service, ep3, client3)
	expectEndpoint(t, loadBalancer, service, ep3, client3)

	expectEndpoint(t, loadBalancer, service, ep1, client1)
	expectEndpoint(t, loadBalancer, service, ep2, client2)
	expectEndpoint(t, loadBalancer, service, ep3, client3)
	expectEndpoint(t, loadBalancer, service, ep1, client1)
	expectEndpoint(t, loadBalancer, service, ep2, client2)
	expectEndpoint(t, loadBalancer, service, ep3, client3)
}

func TestStickyLoadBalanaceWorksWithMultipleEndpointsRemoveOne(t *testing.T) {
	client1 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	client2 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 2), Port: 0}
	client3 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 3), Port: 0}
	client4 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 4), Port: 0}
	client5 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 5), Port: 0}
	client6 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 6), Port: 0}
	loadBalancer := NewLoadBalancerRR()
	service := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, ""}
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	loadBalancer.NewService(service, api.ServiceAffinityClientIP, 0)
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint"}},
				Ports:     []api.EndpointPort{{Port: 1}, {Port: 2}, {Port: 3}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)
	shuffledEndpoints := loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	client1Endpoint := shuffledEndpoints[0]
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	client2Endpoint := shuffledEndpoints[1]
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], client3)
	client3Endpoint := shuffledEndpoints[2]

	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint"}},
				Ports:     []api.EndpointPort{{Port: 1}, {Port: 2}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)
	shuffledEndpoints = loadBalancer.services[service].endpoints
	if client1Endpoint == "endpoint:3" {
		client1Endpoint = shuffledEndpoints[0]
	} else if client2Endpoint == "endpoint:3" {
		client2Endpoint = shuffledEndpoints[0]
	} else if client3Endpoint == "endpoint:3" {
		client3Endpoint = shuffledEndpoints[0]
	}
	expectEndpoint(t, loadBalancer, service, client1Endpoint, client1)
	expectEndpoint(t, loadBalancer, service, client2Endpoint, client2)
	expectEndpoint(t, loadBalancer, service, client3Endpoint, client3)

	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint"}},
				Ports:     []api.EndpointPort{{Port: 1}, {Port: 2}, {Port: 4}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)
	shuffledEndpoints = loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, client1Endpoint, client1)
	expectEndpoint(t, loadBalancer, service, client2Endpoint, client2)
	expectEndpoint(t, loadBalancer, service, client3Endpoint, client3)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client4)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client5)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], client6)
}

func TestStickyLoadBalanceWorksWithMultipleEndpointsAndUpdates(t *testing.T) {
	client1 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	client2 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 2), Port: 0}
	client3 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 3), Port: 0}
	loadBalancer := NewLoadBalancerRR()
	service := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, ""}
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	loadBalancer.NewService(service, api.ServiceAffinityClientIP, 0)
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint"}},
				Ports:     []api.EndpointPort{{Port: 1}, {Port: 2}, {Port: 3}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)
	shuffledEndpoints := loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], client3)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	// Then update the configuration with one fewer endpoints, make sure
	// we start in the beginning again
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint"}},
				Ports:     []api.EndpointPort{{Port: 4}, {Port: 5}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)
	shuffledEndpoints = loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)

	// Clear endpoints
	endpoints[0] = api.Endpoints{ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace}, Subsets: nil}
	loadBalancer.OnEndpointsUpdate(endpoints)

	endpoint, err = loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
}

func TestStickyLoadBalanceWorksWithServiceRemoval(t *testing.T) {
	client1 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	client2 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 2), Port: 0}
	client3 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 3), Port: 0}
	loadBalancer := NewLoadBalancerRR()
	fooService := proxy.ServicePortName{types.NamespacedName{"testnamespace", "foo"}, ""}
	endpoint, err := loadBalancer.NextEndpoint(fooService, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	loadBalancer.NewService(fooService, api.ServiceAffinityClientIP, 0)
	endpoints := make([]api.Endpoints, 2)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: fooService.Name, Namespace: fooService.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint"}},
				Ports:     []api.EndpointPort{{Port: 1}, {Port: 2}, {Port: 3}},
			},
		},
	}
	barService := proxy.ServicePortName{types.NamespacedName{"testnamespace", "bar"}, ""}
	loadBalancer.NewService(barService, api.ServiceAffinityClientIP, 0)
	endpoints[1] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: barService.Name, Namespace: barService.Namespace},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "endpoint"}},
				Ports:     []api.EndpointPort{{Port: 4}, {Port: 5}},
			},
		},
	}
	loadBalancer.OnEndpointsUpdate(endpoints)

	shuffledFooEndpoints := loadBalancer.services[fooService].endpoints
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[2], client3)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[2], client3)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[2], client3)

	shuffledBarEndpoints := loadBalancer.services[barService].endpoints
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[1], client2)

	// Then update the configuration by removing foo
	loadBalancer.OnEndpointsUpdate(endpoints[1:])
	endpoint, err = loadBalancer.NextEndpoint(fooService, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	// but bar is still there, and we continue RR from where we left off.
	shuffledBarEndpoints = loadBalancer.services[barService].endpoints
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], client1)
}
