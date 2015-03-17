/*
Copyright 2014 Google Inc. All rights reserved.

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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

func TestValidateWorks(t *testing.T) {
	if isValidEndpoint(&api.Endpoint{}) {
		t.Errorf("Didn't fail for empty string")
	}
	if isValidEndpoint(&api.Endpoint{IP: "foobar"}) {
		t.Errorf("Didn't fail with no port")
	}
	if isValidEndpoint(&api.Endpoint{IP: "foobar", Port: -1}) {
		t.Errorf("Didn't fail with a negative port")
	}
	if !isValidEndpoint(&api.Endpoint{IP: "foobar", Port: 8080}) {
		t.Errorf("Failed a valid config.")
	}
}

func TestFilterWorks(t *testing.T) {
	endpoints := []api.Endpoint{
		{IP: "foobar", Port: 1},
		{IP: "foobar", Port: 2},
		{IP: "foobar", Port: -1},
		{IP: "foobar", Port: 3},
		{IP: "foobar", Port: -2},
	}
	filtered := filterValidEndpoints(endpoints)

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
	loadBalancer.OnUpdate(endpoints)
	service := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil {
		t.Errorf("Didn't fail with non-existent service")
	}
	if len(endpoint) != 0 {
		t.Errorf("Got an endpoint")
	}
}

func expectEndpoint(t *testing.T, loadBalancer *LoadBalancerRR, service types.NamespacedName, expected string, netaddr net.Addr) {
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
	service := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints:  []api.Endpoint{{IP: "endpoint1", Port: 40}},
	}
	loadBalancer.OnUpdate(endpoints)
	expectEndpoint(t, loadBalancer, service, "endpoint1:40", nil)
	expectEndpoint(t, loadBalancer, service, "endpoint1:40", nil)
	expectEndpoint(t, loadBalancer, service, "endpoint1:40", nil)
	expectEndpoint(t, loadBalancer, service, "endpoint1:40", nil)
}

func TestLoadBalanceWorksWithMultipleEndpoints(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	service := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
			{IP: "endpoint", Port: 3},
		},
	}
	loadBalancer.OnUpdate(endpoints)
	shuffledEndpoints := loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], nil)
}

func TestLoadBalanceWorksWithMultipleEndpointsAndUpdates(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	service := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
			{IP: "endpoint", Port: 3},
		},
	}
	loadBalancer.OnUpdate(endpoints)
	shuffledEndpoints := loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], nil)
	// Then update the configuration with one fewer endpoints, make sure
	// we start in the beginning again
	endpoints[0] = api.Endpoints{ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 8},
			{IP: "endpoint", Port: 9},
		},
	}
	loadBalancer.OnUpdate(endpoints)
	shuffledEndpoints = loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], nil)
	// Clear endpoints
	endpoints[0] = api.Endpoints{ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace}, Endpoints: []api.Endpoint{}}
	loadBalancer.OnUpdate(endpoints)

	endpoint, err = loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
}

func TestLoadBalanceWorksWithServiceRemoval(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	fooService := types.NewNamespacedNameOrDie("testnamespace", "foo")
	barService := types.NewNamespacedNameOrDie("testnamespace", "bar")
	endpoint, err := loadBalancer.NextEndpoint(fooService, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 2)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: fooService.Name, Namespace: fooService.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
			{IP: "endpoint", Port: 3},
		},
	}
	endpoints[1] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: barService.Name, Namespace: barService.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 4},
			{IP: "endpoint", Port: 5},
		},
	}
	loadBalancer.OnUpdate(endpoints)
	shuffledFooEndpoints := loadBalancer.services[fooService].endpoints
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[2], nil)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[1], nil)

	shuffledBarEndpoints := loadBalancer.services[barService].endpoints
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], nil)

	// Then update the configuration by removing foo
	loadBalancer.OnUpdate(endpoints[1:])
	endpoint, err = loadBalancer.NextEndpoint(fooService, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	// but bar is still there, and we continue RR from where we left off.
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], nil)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[1], nil)
	expectEndpoint(t, loadBalancer, barService, shuffledBarEndpoints[0], nil)
}

func TestStickyLoadBalanceWorksWithSingleEndpoint(t *testing.T) {
	client1 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	client2 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 2), Port: 0}
	loadBalancer := NewLoadBalancerRR()
	service := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	loadBalancer.NewService(service, api.AffinityTypeClientIP, 0)
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints:  []api.Endpoint{{IP: "endpoint", Port: 1}},
	}
	loadBalancer.OnUpdate(endpoints)
	expectEndpoint(t, loadBalancer, service, "endpoint:1", client1)
	expectEndpoint(t, loadBalancer, service, "endpoint:1", client1)
	expectEndpoint(t, loadBalancer, service, "endpoint:1", client2)
	expectEndpoint(t, loadBalancer, service, "endpoint:1", client2)
}

func TestStickyLoadBalanaceWorksWithMultipleEndpoints(t *testing.T) {
	client1 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	client2 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 2), Port: 0}
	client3 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 3), Port: 0}
	loadBalancer := NewLoadBalancerRR()
	service := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	loadBalancer.NewService(service, api.AffinityTypeClientIP, 0)
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
			{IP: "endpoint", Port: 3},
		},
	}
	loadBalancer.OnUpdate(endpoints)
	shuffledEndpoints := loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], client3)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], client3)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
}

func TestStickyLoadBalanaceWorksWithMultipleEndpointsStickyNone(t *testing.T) {
	client1 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	client2 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 2), Port: 0}
	client3 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 3), Port: 0}
	loadBalancer := NewLoadBalancerRR()
	service := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	loadBalancer.NewService(service, api.AffinityTypeNone, 0)
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
			{IP: "endpoint", Port: 3},
		},
	}
	loadBalancer.OnUpdate(endpoints)

	shuffledEndpoints := loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client3)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[2], client3)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client1)
}

func TestStickyLoadBalanaceWorksWithMultipleEndpointsRemoveOne(t *testing.T) {
	client1 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	client2 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 2), Port: 0}
	client3 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 3), Port: 0}
	client4 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 4), Port: 0}
	client5 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 5), Port: 0}
	client6 := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 6), Port: 0}
	loadBalancer := NewLoadBalancerRR()
	service := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	loadBalancer.NewService(service, api.AffinityTypeClientIP, 0)
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
			{IP: "endpoint", Port: 3},
		},
	}
	loadBalancer.OnUpdate(endpoints)
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
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
		},
	}
	loadBalancer.OnUpdate(endpoints)
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
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
			{IP: "endpoint", Port: 4},
		},
	}
	loadBalancer.OnUpdate(endpoints)
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
	service := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	loadBalancer.NewService(service, api.AffinityTypeClientIP, 0)
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
			{IP: "endpoint", Port: 3},
		},
	}
	loadBalancer.OnUpdate(endpoints)
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
	endpoints[0] = api.Endpoints{ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 4},
			{IP: "endpoint", Port: 5},
		},
	}
	loadBalancer.OnUpdate(endpoints)
	shuffledEndpoints = loadBalancer.services[service].endpoints
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, service, shuffledEndpoints[1], client2)

	// Clear endpoints
	endpoints[0] = api.Endpoints{ObjectMeta: api.ObjectMeta{Name: service.Name, Namespace: service.Namespace}, Endpoints: []api.Endpoint{}}
	loadBalancer.OnUpdate(endpoints)

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
	fooService := types.NewNamespacedNameOrDie("testnamespace", "foo")
	endpoint, err := loadBalancer.NextEndpoint(fooService, nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	loadBalancer.NewService(fooService, api.AffinityTypeClientIP, 0)
	endpoints := make([]api.Endpoints, 2)
	endpoints[0] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: fooService.Name, Namespace: fooService.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 1},
			{IP: "endpoint", Port: 2},
			{IP: "endpoint", Port: 3},
		},
	}
	barService := types.NewNamespacedNameOrDie("testnamespace", "bar")
	loadBalancer.NewService(barService, api.AffinityTypeClientIP, 0)
	endpoints[1] = api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: barService.Name, Namespace: barService.Namespace},
		Endpoints: []api.Endpoint{
			{IP: "endpoint", Port: 5},
			{IP: "endpoint", Port: 5},
		},
	}
	loadBalancer.OnUpdate(endpoints)

	shuffledFooEndpoints := loadBalancer.services[fooService].endpoints
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[2], client3)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[2], client3)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[1], client2)

	shuffledBarEndpoints := loadBalancer.services[barService].endpoints
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[1], client2)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], client1)
	expectEndpoint(t, loadBalancer, fooService, shuffledFooEndpoints[0], client1)

	// Then update the configuration by removing foo
	loadBalancer.OnUpdate(endpoints[1:])
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
