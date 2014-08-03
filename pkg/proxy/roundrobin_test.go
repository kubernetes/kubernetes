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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestLoadBalanceValidateWorks(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	if loadBalancer.isValid("") {
		t.Errorf("Didn't fail for empty string")
	}
	if loadBalancer.isValid("foobar") {
		t.Errorf("Didn't fail with no port")
	}
	if loadBalancer.isValid("foobar:-1") {
		t.Errorf("Didn't fail with a negative port")
	}
	if !loadBalancer.isValid("foobar:8080") {
		t.Errorf("Failed a valid config.")
	}
}

func TestLoadBalanceFilterWorks(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	endpoints := []string{"foobar:1", "foobar:2", "foobar:-1", "foobar:3", "foobar:-2"}
	filtered := loadBalancer.filterValidEndpoints(endpoints)

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
	endpoint, err := loadBalancer.NextEndpoint("foo", nil)
	if err == nil {
		t.Errorf("Didn't fail with non-existent service")
	}
	if len(endpoint) != 0 {
		t.Errorf("Got an endpoint")
	}
}

func expectEndpoint(t *testing.T, loadBalancer *LoadBalancerRR, service string, expected string) {
	endpoint, err := loadBalancer.NextEndpoint(service, nil)
	if err != nil {
		t.Errorf("Didn't find a service for %s, expected %s, failed with: %v", service, expected, err)
	}
	if endpoint != expected {
		t.Errorf("Didn't get expected endpoint for service %s, expected %s, got: %s", service, expected, endpoint)
	}
}

func TestLoadBalanceWorksWithSingleEndpoint(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	endpoint, err := loadBalancer.NextEndpoint("foo", nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		JSONBase:  api.JSONBase{ID: "foo"},
		Endpoints: []string{"endpoint1:40"},
	}
	loadBalancer.OnUpdate(endpoints)
	expectEndpoint(t, loadBalancer, "foo", "endpoint1:40")
	expectEndpoint(t, loadBalancer, "foo", "endpoint1:40")
	expectEndpoint(t, loadBalancer, "foo", "endpoint1:40")
	expectEndpoint(t, loadBalancer, "foo", "endpoint1:40")
}

func TestLoadBalanceWorksWithMultipleEndpoints(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	endpoint, err := loadBalancer.NextEndpoint("foo", nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		JSONBase:  api.JSONBase{ID: "foo"},
		Endpoints: []string{"endpoint:1", "endpoint:2", "endpoint:3"},
	}
	loadBalancer.OnUpdate(endpoints)
	expectEndpoint(t, loadBalancer, "foo", "endpoint:1")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:2")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:3")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:1")
}

func TestLoadBalanceWorksWithMultipleEndpointsAndUpdates(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	endpoint, err := loadBalancer.NextEndpoint("foo", nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 1)
	endpoints[0] = api.Endpoints{
		JSONBase:  api.JSONBase{ID: "foo"},
		Endpoints: []string{"endpoint:1", "endpoint:2", "endpoint:3"},
	}
	loadBalancer.OnUpdate(endpoints)
	expectEndpoint(t, loadBalancer, "foo", "endpoint:1")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:2")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:3")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:1")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:2")
	// Then update the configuration with one fewer endpoints, make sure
	// we start in the beginning again
	endpoints[0] = api.Endpoints{JSONBase: api.JSONBase{ID: "foo"},
		Endpoints: []string{"endpoint:8", "endpoint:9"},
	}
	loadBalancer.OnUpdate(endpoints)
	expectEndpoint(t, loadBalancer, "foo", "endpoint:8")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:9")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:8")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:9")
	// Clear endpoints
	endpoints[0] = api.Endpoints{JSONBase: api.JSONBase{ID: "foo"}, Endpoints: []string{}}
	loadBalancer.OnUpdate(endpoints)

	endpoint, err = loadBalancer.NextEndpoint("foo", nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
}

func TestLoadBalanceWorksWithServiceRemoval(t *testing.T) {
	loadBalancer := NewLoadBalancerRR()
	endpoint, err := loadBalancer.NextEndpoint("foo", nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}
	endpoints := make([]api.Endpoints, 2)
	endpoints[0] = api.Endpoints{
		JSONBase:  api.JSONBase{ID: "foo"},
		Endpoints: []string{"endpoint:1", "endpoint:2", "endpoint:3"},
	}
	endpoints[1] = api.Endpoints{
		JSONBase:  api.JSONBase{ID: "bar"},
		Endpoints: []string{"endpoint:4", "endpoint:5"},
	}
	loadBalancer.OnUpdate(endpoints)
	expectEndpoint(t, loadBalancer, "foo", "endpoint:1")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:2")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:3")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:1")
	expectEndpoint(t, loadBalancer, "foo", "endpoint:2")

	expectEndpoint(t, loadBalancer, "bar", "endpoint:4")
	expectEndpoint(t, loadBalancer, "bar", "endpoint:5")
	expectEndpoint(t, loadBalancer, "bar", "endpoint:4")
	expectEndpoint(t, loadBalancer, "bar", "endpoint:5")
	expectEndpoint(t, loadBalancer, "bar", "endpoint:4")

	// Then update the configuration by removing foo
	loadBalancer.OnUpdate(endpoints[1:])
	endpoint, err = loadBalancer.NextEndpoint("foo", nil)
	if err == nil || len(endpoint) != 0 {
		t.Errorf("Didn't fail with non-existent service")
	}

	// but bar is still there, and we continue RR from where we left off.
	expectEndpoint(t, loadBalancer, "bar", "endpoint:5")
	expectEndpoint(t, loadBalancer, "bar", "endpoint:4")
	expectEndpoint(t, loadBalancer, "bar", "endpoint:5")
	expectEndpoint(t, loadBalancer, "bar", "endpoint:4")
}
