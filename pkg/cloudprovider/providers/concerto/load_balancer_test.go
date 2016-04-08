/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package concerto_cloud

import (
	"net"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
)

func TestGetLoadBalancer(t *testing.T) {
	apiMock := &ConcertoAPIServiceMock{
		balancers: []ConcertoLoadBalancer{
			{Id: "123456", Name: "mybalancer", FQDN: "mybalancer.concerto.mock"},
		},
	}
	concerto := ConcertoCloud{service: apiMock}
	status, exists, err := concerto.GetLoadBalancer("mybalancer", "whatever region")
	if err != nil {
		t.Errorf("GetLoadBalancer: should not have returned error")
	}
	if !exists {
		t.Errorf("GetLoadBalancer: should have found the LB")
	}
	if status.Ingress[0].Hostname != "mybalancer.concerto.mock" {
		t.Errorf("GetLoadBalancer: should have returned the correct status")
	}
}

func TestGetLoadBalancer_NonExisting(t *testing.T) {
	apiMock := &ConcertoAPIServiceMock{
		balancers: []ConcertoLoadBalancer{
			{Id: "123456", Name: "mybalancer", FQDN: "mybalancer.concerto.mock"},
		},
	}
	concerto := ConcertoCloud{service: apiMock}
	_, exists, err := concerto.GetLoadBalancer("otherbalancer", "whatever region")
	if err != nil {
		t.Errorf("GetLoadBalancer: should not have returned error")
	}
	if exists {
		t.Errorf("GetLoadBalancer: should not have found the LB")
	}
}

func TestEnsureLoadBalancer_CreatesTheLBInConcerto(t *testing.T) {
	apiMock := &ConcertoAPIServiceMock{
		balancers:         []ConcertoLoadBalancer{},
		balancedInstances: map[string][]string{},
	}
	ports := []*api.ServicePort{
		{Port: 1234},
	}
	concerto := ConcertoCloud{service: apiMock}
	concerto.EnsureLoadBalancer("myloadbalancer",
		"region",
		nil,   // external ip
		ports, // ports
		[]string{"host1", "host2"},
		types.NamespacedName{}, // namespaced name
		api.ServiceAffinityNone,
		nil, // annotations
	)
	if len(apiMock.balancers) != 1 {
		t.Errorf("EnsureLoadBalancer: should have created the LB")
	} else {
		lb := apiMock.balancers[0]
		if lb.Name != "myloadbalancer" || lb.Port != 1234 {
			t.Errorf("EnsureLoadBalancer: should have created the LB with correct data")
		}
	}
}

func TestEnsureLoadBalancerAddsTheNodes(t *testing.T) {
	apiMock := &ConcertoAPIServiceMock{
		balancers:         []ConcertoLoadBalancer{},
		balancedInstances: map[string][]string{},
		instances: []ConcertoInstance{
			{Name: "host1", Id: "11235813", PublicIP: "123.123.123.123"},
			{Name: "host2", Id: "11235815", PublicIP: "123.123.123.124"},
		},
	}
	ports := []*api.ServicePort{
		{Port: 1234},
	}
	concerto := ConcertoCloud{service: apiMock}
	_, err := concerto.EnsureLoadBalancer("myloadbalancer",
		"region",
		nil,   // external ip
		ports, // ports
		[]string{"host1", "host2"},
		types.NamespacedName{}, // namespaced name
		api.ServiceAffinityNone,
		nil, // annotations
	)
	if err != nil {
		t.Errorf("EnsureLoadBalancer: should not have returned any errors")
	}
	lb, _ := apiMock.GetLoadBalancerByName("myloadbalancer")
	if len(apiMock.balancedInstances[lb.Id]) != 2 {
		t.Errorf("EnsureLoadBalancer: should have registered the nodes with the LB")
	}
}

func TestEnsureLoadBalancerWithUnsupportedAffinity(t *testing.T) {
	concerto := ConcertoCloud{}
	ports := []*api.ServicePort{
		{},
	}
	_, err := concerto.EnsureLoadBalancer("name",
		"region",
		nil,   // external ip
		ports, // ports
		[]string{"host1", "host2"},
		types.NamespacedName{}, // namespaced name
		api.ServiceAffinityClientIP,
		nil, // annotations
	)
	if err == nil {
		t.Errorf("EnsureLoadBalancer: should not support ServiceAffinity")
	}
}

func TestEnsureLoadBalancerWithNoPort(t *testing.T) {
	concerto := ConcertoCloud{}
	ports := []*api.ServicePort{}
	_, err := concerto.EnsureLoadBalancer("name",
		"region",
		nil,   // external ip
		ports, // ports
		[]string{"host1", "host2"},
		types.NamespacedName{}, // namespaced name
		api.ServiceAffinityNone,
		nil, // annotations
	)
	if err == nil {
		t.Errorf("EnsureLoadBalancer: should return error if no port specified")
	}
}

func TestEnsureLoadBalancerWithMultiplePorts(t *testing.T) {
	concerto := ConcertoCloud{}
	ports := []*api.ServicePort{
		{},
		{},
	}
	_, err := concerto.EnsureLoadBalancer("name",
		"region",
		nil,   // external ip
		ports, // ports
		[]string{"host1", "host2"},
		types.NamespacedName{}, // namespaced name
		api.ServiceAffinityNone,
		nil, // annotations
	)
	if err == nil {
		t.Errorf("EnsureLoadBalancer: should return error if no port specified")
	}
}

func TestEnsureLoadBalancerWithExternalIP(t *testing.T) {
	concerto := ConcertoCloud{}
	ports := []*api.ServicePort{
		{},
	}
	_, err := concerto.EnsureLoadBalancer("name",
		"region",
		net.IP{}, // external ip
		ports,    // ports
		[]string{"host1", "host2"},
		types.NamespacedName{}, // namespaced name
		api.ServiceAffinityNone,
		nil, // annotations
	)
	if err == nil {
		t.Errorf("EnsureLoadBalancer: should not support ExternalIP specification")
	}
}

func TestEnsureLoadBalancer_UpdatesTheLBInConcerto(t *testing.T) {
	apiMock := &ConcertoAPIServiceMock{
		balancers: []ConcertoLoadBalancer{
			{Id: "LB1", Name: "mybalancer"},
		},
		balancedInstances: map[string][]string{
			"LB1": {"123.123.123.123", "123.123.123.124"},
		},
		instances: []ConcertoInstance{
			{Name: "host1", Id: "11235813", PublicIP: "123.123.123.123"},
			{Name: "host2", Id: "11235815", PublicIP: "123.123.123.124"},
			{Name: "host3", Id: "11235817", PublicIP: "123.123.123.125"},
		},
	}
	ports := []*api.ServicePort{
		{Port: 1234},
	}
	concerto := ConcertoCloud{service: apiMock}
	_, err := concerto.EnsureLoadBalancer("mybalancer",
		"region",
		nil,   // external ip
		ports, // ports
		[]string{"host1", "host3"},
		types.NamespacedName{}, // namespaced name
		api.ServiceAffinityNone,
		nil, // annotations
	)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(apiMock.balancers) > 1 {
		t.Errorf("EnsureLoadBalancer: should not have created the LB")
	} else if len(apiMock.balancers) < 1 {
		t.Errorf("EnsureLoadBalancer: should not have deleted the LB")
	} else {
		if apiMock.balancedInstances["LB1"][0] != "123.123.123.123" || apiMock.balancedInstances["LB1"][1] != "123.123.123.125" {
			t.Errorf("EnsureLoadBalancer: should have updated the load balancer: %v", apiMock.balancedInstances["LB1"])
		}
	}
}

func TestUpdateLoadBalancer(t *testing.T) {
	apiMock := &ConcertoAPIServiceMock{
		balancers: []ConcertoLoadBalancer{
			{Id: "LB1", Name: "mybalancer"},
		},
		balancedInstances: map[string][]string{
			"LB1": {"123.123.123.123", "123.123.123.124"},
		},
		instances: []ConcertoInstance{
			{Name: "host1", Id: "11235813", PublicIP: "123.123.123.123"},
			{Name: "host2", Id: "11235815", PublicIP: "123.123.123.124"},
			{Name: "host3", Id: "11235817", PublicIP: "123.123.123.125"},
		},
	}
	concerto := ConcertoCloud{service: apiMock}
	err := concerto.UpdateLoadBalancer("mybalancer", "noregion", []string{"host1", "host3"})
	if err != nil {
		t.Errorf("UpdateLoadBalancer: should not have returned error")
	}
	if apiMock.balancedInstances["LB1"][0] != "123.123.123.123" || apiMock.balancedInstances["LB1"][1] != "123.123.123.125" {
		t.Errorf("UpdateLoadBalancer: should have updated the load balancer: %v", apiMock.balancedInstances["LB1"])
	}
}

func TestEnsureLoadBalancerDeleted(t *testing.T) {
	apiMock := &ConcertoAPIServiceMock{
		balancers: []ConcertoLoadBalancer{
			{Id: "123456", Name: "mybalancer"},
		},
	}
	concerto := ConcertoCloud{service: apiMock}
	err := concerto.EnsureLoadBalancerDeleted("mybalancer", "whatever region")
	if err != nil {
		t.Errorf("EnsureLoadBalancerDeleted: should not have returned error")
	}
	if len(apiMock.balancers) > 0 {
		t.Errorf("EnsureLoadBalancerDeleted: should have deleted the load balancer")
	}
}

func Test_EnsureLoadBalancerDeleted_NonExisting(t *testing.T) {
	apiMock := &ConcertoAPIServiceMock{
		balancers: []ConcertoLoadBalancer{
			{Id: "123456", Name: "mybalancer"},
		},
	}
	concerto := ConcertoCloud{service: apiMock}
	err := concerto.EnsureLoadBalancerDeleted("anotherbalancer", "whatever region")
	if err != nil {
		t.Errorf("EnsureLoadBalancerDeleted: should not have returned error")
	}
	if len(apiMock.balancers) == 0 {
		t.Errorf("EnsureLoadBalancerDeleted: should not have deleted any load balancer")
	}
}

func Test_EnsureLoadBalancerDeleted_Error(t *testing.T) {
	apiMock := &ConcertoAPIServiceMock{
		balancers: []ConcertoLoadBalancer{
			{Id: "123456", Name: "mybalancer"},
		},
	}
	concerto := ConcertoCloud{service: apiMock}
	err := concerto.EnsureLoadBalancerDeleted("GiveMeAnError", "whatever region")
	if err == nil {
		t.Errorf("Error was expected but got none")
	}
}
