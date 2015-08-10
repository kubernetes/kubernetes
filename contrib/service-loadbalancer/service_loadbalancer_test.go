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

package main

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

const ns = "default"

// storeEps stores the given endpoints in a store.
func storeEps(eps []*api.Endpoints) cache.Store {
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	found := make([]interface{}, 0, len(eps))
	for i := range eps {
		found = append(found, eps[i])
	}
	if err := store.Replace(found); err != nil {
		glog.Fatalf("Unable to replace endpoints %v", err)
	}
	return store
}

// storeServices stores the given services in a store.
func storeServices(svcs []*api.Service) cache.Store {
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	found := make([]interface{}, 0, len(svcs))
	for i := range svcs {
		found = append(found, svcs[i])
	}
	if err := store.Replace(found); err != nil {
		glog.Fatalf("Unable to replace services %v", err)
	}
	return store
}

func getEndpoints(svc *api.Service, endpointAddresses []api.EndpointAddress, endpointPorts []api.EndpointPort) *api.Endpoints {
	return &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:      svc.Name,
			Namespace: svc.Namespace,
		},
		Subsets: []api.EndpointSubset{{
			Addresses: endpointAddresses,
			Ports:     endpointPorts,
		}},
	}
}

func getService(servicePorts []api.ServicePort) *api.Service {
	return &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: string(util.NewUUID()), Namespace: ns},
		Spec: api.ServiceSpec{
			Ports: servicePorts,
		},
	}
}

func newFakeLoadBalancerController(endpoints []*api.Endpoints, services []*api.Service) *loadBalancerController {
	flb := loadBalancerController{}
	flb.epLister.Store = storeEps(endpoints)
	flb.svcLister.Store = storeServices(services)
	flb.httpPort = 80
	return &flb
}

func TestGetEndpoints(t *testing.T) {
	// 2 pods each of which have 3 targetPorts exposed via a single service
	endpointAddresses := []api.EndpointAddress{
		{IP: "1.2.3.4"},
		{IP: "6.7.8.9"},
	}
	ports := []int{80, 443, 3306}
	endpointPorts := []api.EndpointPort{
		{Port: ports[0], Protocol: "TCP"},
		{Port: ports[1], Protocol: "TCP"},
		{Port: ports[2], Protocol: "TCP", Name: "mysql"},
	}
	servicePorts := []api.ServicePort{
		{Port: 10, TargetPort: util.NewIntOrStringFromInt(ports[0])},
		{Port: 20, TargetPort: util.NewIntOrStringFromInt(ports[1])},
		{Port: 30, TargetPort: util.NewIntOrStringFromString("mysql")},
	}

	svc := getService(servicePorts)
	endpoints := []*api.Endpoints{getEndpoints(svc, endpointAddresses, endpointPorts)}
	flb := newFakeLoadBalancerController(endpoints, []*api.Service{svc})

	for i := range ports {
		eps := flb.getEndpoints(svc, &svc.Spec.Ports[i])
		expectedEps := util.NewStringSet()
		for _, address := range endpointAddresses {
			expectedEps.Insert(fmt.Sprintf("%v:%v", address.IP, ports[i]))
		}

		receivedEps := util.NewStringSet()
		for _, ep := range eps {
			receivedEps.Insert(ep)
		}
		if len(receivedEps) != len(expectedEps) || !expectedEps.IsSuperset(receivedEps) {
			t.Fatalf("Unexpected endpoints, received %+v, expected %+v", receivedEps, expectedEps)
		}
		glog.Infof("Got endpoints %+v", receivedEps)
	}
}

func TestGetServices(t *testing.T) {
	endpointAddresses := []api.EndpointAddress{
		{IP: "1.2.3.4"},
		{IP: "6.7.8.9"},
	}
	ports := []int{80, 443}
	endpointPorts := []api.EndpointPort{
		{Port: ports[0], Protocol: "TCP"},
		{Port: ports[1], Protocol: "TCP"},
	}
	servicePorts := []api.ServicePort{
		{Port: 10, TargetPort: util.NewIntOrStringFromInt(ports[0])},
		{Port: 20, TargetPort: util.NewIntOrStringFromInt(ports[1])},
	}

	// 2 services targeting the same endpoints, one of which is declared as a tcp service.
	svc1 := getService(servicePorts)
	svc2 := getService(servicePorts)
	endpoints := []*api.Endpoints{
		getEndpoints(svc1, endpointAddresses, endpointPorts),
		getEndpoints(svc2, endpointAddresses, endpointPorts),
	}
	flb := newFakeLoadBalancerController(endpoints, []*api.Service{svc1, svc2})
	flb.tcpServices = map[string]int{
		svc1.Name: 20,
	}
	http, tcp := flb.getServices()
	serviceURLEp := fmt.Sprintf("%v:%v", svc1.Name, 20)
	if len(tcp) != 1 || tcp[0].Name != serviceURLEp || tcp[0].FrontendPort != 20 {
		t.Fatalf("Unexpected tcp service %+v expected %+v", tcp, svc1.Name)
	}

	// All pods of svc1 exposed under servicePort 20 are tcp
	expectedTCPEps := util.NewStringSet()
	for _, address := range endpointAddresses {
		expectedTCPEps.Insert(fmt.Sprintf("%v:%v", address.IP, 443))
	}
	receivedTCPEps := util.NewStringSet()
	for _, ep := range tcp[0].Ep {
		receivedTCPEps.Insert(ep)
	}
	if len(receivedTCPEps) != len(expectedTCPEps) || !expectedTCPEps.IsSuperset(receivedTCPEps) {
		t.Fatalf("Unexpected tcp serice %+v", tcp)
	}

	// All pods of either service not mentioned in the tcpmap are multiplexed on port  :80 as http services.
	expectedURLMapping := map[string]util.StringSet{
		fmt.Sprintf("%v:%v", svc1.Name, 10): util.NewStringSet("1.2.3.4:80", "6.7.8.9:80"),
		fmt.Sprintf("%v:%v", svc2.Name, 10): util.NewStringSet("1.2.3.4:80", "6.7.8.9:80"),
		fmt.Sprintf("%v:%v", svc2.Name, 20): util.NewStringSet("1.2.3.4:443", "6.7.8.9:443"),
	}
	for _, s := range http {
		if s.FrontendPort != 80 {
			t.Fatalf("All http services should get multiplexed via the same frontend port: %+v", s)
		}
		expectedEps, ok := expectedURLMapping[s.Name]
		if !ok {
			t.Fatalf("Expected url endpoint %v, found %+v", s.Name, expectedURLMapping)
		}
		receivedEp := util.NewStringSet()
		for i := range s.Ep {
			receivedEp.Insert(s.Ep[i])
		}
		if len(receivedEp) != len(expectedEps) && !receivedEp.IsSuperset(expectedEps) {
			t.Fatalf("Expected %+v, got %+v", expectedEps, receivedEp)
		}
	}
}
