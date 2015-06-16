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

package servicecontroller

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	fake_cloud "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

const region = "us-central"

func newService(name string, uid types.UID, serviceType api.ServiceType) *api.Service {
	return &api.Service{ObjectMeta: api.ObjectMeta{Name: name, Namespace: "namespace", UID: uid}, Spec: api.ServiceSpec{Type: serviceType}}
}

func TestCreateExternalLoadBalancer(t *testing.T) {
	table := []struct {
		service             *api.Service
		expectErr           bool
		expectCreateAttempt bool
	}{
		{
			service: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "no-external-balancer",
					Namespace: "default",
				},
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeClusterIP,
				},
			},
			expectErr:           false,
			expectCreateAttempt: false,
		},
		{
			service: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "udp-service",
					Namespace: "default",
				},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{{
						Port:     80,
						Protocol: api.ProtocolUDP,
					}},
					Type: api.ServiceTypeLoadBalancer,
				},
			},
			expectErr:           true,
			expectCreateAttempt: false,
		},
		{
			service: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "basic-service1",
					Namespace: "default",
				},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{{
						Port:     80,
						Protocol: api.ProtocolTCP,
					}},
					Type: api.ServiceTypeLoadBalancer,
				},
			},
			expectErr:           false,
			expectCreateAttempt: true,
		},
	}

	for _, item := range table {
		cloud := &fake_cloud.FakeCloud{}
		cloud.Region = region
		client := &testclient.Fake{}
		controller := New(cloud, client, "test-cluster")
		controller.init()
		cloud.Calls = nil    // ignore any cloud calls made in init()
		client.Actions = nil // ignore any client calls made in init()
		err, _ := controller.createLoadBalancerIfNeeded(types.NamespacedName{"foo", "bar"}, item.service, nil)
		if !item.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		} else if item.expectErr && err == nil {
			t.Errorf("expected error creating %v, got nil", item.service)
		}
		if !item.expectCreateAttempt {
			if len(cloud.Calls) > 0 {
				t.Errorf("unexpected cloud provider calls: %v", cloud.Calls)
			}
			if len(client.Actions) > 0 {
				t.Errorf("unexpected client actions: %v", client.Actions)
			}
		} else {
			if len(cloud.Balancers) != 1 {
				t.Errorf("expected one load balancer to be created, got %v", cloud.Balancers)
			} else if cloud.Balancers[0].Name != controller.loadBalancerName(item.service) ||
				cloud.Balancers[0].Region != region ||
				cloud.Balancers[0].Ports[0].Port != item.service.Spec.Ports[0].Port {
				t.Errorf("created load balancer has incorrect parameters: %v", cloud.Balancers[0])
			}
			actionFound := false
			for _, action := range client.Actions {
				if action.Action == "update-service" {
					actionFound = true
				}
			}
			if !actionFound {
				t.Errorf("expected updated service to be sent to client, got these actions instead: %v", client.Actions)
			}
		}
	}
}

// TODO: Finish converting and update comments
func TestUpdateNodesInExternalLoadBalancer(t *testing.T) {
	hosts := []string{"node0", "node1", "node73"}
	table := []struct {
		services            []*api.Service
		expectedUpdateCalls []fake_cloud.FakeUpdateBalancerCall
	}{
		{
			// No services present: no calls should be made.
			services:            []*api.Service{},
			expectedUpdateCalls: nil,
		},
		{
			// Services do not have external load balancers: no calls should be made.
			services: []*api.Service{
				newService("s0", "111", api.ServiceTypeClusterIP),
				newService("s1", "222", api.ServiceTypeNodePort),
			},
			expectedUpdateCalls: nil,
		},
		{
			// Services does have an external load balancer: one call should be made.
			services: []*api.Service{
				newService("s0", "333", api.ServiceTypeLoadBalancer),
			},
			expectedUpdateCalls: []fake_cloud.FakeUpdateBalancerCall{
				{Name: "a333", Region: region, Hosts: []string{"node0", "node1", "node73"}},
			},
		},
		{
			// Three services have an external load balancer: three calls.
			services: []*api.Service{
				newService("s0", "444", api.ServiceTypeLoadBalancer),
				newService("s1", "555", api.ServiceTypeLoadBalancer),
				newService("s2", "666", api.ServiceTypeLoadBalancer),
			},
			expectedUpdateCalls: []fake_cloud.FakeUpdateBalancerCall{
				{Name: "a444", Region: region, Hosts: []string{"node0", "node1", "node73"}},
				{Name: "a555", Region: region, Hosts: []string{"node0", "node1", "node73"}},
				{Name: "a666", Region: region, Hosts: []string{"node0", "node1", "node73"}},
			},
		},
		{
			// Two services have an external load balancer and two don't: two calls.
			services: []*api.Service{
				newService("s0", "777", api.ServiceTypeNodePort),
				newService("s1", "888", api.ServiceTypeLoadBalancer),
				newService("s3", "999", api.ServiceTypeLoadBalancer),
				newService("s4", "123", api.ServiceTypeClusterIP),
			},
			expectedUpdateCalls: []fake_cloud.FakeUpdateBalancerCall{
				{Name: "a888", Region: region, Hosts: []string{"node0", "node1", "node73"}},
				{Name: "a999", Region: region, Hosts: []string{"node0", "node1", "node73"}},
			},
		},
		{
			// One service has an external load balancer and one is nil: one call.
			services: []*api.Service{
				newService("s0", "234", api.ServiceTypeLoadBalancer),
				nil,
			},
			expectedUpdateCalls: []fake_cloud.FakeUpdateBalancerCall{
				{Name: "a234", Region: region, Hosts: []string{"node0", "node1", "node73"}},
			},
		},
	}
	for _, item := range table {
		cloud := &fake_cloud.FakeCloud{}

		cloud.Region = region
		client := &testclient.Fake{}
		controller := New(cloud, client, "test-cluster2")
		controller.init()
		cloud.Calls = nil // ignore any cloud calls made in init()

		var services []*cachedService
		for _, service := range item.services {
			services = append(services, &cachedService{lastState: service, appliedState: service})
		}
		if err := controller.updateLoadBalancerHosts(services, hosts); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(item.expectedUpdateCalls, cloud.UpdateCalls) {
			t.Errorf("expected update calls mismatch, expected %+v, got %+v", item.expectedUpdateCalls, cloud.UpdateCalls)
		}
	}
}

// TODO(a-robinson): Add tests for update/sync/delete.
