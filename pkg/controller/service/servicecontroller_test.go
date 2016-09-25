/*
Copyright 2015 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/types"
)

const region = "us-central"

func newService(name string, uid types.UID, serviceType api.ServiceType) *api.Service {
	return &api.Service{ObjectMeta: api.ObjectMeta{Name: name, Namespace: "namespace", UID: uid, SelfLink: testapi.Default.SelfLink("services", name)}, Spec: api.ServiceSpec{Type: serviceType}}
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
					SelfLink:  testapi.Default.SelfLink("services", "udp-service"),
				},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{{
						Port:     80,
						Protocol: api.ProtocolUDP,
					}},
					Type: api.ServiceTypeLoadBalancer,
				},
			},
			expectErr:           false,
			expectCreateAttempt: true,
		},
		{
			service: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "basic-service1",
					Namespace: "default",
					SelfLink:  testapi.Default.SelfLink("services", "basic-service1"),
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
		cloud := &fakecloud.FakeCloud{}
		cloud.Region = region
		client := &fake.Clientset{}
		controller, _ := New(cloud, client, "test-cluster")
		controller.init()
		cloud.Calls = nil     // ignore any cloud calls made in init()
		client.ClearActions() // ignore any client calls made in init()
		err, _ := controller.createLoadBalancerIfNeeded("foo/bar", item.service)
		if !item.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		} else if item.expectErr && err == nil {
			t.Errorf("expected error creating %v, got nil", item.service)
		}
		actions := client.Actions()
		if !item.expectCreateAttempt {
			if len(cloud.Calls) > 0 {
				t.Errorf("unexpected cloud provider calls: %v", cloud.Calls)
			}
			if len(actions) > 0 {
				t.Errorf("unexpected client actions: %v", actions)
			}
		} else {
			var balancer *fakecloud.FakeBalancer
			for k := range cloud.Balancers {
				if balancer == nil {
					b := cloud.Balancers[k]
					balancer = &b
				} else {
					t.Errorf("expected one load balancer to be created, got %v", cloud.Balancers)
					break
				}
			}
			if balancer == nil {
				t.Errorf("expected one load balancer to be created, got none")
			} else if balancer.Name != controller.loadBalancerName(item.service) ||
				balancer.Region != region ||
				balancer.Ports[0].Port != item.service.Spec.Ports[0].Port {
				t.Errorf("created load balancer has incorrect parameters: %v", balancer)
			}
			actionFound := false
			for _, action := range actions {
				if action.GetVerb() == "update" && action.GetResource().Resource == "services" {
					actionFound = true
				}
			}
			if !actionFound {
				t.Errorf("expected updated service to be sent to client, got these actions instead: %v", actions)
			}
		}
	}
}

// TODO: Finish converting and update comments
func TestUpdateNodesInExternalLoadBalancer(t *testing.T) {
	hosts := []string{"node0", "node1", "node73"}
	table := []struct {
		services            []*api.Service
		expectedUpdateCalls []fakecloud.FakeUpdateBalancerCall
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
			expectedUpdateCalls: []fakecloud.FakeUpdateBalancerCall{
				{newService("s0", "333", api.ServiceTypeLoadBalancer), hosts},
			},
		},
		{
			// Three services have an external load balancer: three calls.
			services: []*api.Service{
				newService("s0", "444", api.ServiceTypeLoadBalancer),
				newService("s1", "555", api.ServiceTypeLoadBalancer),
				newService("s2", "666", api.ServiceTypeLoadBalancer),
			},
			expectedUpdateCalls: []fakecloud.FakeUpdateBalancerCall{
				{newService("s0", "444", api.ServiceTypeLoadBalancer), hosts},
				{newService("s1", "555", api.ServiceTypeLoadBalancer), hosts},
				{newService("s2", "666", api.ServiceTypeLoadBalancer), hosts},
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
			expectedUpdateCalls: []fakecloud.FakeUpdateBalancerCall{
				{newService("s1", "888", api.ServiceTypeLoadBalancer), hosts},
				{newService("s3", "999", api.ServiceTypeLoadBalancer), hosts},
			},
		},
		{
			// One service has an external load balancer and one is nil: one call.
			services: []*api.Service{
				newService("s0", "234", api.ServiceTypeLoadBalancer),
				nil,
			},
			expectedUpdateCalls: []fakecloud.FakeUpdateBalancerCall{
				{newService("s0", "234", api.ServiceTypeLoadBalancer), hosts},
			},
		},
	}
	for _, item := range table {
		cloud := &fakecloud.FakeCloud{}

		cloud.Region = region
		client := &fake.Clientset{}
		controller, _ := New(cloud, client, "test-cluster2")
		controller.init()
		cloud.Calls = nil // ignore any cloud calls made in init()

		var services []*api.Service
		for _, service := range item.services {
			services = append(services, service)
		}
		if err := controller.updateLoadBalancerHosts(services, hosts); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(item.expectedUpdateCalls, cloud.UpdateCalls) {
			t.Errorf("expected update calls mismatch, expected %+v, got %+v", item.expectedUpdateCalls, cloud.UpdateCalls)
		}
	}
}

func TestHostsFromNodeList(t *testing.T) {
	tests := []struct {
		nodes         *api.NodeList
		expectedHosts []string
	}{
		{
			nodes:         &api.NodeList{},
			expectedHosts: []string{},
		},
		{
			nodes: &api.NodeList{
				Items: []api.Node{
					{
						ObjectMeta: api.ObjectMeta{Name: "foo"},
						Status:     api.NodeStatus{Phase: api.NodeRunning},
					},
					{
						ObjectMeta: api.ObjectMeta{Name: "bar"},
						Status:     api.NodeStatus{Phase: api.NodeRunning},
					},
				},
			},
			expectedHosts: []string{"foo", "bar"},
		},
		{
			nodes: &api.NodeList{
				Items: []api.Node{
					{
						ObjectMeta: api.ObjectMeta{Name: "foo"},
						Status:     api.NodeStatus{Phase: api.NodeRunning},
					},
					{
						ObjectMeta: api.ObjectMeta{Name: "bar"},
						Status:     api.NodeStatus{Phase: api.NodeRunning},
					},
					{
						ObjectMeta: api.ObjectMeta{Name: "unschedulable"},
						Spec:       api.NodeSpec{Unschedulable: true},
						Status:     api.NodeStatus{Phase: api.NodeRunning},
					},
				},
			},
			expectedHosts: []string{"foo", "bar"},
		},
	}

	for _, test := range tests {
		hosts := hostsFromNodeList(test.nodes)
		if !reflect.DeepEqual(hosts, test.expectedHosts) {
			t.Errorf("expected: %v, saw: %v", test.expectedHosts, hosts)
		}
	}
}

func TestGetNodeConditionPredicate(t *testing.T) {
	tests := []struct {
		node         api.Node
		expectAccept bool
		name         string
	}{
		{
			node:         api.Node{},
			expectAccept: false,
			name:         "empty",
		},
		{
			node: api.Node{
				Status: api.NodeStatus{
					Conditions: []api.NodeCondition{
						{Type: api.NodeReady, Status: api.ConditionTrue},
					},
				},
			},
			expectAccept: true,
			name:         "basic",
		},
		{
			node: api.Node{
				Spec: api.NodeSpec{Unschedulable: true},
				Status: api.NodeStatus{
					Conditions: []api.NodeCondition{
						{Type: api.NodeReady, Status: api.ConditionTrue},
					},
				},
			},
			expectAccept: false,
			name:         "unschedulable",
		},
	}
	pred := getNodeConditionPredicate()
	for _, test := range tests {
		accept := pred(&test.node)
		if accept != test.expectAccept {
			t.Errorf("Test failed for %s, expected %v, saw %v", test.name, test.expectAccept, accept)
		}
	}
}

// TODO(a-robinson): Add tests for update/sync/delete.
