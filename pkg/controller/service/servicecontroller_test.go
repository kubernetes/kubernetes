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

	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/types"
)

const region = "us-central"

func newService(name string, uid types.UID, serviceType v1.ServiceType) *v1.Service {
	return &v1.Service{ObjectMeta: v1.ObjectMeta{Name: name, Namespace: "namespace", UID: uid, SelfLink: testapi.Default.SelfLink("services", name)}, Spec: v1.ServiceSpec{Type: serviceType}}
}

func TestCreateExternalLoadBalancer(t *testing.T) {
	table := []struct {
		service             *v1.Service
		expectErr           bool
		expectCreateAttempt bool
	}{
		{
			service: &v1.Service{
				ObjectMeta: v1.ObjectMeta{
					Name:      "no-external-balancer",
					Namespace: "default",
				},
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeClusterIP,
				},
			},
			expectErr:           false,
			expectCreateAttempt: false,
		},
		{
			service: &v1.Service{
				ObjectMeta: v1.ObjectMeta{
					Name:      "udp-service",
					Namespace: "default",
					SelfLink:  testapi.Default.SelfLink("services", "udp-service"),
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:     80,
						Protocol: v1.ProtocolUDP,
					}},
					Type: v1.ServiceTypeLoadBalancer,
				},
			},
			expectErr:           false,
			expectCreateAttempt: true,
		},
		{
			service: &v1.Service{
				ObjectMeta: v1.ObjectMeta{
					Name:      "basic-service1",
					Namespace: "default",
					SelfLink:  testapi.Default.SelfLink("services", "basic-service1"),
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:     80,
						Protocol: v1.ProtocolTCP,
					}},
					Type: v1.ServiceTypeLoadBalancer,
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
	nodes := []*v1.Node{
		{ObjectMeta: v1.ObjectMeta{Name: "node0"}},
		{ObjectMeta: v1.ObjectMeta{Name: "node1"}},
		{ObjectMeta: v1.ObjectMeta{Name: "node73"}},
	}
	table := []struct {
		services            []*v1.Service
		expectedUpdateCalls []fakecloud.FakeUpdateBalancerCall
	}{
		{
			// No services present: no calls should be made.
			services:            []*v1.Service{},
			expectedUpdateCalls: nil,
		},
		{
			// Services do not have external load balancers: no calls should be made.
			services: []*v1.Service{
				newService("s0", "111", v1.ServiceTypeClusterIP),
				newService("s1", "222", v1.ServiceTypeNodePort),
			},
			expectedUpdateCalls: nil,
		},
		{
			// Services does have an external load balancer: one call should be made.
			services: []*v1.Service{
				newService("s0", "333", v1.ServiceTypeLoadBalancer),
			},
			expectedUpdateCalls: []fakecloud.FakeUpdateBalancerCall{
				{newService("s0", "333", v1.ServiceTypeLoadBalancer), nodes},
			},
		},
		{
			// Three services have an external load balancer: three calls.
			services: []*v1.Service{
				newService("s0", "444", v1.ServiceTypeLoadBalancer),
				newService("s1", "555", v1.ServiceTypeLoadBalancer),
				newService("s2", "666", v1.ServiceTypeLoadBalancer),
			},
			expectedUpdateCalls: []fakecloud.FakeUpdateBalancerCall{
				{newService("s0", "444", v1.ServiceTypeLoadBalancer), nodes},
				{newService("s1", "555", v1.ServiceTypeLoadBalancer), nodes},
				{newService("s2", "666", v1.ServiceTypeLoadBalancer), nodes},
			},
		},
		{
			// Two services have an external load balancer and two don't: two calls.
			services: []*v1.Service{
				newService("s0", "777", v1.ServiceTypeNodePort),
				newService("s1", "888", v1.ServiceTypeLoadBalancer),
				newService("s3", "999", v1.ServiceTypeLoadBalancer),
				newService("s4", "123", v1.ServiceTypeClusterIP),
			},
			expectedUpdateCalls: []fakecloud.FakeUpdateBalancerCall{
				{newService("s1", "888", v1.ServiceTypeLoadBalancer), nodes},
				{newService("s3", "999", v1.ServiceTypeLoadBalancer), nodes},
			},
		},
		{
			// One service has an external load balancer and one is nil: one call.
			services: []*v1.Service{
				newService("s0", "234", v1.ServiceTypeLoadBalancer),
				nil,
			},
			expectedUpdateCalls: []fakecloud.FakeUpdateBalancerCall{
				{newService("s0", "234", v1.ServiceTypeLoadBalancer), nodes},
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

		var services []*v1.Service
		for _, service := range item.services {
			services = append(services, service)
		}
		if err := controller.updateLoadBalancerHosts(services, nodes); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(item.expectedUpdateCalls, cloud.UpdateCalls) {
			t.Errorf("expected update calls mismatch, expected %+v, got %+v", item.expectedUpdateCalls, cloud.UpdateCalls)
		}
	}
}

func TestGetNodeConditionPredicate(t *testing.T) {
	tests := []struct {
		node         v1.Node
		expectAccept bool
		name         string
	}{
		{
			node:         v1.Node{},
			expectAccept: false,
			name:         "empty",
		},
		{
			node: v1.Node{
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{Type: v1.NodeReady, Status: v1.ConditionTrue},
					},
				},
			},
			expectAccept: true,
			name:         "basic",
		},
		{
			node: v1.Node{
				Spec: v1.NodeSpec{Unschedulable: true},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{Type: v1.NodeReady, Status: v1.ConditionTrue},
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
