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
	"encoding/json"
	"fmt"
	"net"
	"reflect"
	"testing"
	"time"

	corelisters "k8s.io/client-go/listers/core/v1"
	testingutil "k8s.io/client-go/testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/testapi"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller"
)

const region = "us-central"

func newService(name string, uid types.UID, serviceType v1.ServiceType) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
			UID:       uid,
			SelfLink:  testapi.Default.SelfLink("services", name),
		},
		Spec: v1.ServiceSpec{
			Type: serviceType,
		},
	}
}

//Wrap newService so that you dont have to call default argumetns again and again.
func defaultExternalService() *v1.Service {
	return newService("external-balancer", types.UID("123"), v1.ServiceTypeLoadBalancer)
}

func alwaysReady() bool { return true }

func newController() (*ServiceController, *fakecloud.FakeCloud, *fake.Clientset) {
	cloud := &fakecloud.FakeCloud{}
	cloud.Region = region

	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	serviceInformer := informerFactory.Core().V1().Services()
	nodeInformer := informerFactory.Core().V1().Nodes()

	controller, _ := New(cloud, client, serviceInformer, nodeInformer, "test-cluster")
	controller.nodeListerSynced = alwaysReady
	controller.serviceListerSynced = alwaysReady
	controller.eventRecorder = record.NewFakeRecorder(100)

	controller.init()
	cloud.Calls = nil     // ignore any cloud calls made in init()
	client.ClearActions() // ignore any client calls made in init()

	return controller, cloud, client
}

func TestCreateExternalLoadBalancer(t *testing.T) {
	table := []struct {
		service             *v1.Service
		expectErr           bool
		expectCreateAttempt bool
	}{
		{
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
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
				ObjectMeta: metav1.ObjectMeta{
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
				ObjectMeta: metav1.ObjectMeta{
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
		controller, cloud, client := newController()
		err := controller.createLoadBalancerIfNeeded(item.service, "foo/bar")
		if !item.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		} else if item.expectErr && err == nil {
			t.Errorf("expected error creating %v, got nil", item.service)
		}
		actions := client.Actions()
		if !item.expectCreateAttempt {
			expectedCloudCalls := []string{"get"}
			if !reflect.DeepEqual(expectedCloudCalls, cloud.Calls) {
				t.Errorf("got cloud provider calls %+v but wanted %+v", cloud.Calls, expectedCloudCalls)
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
				if action.GetVerb() == "patch" && action.GetResource().Resource == "services" {
					actionFound = true
				}
			}
			if !actionFound {
				t.Errorf("expected patch service to be sent to client, got these actions instead: %v", actions)
			}
		}
	}
}

// TODO: Finish converting and update comments
func TestUpdateNodesInExternalLoadBalancer(t *testing.T) {
	nodes := []*v1.Node{
		{ObjectMeta: metav1.ObjectMeta{Name: "node0"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "node73"}},
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
				{Service: newService("s0", "333", v1.ServiceTypeLoadBalancer), Hosts: nodes},
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
				{Service: newService("s0", "444", v1.ServiceTypeLoadBalancer), Hosts: nodes},
				{Service: newService("s1", "555", v1.ServiceTypeLoadBalancer), Hosts: nodes},
				{Service: newService("s2", "666", v1.ServiceTypeLoadBalancer), Hosts: nodes},
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
				{Service: newService("s1", "888", v1.ServiceTypeLoadBalancer), Hosts: nodes},
				{Service: newService("s3", "999", v1.ServiceTypeLoadBalancer), Hosts: nodes},
			},
		},
		{
			// One service has an external load balancer and one is nil: one call.
			services: []*v1.Service{
				newService("s0", "234", v1.ServiceTypeLoadBalancer),
				nil,
			},
			expectedUpdateCalls: []fakecloud.FakeUpdateBalancerCall{
				{Service: newService("s0", "234", v1.ServiceTypeLoadBalancer), Hosts: nodes},
			},
		},
	}
	for _, item := range table {
		controller, cloud, _ := newController()

		var services []*v1.Service
		services = append(services, item.services...)

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

// getServiceFromPatchAction ensures that services/patch has been called and returns
// the patched service object so it can be validated.
// An error is returned if action not found.
func getServiceFromPatchAction(actions []testingutil.Action) (*v1.Service, error) {
	for _, action := range actions {
		if action.GetVerb() == "patch" && action.GetResource().Resource == "services" {
			patchAction, ok := action.(testingutil.PatchAction)
			if !ok {
				return nil, fmt.Errorf("action %+v is not a patch", action)
			}

			svc := &v1.Service{}
			err := json.Unmarshal(patchAction.GetPatch(), svc)
			if err != nil {
				return nil, err
			}

			return svc, nil
		}
	}

	return nil, fmt.Errorf("expected patch service to be sent to client, got these actions instead: %v", actions)
}

// TODO(a-robinson): Add tests for update/sync/delete.

func TestProcessServiceUpdate(t *testing.T) {

	var controller *ServiceController
	var cloud *fakecloud.FakeCloud
	var client *fake.Clientset

	//A pair of old and new loadbalancer IP address
	oldLBIP := "192.168.1.1"
	newLBIP := "192.168.1.11"

	ensureCreateCalled := func() error {
		if len(cloud.Calls) != 1 && cloud.Calls[0] != "create" {
			return fmt.Errorf("expected 'create' to be called but got %q", cloud.Calls)
		}
		return nil
	}

	testCases := []struct {
		testName   string
		key        string
		updateFn   func(*v1.Service) *v1.Service //Manipulate the structure
		svc        *v1.Service
		expectedFn func(*v1.Service, error) error //Error comparison function
	}{
		{
			testName: "If updating a valid service",
			key:      "validKey",
			svc:      defaultExternalService(),
			updateFn: func(svc *v1.Service) *v1.Service {
				return svc
			},
			expectedFn: func(svc *v1.Service, err error) error {
				if err != nil {
					return err
				}

				return ensureCreateCalled()
			},
		},
		{
			testName: "If Updating Loadbalancer IP",
			key:      "default/sync-test-name",
			svc:      newService("sync-test-name", types.UID("sync-test-uid"), v1.ServiceTypeLoadBalancer),
			updateFn: func(svc *v1.Service) *v1.Service {
				svc.Status.LoadBalancer = v1.LoadBalancerStatus{
					Ingress: []v1.LoadBalancerIngress{
						{IP: oldLBIP},
					},
				}

				cloud.ExternalIP = net.ParseIP(newLBIP)

				newService := svc.DeepCopy()

				// Set the nodes for the cloud's UpdateLoadBalancer call to use.
				nodeIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
				nodeIndexer.Add(&v1.Node{
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{Type: v1.NodeReady, Status: v1.ConditionTrue},
						},
					},
				})
				controller.nodeLister = corelisters.NewNodeLister(nodeIndexer)

				return newService
			},
			expectedFn: func(svc *v1.Service, err error) error {

				if err != nil {
					return err
				}

				if err := ensureCreateCalled(); err != nil {
					return err
				}

				actions := client.Actions()
				patchedSvc, err := getServiceFromPatchAction(actions)
				if err != nil {
					return err
				}

				if len(patchedSvc.Status.LoadBalancer.Ingress) != 1 {
					return fmt.Errorf("expected 1 load balancer ingress but got %+v", patchedSvc.Status.LoadBalancer.Ingress)
				}

				gotIP := patchedSvc.Status.LoadBalancer.Ingress[0].IP
				if gotIP != newLBIP {
					return fmt.Errorf("expected load balancer ip %q but got %q", newLBIP, gotIP)
				}

				return nil
			},
		},
		{
			testName: "If no patch required",
			key:      "default/sync-test-name",
			svc:      newService("sync-test-name", types.UID("sync-test-uid"), v1.ServiceTypeLoadBalancer),
			updateFn: func(svc *v1.Service) *v1.Service {
				svc.Status.LoadBalancer = v1.LoadBalancerStatus{
					Ingress: []v1.LoadBalancerIngress{
						{IP: oldLBIP},
					},
				}

				cloud.ExternalIP = net.ParseIP(oldLBIP)
				return svc
			},
			expectedFn: func(_ *v1.Service, err error) error {
				if err != nil {
					return err
				}

				if err := ensureCreateCalled(); err != nil {
					return err
				}

				actions := client.Actions()
				if len(actions) != 0 {
					return fmt.Errorf("expected no client actions but got %+v", actions)
				}

				return nil
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			controller, cloud, client = newController()
			newSvc := tc.updateFn(tc.svc)
			obtErr := controller.processServiceUpdate(newSvc, tc.key)
			if err := tc.expectedFn(newSvc, obtErr); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestProcessServiceDeletion(t *testing.T) {
	var controller *ServiceController
	var cloud *fakecloud.FakeCloud

	svcKey := "external-balancer"

	testCases := map[string]struct {
		setup      func() *v1.Service
		expectedFn func(svcErr error) error
	}{
		"If an non-existent service is deleted": {
			setup: func() *v1.Service {
				return defaultExternalService()
			},
			expectedFn: func(svcErr error) error {
				return svcErr
			},
		},
		"load balancer deleted successfully": {
			setup: func() *v1.Service {
				return defaultExternalService()
			},
			expectedFn: func(svcErr error) error {
				if svcErr != nil {
					return fmt.Errorf("Error Expected=%v Obtained=%v", nil, svcErr)
				}

				expectedCalls := []string{"delete"}
				if !reflect.DeepEqual(expectedCalls, cloud.Calls) {
					return fmt.Errorf("CloudCalls Obtained=%+v Expected=%+v", cloud.Calls, expectedCalls)
				}

				return nil
			},
		},
		"delete load balancer failed": {
			setup: func() *v1.Service {
				cloud.Err = fmt.Errorf("Error Deleting the Loadbalancer")
				svc := defaultExternalService()
				return svc
			},
			expectedFn: func(svcErr error) error {
				expectedError := "Error Deleting the Loadbalancer"

				if svcErr == nil || svcErr.Error() != expectedError {
					return fmt.Errorf("Error Expected=%v Obtained=%v", expectedError, svcErr)
				}

				return nil
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			controller, cloud, _ = newController()
			svc := tc.setup()
			obtainedErr := controller.processServiceDeletion(svc, svcKey)
			if err := tc.expectedFn(obtainedErr); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestDoesExternalLoadBalancerNeedsUpdate(t *testing.T) {
	var oldSvc, newSvc *v1.Service

	testCases := []struct {
		testName            string //Name of the test case
		updateFn            func() //Function to update the service object
		expectedNeedsUpdate bool   //needsupdate always returns bool

	}{
		{
			testName: "If the service type is changed from LoadBalancer to ClusterIP",
			updateFn: func() {
				oldSvc = defaultExternalService()
				newSvc = defaultExternalService()
				newSvc.Spec.Type = v1.ServiceTypeClusterIP
			},
			expectedNeedsUpdate: true,
		},
		{
			testName: "If service is missing finalizer",
			updateFn: func() {
				oldSvc = defaultExternalService()
				newSvc = defaultExternalService()
			},
			expectedNeedsUpdate: true,
		},
		{
			testName: "If service is marked for deletion without finalizer",
			updateFn: func() {
				oldSvc = defaultExternalService()
				newSvc = defaultExternalService()

				t := metav1.NewTime(time.Now())
				newSvc.DeletionTimestamp = &t
			},
			expectedNeedsUpdate: false,
		},
		{
			testName: "If the Ports are different",
			updateFn: func() {
				oldSvc = defaultExternalService()
				newSvc = defaultExternalService()
				oldSvc.Spec.Ports = []v1.ServicePort{
					{
						Port: 8000,
					},
					{
						Port: 9000,
					},
					{
						Port: 10000,
					},
				}
				newSvc.Spec.Ports = []v1.ServicePort{
					{
						Port: 8001,
					},
					{
						Port: 9001,
					},
					{
						Port: 10001,
					},
				}

			},
			expectedNeedsUpdate: true,
		},
		{
			testName: "If externel ip counts are different",
			updateFn: func() {
				oldSvc = defaultExternalService()
				newSvc = defaultExternalService()
				oldSvc.Spec.ExternalIPs = []string{"old.IP.1"}
				newSvc.Spec.ExternalIPs = []string{"new.IP.1", "new.IP.2"}
			},
			expectedNeedsUpdate: true,
		},
		{
			testName: "If externel ips are different",
			updateFn: func() {
				oldSvc = defaultExternalService()
				newSvc = defaultExternalService()
				oldSvc.Spec.ExternalIPs = []string{"old.IP.1", "old.IP.2"}
				newSvc.Spec.ExternalIPs = []string{"new.IP.1", "new.IP.2"}
			},
			expectedNeedsUpdate: true,
		},
		{
			testName: "If UID is different",
			updateFn: func() {
				oldSvc = defaultExternalService()
				newSvc = defaultExternalService()
				oldSvc.UID = types.UID("UID old")
				newSvc.UID = types.UID("UID new")
			},
			expectedNeedsUpdate: true,
		},
		{
			testName: "If ExternalTrafficPolicy is different",
			updateFn: func() {
				oldSvc = defaultExternalService()
				newSvc = defaultExternalService()
				newSvc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			},
			expectedNeedsUpdate: true,
		},
		{
			testName: "If HealthCheckNodePort is different",
			updateFn: func() {
				oldSvc = defaultExternalService()
				newSvc = defaultExternalService()
				newSvc.Spec.HealthCheckNodePort = 30123
			},
			expectedNeedsUpdate: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			controller, _, _ := newController()
			tc.updateFn()
			obtainedResult := controller.needsUpdate(oldSvc, newSvc)
			if obtainedResult != tc.expectedNeedsUpdate {
				t.Errorf("should have returned %v but returned %v", tc.expectedNeedsUpdate, obtainedResult)
			}
		})
	}
}

//Test a utility functions as its not easy to unit test nodeSyncLoop directly
func TestNodeSlicesEqualForLB(t *testing.T) {
	numNodes := 10
	nArray := make([]*v1.Node, numNodes)
	mArray := make([]*v1.Node, numNodes)
	for i := 0; i < numNodes; i++ {
		nArray[i] = &v1.Node{}
		nArray[i].Name = fmt.Sprintf("node%d", i)
	}
	for i := 0; i < numNodes; i++ {
		mArray[i] = &v1.Node{}
		mArray[i].Name = fmt.Sprintf("node%d", i+1)
	}

	if !nodeSlicesEqualForLB(nArray, nArray) {
		t.Errorf("nodeSlicesEqualForLB() Expected=true Obtained=false")
	}
	if nodeSlicesEqualForLB(nArray, mArray) {
		t.Errorf("nodeSlicesEqualForLB() Expected=false Obtained=true")
	}
}
