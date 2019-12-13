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
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	fakecloud "k8s.io/cloud-provider/fake"
	servicehelper "k8s.io/cloud-provider/service/helpers"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/controller"
)

const region = "us-central"

func newService(name string, uid types.UID, serviceType v1.ServiceType) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
			UID:       uid,
			SelfLink:  "/api/v1/namespaces/default/services/" + name,
		},
		Spec: v1.ServiceSpec{
			Type: serviceType,
		},
	}
}

//Wrap newService so that you don't have to call default arguments again and again.
func defaultExternalService() *v1.Service {
	return newService("external-balancer", types.UID("123"), v1.ServiceTypeLoadBalancer)
}

func alwaysReady() bool { return true }

func newController() (*Controller, *fakecloud.Cloud, *fake.Clientset) {
	cloud := &fakecloud.Cloud{}
	cloud.Region = region

	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	serviceInformer := informerFactory.Core().V1().Services()
	nodeInformer := informerFactory.Core().V1().Nodes()

	controller, _ := New(cloud, client, serviceInformer, nodeInformer, "test-cluster")
	controller.nodeListerSynced = alwaysReady
	controller.serviceListerSynced = alwaysReady
	controller.eventRecorder = record.NewFakeRecorder(100)

	cloud.Calls = nil     // ignore any cloud calls made in init()
	client.ClearActions() // ignore any client calls made in init()

	return controller, cloud, client
}

// TODO(@MrHohn): Verify the end state when below issue is resolved:
// https://github.com/kubernetes/client-go/issues/607
func TestSyncLoadBalancerIfNeeded(t *testing.T) {
	testCases := []struct {
		desc                 string
		service              *v1.Service
		lbExists             bool
		expectOp             loadBalancerOperation
		expectCreateAttempt  bool
		expectDeleteAttempt  bool
		expectPatchStatus    bool
		expectPatchFinalizer bool
	}{
		{
			desc: "service doesn't want LB",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "no-external-balancer",
					Namespace: "default",
				},
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeClusterIP,
				},
			},
			expectOp:          deleteLoadBalancer,
			expectPatchStatus: false,
		},
		{
			desc: "service no longer wants LB",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "no-external-balancer",
					Namespace: "default",
				},
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeClusterIP,
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: "8.8.8.8"},
						},
					},
				},
			},
			lbExists:            true,
			expectOp:            deleteLoadBalancer,
			expectDeleteAttempt: true,
			expectPatchStatus:   true,
		},
		{
			desc: "udp service that wants LB",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "udp-service",
					Namespace: "default",
					SelfLink:  "/api/v1/namespaces/default/services/udp-service",
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:     80,
						Protocol: v1.ProtocolUDP,
					}},
					Type: v1.ServiceTypeLoadBalancer,
				},
			},
			expectOp:             ensureLoadBalancer,
			expectCreateAttempt:  true,
			expectPatchStatus:    true,
			expectPatchFinalizer: true,
		},
		{
			desc: "tcp service that wants LB",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "basic-service1",
					Namespace: "default",
					SelfLink:  "/api/v1/namespaces/default/services/basic-service1",
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:     80,
						Protocol: v1.ProtocolTCP,
					}},
					Type: v1.ServiceTypeLoadBalancer,
				},
			},
			expectOp:             ensureLoadBalancer,
			expectCreateAttempt:  true,
			expectPatchStatus:    true,
			expectPatchFinalizer: true,
		},
		{
			desc: "sctp service that wants LB",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "sctp-service",
					Namespace: "default",
					SelfLink:  "/api/v1/namespaces/default/services/sctp-service",
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:     80,
						Protocol: v1.ProtocolSCTP,
					}},
					Type: v1.ServiceTypeLoadBalancer,
				},
			},
			expectOp:             ensureLoadBalancer,
			expectCreateAttempt:  true,
			expectPatchStatus:    true,
			expectPatchFinalizer: true,
		},
		// Finalizer test cases below.
		{
			desc: "service with finalizer that no longer wants LB",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "no-external-balancer",
					Namespace:  "default",
					Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
				},
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeClusterIP,
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: "8.8.8.8"},
						},
					},
				},
			},
			lbExists:             true,
			expectOp:             deleteLoadBalancer,
			expectDeleteAttempt:  true,
			expectPatchStatus:    true,
			expectPatchFinalizer: true,
		},
		{
			desc: "service that needs cleanup",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "basic-service1",
					Namespace: "default",
					SelfLink:  "/api/v1/namespaces/default/services/basic-service1",
					DeletionTimestamp: &metav1.Time{
						Time: time.Now(),
					},
					Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:     80,
						Protocol: v1.ProtocolTCP,
					}},
					Type: v1.ServiceTypeLoadBalancer,
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: "8.8.8.8"},
						},
					},
				},
			},
			lbExists:             true,
			expectOp:             deleteLoadBalancer,
			expectDeleteAttempt:  true,
			expectPatchStatus:    true,
			expectPatchFinalizer: true,
		},
		{
			desc: "service without finalizer that wants LB",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "basic-service1",
					Namespace: "default",
					SelfLink:  "/api/v1/namespaces/default/services/basic-service1",
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:     80,
						Protocol: v1.ProtocolTCP,
					}},
					Type: v1.ServiceTypeLoadBalancer,
				},
			},
			expectOp:             ensureLoadBalancer,
			expectCreateAttempt:  true,
			expectPatchStatus:    true,
			expectPatchFinalizer: true,
		},
		{
			desc: "service with finalizer that wants LB",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "basic-service1",
					Namespace:  "default",
					SelfLink:   "/api/v1/namespaces/default/services/basic-service1",
					Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:     80,
						Protocol: v1.ProtocolTCP,
					}},
					Type: v1.ServiceTypeLoadBalancer,
				},
			},
			expectOp:             ensureLoadBalancer,
			expectCreateAttempt:  true,
			expectPatchStatus:    true,
			expectPatchFinalizer: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			controller, cloud, client := newController()
			cloud.Exists = tc.lbExists
			key := fmt.Sprintf("%s/%s", tc.service.Namespace, tc.service.Name)
			if _, err := client.CoreV1().Services(tc.service.Namespace).Create(tc.service); err != nil {
				t.Fatalf("Failed to prepare service %s for testing: %v", key, err)
			}
			client.ClearActions()

			op, err := controller.syncLoadBalancerIfNeeded(tc.service, key)
			if err != nil {
				t.Errorf("Got error: %v, want nil", err)
			}
			if op != tc.expectOp {
				t.Errorf("Got operation %v, want %v", op, tc.expectOp)
			}
			// Capture actions from test so it won't be messed up.
			actions := client.Actions()

			if !tc.expectCreateAttempt && !tc.expectDeleteAttempt {
				if len(cloud.Calls) > 0 {
					t.Errorf("Unexpected cloud provider calls: %v", cloud.Calls)
				}
				if len(actions) > 0 {
					t.Errorf("Unexpected client actions: %v", actions)
				}
				return
			}

			if tc.expectCreateAttempt {
				createCallFound := false
				for _, call := range cloud.Calls {
					if call == "create" {
						createCallFound = true
					}
				}
				if !createCallFound {
					t.Errorf("Got no create call for load balancer, expected one")
				}
				// TODO(@MrHohn): Clean up the awkward pattern here.
				var balancer *fakecloud.Balancer
				for k := range cloud.Balancers {
					if balancer == nil {
						b := cloud.Balancers[k]
						balancer = &b
					} else {
						t.Errorf("Got load balancer %v, expected one to be created", cloud.Balancers)
						break
					}
				}
				if balancer == nil {
					t.Errorf("Got no load balancer, expected one to be created")
				} else if balancer.Name != controller.balancer.GetLoadBalancerName(context.Background(), "", tc.service) ||
					balancer.Region != region ||
					balancer.Ports[0].Port != tc.service.Spec.Ports[0].Port {
					t.Errorf("Created load balancer has incorrect parameters: %v", balancer)
				}
			}
			if tc.expectDeleteAttempt {
				deleteCallFound := false
				for _, call := range cloud.Calls {
					if call == "delete" {
						deleteCallFound = true
					}
				}
				if !deleteCallFound {
					t.Errorf("Got no delete call for load balancer, expected one")
				}
			}

			expectNumPatches := 0
			if tc.expectPatchStatus {
				expectNumPatches++
			}
			if tc.expectPatchFinalizer {
				expectNumPatches++
			}
			numPatches := 0
			for _, action := range actions {
				if action.Matches("patch", "services") {
					numPatches++
				}
			}
			if numPatches != expectNumPatches {
				t.Errorf("Got %d patches, expect %d instead. Actions: %v", numPatches, expectNumPatches, actions)
			}
		})
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
		expectedUpdateCalls []fakecloud.UpdateBalancerCall
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
			expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
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
			expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
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
			expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
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
			expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
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

func TestProcessServiceCreateOrUpdate(t *testing.T) {
	controller, _, client := newController()

	//A pair of old and new loadbalancer IP address
	oldLBIP := "192.168.1.1"
	newLBIP := "192.168.1.11"

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

				controller.cache.getOrCreate("validKey")
				return svc

			},
			expectedFn: func(svc *v1.Service, err error) error {
				return err
			},
		},
		{
			testName: "If Updating Loadbalancer IP",
			key:      "default/sync-test-name",
			svc:      newService("sync-test-name", types.UID("sync-test-uid"), v1.ServiceTypeLoadBalancer),
			updateFn: func(svc *v1.Service) *v1.Service {

				svc.Spec.LoadBalancerIP = oldLBIP

				keyExpected := svc.GetObjectMeta().GetNamespace() + "/" + svc.GetObjectMeta().GetName()
				controller.enqueueService(svc)
				cachedServiceTest := controller.cache.getOrCreate(keyExpected)
				cachedServiceTest.state = svc
				controller.cache.set(keyExpected, cachedServiceTest)

				keyGot, quit := controller.queue.Get()
				if quit {
					t.Fatalf("get no queue element")
				}
				if keyExpected != keyGot.(string) {
					t.Fatalf("get service key error, expected: %s, got: %s", keyExpected, keyGot.(string))
				}

				newService := svc.DeepCopy()

				newService.Spec.LoadBalancerIP = newLBIP
				return newService

			},
			expectedFn: func(svc *v1.Service, err error) error {

				if err != nil {
					return err
				}

				keyExpected := svc.GetObjectMeta().GetNamespace() + "/" + svc.GetObjectMeta().GetName()

				cachedServiceGot, exist := controller.cache.get(keyExpected)
				if !exist {
					return fmt.Errorf("update service error, queue should contain service: %s", keyExpected)
				}
				if cachedServiceGot.state.Spec.LoadBalancerIP != newLBIP {
					return fmt.Errorf("update LoadBalancerIP error, expected: %s, got: %s", newLBIP, cachedServiceGot.state.Spec.LoadBalancerIP)
				}
				return nil
			},
		},
	}

	for _, tc := range testCases {
		newSvc := tc.updateFn(tc.svc)
		if _, err := client.CoreV1().Services(tc.svc.Namespace).Create(tc.svc); err != nil {
			t.Fatalf("Failed to prepare service %s for testing: %v", tc.key, err)
		}
		obtErr := controller.processServiceCreateOrUpdate(newSvc, tc.key)
		if err := tc.expectedFn(newSvc, obtErr); err != nil {
			t.Errorf("%v processServiceCreateOrUpdate() %v", tc.testName, err)
		}
	}

}

// TestProcessServiceCreateOrUpdateK8sError tests processServiceCreateOrUpdate
// with various kubernetes errors when patching status.
func TestProcessServiceCreateOrUpdateK8sError(t *testing.T) {
	svcName := "svc-k8s-err"
	conflictErr := apierrors.NewConflict(schema.GroupResource{}, svcName, errors.New("object conflict"))
	notFoundErr := apierrors.NewNotFound(schema.GroupResource{}, svcName)

	testCases := []struct {
		desc      string
		k8sErr    error
		expectErr error
	}{
		{
			desc:      "conflict error",
			k8sErr:    conflictErr,
			expectErr: fmt.Errorf("failed to update load balancer status: %v", conflictErr),
		},
		{
			desc:      "not found error",
			k8sErr:    notFoundErr,
			expectErr: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			svc := newService(svcName, types.UID("123"), v1.ServiceTypeLoadBalancer)
			// Preset finalizer so k8s error only happens when patching status.
			svc.Finalizers = []string{servicehelper.LoadBalancerCleanupFinalizer}
			controller, _, client := newController()
			client.PrependReactor("patch", "services", func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, tc.k8sErr
			})

			if err := controller.processServiceCreateOrUpdate(svc, svcName); !reflect.DeepEqual(err, tc.expectErr) {
				t.Fatalf("processServiceCreateOrUpdate() = %v, want %v", err, tc.expectErr)
			}
			if tc.expectErr == nil {
				return
			}

			errMsg := "Error syncing load balancer"
			if gotEvent := func() bool {
				events := controller.eventRecorder.(*record.FakeRecorder).Events
				for len(events) > 0 {
					e := <-events
					if strings.Contains(e, errMsg) {
						return true
					}
				}
				return false
			}(); !gotEvent {
				t.Errorf("processServiceCreateOrUpdate() = can't find sync error event, want event contains %q", errMsg)
			}
		})
	}

}

func TestSyncService(t *testing.T) {

	var controller *Controller

	testCases := []struct {
		testName   string
		key        string
		updateFn   func()            //Function to manipulate the controller element to simulate error
		expectedFn func(error) error //Expected function if returns nil then test passed, failed otherwise
	}{
		{
			testName: "if an invalid service name is synced",
			key:      "invalid/key/string",
			updateFn: func() {
				controller, _, _ = newController()
			},
			expectedFn: func(e error) error {
				//TODO: should find a way to test for dependent package errors in such a way that it won't break
				//TODO:	our tests, currently we only test if there is an error.
				//Error should be unexpected key format: "invalid/key/string"
				expectedError := fmt.Sprintf("unexpected key format: %q", "invalid/key/string")
				if e == nil || e.Error() != expectedError {
					return fmt.Errorf("Expected=unexpected key format: %q, Obtained=%v", "invalid/key/string", e)
				}
				return nil
			},
		},
		/* We cannot open this test case as syncService(key) currently runtime.HandleError(err) and suppresses frequently occurring errors
		{
			testName: "if an invalid service is synced",
			key: "somethingelse",
			updateFn: func() {
				controller, _, _ = newController()
				srv := controller.cache.getOrCreate("external-balancer")
				srv.state = defaultExternalService()
			},
			expectedErr: fmt.Errorf("service somethingelse not in cache even though the watcher thought it was. Ignoring the deletion."),
		},
		*/

		//TODO: see if we can add a test for valid but error throwing service, its difficult right now because synCService() currently runtime.HandleError
		{
			testName: "if valid service",
			key:      "external-balancer",
			updateFn: func() {
				testSvc := defaultExternalService()
				controller, _, _ = newController()
				controller.enqueueService(testSvc)
				svc := controller.cache.getOrCreate("external-balancer")
				svc.state = testSvc
			},
			expectedFn: func(e error) error {
				//error should be nil
				if e != nil {
					return fmt.Errorf("Expected=nil, Obtained=%v", e)
				}
				return nil
			},
		},
	}

	for _, tc := range testCases {

		tc.updateFn()
		obtainedErr := controller.syncService(tc.key)

		//expected matches obtained ??.
		if exp := tc.expectedFn(obtainedErr); exp != nil {
			t.Errorf("%v Error:%v", tc.testName, exp)
		}

		//Post processing, the element should not be in the sync queue.
		_, exist := controller.cache.get(tc.key)
		if exist {
			t.Fatalf("%v working Queue should be empty, but contains %s", tc.testName, tc.key)
		}
	}
}

func TestProcessServiceDeletion(t *testing.T) {

	var controller *Controller
	var cloud *fakecloud.Cloud
	// Add a global svcKey name
	svcKey := "external-balancer"

	testCases := []struct {
		testName   string
		updateFn   func(*Controller)        // Update function used to manipulate srv and controller values
		expectedFn func(svcErr error) error // Function to check if the returned value is expected
	}{
		{
			testName: "If a non-existent service is deleted",
			updateFn: func(controller *Controller) {
				// Does not do anything
			},
			expectedFn: func(svcErr error) error {
				return svcErr
			},
		},
		{
			testName: "If cloudprovided failed to delete the service",
			updateFn: func(controller *Controller) {

				svc := controller.cache.getOrCreate(svcKey)
				svc.state = defaultExternalService()
				cloud.Err = fmt.Errorf("error Deleting the Loadbalancer")

			},
			expectedFn: func(svcErr error) error {

				expectedError := "error Deleting the Loadbalancer"

				if svcErr == nil || svcErr.Error() != expectedError {
					return fmt.Errorf("Expected=%v Obtained=%v", expectedError, svcErr)
				}

				return nil
			},
		},
		{
			testName: "If delete was successful",
			updateFn: func(controller *Controller) {

				testSvc := defaultExternalService()
				controller.enqueueService(testSvc)
				svc := controller.cache.getOrCreate(svcKey)
				svc.state = testSvc
				controller.cache.set(svcKey, svc)

			},
			expectedFn: func(svcErr error) error {
				if svcErr != nil {
					return fmt.Errorf("Expected=nil Obtained=%v", svcErr)
				}

				// It should no longer be in the workqueue.
				_, exist := controller.cache.get(svcKey)
				if exist {
					return fmt.Errorf("delete service error, queue should not contain service: %s any more", svcKey)
				}

				return nil
			},
		},
	}

	for _, tc := range testCases {
		//Create a new controller.
		controller, cloud, _ = newController()
		tc.updateFn(controller)
		obtainedErr := controller.processServiceDeletion(svcKey)
		if err := tc.expectedFn(obtainedErr); err != nil {
			t.Errorf("%v processServiceDeletion() %v", tc.testName, err)
		}
	}

}

// Test cases:
// index    finalizer    timestamp    wantLB  |  clean-up
//   0         0           0            0     |   false    (No finalizer, no clean up)
//   1         0           0            1     |   false    (Ignored as same with case 0)
//   2         0           1            0     |   false    (Ignored as same with case 0)
//   3         0           1            1     |   false    (Ignored as same with case 0)
//   4         1           0            0     |   true
//   5         1           0            1     |   false
//   6         1           1            0     |   true    (Service is deleted, needs clean up)
//   7         1           1            1     |   true    (Ignored as same with case 6)
func TestNeedsCleanup(t *testing.T) {
	testCases := []struct {
		desc               string
		svc                *v1.Service
		expectNeedsCleanup bool
	}{
		{
			desc:               "service without finalizer",
			svc:                &v1.Service{},
			expectNeedsCleanup: false,
		},
		{
			desc: "service with finalizer without timestamp without LB",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
				},
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeNodePort,
				},
			},
			expectNeedsCleanup: true,
		},
		{
			desc: "service with finalizer without timestamp with LB",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
				},
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeLoadBalancer,
				},
			},
			expectNeedsCleanup: false,
		},
		{
			desc: "service with finalizer with timestamp",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
					DeletionTimestamp: &metav1.Time{
						Time: time.Now(),
					},
				},
			},
			expectNeedsCleanup: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			if gotNeedsCleanup := needsCleanup(tc.svc); gotNeedsCleanup != tc.expectNeedsCleanup {
				t.Errorf("needsCleanup() = %t, want %t", gotNeedsCleanup, tc.expectNeedsCleanup)
			}
		})
	}

}

func TestNeedsUpdate(t *testing.T) {

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
		{
			testName: "If TargetGroup is different 1",
			updateFn: func() {
				oldSvc = &v1.Service{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "tcp-service",
						Namespace: "default",
					},
					Spec: v1.ServiceSpec{
						Ports: []v1.ServicePort{{
							Port:       80,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.Parse("20"),
						}},
						Type: v1.ServiceTypeLoadBalancer,
					},
				}
				newSvc = oldSvc.DeepCopy()
				newSvc.Spec.Ports[0].TargetPort = intstr.Parse("21")
			},
			expectedNeedsUpdate: true,
		},
		{
			testName: "If TargetGroup is different 2",
			updateFn: func() {
				oldSvc = &v1.Service{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "tcp-service",
						Namespace: "default",
					},
					Spec: v1.ServiceSpec{
						Ports: []v1.ServicePort{{
							Port:       80,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.Parse("22"),
						}},
						Type: v1.ServiceTypeLoadBalancer,
					},
				}
				newSvc = oldSvc.DeepCopy()
				newSvc.Spec.Ports[0].TargetPort = intstr.Parse("dns")
			},
			expectedNeedsUpdate: true,
		},
	}

	controller, _, _ := newController()
	for _, tc := range testCases {
		tc.updateFn()
		obtainedResult := controller.needsUpdate(oldSvc, newSvc)
		if obtainedResult != tc.expectedNeedsUpdate {
			t.Errorf("%v needsUpdate() should have returned %v but returned %v", tc.testName, tc.expectedNeedsUpdate, obtainedResult)
		}
	}
}

//All the test cases for ServiceCache uses a single cache, these below test cases should be run in order,
//as tc1 (addCache would add elements to the cache)
//and tc2 (delCache would remove element from the cache without it adding automatically)
//Please keep this in mind while adding new test cases.
func TestServiceCache(t *testing.T) {

	//ServiceCache a common service cache for all the test cases
	sc := &serviceCache{serviceMap: make(map[string]*cachedService)}

	testCases := []struct {
		testName     string
		setCacheFn   func()
		checkCacheFn func() error
	}{
		{
			testName: "Add",
			setCacheFn: func() {
				cS := sc.getOrCreate("addTest")
				cS.state = defaultExternalService()
			},
			checkCacheFn: func() error {
				//There must be exactly one element
				if len(sc.serviceMap) != 1 {
					return fmt.Errorf("Expected=1 Obtained=%d", len(sc.serviceMap))
				}
				return nil
			},
		},
		{
			testName: "Del",
			setCacheFn: func() {
				sc.delete("addTest")

			},
			checkCacheFn: func() error {
				//Now it should have no element
				if len(sc.serviceMap) != 0 {
					return fmt.Errorf("Expected=0 Obtained=%d", len(sc.serviceMap))
				}
				return nil
			},
		},
		{
			testName: "Set and Get",
			setCacheFn: func() {
				sc.set("addTest", &cachedService{state: defaultExternalService()})
			},
			checkCacheFn: func() error {
				//Now it should have one element
				Cs, bool := sc.get("addTest")
				if !bool {
					return fmt.Errorf("is Available Expected=true Obtained=%v", bool)
				}
				if Cs == nil {
					return fmt.Errorf("cachedService expected:non-nil Obtained=nil")
				}
				return nil
			},
		},
		{
			testName: "ListKeys",
			setCacheFn: func() {
				//Add one more entry here
				sc.set("addTest1", &cachedService{state: defaultExternalService()})
			},
			checkCacheFn: func() error {
				//It should have two elements
				keys := sc.ListKeys()
				if len(keys) != 2 {
					return fmt.Errorf("elements Expected=2 Obtained=%v", len(keys))
				}
				return nil
			},
		},
		{
			testName:   "GetbyKeys",
			setCacheFn: nil, //Nothing to set
			checkCacheFn: func() error {
				//It should have two elements
				svc, isKey, err := sc.GetByKey("addTest")
				if svc == nil || isKey == false || err != nil {
					return fmt.Errorf("Expected(non-nil, true, nil) Obtained(%v,%v,%v)", svc, isKey, err)
				}
				return nil
			},
		},
		{
			testName:   "allServices",
			setCacheFn: nil, //Nothing to set
			checkCacheFn: func() error {
				//It should return two elements
				svcArray := sc.allServices()
				if len(svcArray) != 2 {
					return fmt.Errorf("Expected(2) Obtained(%v)", len(svcArray))
				}
				return nil
			},
		},
	}

	for _, tc := range testCases {
		if tc.setCacheFn != nil {
			tc.setCacheFn()
		}
		if err := tc.checkCacheFn(); err != nil {
			t.Errorf("%v returned %v", tc.testName, err)
		}
	}
}

//Test a utility functions as it's not easy to unit test nodeSyncLoop directly
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

// TODO(@MrHohn): Verify the end state when below issue is resolved:
// https://github.com/kubernetes/client-go/issues/607
func TestAddFinalizer(t *testing.T) {
	testCases := []struct {
		desc        string
		svc         *v1.Service
		expectPatch bool
	}{
		{
			desc: "no-op add finalizer",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "test-patch-finalizer",
					Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
				},
			},
			expectPatch: false,
		},
		{
			desc: "add finalizer",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-patch-finalizer",
				},
			},
			expectPatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			c := fake.NewSimpleClientset()
			s := &Controller{
				kubeClient: c,
			}
			if _, err := s.kubeClient.CoreV1().Services(tc.svc.Namespace).Create(tc.svc); err != nil {
				t.Fatalf("Failed to prepare service for testing: %v", err)
			}
			if err := s.addFinalizer(tc.svc); err != nil {
				t.Fatalf("addFinalizer() = %v, want nil", err)
			}
			patchActionFound := false
			for _, action := range c.Actions() {
				if action.Matches("patch", "services") {
					patchActionFound = true
				}
			}
			if patchActionFound != tc.expectPatch {
				t.Errorf("Got patchActionFound = %t, want %t", patchActionFound, tc.expectPatch)
			}
		})
	}
}

// TODO(@MrHohn): Verify the end state when below issue is resolved:
// https://github.com/kubernetes/client-go/issues/607
func TestRemoveFinalizer(t *testing.T) {
	testCases := []struct {
		desc        string
		svc         *v1.Service
		expectPatch bool
	}{
		{
			desc: "no-op remove finalizer",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-patch-finalizer",
				},
			},
			expectPatch: false,
		},
		{
			desc: "remove finalizer",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "test-patch-finalizer",
					Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
				},
			},
			expectPatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			c := fake.NewSimpleClientset()
			s := &Controller{
				kubeClient: c,
			}
			if _, err := s.kubeClient.CoreV1().Services(tc.svc.Namespace).Create(tc.svc); err != nil {
				t.Fatalf("Failed to prepare service for testing: %v", err)
			}
			if err := s.removeFinalizer(tc.svc); err != nil {
				t.Fatalf("removeFinalizer() = %v, want nil", err)
			}
			patchActionFound := false
			for _, action := range c.Actions() {
				if action.Matches("patch", "services") {
					patchActionFound = true
				}
			}
			if patchActionFound != tc.expectPatch {
				t.Errorf("Got patchActionFound = %t, want %t", patchActionFound, tc.expectPatch)
			}
		})
	}
}

// TODO(@MrHohn): Verify the end state when below issue is resolved:
// https://github.com/kubernetes/client-go/issues/607
func TestPatchStatus(t *testing.T) {
	testCases := []struct {
		desc        string
		svc         *v1.Service
		newStatus   *v1.LoadBalancerStatus
		expectPatch bool
	}{
		{
			desc: "no-op add status",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-patch-status",
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: "8.8.8.8"},
						},
					},
				},
			},
			newStatus: &v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "8.8.8.8"},
				},
			},
			expectPatch: false,
		},
		{
			desc: "add status",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-patch-status",
				},
				Status: v1.ServiceStatus{},
			},
			newStatus: &v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "8.8.8.8"},
				},
			},
			expectPatch: true,
		},
		{
			desc: "no-op clear status",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-patch-status",
				},
				Status: v1.ServiceStatus{},
			},
			newStatus:   &v1.LoadBalancerStatus{},
			expectPatch: false,
		},
		{
			desc: "clear status",
			svc: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-patch-status",
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: "8.8.8.8"},
						},
					},
				},
			},
			newStatus:   &v1.LoadBalancerStatus{},
			expectPatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			c := fake.NewSimpleClientset()
			s := &Controller{
				kubeClient: c,
			}
			if _, err := s.kubeClient.CoreV1().Services(tc.svc.Namespace).Create(tc.svc); err != nil {
				t.Fatalf("Failed to prepare service for testing: %v", err)
			}
			if err := s.patchStatus(tc.svc, &tc.svc.Status.LoadBalancer, tc.newStatus); err != nil {
				t.Fatalf("patchStatus() = %v, want nil", err)
			}
			patchActionFound := false
			for _, action := range c.Actions() {
				if action.Matches("patch", "services") {
					patchActionFound = true
				}
			}
			if patchActionFound != tc.expectPatch {
				t.Errorf("Got patchActionFound = %t, want %t", patchActionFound, tc.expectPatch)
			}
		})
	}
}

func Test_getNodeConditionPredicate(t *testing.T) {
	validNodeStatus := v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: "Test"}}}
	tests := []struct {
		name string

		enableExclusion bool
		enableLegacy    bool
		input           *v1.Node
		want            bool
	}{
		{want: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{}}}},
		{want: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeRoleMaster: ""}}}},
		{want: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeRoleExcludeBalancer: ""}}}},
		{want: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelAlphaNodeRoleExcludeBalancer: ""}}}},

		{want: true, enableExclusion: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeRoleMaster: ""}}}},
		{want: true, enableLegacy: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeRoleExcludeBalancer: ""}}}},

		{want: false, enableLegacy: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeRoleMaster: ""}}}},
		{want: false, enableExclusion: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelAlphaNodeRoleExcludeBalancer: ""}}}},
		{want: false, enableExclusion: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeRoleExcludeBalancer: ""}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, serviceNodeExclusionFeature, tt.enableExclusion)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, legacyNodeRoleBehaviorFeature, tt.enableLegacy)()

			if result := getNodeConditionPredicate()(tt.input); result != tt.want {
				t.Errorf("getNodeConditionPredicate() = %v, want %v", result, tt.want)
			}
		})
	}
}

func TestListWithPredicate(t *testing.T) {
	fakeInformerFactory := informers.NewSharedInformerFactory(&fake.Clientset{}, 0*time.Second)
	var nodes []*v1.Node
	for i := 0; i < 5; i++ {
		var phase v1.NodePhase
		if i%2 == 0 {
			phase = v1.NodePending
		} else {
			phase = v1.NodeRunning
		}
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("node-%d", i),
			},
			Status: v1.NodeStatus{
				Phase: phase,
			},
		}
		nodes = append(nodes, node)
		fakeInformerFactory.Core().V1().Nodes().Informer().GetStore().Add(node)
	}

	tests := []struct {
		name      string
		predicate NodeConditionPredicate
		expect    []*v1.Node
	}{
		{
			name: "ListWithPredicate filter Running node",
			predicate: func(node *v1.Node) bool {
				return node.Status.Phase == v1.NodeRunning
			},
			expect: []*v1.Node{nodes[1], nodes[3]},
		},
		{
			name: "ListWithPredicate filter Pending node",
			predicate: func(node *v1.Node) bool {
				return node.Status.Phase == v1.NodePending
			},
			expect: []*v1.Node{nodes[0], nodes[2], nodes[4]},
		},
		{
			name: "ListWithPredicate filter Terminated node",
			predicate: func(node *v1.Node) bool {
				return node.Status.Phase == v1.NodeTerminated
			},
			expect: nil,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			get, err := listWithPredicate(fakeInformerFactory.Core().V1().Nodes().Lister(), test.predicate)
			sort.Slice(get, func(i, j int) bool {
				return get[i].Name < get[j].Name
			})
			if err != nil {
				t.Errorf("Error from ListWithPredicate: %v", err)
			} else if !reflect.DeepEqual(get, test.expect) {
				t.Errorf("Expect nodes %v, but got %v", test.expect, get)
			}
		})
	}
}
