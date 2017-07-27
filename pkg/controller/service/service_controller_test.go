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
	"fmt"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/testapi"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller"
)

const region = "us-central"

func newService(name string, uid types.UID, serviceType v1.ServiceType) *v1.Service {
	return &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default", UID: uid, SelfLink: testapi.Default.SelfLink("services", name)}, Spec: v1.ServiceSpec{Type: serviceType}}
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

func TestProcessServiceUpdate(t *testing.T) {

	var controller *ServiceController
	var cloud *fakecloud.FakeCloud

	//A pair of old and new loadbalancer IP address
	oldLBIP := "192.168.1.1"
	newLBIP := "192.168.1.11"

	testCases := []struct {
		testName   string
		key        string
		updateFn   func(*v1.Service) *v1.Service //Manipulate the structure
		svc        *v1.Service
		expectedFn func(*v1.Service, error, time.Duration) error //Error comparision function
	}{
		{
			testName: "If updating a valid service",
			key:      "validKey",
			svc:      defaultExternalService(),
			updateFn: func(svc *v1.Service) *v1.Service {

				controller, cloud, _ = newController()
				controller.cache.getOrCreate("validKey")
				return svc

			},
			expectedFn: func(svc *v1.Service, err error, retryDuration time.Duration) error {

				if err != nil {
					return err
				}
				if retryDuration != doNotRetry {
					return fmt.Errorf("retryDuration Expected=%v Obtained=%v", doNotRetry, retryDuration)
				}
				return nil
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

				keyGot, quit := controller.workingQueue.Get()
				if quit {
					t.Fatalf("get no workingQueue element")
				}
				if keyExpected != keyGot.(string) {
					t.Fatalf("get service key error, expected: %s, got: %s", keyExpected, keyGot.(string))
				}

				copy, err := scheme.Scheme.DeepCopy(svc)
				if err != nil {
					t.Fatalf("copy service error: %v", err)
				}
				newService := copy.(*v1.Service)

				newService.Spec.LoadBalancerIP = newLBIP
				return newService

			},
			expectedFn: func(svc *v1.Service, err error, retryDuration time.Duration) error {

				if err != nil {
					return err
				}
				if retryDuration != doNotRetry {
					return fmt.Errorf("retryDuration Expected=%v Obtained=%v", doNotRetry, retryDuration)
				}

				keyExpected := svc.GetObjectMeta().GetNamespace() + "/" + svc.GetObjectMeta().GetName()

				cachedServiceGot, exist := controller.cache.get(keyExpected)
				if !exist {
					return fmt.Errorf("update service error, workingQueue should contain service: %s", keyExpected)
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
		svcCache := controller.cache.getOrCreate(tc.key)
		obtErr, retryDuration := controller.processServiceUpdate(svcCache, newSvc, tc.key)
		if err := tc.expectedFn(newSvc, obtErr, retryDuration); err != nil {
			t.Errorf("%v processServiceUpdate() %v", tc.testName, err)
		}
	}

}

func TestSyncService(t *testing.T) {

	var controller *ServiceController
	var cloud *fakecloud.FakeCloud

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
				controller, cloud, _ = newController()

			},
			expectedFn: func(e error) error {
				//TODO: Expected error is of the format fmt.Errorf("unexpected key format: %q", "invalid/key/string"),
				//TODO: should find a way to test for dependent package errors in such a way that it wont break
				//TODO:	our tests, currently we only test if there is an error.
				//Error should be non-nil
				if e == nil {
					return fmt.Errorf("Expected=unexpected key format: %q, Obtained=nil", "invalid/key/string")
				}
				return nil
			},
		},
		/* We cannot open this test case as syncService(key) currently runtime.HandleError(err) and suppresses frequently occurring errors
		{
			testName: "if an invalid service is synced",
			key: "somethingelse",
			updateFn: func() {
				controller, cloud, _ = newController()
				srv := controller.cache.getOrCreate("external-balancer")
				srv.state = defaultExternalService()
			},
			expectedErr: fmt.Errorf("Service somethingelse not in cache even though the watcher thought it was. Ignoring the deletion."),
		},
		*/

		//TODO: see if we can add a test for valid but error throwing service, its difficult right now because synCService() currently runtime.HandleError
		{
			testName: "if valid service",
			key:      "external-balancer",
			updateFn: func() {
				testSvc := defaultExternalService()
				controller, cloud, _ = newController()
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

	var controller *ServiceController
	var cloud *fakecloud.FakeCloud
	//Add a global svcKey name
	svcKey := "external-balancer"

	testCases := []struct {
		testName   string
		updateFn   func(*ServiceController)                              //Update function used to manupulate srv and controller values
		expectedFn func(svcErr error, retryDuration time.Duration) error //Function to check if the returned value is expected
	}{
		{
			testName: "If an non-existant service is deleted",
			updateFn: func(controller *ServiceController) {
				//Does not do anything
			},
			expectedFn: func(svcErr error, retryDuration time.Duration) error {

				expectedError := "Service external-balancer not in cache even though the watcher thought it was. Ignoring the deletion."
				if svcErr == nil || svcErr.Error() != expectedError {
					//cannot be nil or Wrong error message
					return fmt.Errorf("Expected=%v Obtained=%v", expectedError, svcErr)
				}

				if retryDuration != doNotRetry {
					//Retry duration should match
					return fmt.Errorf("RetryDuration Expected=%v Obtained=%v", doNotRetry, retryDuration)
				}

				return nil
			},
		},
		{
			testName: "If cloudprovided failed to delete the service",
			updateFn: func(controller *ServiceController) {

				svc := controller.cache.getOrCreate(svcKey)
				svc.state = defaultExternalService()
				cloud.Err = fmt.Errorf("Error Deleting the Loadbalancer")

			},
			expectedFn: func(svcErr error, retryDuration time.Duration) error {

				expectedError := "Error Deleting the Loadbalancer"

				if svcErr == nil || svcErr.Error() != expectedError {
					return fmt.Errorf("Expected=%v Obtained=%v", expectedError, svcErr)
				}

				if retryDuration != minRetryDelay {
					return fmt.Errorf("RetryDuration Expected=%v Obtained=%v", minRetryDelay, retryDuration)
				}
				return nil
			},
		},
		{
			testName: "If delete was successful",
			updateFn: func(controller *ServiceController) {

				testSvc := defaultExternalService()
				controller.enqueueService(testSvc)
				svc := controller.cache.getOrCreate(svcKey)
				svc.state = testSvc
				controller.cache.set(svcKey, svc)

			},
			expectedFn: func(svcErr error, retryDuration time.Duration) error {

				if svcErr != nil {
					return fmt.Errorf("Expected=nil Obtained=%v", svcErr)
				}

				if retryDuration != doNotRetry {
					//Retry duration should match
					return fmt.Errorf("RetryDuration Expected=%v Obtained=%v", doNotRetry, retryDuration)
				}

				//It should no longer be in the workqueue.
				_, exist := controller.cache.get(svcKey)
				if exist {
					return fmt.Errorf("delete service error, workingQueue should not contain service: %s any more", svcKey)
				}

				return nil
			},
		},
	}

	for _, tc := range testCases {
		//Create a new controller.
		controller, cloud, _ = newController()
		tc.updateFn(controller)
		obtainedErr, retryDuration := controller.processServiceDeletion(svcKey)
		if err := tc.expectedFn(obtainedErr, retryDuration); err != nil {
			t.Errorf("%v processServiceDeletion() %v", tc.testName, err)
		}
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

	controller, _, _ := newController()
	for _, tc := range testCases {
		tc.updateFn()
		obtainedResult := controller.needsUpdate(oldSvc, newSvc)
		if obtainedResult != tc.expectedNeedsUpdate {
			t.Errorf("%v needsUpdate() should have returned %v but returned %v", tc.testName, tc.expectedNeedsUpdate, obtainedResult)
		}
	}
}

//All the testcases for ServiceCache uses a single cache, these below test cases should be run in order,
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
					return fmt.Errorf("CachedService expected:non-nil Obtained=nil")
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
					return fmt.Errorf("Elementes Expected=2 Obtained=%v", len(keys))
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

//Test a utility functions as its not easy to unit test nodeSyncLoop directly
func TestNodeSlicesEqualForLB(t *testing.T) {
	numNodes := 10
	nArray := make([]*v1.Node, 10)

	for i := 0; i < numNodes; i++ {
		nArray[i] = &v1.Node{}
		nArray[i].Name = fmt.Sprintf("node1")
	}
	if !nodeSlicesEqualForLB(nArray, nArray) {
		t.Errorf("nodeSlicesEqualForLB() Expected=true Obtained=false")
	}
}
