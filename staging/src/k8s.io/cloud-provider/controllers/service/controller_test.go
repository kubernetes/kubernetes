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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/cloud-provider/api"
	fakecloud "k8s.io/cloud-provider/fake"
	servicehelper "k8s.io/cloud-provider/service/helpers"
	_ "k8s.io/controller-manager/pkg/features/register"
	"k8s.io/klog/v2/ktesting"

	utilpointer "k8s.io/utils/pointer"
)

const region = "us-central"

type serviceTweak func(s *v1.Service)

func newService(name string, serviceType v1.ServiceType, tweaks ...serviceTweak) *v1.Service {
	s := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: v1.ServiceSpec{
			Type:  serviceType,
			Ports: makeServicePort(v1.ProtocolTCP, 0),
		},
	}
	for _, tw := range tweaks {
		tw(s)
	}
	return s
}

func copyService(oldSvc *v1.Service, tweaks ...serviceTweak) *v1.Service {
	newSvc := oldSvc.DeepCopy()
	for _, tw := range tweaks {
		tw(newSvc)
	}
	return newSvc
}

func tweakAddETP(etpType v1.ServiceExternalTrafficPolicyType) serviceTweak {
	return func(s *v1.Service) {
		s.Spec.ExternalTrafficPolicy = etpType
	}
}

func tweakAddLBIngress(ip string) serviceTweak {
	return func(s *v1.Service) {
		s.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: ip}}
	}
}

func makeServicePort(protocol v1.Protocol, targetPort int) []v1.ServicePort {
	sp := v1.ServicePort{Port: 80, Protocol: protocol}
	if targetPort > 0 {
		sp.TargetPort = intstr.FromInt32(int32(targetPort))
	}
	return []v1.ServicePort{sp}
}

func tweakAddPorts(protocol v1.Protocol, targetPort int) serviceTweak {
	return func(s *v1.Service) {
		s.Spec.Ports = makeServicePort(protocol, targetPort)
	}
}

func tweakAddLBClass(loadBalancerClass *string) serviceTweak {
	return func(s *v1.Service) {
		s.Spec.LoadBalancerClass = loadBalancerClass
	}
}

func tweakAddFinalizers(finalizers ...string) serviceTweak {
	return func(s *v1.Service) {
		s.ObjectMeta.Finalizers = finalizers
	}
}

func tweakAddDeletionTimestamp(time time.Time) serviceTweak {
	return func(s *v1.Service) {
		s.ObjectMeta.DeletionTimestamp = &metav1.Time{Time: time}
	}
}

func tweakAddAppProtocol(appProtocol string) serviceTweak {
	return func(s *v1.Service) {
		s.Spec.Ports[0].AppProtocol = &appProtocol
	}
}

func tweakSetIPFamilies(families ...v1.IPFamily) serviceTweak {
	return func(s *v1.Service) {
		s.Spec.IPFamilies = families
	}
}

// Wrap newService so that you don't have to call default arguments again and again.
func defaultExternalService() *v1.Service {
	return newService("external-balancer", v1.ServiceTypeLoadBalancer)
}

// newController creates a new service controller. Callers have the option to
// specify `stopChan` for test cases which might require running the
// node/service informers and reacting to resource events. Callers can also
// specify `objects` which represent the initial state of objects, used to
// populate the client set / informer cache at start-up.
func newController(ctx context.Context, objects ...runtime.Object) (*Controller, *fakecloud.Cloud, *fake.Clientset) {
	stopCh := ctx.Done()
	cloud := &fakecloud.Cloud{}
	cloud.Region = region

	kubeClient := fake.NewSimpleClientset(objects...)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)
	serviceInformer := informerFactory.Core().V1().Services()
	nodeInformer := informerFactory.Core().V1().Nodes()
	broadcaster := record.NewBroadcaster(record.WithContext(ctx))
	broadcaster.StartStructuredLogging(0)
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "service-controller"})

	controller := &Controller{
		cloud:               cloud,
		kubeClient:          kubeClient,
		clusterName:         "test-cluster",
		eventBroadcaster:    broadcaster,
		eventRecorder:       recorder,
		serviceLister:       serviceInformer.Lister(),
		serviceListerSynced: serviceInformer.Informer().HasSynced,
		nodeLister:          nodeInformer.Lister(),
		nodeListerSynced:    nodeInformer.Informer().HasSynced,
		serviceQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.NewTypedItemExponentialFailureRateLimiter[string](minRetryDelay, maxRetryDelay),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "service"},
		),
		nodeQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.NewTypedItemExponentialFailureRateLimiter[string](minRetryDelay, maxRetryDelay),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "node"},
		),
		lastSyncedNodes: make(map[string][]*v1.Node),
	}

	informerFactory.Start(stopCh)
	informerFactory.WaitForCacheSync(stopCh)

	serviceMap := make(map[string]*cachedService)
	services, _ := serviceInformer.Lister().List(labels.Everything())
	for _, service := range services {
		serviceMap[service.Name] = &cachedService{
			state: service,
		}
	}

	controller.cache = &serviceCache{serviceMap: serviceMap}

	balancer, _ := cloud.LoadBalancer()
	controller.balancer = balancer

	controller.eventRecorder = record.NewFakeRecorder(100)

	cloud.Calls = nil         // ignore any cloud calls made in init()
	kubeClient.ClearActions() // ignore any client calls made in init()

	return controller, cloud, kubeClient
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
	}{{
		desc:              "service doesn't want LB",
		service:           newService("no-external-balancer", v1.ServiceTypeClusterIP),
		expectOp:          deleteLoadBalancer,
		expectPatchStatus: false,
	}, {
		desc:                 "udp service that wants LB",
		service:              newService("udp-service", v1.ServiceTypeLoadBalancer, tweakAddPorts(v1.ProtocolUDP, 0)),
		expectOp:             ensureLoadBalancer,
		expectCreateAttempt:  true,
		expectPatchStatus:    true,
		expectPatchFinalizer: true,
	}, {
		desc:                 "tcp service that wants LB",
		service:              newService("basic-service1", v1.ServiceTypeLoadBalancer),
		expectOp:             ensureLoadBalancer,
		expectCreateAttempt:  true,
		expectPatchStatus:    true,
		expectPatchFinalizer: true,
	}, {
		desc:                 "sctp service that wants LB",
		service:              newService("sctp-service", v1.ServiceTypeLoadBalancer, tweakAddPorts(v1.ProtocolSCTP, 0)),
		expectOp:             ensureLoadBalancer,
		expectCreateAttempt:  true,
		expectPatchStatus:    true,
		expectPatchFinalizer: true,
	}, {
		desc:                 "service specifies loadBalancerClass",
		service:              newService("with-external-balancer", v1.ServiceTypeLoadBalancer, tweakAddLBClass(utilpointer.String("custom-loadbalancer"))),
		expectOp:             deleteLoadBalancer,
		expectCreateAttempt:  false,
		expectPatchStatus:    false,
		expectPatchFinalizer: false,
	}, {
		// Finalizer test cases below.
		desc:                 "service with finalizer that no longer wants LB",
		service:              newService("no-external-balancer", v1.ServiceTypeClusterIP, tweakAddLBIngress("8.8.8.8"), tweakAddFinalizers(servicehelper.LoadBalancerCleanupFinalizer)),
		lbExists:             true,
		expectOp:             deleteLoadBalancer,
		expectDeleteAttempt:  true,
		expectPatchStatus:    true,
		expectPatchFinalizer: true,
	}, {
		desc:                 "service that needs cleanup",
		service:              newService("basic-service1", v1.ServiceTypeLoadBalancer, tweakAddLBIngress("8.8.8.8"), tweakAddFinalizers(servicehelper.LoadBalancerCleanupFinalizer), tweakAddDeletionTimestamp(time.Now())),
		lbExists:             true,
		expectOp:             deleteLoadBalancer,
		expectDeleteAttempt:  true,
		expectPatchStatus:    true,
		expectPatchFinalizer: true,
	}, {
		desc:                 "service with finalizer that wants LB",
		service:              newService("basic-service1", v1.ServiceTypeLoadBalancer, tweakAddFinalizers(servicehelper.LoadBalancerCleanupFinalizer)),
		expectOp:             ensureLoadBalancer,
		expectCreateAttempt:  true,
		expectPatchStatus:    true,
		expectPatchFinalizer: false,
	}}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			controller, cloud, client := newController(ctx)
			cloud.Exists = tc.lbExists
			key := fmt.Sprintf("%s/%s", tc.service.Namespace, tc.service.Name)
			if _, err := client.CoreV1().Services(tc.service.Namespace).Create(ctx, tc.service, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to prepare service %s for testing: %v", key, err)
			}
			client.ClearActions()

			op, err := controller.syncLoadBalancerIfNeeded(ctx, tc.service, key)
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

				if len(cloud.Balancers) == 0 {
					t.Errorf("Got no load balancer: %v, expected one to be created", cloud.Balancers)
				}

				for _, balancer := range cloud.Balancers {
					if balancer.Name != controller.balancer.GetLoadBalancerName(ctx, "", tc.service) ||
						balancer.Region != region ||
						balancer.Ports[0].Port != tc.service.Spec.Ports[0].Port {
						t.Errorf("Created load balancer has incorrect parameters: %v", balancer)
					}
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
		makeNode(tweakName("node1")),
		makeNode(tweakName("node2")),
		makeNode(tweakName("node3")),
	}
	table := []struct {
		desc                string
		services            []*v1.Service
		expectedUpdateCalls []fakecloud.UpdateBalancerCall
		workers             int
	}{{
		desc:                "No services present: no calls should be made.",
		services:            []*v1.Service{},
		expectedUpdateCalls: nil,
		workers:             1,
	}, {
		desc: "Services do not have external load balancers: no calls should be made.",
		services: []*v1.Service{
			newService("s0", v1.ServiceTypeClusterIP),
			newService("s1", v1.ServiceTypeNodePort),
		},
		expectedUpdateCalls: nil,
		workers:             2,
	}, {
		desc: "Services does have an external load balancer: one call should be made.",
		services: []*v1.Service{
			newService("s0", v1.ServiceTypeLoadBalancer),
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: newService("s0", v1.ServiceTypeLoadBalancer), Hosts: nodes},
		},
		workers: 3,
	}, {
		desc: "Three services have an external load balancer: three calls.",
		services: []*v1.Service{
			newService("s0", v1.ServiceTypeLoadBalancer),
			newService("s1", v1.ServiceTypeLoadBalancer),
			newService("s2", v1.ServiceTypeLoadBalancer),
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: newService("s0", v1.ServiceTypeLoadBalancer), Hosts: nodes},
			{Service: newService("s1", v1.ServiceTypeLoadBalancer), Hosts: nodes},
			{Service: newService("s2", v1.ServiceTypeLoadBalancer), Hosts: nodes},
		},
		workers: 4,
	}, {
		desc: "Two services have an external load balancer and two don't: two calls.",
		services: []*v1.Service{
			newService("s0", v1.ServiceTypeNodePort),
			newService("s1", v1.ServiceTypeLoadBalancer),
			newService("s3", v1.ServiceTypeLoadBalancer),
			newService("s4", v1.ServiceTypeClusterIP),
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: newService("s1", v1.ServiceTypeLoadBalancer), Hosts: nodes},
			{Service: newService("s3", v1.ServiceTypeLoadBalancer), Hosts: nodes},
		},
		workers: 5,
	}, {
		desc: "One service has an external load balancer and one is nil: one call.",
		services: []*v1.Service{
			newService("s0", v1.ServiceTypeLoadBalancer),
			nil,
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: newService("s0", v1.ServiceTypeLoadBalancer), Hosts: nodes},
		},
		workers: 6,
	}, {
		desc: "Four services have external load balancer with only 2 workers",
		services: []*v1.Service{
			newService("s0", v1.ServiceTypeLoadBalancer),
			newService("s1", v1.ServiceTypeLoadBalancer),
			newService("s3", v1.ServiceTypeLoadBalancer),
			newService("s4", v1.ServiceTypeLoadBalancer),
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: newService("s0", v1.ServiceTypeLoadBalancer), Hosts: nodes},
			{Service: newService("s1", v1.ServiceTypeLoadBalancer), Hosts: nodes},
			{Service: newService("s3", v1.ServiceTypeLoadBalancer), Hosts: nodes},
			{Service: newService("s4", v1.ServiceTypeLoadBalancer), Hosts: nodes},
		},
		workers: 2,
	}}
	for _, item := range table {
		t.Run(item.desc, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			controller, cloud, _ := newController(ctx)
			controller.nodeLister = newFakeNodeLister(nil, nodes...)
			if servicesToRetry := controller.updateLoadBalancerHosts(ctx, item.services, item.workers); len(servicesToRetry) != 0 {
				t.Errorf("for case %q, unexpected servicesToRetry: %v", item.desc, servicesToRetry)
			}
			compareUpdateCalls(t, item.expectedUpdateCalls, cloud.UpdateCalls)
		})
	}
}

func TestNodeChangesForStableNodeSetEnabled(t *testing.T) {
	node1 := makeNode(tweakName("node1"), tweakSetCondition(v1.NodeReady, v1.ConditionTrue))
	node2 := makeNode(tweakName("node2"), tweakSetCondition(v1.NodeReady, v1.ConditionTrue))
	node3 := makeNode(tweakName("node3"), tweakSetCondition(v1.NodeReady, v1.ConditionTrue))
	node2NotReady := makeNode(tweakName("node2"), tweakSetCondition(v1.NodeReady, v1.ConditionFalse))
	node2Tainted := makeNode(tweakName("node2"), tweakAddTaint(ToBeDeletedTaint), tweakSetCondition(v1.NodeReady, v1.ConditionTrue))
	node2SpuriousChange := makeNode(tweakName("node2"), tweakAddTaint("Other"), tweakSetCondition(v1.NodeReady, v1.ConditionTrue))
	node2Exclude := makeNode(tweakName("node2"), tweakSetLabel(v1.LabelNodeExcludeBalancers, ""), tweakSetCondition(v1.NodeReady, v1.ConditionTrue))
	node2Deleted := makeNode(tweakName("node2"), tweakDeleted())

	type stateChanges struct {
		nodes       []*v1.Node
		syncCallErr bool
	}

	etpLocalservice1 := newService("s0", v1.ServiceTypeLoadBalancer, tweakAddETP(v1.ServiceExternalTrafficPolicyLocal))
	etpLocalservice2 := newService("s1", v1.ServiceTypeLoadBalancer, tweakAddETP(v1.ServiceExternalTrafficPolicyLocal))
	service3 := defaultExternalService()

	services := []*v1.Service{etpLocalservice1, etpLocalservice2, service3}

	for _, tc := range []struct {
		desc                string
		expectedUpdateCalls []fakecloud.UpdateBalancerCall
		stateChanges        []stateChanges
		initialState        []*v1.Node
	}{{
		desc:         "No node changes",
		initialState: []*v1.Node{node1, node2, node3},
		stateChanges: []stateChanges{
			{
				nodes: []*v1.Node{node1, node2, node3},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{},
	}, {
		desc:         "1 new node gets added",
		initialState: []*v1.Node{node1, node2},
		stateChanges: []stateChanges{
			{
				nodes: []*v1.Node{node1, node2, node3},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: etpLocalservice1, Hosts: []*v1.Node{node1, node2, node3}},
			{Service: etpLocalservice2, Hosts: []*v1.Node{node1, node2, node3}},
			{Service: service3, Hosts: []*v1.Node{node1, node2, node3}},
		},
	}, {
		desc:         "1 new node gets added - with retries",
		initialState: []*v1.Node{node1, node2},
		stateChanges: []stateChanges{
			{
				nodes:       []*v1.Node{node1, node2, node3},
				syncCallErr: true,
			},
			{
				nodes: []*v1.Node{node1, node2, node3},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: etpLocalservice1, Hosts: []*v1.Node{node1, node2, node3}},
			{Service: etpLocalservice2, Hosts: []*v1.Node{node1, node2, node3}},
			{Service: service3, Hosts: []*v1.Node{node1, node2, node3}},
		},
	}, {
		desc:         "1 node goes NotReady",
		initialState: []*v1.Node{node1, node2, node3},
		stateChanges: []stateChanges{
			{
				nodes: []*v1.Node{node1, node2NotReady, node3},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{},
	}, {
		desc:         "1 node gets Tainted",
		initialState: []*v1.Node{node1, node2, node3},
		stateChanges: []stateChanges{
			{
				nodes: []*v1.Node{node1, node2Tainted, node3},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: etpLocalservice1, Hosts: []*v1.Node{node1, node3}},
			{Service: etpLocalservice2, Hosts: []*v1.Node{node1, node3}},
			{Service: service3, Hosts: []*v1.Node{node1, node3}},
		},
	}, {
		desc:         "1 node goes Ready",
		initialState: []*v1.Node{node1, node2NotReady, node3},
		stateChanges: []stateChanges{
			{
				nodes: []*v1.Node{node1, node2, node3},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{},
	}, {
		desc:         "1 node get excluded",
		initialState: []*v1.Node{node1, node2, node3},
		stateChanges: []stateChanges{
			{
				nodes: []*v1.Node{node1, node2Exclude, node3},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: etpLocalservice1, Hosts: []*v1.Node{node1, node3}},
			{Service: etpLocalservice2, Hosts: []*v1.Node{node1, node3}},
			{Service: service3, Hosts: []*v1.Node{node1, node3}},
		},
	}, {
		desc:         "1 old node gets deleted",
		initialState: []*v1.Node{node1, node2, node3},
		stateChanges: []stateChanges{
			{
				nodes: []*v1.Node{node1, node2},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: etpLocalservice1, Hosts: []*v1.Node{node1, node2}},
			{Service: etpLocalservice2, Hosts: []*v1.Node{node1, node2}},
			{Service: service3, Hosts: []*v1.Node{node1, node2}},
		},
	}, {
		desc:         "1 node marked for deletion",
		initialState: []*v1.Node{node1, node2, node3},
		stateChanges: []stateChanges{
			{
				nodes: []*v1.Node{node1, node2Deleted, node3},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: etpLocalservice1, Hosts: []*v1.Node{node1, node3}},
			{Service: etpLocalservice2, Hosts: []*v1.Node{node1, node3}},
			{Service: service3, Hosts: []*v1.Node{node1, node3}},
		},
	}, {
		desc:         "1 spurious node update",
		initialState: []*v1.Node{node1, node2, node3},
		stateChanges: []stateChanges{
			{
				nodes: []*v1.Node{node1, node2SpuriousChange, node3},
			},
		},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{},
	}} {
		t.Run(tc.desc, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			controller, cloud, _ := newController(ctx)

			for _, svc := range services {
				key, _ := cache.MetaNamespaceKeyFunc(svc)
				controller.lastSyncedNodes[key] = tc.initialState
			}

			for _, state := range tc.stateChanges {
				setupState := func() {
					controller.nodeLister = newFakeNodeLister(nil, state.nodes...)
					if state.syncCallErr {
						cloud.Err = fmt.Errorf("error please")
					}
				}
				cleanupState := func() {
					cloud.Err = nil
				}
				setupState()
				controller.updateLoadBalancerHosts(ctx, services, 3)
				cleanupState()
			}

			compareUpdateCalls(t, tc.expectedUpdateCalls, cloud.UpdateCalls)
		})
	}
}

func TestNodeChangesInExternalLoadBalancer(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	node1 := makeNode(tweakName("node1"))
	node2 := makeNode(tweakName("node2"))
	node3 := makeNode(tweakName("node3"))
	node4 := makeNode(tweakName("node4"))

	services := []*v1.Service{
		newService("s0", v1.ServiceTypeLoadBalancer),
		newService("s1", v1.ServiceTypeLoadBalancer),
		newService("s3", v1.ServiceTypeLoadBalancer),
		newService("s4", v1.ServiceTypeLoadBalancer),
	}

	serviceNames := sets.NewString()
	for _, svc := range services {
		serviceNames.Insert(fmt.Sprintf("%s/%s", svc.GetObjectMeta().GetNamespace(), svc.GetObjectMeta().GetName()))
	}

	controller, cloud, _ := newController(ctx)
	for _, tc := range []struct {
		desc                  string
		nodes                 []*v1.Node
		expectedUpdateCalls   []fakecloud.UpdateBalancerCall
		worker                int
		nodeListerErr         error
		expectedRetryServices sets.String
	}{{
		desc:  "only 1 node",
		nodes: []*v1.Node{node1},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: newService("s0", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1}},
			{Service: newService("s1", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1}},
			{Service: newService("s3", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1}},
			{Service: newService("s4", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1}},
		},
		worker:                3,
		nodeListerErr:         nil,
		expectedRetryServices: sets.NewString(),
	}, {
		desc:  "2 nodes",
		nodes: []*v1.Node{node1, node2},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: newService("s0", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1, node2}},
			{Service: newService("s1", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1, node2}},
			{Service: newService("s3", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1, node2}},
			{Service: newService("s4", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1, node2}},
		},
		worker:                1,
		nodeListerErr:         nil,
		expectedRetryServices: sets.NewString(),
	}, {
		desc:  "4 nodes",
		nodes: []*v1.Node{node1, node2, node3, node4},
		expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
			{Service: newService("s0", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1, node2, node3, node4}},
			{Service: newService("s1", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1, node2, node3, node4}},
			{Service: newService("s3", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1, node2, node3, node4}},
			{Service: newService("s4", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1, node2, node3, node4}},
		},
		worker:                3,
		nodeListerErr:         nil,
		expectedRetryServices: sets.NewString(),
	}, {
		desc:                  "error occur during sync",
		nodes:                 []*v1.Node{node1, node2, node3, node4},
		expectedUpdateCalls:   []fakecloud.UpdateBalancerCall{},
		worker:                3,
		nodeListerErr:         fmt.Errorf("random error"),
		expectedRetryServices: serviceNames,
	}, {
		desc:                  "error occur during sync with 1 workers",
		nodes:                 []*v1.Node{node1, node2, node3, node4},
		expectedUpdateCalls:   []fakecloud.UpdateBalancerCall{},
		worker:                1,
		nodeListerErr:         fmt.Errorf("random error"),
		expectedRetryServices: serviceNames,
	}} {
		t.Run(tc.desc, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			controller.nodeLister = newFakeNodeLister(tc.nodeListerErr, tc.nodes...)
			servicesToRetry := controller.updateLoadBalancerHosts(ctx, services, tc.worker)
			assert.Truef(t, tc.expectedRetryServices.Equal(servicesToRetry), "Services to retry are not expected")
			compareUpdateCalls(t, tc.expectedUpdateCalls, cloud.UpdateCalls)
			cloud.UpdateCalls = []fakecloud.UpdateBalancerCall{}
		})
	}
}

// compareUpdateCalls compares if the same update calls were made in both left and right inputs despite the order.
func compareUpdateCalls(t *testing.T, left, right []fakecloud.UpdateBalancerCall) {
	if len(left) != len(right) {
		t.Errorf("expect len(left) == len(right), but got %v != %v", len(left), len(right))
	}

	mismatch := false
	for _, l := range left {
		found := false
		for _, r := range right {
			if reflect.DeepEqual(l, r) {
				found = true
			}
		}
		if !found {
			mismatch = true
			break
		}
	}
	if mismatch {
		t.Errorf("expected update calls to match, expected %+v, got %+v", left, right)
	}
}

// compareHostSets compares if the nodes in left are in right, despite the order.
func compareHostSets(t *testing.T, left, right []*v1.Node) bool {
	if len(left) != len(right) {
		return false
	}
	for _, lHost := range left {
		found := false
		for _, rHost := range right {
			if reflect.DeepEqual(lHost, rHost) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func TestNodesNotEqual(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	controller, cloud, _ := newController(ctx)

	services := []*v1.Service{
		newService("s0", v1.ServiceTypeLoadBalancer),
		newService("s1", v1.ServiceTypeLoadBalancer),
	}

	node1 := makeNode(tweakName("node1"))
	node2 := makeNode(tweakName("node2"))
	node3 := makeNode(tweakName("node3"))
	node1WithProviderID := makeNode(tweakName("node1"), tweakProviderID("cumulus/1"))
	node2WithProviderID := makeNode(tweakName("node2"), tweakProviderID("cumulus/2"))

	testCases := []struct {
		desc                string
		lastSyncNodes       []*v1.Node
		newNodes            []*v1.Node
		expectedUpdateCalls []fakecloud.UpdateBalancerCall
	}{
		{
			desc:          "Nodes with updated providerID",
			lastSyncNodes: []*v1.Node{node1, node2},
			newNodes:      []*v1.Node{node1WithProviderID, node2WithProviderID},
			expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
				{Service: newService("s0", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1WithProviderID, node2WithProviderID}},
				{Service: newService("s1", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1WithProviderID, node2WithProviderID}},
			},
		},
		{
			desc:                "Nodes unchanged",
			lastSyncNodes:       []*v1.Node{node1WithProviderID, node2},
			newNodes:            []*v1.Node{node1WithProviderID, node2},
			expectedUpdateCalls: []fakecloud.UpdateBalancerCall{},
		},
		{
			desc:          "Change node with empty providerID",
			lastSyncNodes: []*v1.Node{node1WithProviderID, node2},
			newNodes:      []*v1.Node{node1WithProviderID, node3},
			expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
				{Service: newService("s0", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1WithProviderID, node3}},
				{Service: newService("s1", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1WithProviderID, node3}},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			controller.nodeLister = newFakeNodeLister(nil, tc.newNodes...)

			for _, svc := range services {
				key, _ := cache.MetaNamespaceKeyFunc(svc)
				controller.lastSyncedNodes[key] = tc.lastSyncNodes
			}

			controller.updateLoadBalancerHosts(ctx, services, 5)
			compareUpdateCalls(t, tc.expectedUpdateCalls, cloud.UpdateCalls)
			cloud.UpdateCalls = []fakecloud.UpdateBalancerCall{}
		})
	}
}

func TestProcessServiceCreateOrUpdate(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	controller, _, client := newController(ctx)

	//A pair of old and new loadbalancer IP address
	oldLBIP := "192.168.1.1"
	newLBIP := "192.168.1.11"

	testCases := []struct {
		testName   string
		key        string
		updateFn   func(*v1.Service) *v1.Service //Manipulate the structure
		svc        *v1.Service
		expectedFn func(*v1.Service, error) error //Error comparison function
	}{{
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
	}, {
		testName: "If Updating Loadbalancer IP",
		key:      "default/sync-test-name",
		svc:      newService("sync-test-name", v1.ServiceTypeLoadBalancer),
		updateFn: func(svc *v1.Service) *v1.Service {

			svc.Spec.LoadBalancerIP = oldLBIP

			keyExpected := svc.GetObjectMeta().GetNamespace() + "/" + svc.GetObjectMeta().GetName()
			controller.enqueueService(svc)
			cachedServiceTest := controller.cache.getOrCreate(keyExpected)
			cachedServiceTest.state = svc
			controller.cache.set(keyExpected, cachedServiceTest)

			keyGot, quit := controller.serviceQueue.Get()
			if quit {
				t.Fatalf("get no queue element")
			}
			if keyExpected != keyGot {
				t.Fatalf("get service key error, expected: %s, got: %s", keyExpected, keyGot)
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
	}}

	for _, tc := range testCases {
		_, ctx := ktesting.NewTestContext(t)
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		newSvc := tc.updateFn(tc.svc)
		if _, err := client.CoreV1().Services(tc.svc.Namespace).Create(ctx, tc.svc, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to prepare service %s for testing: %v", tc.key, err)
		}
		obtErr := controller.processServiceCreateOrUpdate(ctx, newSvc, tc.key)
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
	}{{
		desc:      "conflict error",
		k8sErr:    conflictErr,
		expectErr: fmt.Errorf("failed to update load balancer status: %v", conflictErr),
	}, {
		desc:      "not found error",
		k8sErr:    notFoundErr,
		expectErr: nil,
	}}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			svc := newService(svcName, v1.ServiceTypeLoadBalancer)
			// Preset finalizer so k8s error only happens when patching status.
			svc.Finalizers = []string{servicehelper.LoadBalancerCleanupFinalizer}
			controller, _, client := newController(ctx)
			client.PrependReactor("patch", "services", func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, tc.k8sErr
			})

			if err := controller.processServiceCreateOrUpdate(ctx, svc, svcName); !reflect.DeepEqual(err, tc.expectErr) {
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
		updateFn   func(context.Context) // Function to manipulate the controller element to simulate error
		expectedFn func(error) error     // Expected function if returns nil then test passed, failed otherwise
	}{
		{
			testName: "if an invalid service name is synced",
			key:      "invalid/key/string",
			updateFn: func(ctx context.Context) {
				controller, _, _ = newController(ctx)
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
			updateFn: func(ctx context.Context) {
				testSvc := defaultExternalService()
				controller, _, _ = newController(ctx)
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
		t.Run(tc.testName, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			tc.updateFn(ctx)
			obtainedErr := controller.syncService(ctx, tc.key)

			//expected matches obtained ??.
			if exp := tc.expectedFn(obtainedErr); exp != nil {
				t.Errorf("%v Error:%v", tc.testName, exp)
			}

			//Post processing, the element should not be in the sync queue.
			_, exist := controller.cache.get(tc.key)
			if exist {
				t.Fatalf("%v working Queue should be empty, but contains %s", tc.testName, tc.key)
			}
		})
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
	}{{
		testName: "If a non-existent service is deleted",
		updateFn: func(controller *Controller) {
			// Does not do anything
		},
		expectedFn: func(svcErr error) error {
			return svcErr
		},
	}, {
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
	}, {
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
	}}

	for _, tc := range testCases {
		_, ctx := ktesting.NewTestContext(t)
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		//Create a new controller.
		controller, cloud, _ = newController(ctx)
		tc.updateFn(controller)
		obtainedErr := controller.processServiceDeletion(ctx, svcKey)
		if err := tc.expectedFn(obtainedErr); err != nil {
			t.Errorf("%v processServiceDeletion() %v", tc.testName, err)
		}
	}

}

// Test cases:
// index    finalizer    timestamp    wantLB  |  clean-up
//
//	0         0           0            0     |   false    (No finalizer, no clean up)
//	1         0           0            1     |   false    (Ignored as same with case 0)
//	2         0           1            0     |   false    (Ignored as same with case 0)
//	3         0           1            1     |   false    (Ignored as same with case 0)
//	4         1           0            0     |   true
//	5         1           0            1     |   false
//	6         1           1            0     |   true    (Service is deleted, needs clean up)
//	7         1           1            1     |   true    (Ignored as same with case 6)
func TestNeedsCleanup(t *testing.T) {
	testCases := []struct {
		desc               string
		svc                *v1.Service
		expectNeedsCleanup bool
	}{{
		desc:               "service without finalizer",
		svc:                &v1.Service{},
		expectNeedsCleanup: false,
	}, {
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
	}, {
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
	}, {
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
	}}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			if gotNeedsCleanup := needsCleanup(tc.svc); gotNeedsCleanup != tc.expectNeedsCleanup {
				t.Errorf("needsCleanup() = %t, want %t", gotNeedsCleanup, tc.expectNeedsCleanup)
			}
		})
	}

}

// This tests a service update while a slow node sync is happening. If we have multiple
// services to process from a node sync: each service will experience a sync delta.
// If a new Node is added and a service is synced while this happens: we want to
// make sure that the slow node sync never removes the Node from LB set because it
// has stale data.
func TestSlowNodeSync(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	syncServiceDone, syncService := make(chan string), make(chan string)
	defer close(syncService)

	node1 := makeNode(tweakName("node1"))
	node2 := makeNode(tweakName("node2"))
	node3 := makeNode(tweakName("node3"))
	service1 := newService("service1", v1.ServiceTypeLoadBalancer)
	service2 := newService("service2", v1.ServiceTypeLoadBalancer)

	sKey1, _ := cache.MetaNamespaceKeyFunc(service1)
	sKey2, _ := cache.MetaNamespaceKeyFunc(service2)
	serviceKeys := sets.New(sKey1, sKey2)

	controller, cloudProvider, kubeClient := newController(ctx, node1, node2, service1, service2)
	cloudProvider.UpdateCallCb = func(update fakecloud.UpdateBalancerCall) {
		key, _ := cache.MetaNamespaceKeyFunc(update.Service)
		impactedService := serviceKeys.Difference(sets.New(key)).UnsortedList()[0]
		syncService <- impactedService
		<-syncServiceDone

	}
	cloudProvider.EnsureCallCb = func(update fakecloud.UpdateBalancerCall) {
		syncServiceDone <- update.Service.Name
	}
	// Two update calls are expected. This is because this test calls
	// controller.syncNodes once with two existing services, but with one
	// controller.syncService while that is happening. The end result is
	// therefore two update calls - since the second controller.syncNodes won't
	// trigger an update call because the syncService already did. Each update
	// call takes cloudProvider.RequestDelay to process. The test asserts that
	// the order of the Hosts defined by the update calls is respected, but
	// doesn't necessarily assert the order of the Service. This is because the
	// controller implementation doesn't use a deterministic order when syncing
	// services. The test therefor works out which service is impacted by the
	// slow node sync (which will be whatever service is not synced first) and
	// then validates that the Hosts for each update call is respected.
	expectedUpdateCalls := []fakecloud.UpdateBalancerCall{
		// First update call for first service from controller.syncNodes
		{Service: service1, Hosts: []*v1.Node{node1, node2}},
	}
	expectedEnsureCalls := []fakecloud.UpdateBalancerCall{
		// Second update call for impacted service from controller.syncService
		{Service: service2, Hosts: []*v1.Node{node1, node2, node3}},
	}

	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		controller.syncNodes(ctx, 1)
	}()

	key := <-syncService
	if _, err := kubeClient.CoreV1().Nodes().Create(ctx, node3, metav1.CreateOptions{}); err != nil {
		t.Fatalf("error creating node3, err: %v", err)
	}

	// Allow a bit of time for the informer cache to get populated with the new
	// node
	if err := wait.PollUntilContextCancel(ctx, 10*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
		n3, _ := controller.nodeLister.Get("node3")
		return n3 != nil, nil
	}); err != nil {
		t.Fatalf("informer cache was never populated with node3")
	}

	// Sync the service
	if err := controller.syncService(ctx, key); err != nil {
		t.Fatalf("unexpected service sync error, err: %v", err)
	}

	wg.Wait()

	if len(expectedUpdateCalls) != len(cloudProvider.UpdateCalls) {
		t.Fatalf("unexpected amount of update calls, expected: %v, got: %v", len(expectedUpdateCalls), len(cloudProvider.UpdateCalls))
	}
	for idx, update := range cloudProvider.UpdateCalls {
		if !compareHostSets(t, expectedUpdateCalls[idx].Hosts, update.Hosts) {
			t.Fatalf("unexpected updated hosts for update: %v, expected: %v, got: %v", idx, expectedUpdateCalls[idx].Hosts, update.Hosts)
		}
	}
	if len(expectedEnsureCalls) != len(cloudProvider.EnsureCalls) {
		t.Fatalf("unexpected amount of ensure calls, expected: %v, got: %v", len(expectedEnsureCalls), len(cloudProvider.EnsureCalls))
	}
	for idx, ensure := range cloudProvider.EnsureCalls {
		if !compareHostSets(t, expectedEnsureCalls[idx].Hosts, ensure.Hosts) {
			t.Fatalf("unexpected updated hosts for ensure: %v, expected: %v, got: %v", idx, expectedEnsureCalls[idx].Hosts, ensure.Hosts)
		}
	}
}

func TestNeedsUpdate(t *testing.T) {
	testCases := []struct {
		testName            string                            //Name of the test case
		updateFn            func() (*v1.Service, *v1.Service) //Function to update the service object
		expectedNeedsUpdate bool                              //needsupdate always returns bool

	}{{
		testName: "If the service type is changed from LoadBalancer to ClusterIP",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = defaultExternalService()
			newSvc = defaultExternalService()
			newSvc.Spec.Type = v1.ServiceTypeClusterIP
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If the Ports are different",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
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
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If external ip counts are different",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = defaultExternalService()
			newSvc = defaultExternalService()
			oldSvc.Spec.ExternalIPs = []string{"old.IP.1"}
			newSvc.Spec.ExternalIPs = []string{"new.IP.1", "new.IP.2"}
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If external ips are different",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = defaultExternalService()
			newSvc = defaultExternalService()
			oldSvc.Spec.ExternalIPs = []string{"old.IP.1", "old.IP.2"}
			newSvc.Spec.ExternalIPs = []string{"new.IP.1", "new.IP.2"}
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If UID is different",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = defaultExternalService()
			newSvc = defaultExternalService()
			oldSvc.UID = types.UID("UID old")
			newSvc.UID = types.UID("UID new")
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If ExternalTrafficPolicy is different",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = defaultExternalService()
			newSvc = defaultExternalService()
			newSvc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If HealthCheckNodePort is different",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = defaultExternalService()
			newSvc = defaultExternalService()
			newSvc.Spec.HealthCheckNodePort = 30123
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If TargetGroup is different 1",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = newService("tcp-service", v1.ServiceTypeLoadBalancer, tweakAddPorts(v1.ProtocolTCP, 20))
			newSvc = copyService(oldSvc, tweakAddPorts(v1.ProtocolTCP, 21))
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If TargetGroup is different 2",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = newService("tcp-service", v1.ServiceTypeLoadBalancer, tweakAddPorts(v1.ProtocolTCP, 22))
			newSvc = oldSvc.DeepCopy()
			newSvc.Spec.Ports[0].TargetPort = intstr.Parse("dns")
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If appProtocol is the same",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = newService("tcp-service", v1.ServiceTypeLoadBalancer)
			newSvc = copyService(oldSvc)
			return
		},
		expectedNeedsUpdate: false,
	}, {
		testName: "If service IPFamilies from single stack to dual stack",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = newService("tcp-service", v1.ServiceTypeLoadBalancer, tweakSetIPFamilies(v1.IPv4Protocol))
			newSvc = copyService(oldSvc, tweakSetIPFamilies(v1.IPv4Protocol, v1.IPv6Protocol))
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If service IPFamilies from dual stack to single stack",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = newService("tcp-service", v1.ServiceTypeLoadBalancer, tweakSetIPFamilies(v1.IPv4Protocol, v1.IPv6Protocol))
			newSvc = copyService(oldSvc, tweakSetIPFamilies(v1.IPv4Protocol))
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If service IPFamilies not change",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = newService("tcp-service", v1.ServiceTypeLoadBalancer, tweakSetIPFamilies(v1.IPv4Protocol))
			newSvc = copyService(oldSvc)
			return
		},
		expectedNeedsUpdate: false,
	}, {
		testName: "If appProtocol is set when previously unset",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = newService("tcp-service", v1.ServiceTypeLoadBalancer)
			newSvc = copyService(oldSvc, tweakAddAppProtocol("http"))
			return
		},
		expectedNeedsUpdate: true,
	}, {
		testName: "If appProtocol is set to a different value",
		updateFn: func() (oldSvc *v1.Service, newSvc *v1.Service) {
			oldSvc = newService("tcp-service", v1.ServiceTypeLoadBalancer, tweakAddAppProtocol("http"))
			newSvc = copyService(oldSvc, tweakAddAppProtocol("tcp"))
			return
		},
		expectedNeedsUpdate: true,
	}}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			oldSvc, newSvc := tc.updateFn()
			obtainedResult := needsUpdate(oldSvc, newSvc)
			if obtainedResult != tc.expectedNeedsUpdate {
				t.Errorf("%v needsUpdate() should have returned %v but returned %v", tc.testName, tc.expectedNeedsUpdate, obtainedResult)
			}
		})
	}
}

// All the test cases for ServiceCache uses a single cache, these below test cases should be run in order,
// as tc1 (addCache would add elements to the cache)
// and tc2 (delCache would remove element from the cache without it adding automatically)
// Please keep this in mind while adding new test cases.
func TestServiceCache(t *testing.T) {

	//ServiceCache a common service cache for all the test cases
	sc := &serviceCache{serviceMap: make(map[string]*cachedService)}

	testCases := []struct {
		testName     string
		setCacheFn   func()
		checkCacheFn func() error
	}{{
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
	}, {
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
	}, {
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
	}, {
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
	}, {
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
	}, {
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
	}}

	for _, tc := range testCases {
		if tc.setCacheFn != nil {
			tc.setCacheFn()
		}
		if err := tc.checkCacheFn(); err != nil {
			t.Errorf("%v returned %v", tc.testName, err)
		}
	}
}

// TODO(@MrHohn): Verify the end state when below issue is resolved:
// https://github.com/kubernetes/client-go/issues/607
func TestAddFinalizer(t *testing.T) {
	testCases := []struct {
		desc        string
		svc         *v1.Service
		expectPatch bool
	}{{
		desc: "no-op add finalizer",
		svc: &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "test-patch-finalizer",
				Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
			},
		},
		expectPatch: false,
	}, {
		desc: "add finalizer",
		svc: &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-patch-finalizer",
			},
		},
		expectPatch: true,
	}}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			c := fake.NewSimpleClientset()
			s := &Controller{
				kubeClient: c,
			}
			if _, err := s.kubeClient.CoreV1().Services(tc.svc.Namespace).Create(ctx, tc.svc, metav1.CreateOptions{}); err != nil {
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
	}{{
		desc: "no-op remove finalizer",
		svc: &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-patch-finalizer",
			},
		},
		expectPatch: false,
	}, {
		desc: "remove finalizer",
		svc: &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "test-patch-finalizer",
				Finalizers: []string{servicehelper.LoadBalancerCleanupFinalizer},
			},
		},
		expectPatch: true,
	}}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			c := fake.NewSimpleClientset()
			s := &Controller{
				kubeClient: c,
			}
			if _, err := s.kubeClient.CoreV1().Services(tc.svc.Namespace).Create(ctx, tc.svc, metav1.CreateOptions{}); err != nil {
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
	}{{
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
	}, {
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
	}, {
		desc: "no-op clear status",
		svc: &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-patch-status",
			},
			Status: v1.ServiceStatus{},
		},
		newStatus:   &v1.LoadBalancerStatus{},
		expectPatch: false,
	}, {
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
	}}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			c := fake.NewSimpleClientset()
			s := &Controller{
				kubeClient: c,
			}
			if _, err := s.kubeClient.CoreV1().Services(tc.svc.Namespace).Create(ctx, tc.svc, metav1.CreateOptions{}); err != nil {
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

func Test_respectsPredicates(t *testing.T) {
	tests := []struct {
		name string

		input *v1.Node
		want  bool
	}{
		{want: false, input: &v1.Node{}},
		{want: true, input: &v1.Node{Spec: v1.NodeSpec{ProviderID: providerID}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}}}},
		{want: true, input: &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}}}},
		{want: false, input: &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}}}}},
		{want: true, input: &v1.Node{Spec: v1.NodeSpec{ProviderID: providerID}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{}}}},
		{want: false, input: &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelNodeExcludeBalancers: ""}}}},
		{want: true, input: &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelNodeExcludeBalancers: "false"}}}},
		{want: false, input: &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelNodeExcludeBalancers: "true"}}}},
		{want: false, input: &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{v1.LabelNodeExcludeBalancers: "foo"}}}},

		{want: false, input: &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}},
			Spec: v1.NodeSpec{Taints: []v1.Taint{{Key: ToBeDeletedTaint, Value: fmt.Sprint(time.Now().Unix()), Effect: v1.TaintEffectNoSchedule}}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if result := respectsPredicates(tt.input, allNodePredicates...); result != tt.want {
				t.Errorf("matchesPredicates() = %v, want %v", result, tt.want)
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
	}{{
		name: "ListWithPredicate filter Running node",
		predicate: func(node *v1.Node) bool {
			return node.Status.Phase == v1.NodeRunning
		},
		expect: []*v1.Node{nodes[1], nodes[3]},
	}, {
		name: "ListWithPredicate filter Pending node",
		predicate: func(node *v1.Node) bool {
			return node.Status.Phase == v1.NodePending
		},
		expect: []*v1.Node{nodes[0], nodes[2], nodes[4]},
	}, {
		name: "ListWithPredicate filter Terminated node",
		predicate: func(node *v1.Node) bool {
			return node.Status.Phase == v1.NodeTerminated
		},
		expect: nil,
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			get, err := listWithPredicates(fakeInformerFactory.Core().V1().Nodes().Lister(), test.predicate)
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

var providerID = "providerID"

type nodeTweak func(n *v1.Node)

// TODO: use this pattern in all the tests above.
func makeNode(tweaks ...nodeTweak) *v1.Node {
	n := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node",
			Labels: map[string]string{},
		},
		Spec: v1.NodeSpec{
			Taints:     []v1.Taint{},
			ProviderID: providerID,
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{{
				Type:   v1.NodeReady,
				Status: v1.ConditionTrue,
			}},
		},
	}

	for _, tw := range tweaks {
		tw(n)
	}
	return n
}

func tweakName(name string) nodeTweak {
	return func(n *v1.Node) {
		n.Name = name
	}
}

func tweakAddTaint(key string) nodeTweak {
	return func(n *v1.Node) {
		n.Spec.Taints = append(n.Spec.Taints, v1.Taint{Key: key})
	}
}

func tweakSetLabel(key, val string) nodeTweak {
	return func(n *v1.Node) {
		n.Labels[key] = val
	}
}

func tweakSetCondition(condType v1.NodeConditionType, condStatus v1.ConditionStatus) nodeTweak {
	return func(n *v1.Node) {
		var cond *v1.NodeCondition
		for i := range n.Status.Conditions {
			c := &n.Status.Conditions[i]
			if c.Type == condType {
				cond = c
				break
			}
		}
		if cond == nil {
			n.Status.Conditions = append(n.Status.Conditions, v1.NodeCondition{})
			cond = &n.Status.Conditions[len(n.Status.Conditions)-1]
		}
		*cond = v1.NodeCondition{
			Type:   condType,
			Status: condStatus,
		}
	}
}

func tweakSetReady(val bool) nodeTweak {
	var condStatus v1.ConditionStatus

	if val {
		condStatus = v1.ConditionTrue
	} else {
		condStatus = v1.ConditionFalse
	}

	return tweakSetCondition(v1.NodeReady, condStatus)
}

func tweakUnsetCondition(condType v1.NodeConditionType) nodeTweak {
	return func(n *v1.Node) {
		for i := range n.Status.Conditions {
			c := &n.Status.Conditions[i]
			if c.Type == condType {
				// Hacky but easy.
				c.Type = "SomethingElse"
				break
			}
		}
	}
}

func tweakDeleted() nodeTweak {
	return func(n *v1.Node) {
		n.DeletionTimestamp = &metav1.Time{
			Time: time.Now(),
		}
	}
}

func tweakProviderID(id string) nodeTweak {
	return func(n *v1.Node) {
		n.Spec.ProviderID = id
	}
}

func Test_shouldSyncUpdatedNode_individualPredicates(t *testing.T) {
	testcases := []struct {
		name       string
		oldNode    *v1.Node
		newNode    *v1.Node
		shouldSync bool
	}{{
		name:       "nothing changed",
		oldNode:    makeNode(),
		newNode:    makeNode(),
		shouldSync: false,
	}, {
		name:       "excluded F->T",
		oldNode:    makeNode(),
		newNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
		shouldSync: true,
	}, {
		name:       "excluded changed T->F",
		oldNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
		newNode:    makeNode(),
		shouldSync: true,
	}, {
		name:       "excluded changed T->T",
		oldNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
		newNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
		shouldSync: false,
	}, {
		name:       "other taint F->T",
		oldNode:    makeNode(),
		newNode:    makeNode(tweakAddTaint("other")),
		shouldSync: false,
	}, {
		name:       "other taint T->F",
		oldNode:    makeNode(tweakAddTaint("other")),
		newNode:    makeNode(),
		shouldSync: false,
	}, {
		name:       "other label changed F->T",
		oldNode:    makeNode(),
		newNode:    makeNode(tweakSetLabel("other", "")),
		shouldSync: false,
	}, {
		name:       "other label changed T->F",
		oldNode:    makeNode(tweakSetLabel("other", "")),
		newNode:    makeNode(),
		shouldSync: false,
	}, {
		name:       "readiness changed F->F",
		oldNode:    makeNode(tweakSetReady(false)),
		newNode:    makeNode(tweakSetReady(false)),
		shouldSync: false,
	}, {
		name:       "readiness changed F->unset",
		oldNode:    makeNode(tweakSetReady(false)),
		newNode:    makeNode(tweakUnsetCondition(v1.NodeReady)),
		shouldSync: false,
	}, {
		name:       "readiness changed unset->F",
		oldNode:    makeNode(tweakUnsetCondition(v1.NodeReady)),
		newNode:    makeNode(tweakSetReady(false)),
		shouldSync: false,
	}, {
		name:       "readiness changed unset->unset",
		oldNode:    makeNode(tweakUnsetCondition(v1.NodeReady)),
		newNode:    makeNode(tweakUnsetCondition(v1.NodeReady)),
		shouldSync: false,
	}, {
		name:       "ready F, other condition changed F->T",
		oldNode:    makeNode(tweakSetReady(false), tweakSetCondition(v1.NodeDiskPressure, v1.ConditionFalse)),
		newNode:    makeNode(tweakSetReady(false), tweakSetCondition(v1.NodeDiskPressure, v1.ConditionTrue)),
		shouldSync: false,
	}, {
		name:       "ready F, other condition changed T->F",
		oldNode:    makeNode(tweakSetReady(false), tweakSetCondition(v1.NodeDiskPressure, v1.ConditionTrue)),
		newNode:    makeNode(tweakSetReady(false), tweakSetCondition(v1.NodeDiskPressure, v1.ConditionFalse)),
		shouldSync: false,
	}, {
		name:       "ready T, other condition changed F->T",
		oldNode:    makeNode(tweakSetCondition("Other", v1.ConditionFalse)),
		newNode:    makeNode(tweakSetCondition("Other", v1.ConditionTrue)),
		shouldSync: false,
	}, {
		name:       "ready T, other condition changed T->F",
		oldNode:    makeNode(tweakSetCondition("Other", v1.ConditionTrue)),
		newNode:    makeNode(tweakSetCondition("Other", v1.ConditionFalse)),
		shouldSync: false,
	}, {
		name:       "deletionTimestamp F -> T",
		oldNode:    makeNode(),
		newNode:    makeNode(tweakDeleted()),
		shouldSync: false,
	}, {
		name:       "providerID set F -> T",
		oldNode:    makeNode(tweakProviderID("")),
		newNode:    makeNode(),
		shouldSync: true,
	}, {
		name:       "providerID set F -> T",
		oldNode:    makeNode(tweakProviderID("")),
		newNode:    makeNode(),
		shouldSync: true,
	}, {
		name:       "providerID set T-> F",
		oldNode:    makeNode(),
		newNode:    makeNode(tweakProviderID("")),
		shouldSync: true,
	}, {
		name:       "providerID change",
		oldNode:    makeNode(),
		newNode:    makeNode(tweakProviderID(providerID + "-2")),
		shouldSync: true,
	}}
	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			shouldSync := shouldSyncUpdatedNode(testcase.oldNode, testcase.newNode)
			if shouldSync != testcase.shouldSync {
				t.Errorf("unexpected result from shouldSyncNode, expected: %v, actual: %v", testcase.shouldSync, shouldSync)
			}
		})
	}
}

func Test_shouldSyncUpdatedNode_compoundedPredicates(t *testing.T) {
	type testCase struct {
		name       string
		oldNode    *v1.Node
		newNode    *v1.Node
		shouldSync bool
	}
	testcases := []testCase{
		{
			name:       "tainted T, excluded F->T",
			oldNode:    makeNode(tweakAddTaint(ToBeDeletedTaint)),
			newNode:    makeNode(tweakAddTaint(ToBeDeletedTaint), tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			shouldSync: true,
		}, {
			name:       "tainted T, excluded T->F",
			oldNode:    makeNode(tweakAddTaint(ToBeDeletedTaint), tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			newNode:    makeNode(tweakAddTaint(ToBeDeletedTaint)),
			shouldSync: true,
		}, {
			name:       "tainted T, providerID set F->T",
			oldNode:    makeNode(tweakAddTaint(ToBeDeletedTaint), tweakProviderID("")),
			newNode:    makeNode(tweakAddTaint(ToBeDeletedTaint)),
			shouldSync: true,
		}, {
			name:       "tainted T, providerID set T->F",
			oldNode:    makeNode(tweakAddTaint(ToBeDeletedTaint)),
			newNode:    makeNode(tweakAddTaint(ToBeDeletedTaint), tweakProviderID("")),
			shouldSync: true,
		}, {
			name:       "tainted T, providerID change",
			oldNode:    makeNode(tweakAddTaint(ToBeDeletedTaint)),
			newNode:    makeNode(tweakAddTaint(ToBeDeletedTaint), tweakProviderID(providerID+"-2")),
			shouldSync: true,
		}, {
			name:       "tainted T, ready F->T",
			oldNode:    makeNode(tweakAddTaint(ToBeDeletedTaint), tweakSetReady(false)),
			newNode:    makeNode(tweakAddTaint(ToBeDeletedTaint)),
			shouldSync: false,
		}, {
			name:       "tainted T, ready T->F",
			oldNode:    makeNode(tweakAddTaint(ToBeDeletedTaint)),
			newNode:    makeNode(tweakAddTaint(ToBeDeletedTaint), tweakSetReady(false)),
			shouldSync: false,
		}, {
			name:       "excluded T, tainted F->T",
			oldNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			newNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, ""), tweakAddTaint(ToBeDeletedTaint)),
			shouldSync: false,
		}, {
			name:       "excluded T, tainted T->F",
			oldNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, ""), tweakAddTaint(ToBeDeletedTaint)),
			newNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			shouldSync: false,
		}, {
			name:       "excluded T, ready F->T",
			oldNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, ""), tweakSetReady(false)),
			newNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			shouldSync: false,
		}, {
			name:       "excluded T, ready T->F",
			oldNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			newNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, ""), tweakSetReady(false)),
			shouldSync: false,
		}, {
			name:       "excluded T, providerID set F->T",
			oldNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, ""), tweakProviderID("")),
			newNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			shouldSync: true,
		}, {
			name:       "excluded T, providerID set T->F",
			oldNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			newNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, ""), tweakProviderID("")),
			shouldSync: true,
		}, {
			name:       "excluded T, providerID change",
			oldNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			newNode:    makeNode(tweakSetLabel(v1.LabelNodeExcludeBalancers, ""), tweakProviderID(providerID+"-2")),
			shouldSync: true,
		}, {
			name:       "ready F, tainted F->T",
			oldNode:    makeNode(tweakSetReady(false)),
			newNode:    makeNode(tweakSetReady(false), tweakAddTaint(ToBeDeletedTaint)),
			shouldSync: false,
		}, {
			name:       "ready F, tainted T->F",
			oldNode:    makeNode(tweakSetReady(false), tweakAddTaint(ToBeDeletedTaint)),
			newNode:    makeNode(tweakSetReady(false)),
			shouldSync: false,
		}, {
			name:       "ready F, excluded F->T",
			oldNode:    makeNode(tweakSetReady(false)),
			newNode:    makeNode(tweakSetReady(false), tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			shouldSync: true,
		}, {
			name:       "ready F, excluded T->F",
			oldNode:    makeNode(tweakSetReady(false), tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			newNode:    makeNode(tweakSetReady(false)),
			shouldSync: true,
		}, {
			name:       "ready F, providerID set F->T",
			oldNode:    makeNode(tweakSetReady(false), tweakProviderID("")),
			newNode:    makeNode(tweakSetReady(false)),
			shouldSync: true,
		}, {
			name:       "ready F, providerID set T->F",
			oldNode:    makeNode(tweakSetReady(false)),
			newNode:    makeNode(tweakSetReady(false), tweakProviderID("")),
			shouldSync: true,
		}, {
			name:       "ready F, providerID change",
			oldNode:    makeNode(tweakSetReady(false)),
			newNode:    makeNode(tweakSetReady(false), tweakProviderID(providerID+"-2")),
			shouldSync: true,
		}, {
			name:       "providerID unset, excluded F->T",
			oldNode:    makeNode(tweakProviderID("")),
			newNode:    makeNode(tweakProviderID(""), tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			shouldSync: true,
		}, {
			name:       "providerID unset, excluded T->F",
			oldNode:    makeNode(tweakProviderID(""), tweakSetLabel(v1.LabelNodeExcludeBalancers, "")),
			newNode:    makeNode(tweakProviderID("")),
			shouldSync: true,
		}, {
			name:       "providerID unset, ready T->F",
			oldNode:    makeNode(tweakProviderID("")),
			newNode:    makeNode(tweakProviderID(""), tweakSetReady(true)),
			shouldSync: false,
		}, {
			name:       "providerID unset, ready F->T",
			oldNode:    makeNode(tweakProviderID("")),
			newNode:    makeNode(tweakProviderID(""), tweakSetReady(false)),
			shouldSync: false,
		}, {
			name:       "providerID unset, tainted T->F",
			oldNode:    makeNode(tweakProviderID(""), tweakAddTaint(ToBeDeletedTaint)),
			newNode:    makeNode(tweakProviderID("")),
			shouldSync: false,
		}, {
			name:       "providerID unset, tainted F->T",
			oldNode:    makeNode(tweakProviderID("")),
			newNode:    makeNode(tweakProviderID(""), tweakAddTaint(ToBeDeletedTaint)),
			shouldSync: false,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			shouldSync := shouldSyncUpdatedNode(testcase.oldNode, testcase.newNode)
			if shouldSync != testcase.shouldSync {
				t.Errorf("unexpected result from shouldSyncNode, expected: %v, actual: %v", testcase.shouldSync, shouldSync)
			}
		})
	}
}

func TestServiceQueueDelay(t *testing.T) {
	const ns = metav1.NamespaceDefault

	tests := []struct {
		name           string
		lbCloudErr     error
		wantRetryDelay time.Duration
	}{
		{
			name:       "processing successful",
			lbCloudErr: nil,
		},
		{
			name:       "regular error",
			lbCloudErr: errors.New("something went wrong"),
		},
		{
			name:           "retry error",
			lbCloudErr:     api.NewRetryError("LB create in progress", 42*time.Second),
			wantRetryDelay: 42 * time.Second,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			controller, cloud, client := newController(ctx)
			queue := &spyWorkQueue{TypedRateLimitingInterface: workqueue.NewTypedRateLimitingQueueWithConfig(
				workqueue.DefaultTypedControllerRateLimiter[string](),
				workqueue.TypedRateLimitingQueueConfig[string]{Name: "test-service-queue-delay"},
			)}
			controller.serviceQueue = queue
			cloud.Err = tc.lbCloudErr

			serviceCache := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			controller.serviceLister = corelisters.NewServiceLister(serviceCache)

			svc := defaultExternalService()
			if err := serviceCache.Add(svc); err != nil {
				t.Fatalf("adding service %s to cache: %s", svc.Name, err)
			}

			_, err := client.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			key, err := cache.MetaNamespaceKeyFunc(svc)
			if err != nil {
				t.Fatalf("creating meta namespace key: %s", err)
			}
			queue.Add(key)

			done := controller.processNextServiceItem(ctx)
			if !done {
				t.Fatal("processNextServiceItem stopped prematurely")
			}

			// Expect no requeues unless we hit an error that is not a retry
			// error.
			wantNumRequeues := 0
			var re *api.RetryError
			isRetryError := errors.As(tc.lbCloudErr, &re)
			if tc.lbCloudErr != nil && !isRetryError {
				wantNumRequeues = 1
			}

			if gotNumRequeues := queue.NumRequeues(key); gotNumRequeues != wantNumRequeues {
				t.Fatalf("got %d requeue(s), want %d", gotNumRequeues, wantNumRequeues)
			}

			if tc.wantRetryDelay > 0 {
				items := queue.getItems()
				if len(items) != 1 {
					t.Fatalf("got %d item(s), want 1", len(items))
				}
				if gotDelay := items[0].Delay; gotDelay != tc.wantRetryDelay {
					t.Fatalf("got delay %s, want %s", gotDelay, tc.wantRetryDelay)
				}
			}
		})
	}
}

type fakeNodeLister struct {
	cache []*v1.Node
	err   error
}

func newFakeNodeLister(err error, nodes ...*v1.Node) *fakeNodeLister {
	ret := &fakeNodeLister{}
	ret.cache = nodes
	ret.err = err
	return ret
}

// List lists all Nodes in the indexer.
// Objects returned here must be treated as read-only.
func (l *fakeNodeLister) List(selector labels.Selector) (ret []*v1.Node, err error) {
	return l.cache, l.err
}

// Get retrieves the Node from the index for a given name.
// Objects returned here must be treated as read-only.
func (l *fakeNodeLister) Get(name string) (*v1.Node, error) {
	for _, node := range l.cache {
		if node.Name == name {
			return node, nil
		}
	}
	return nil, nil
}

// spyWorkQueue implements a work queue and adds the ability to inspect processed
// items for testing purposes.
type spyWorkQueue struct {
	workqueue.TypedRateLimitingInterface[string]
	items []spyQueueItem
}

// spyQueueItem represents an item that was being processed.
type spyQueueItem struct {
	Key string
	// Delay represents the delayed duration if and only if AddAfter was invoked.
	Delay time.Duration
}

// AddAfter is like workqueue.RateLimitingInterface.AddAfter but records the
// added key and delay internally.
func (f *spyWorkQueue) AddAfter(key string, delay time.Duration) {
	f.items = append(f.items, spyQueueItem{
		Key:   key,
		Delay: delay,
	})

	f.TypedRateLimitingInterface.AddAfter(key, delay)
}

// getItems returns all items that were recorded.
func (f *spyWorkQueue) getItems() []spyQueueItem {
	return f.items
}
