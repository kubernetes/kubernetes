/*
Copyright 2022 The Kubernetes Authors.

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

package controller

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/klog/v2/ktesting/init"
)

func TestController(t *testing.T) {
	claimKey := "claim:default/claim"
	claimName := "claim"
	claimNamespace := "default"
	driverName := "mock-driver"
	className := "mock-class"
	otherDriverName := "other-driver"
	otherClassName := "other-class"
	ourFinalizer := driverName + "/deletion-protection"
	otherFinalizer := otherDriverName + "/deletion-protection"
	classes := []*resourcev1alpha2.ResourceClass{
		createClass(className, driverName),
		createClass(otherClassName, otherDriverName),
	}
	claim := createClaim(claimName, claimNamespace, className)
	otherClaim := createClaim(claimName, claimNamespace, otherClassName)
	delayedClaim := claim.DeepCopy()
	delayedClaim.Spec.AllocationMode = resourcev1alpha2.AllocationModeWaitForFirstConsumer
	podName := "pod"
	podKey := "schedulingCtx:default/pod"
	pod := createPod(podName, claimNamespace, nil)
	podClaimName := "my-pod-claim"
	podSchedulingCtx := createPodSchedulingContexts(pod)
	podWithClaim := createPod(podName, claimNamespace, map[string]string{podClaimName: claimName})
	nodeName := "worker"
	otherNodeName := "worker-2"
	unsuitableNodes := []string{otherNodeName}
	potentialNodes := []string{nodeName, otherNodeName}
	maxNodes := make([]string, resourcev1alpha2.PodSchedulingNodeListMaxSize)
	for i := range maxNodes {
		maxNodes[i] = fmt.Sprintf("node-%d", i)
	}
	withDeletionTimestamp := func(claim *resourcev1alpha2.ResourceClaim) *resourcev1alpha2.ResourceClaim {
		var deleted metav1.Time
		claim = claim.DeepCopy()
		claim.DeletionTimestamp = &deleted
		return claim
	}
	withReservedFor := func(claim *resourcev1alpha2.ResourceClaim, pod *corev1.Pod) *resourcev1alpha2.ResourceClaim {
		claim = claim.DeepCopy()
		claim.Status.ReservedFor = append(claim.Status.ReservedFor, resourcev1alpha2.ResourceClaimConsumerReference{
			Resource: "pods",
			Name:     pod.Name,
			UID:      pod.UID,
		})
		return claim
	}
	withFinalizer := func(claim *resourcev1alpha2.ResourceClaim, finalizer string) *resourcev1alpha2.ResourceClaim {
		claim = claim.DeepCopy()
		claim.Finalizers = append(claim.Finalizers, finalizer)
		return claim
	}
	allocation := resourcev1alpha2.AllocationResult{}
	withAllocate := func(claim *resourcev1alpha2.ResourceClaim) *resourcev1alpha2.ResourceClaim {
		// Any allocated claim must have our finalizer.
		claim = withFinalizer(claim, ourFinalizer)
		claim.Status.Allocation = &allocation
		claim.Status.DriverName = driverName
		return claim
	}
	withDeallocate := func(claim *resourcev1alpha2.ResourceClaim) *resourcev1alpha2.ResourceClaim {
		claim.Status.DeallocationRequested = true
		return claim
	}
	withSelectedNode := func(podSchedulingCtx *resourcev1alpha2.PodSchedulingContext) *resourcev1alpha2.PodSchedulingContext {
		podSchedulingCtx = podSchedulingCtx.DeepCopy()
		podSchedulingCtx.Spec.SelectedNode = nodeName
		return podSchedulingCtx
	}
	withSpecificUnsuitableNodes := func(podSchedulingCtx *resourcev1alpha2.PodSchedulingContext, unsuitableNodes []string) *resourcev1alpha2.PodSchedulingContext {
		podSchedulingCtx = podSchedulingCtx.DeepCopy()
		podSchedulingCtx.Status.ResourceClaims = append(podSchedulingCtx.Status.ResourceClaims,
			resourcev1alpha2.ResourceClaimSchedulingStatus{Name: podClaimName, UnsuitableNodes: unsuitableNodes},
		)
		return podSchedulingCtx
	}
	withUnsuitableNodes := func(podSchedulingCtx *resourcev1alpha2.PodSchedulingContext) *resourcev1alpha2.PodSchedulingContext {
		return withSpecificUnsuitableNodes(podSchedulingCtx, unsuitableNodes)
	}
	withSpecificPotentialNodes := func(podSchedulingCtx *resourcev1alpha2.PodSchedulingContext, potentialNodes []string) *resourcev1alpha2.PodSchedulingContext {
		podSchedulingCtx = podSchedulingCtx.DeepCopy()
		podSchedulingCtx.Spec.PotentialNodes = potentialNodes
		return podSchedulingCtx
	}
	withPotentialNodes := func(podSchedulingCtx *resourcev1alpha2.PodSchedulingContext) *resourcev1alpha2.PodSchedulingContext {
		return withSpecificPotentialNodes(podSchedulingCtx, potentialNodes)
	}

	var m mockDriver

	for name, test := range map[string]struct {
		key                                  string
		driver                               mockDriver
		classes                              []*resourcev1alpha2.ResourceClass
		pod                                  *corev1.Pod
		schedulingCtx, expectedSchedulingCtx *resourcev1alpha2.PodSchedulingContext
		claim, expectedClaim                 *resourcev1alpha2.ResourceClaim
		expectedError                        string
	}{
		"invalid-key": {
			key:           "claim:x/y/z",
			expectedError: `unexpected key format: "x/y/z"`,
		},
		"not-found": {
			key: "claim:default/claim",
		},
		"wrong-driver": {
			key:           claimKey,
			classes:       classes,
			claim:         otherClaim,
			expectedClaim: otherClaim,
			expectedError: errRequeue.Error(), // class might change
		},
		// Immediate allocation:
		// deletion time stamp set, our finalizer set, not allocated  -> remove finalizer
		"immediate-deleted-finalizer-removal": {
			key:           claimKey,
			classes:       classes,
			claim:         withFinalizer(withDeletionTimestamp(claim), ourFinalizer),
			driver:        m.expectDeallocate(map[string]error{claimName: nil}),
			expectedClaim: withDeletionTimestamp(claim),
		},
		// deletion time stamp set, our finalizer set, not allocated, stopping fails  -> requeue
		"immediate-deleted-finalizer-stop-failure": {
			key:           claimKey,
			classes:       classes,
			claim:         withFinalizer(withDeletionTimestamp(claim), ourFinalizer),
			driver:        m.expectDeallocate(map[string]error{claimName: errors.New("fake error")}),
			expectedClaim: withFinalizer(withDeletionTimestamp(claim), ourFinalizer),
			expectedError: "stop allocation: fake error",
		},
		// deletion time stamp set, other finalizer set, not allocated  -> do nothing
		"immediate-deleted-finalizer-no-removal": {
			key:           claimKey,
			classes:       classes,
			claim:         withFinalizer(withDeletionTimestamp(claim), otherFinalizer),
			expectedClaim: withFinalizer(withDeletionTimestamp(claim), otherFinalizer),
		},
		// deletion time stamp set, finalizer set, allocated  -> deallocate
		"immediate-deleted-allocated": {
			key:           claimKey,
			classes:       classes,
			claim:         withAllocate(withDeletionTimestamp(claim)),
			driver:        m.expectDeallocate(map[string]error{claimName: nil}),
			expectedClaim: withDeletionTimestamp(claim),
		},
		// deletion time stamp set, finalizer set, allocated, deallocation fails  -> requeue
		"immediate-deleted-deallocate-failure": {
			key:           claimKey,
			classes:       classes,
			claim:         withAllocate(withDeletionTimestamp(claim)),
			driver:        m.expectDeallocate(map[string]error{claimName: errors.New("fake error")}),
			expectedClaim: withAllocate(withDeletionTimestamp(claim)),
			expectedError: "deallocate: fake error",
		},
		// deletion time stamp set, finalizer not set -> do nothing
		"immediate-deleted-no-finalizer": {
			key:           claimKey,
			classes:       classes,
			claim:         withDeletionTimestamp(claim),
			expectedClaim: withDeletionTimestamp(claim),
		},
		// not deleted, not allocated, no finalizer -> add finalizer, allocate
		"immediate-do-allocation": {
			key:     claimKey,
			classes: classes,
			claim:   claim,
			driver: m.expectClassParameters(map[string]interface{}{className: 1}).
				expectClaimParameters(map[string]interface{}{claimName: 2}).
				expectAllocate(map[string]allocate{claimName: {allocResult: &allocation, allocErr: nil}}),
			expectedClaim: withAllocate(claim),
		},
		// not deleted, not allocated, finalizer -> allocate
		"immediate-continue-allocation": {
			key:     claimKey,
			classes: classes,
			claim:   withFinalizer(claim, ourFinalizer),
			driver: m.expectClassParameters(map[string]interface{}{className: 1}).
				expectClaimParameters(map[string]interface{}{claimName: 2}).
				expectAllocate(map[string]allocate{claimName: {allocResult: &allocation, allocErr: nil}}),
			expectedClaim: withAllocate(claim),
		},
		// not deleted, not allocated, finalizer, fail allocation -> requeue
		"immediate-fail-allocation": {
			key:     claimKey,
			classes: classes,
			claim:   withFinalizer(claim, ourFinalizer),
			driver: m.expectClassParameters(map[string]interface{}{className: 1}).
				expectClaimParameters(map[string]interface{}{claimName: 2}).
				expectAllocate(map[string]allocate{claimName: {allocErr: errors.New("fake error")}}),
			expectedClaim: withFinalizer(claim, ourFinalizer),
			expectedError: "allocate: fake error",
		},
		// not deleted, allocated -> do nothing
		"immediate-allocated-nop": {
			key:           claimKey,
			classes:       classes,
			claim:         withAllocate(claim),
			expectedClaim: withAllocate(claim),
		},

		// not deleted, reallocate -> deallocate
		"immediate-allocated-reallocate": {
			key:           claimKey,
			classes:       classes,
			claim:         withDeallocate(withAllocate(claim)),
			driver:        m.expectDeallocate(map[string]error{claimName: nil}),
			expectedClaim: claim,
		},

		// not deleted, reallocate, deallocate failure -> requeue
		"immediate-allocated-fail-deallocation-during-reallocate": {
			key:           claimKey,
			classes:       classes,
			claim:         withDeallocate(withAllocate(claim)),
			driver:        m.expectDeallocate(map[string]error{claimName: errors.New("fake error")}),
			expectedClaim: withDeallocate(withAllocate(claim)),
			expectedError: "deallocate: fake error",
		},

		// Delayed allocation is similar in some cases, but not quite
		// the same.
		// deletion time stamp set, our finalizer set, not allocated  -> remove finalizer
		"delayed-deleted-finalizer-removal": {
			key:           claimKey,
			classes:       classes,
			claim:         withFinalizer(withDeletionTimestamp(delayedClaim), ourFinalizer),
			driver:        m.expectDeallocate(map[string]error{claimName: nil}),
			expectedClaim: withDeletionTimestamp(delayedClaim),
		},
		// deletion time stamp set, our finalizer set, not allocated, stopping fails  -> requeue
		"delayed-deleted-finalizer-stop-failure": {
			key:           claimKey,
			classes:       classes,
			claim:         withFinalizer(withDeletionTimestamp(delayedClaim), ourFinalizer),
			driver:        m.expectDeallocate(map[string]error{claimName: errors.New("fake error")}),
			expectedClaim: withFinalizer(withDeletionTimestamp(delayedClaim), ourFinalizer),
			expectedError: "stop allocation: fake error",
		},
		// deletion time stamp set, other finalizer set, not allocated  -> do nothing
		"delayed-deleted-finalizer-no-removal": {
			key:           claimKey,
			classes:       classes,
			claim:         withFinalizer(withDeletionTimestamp(delayedClaim), otherFinalizer),
			expectedClaim: withFinalizer(withDeletionTimestamp(delayedClaim), otherFinalizer),
		},
		// deletion time stamp set, finalizer set, allocated  -> deallocate
		"delayed-deleted-allocated": {
			key:           claimKey,
			classes:       classes,
			claim:         withAllocate(withDeletionTimestamp(delayedClaim)),
			driver:        m.expectDeallocate(map[string]error{claimName: nil}),
			expectedClaim: withDeletionTimestamp(delayedClaim),
		},
		// deletion time stamp set, finalizer set, allocated, deallocation fails  -> requeue
		"delayed-deleted-deallocate-failure": {
			key:           claimKey,
			classes:       classes,
			claim:         withAllocate(withDeletionTimestamp(delayedClaim)),
			driver:        m.expectDeallocate(map[string]error{claimName: errors.New("fake error")}),
			expectedClaim: withAllocate(withDeletionTimestamp(delayedClaim)),
			expectedError: "deallocate: fake error",
		},
		// deletion time stamp set, finalizer not set -> do nothing
		"delayed-deleted-no-finalizer": {
			key:           claimKey,
			classes:       classes,
			claim:         withDeletionTimestamp(delayedClaim),
			expectedClaim: withDeletionTimestamp(delayedClaim),
		},
		// waiting for first consumer -> do nothing
		"delayed-pending": {
			key:           claimKey,
			classes:       classes,
			claim:         delayedClaim,
			expectedClaim: delayedClaim,
		},

		// pod with no claims -> shouldn't occur, check again anyway
		"pod-nop": {
			key:                   podKey,
			pod:                   pod,
			schedulingCtx:         withSelectedNode(podSchedulingCtx),
			expectedSchedulingCtx: withSelectedNode(podSchedulingCtx),
			expectedError:         errPeriodic.Error(),
		},

		// pod with immediate allocation and selected node -> shouldn't occur, check again in case that claim changes
		"pod-immediate": {
			key:                   podKey,
			claim:                 claim,
			expectedClaim:         claim,
			pod:                   podWithClaim,
			schedulingCtx:         withSelectedNode(podSchedulingCtx),
			expectedSchedulingCtx: withSelectedNode(podSchedulingCtx),
			expectedError:         errPeriodic.Error(),
		},

		// pod with delayed allocation, no potential nodes -> shouldn't occur
		"pod-delayed-no-nodes": {
			key:                   podKey,
			classes:               classes,
			claim:                 delayedClaim,
			expectedClaim:         delayedClaim,
			pod:                   podWithClaim,
			schedulingCtx:         podSchedulingCtx,
			expectedSchedulingCtx: podSchedulingCtx,
		},

		// pod with delayed allocation, potential nodes -> provide unsuitable nodes
		"pod-delayed-info": {
			key:           podKey,
			classes:       classes,
			claim:         delayedClaim,
			expectedClaim: delayedClaim,
			pod:           podWithClaim,
			schedulingCtx: withPotentialNodes(podSchedulingCtx),
			driver: m.expectClassParameters(map[string]interface{}{className: 1}).
				expectClaimParameters(map[string]interface{}{claimName: 2}).
				expectUnsuitableNodes(map[string][]string{podClaimName: unsuitableNodes}, nil),
			expectedSchedulingCtx: withUnsuitableNodes(withPotentialNodes(podSchedulingCtx)),
			expectedError:         errPeriodic.Error(),
		},

		// pod with delayed allocation, potential nodes, selected node, missing class -> failure
		"pod-delayed-missing-class": {
			key:                   podKey,
			claim:                 delayedClaim,
			expectedClaim:         delayedClaim,
			pod:                   podWithClaim,
			schedulingCtx:         withSelectedNode(withPotentialNodes(podSchedulingCtx)),
			expectedSchedulingCtx: withSelectedNode(withPotentialNodes(podSchedulingCtx)),
			expectedError:         `pod claim my-pod-claim: resourceclass.resource.k8s.io "mock-class" not found`,
		},

		// pod with delayed allocation, potential nodes, selected node -> allocate
		"pod-delayed-allocate": {
			key:           podKey,
			classes:       classes,
			claim:         delayedClaim,
			expectedClaim: withReservedFor(withAllocate(delayedClaim), pod),
			pod:           podWithClaim,
			schedulingCtx: withSelectedNode(withPotentialNodes(podSchedulingCtx)),
			driver: m.expectClassParameters(map[string]interface{}{className: 1}).
				expectClaimParameters(map[string]interface{}{claimName: 2}).
				expectUnsuitableNodes(map[string][]string{podClaimName: unsuitableNodes}, nil).
				expectAllocate(map[string]allocate{claimName: {allocResult: &allocation, selectedNode: nodeName, allocErr: nil}}),
			expectedSchedulingCtx: withUnsuitableNodes(withSelectedNode(withPotentialNodes(podSchedulingCtx))),
			expectedError:         errPeriodic.Error(),
		},
		// pod with delayed allocation, potential nodes, selected node, all unsuitable -> update unsuitable nodes
		"pod-selected-is-potential-node": {
			key:           podKey,
			classes:       classes,
			claim:         delayedClaim,
			expectedClaim: delayedClaim,
			pod:           podWithClaim,
			schedulingCtx: withPotentialNodes(withSelectedNode(withPotentialNodes(podSchedulingCtx))),
			driver: m.expectClassParameters(map[string]interface{}{className: 1}).
				expectClaimParameters(map[string]interface{}{claimName: 2}).
				expectUnsuitableNodes(map[string][]string{podClaimName: potentialNodes}, nil),
			expectedSchedulingCtx: withSpecificUnsuitableNodes(withSelectedNode(withPotentialNodes(podSchedulingCtx)), potentialNodes),
			expectedError:         errPeriodic.Error(),
		},
		// pod with delayed allocation, max potential nodes, other selected node, all unsuitable -> update unsuitable nodes with truncation at start
		"pod-selected-is-potential-node-truncate-first": {
			key:           podKey,
			classes:       classes,
			claim:         delayedClaim,
			expectedClaim: delayedClaim,
			pod:           podWithClaim,
			schedulingCtx: withSpecificPotentialNodes(withSelectedNode(withSpecificPotentialNodes(podSchedulingCtx, maxNodes)), maxNodes),
			driver: m.expectClassParameters(map[string]interface{}{className: 1}).
				expectClaimParameters(map[string]interface{}{claimName: 2}).
				expectUnsuitableNodes(map[string][]string{podClaimName: append(maxNodes, nodeName)}, nil),
			expectedSchedulingCtx: withSpecificUnsuitableNodes(withSelectedNode(withSpecificPotentialNodes(podSchedulingCtx, maxNodes)), append(maxNodes[1:], nodeName)),
			expectedError:         errPeriodic.Error(),
		},
		// pod with delayed allocation, max potential nodes, other selected node, all unsuitable (but in reverse order) -> update unsuitable nodes with truncation at end
		"pod-selected-is-potential-node-truncate-last": {
			key:           podKey,
			classes:       classes,
			claim:         delayedClaim,
			expectedClaim: delayedClaim,
			pod:           podWithClaim,
			schedulingCtx: withSpecificPotentialNodes(withSelectedNode(withSpecificPotentialNodes(podSchedulingCtx, maxNodes)), maxNodes),
			driver: m.expectClassParameters(map[string]interface{}{className: 1}).
				expectClaimParameters(map[string]interface{}{claimName: 2}).
				expectUnsuitableNodes(map[string][]string{podClaimName: append([]string{nodeName}, maxNodes...)}, nil),
			expectedSchedulingCtx: withSpecificUnsuitableNodes(withSelectedNode(withSpecificPotentialNodes(podSchedulingCtx, maxNodes)), append([]string{nodeName}, maxNodes[:len(maxNodes)-1]...)),
			expectedError:         errPeriodic.Error(),
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)

			initialObjects := []runtime.Object{}
			for _, class := range test.classes {
				initialObjects = append(initialObjects, class)
			}
			if test.pod != nil {
				initialObjects = append(initialObjects, test.pod)
			}
			if test.schedulingCtx != nil {
				initialObjects = append(initialObjects, test.schedulingCtx)
			}
			if test.claim != nil {
				initialObjects = append(initialObjects, test.claim)
			}
			kubeClient, informerFactory := fakeK8s(initialObjects)
			rcInformer := informerFactory.Resource().V1alpha2().ResourceClasses()
			claimInformer := informerFactory.Resource().V1alpha2().ResourceClaims()
			podInformer := informerFactory.Core().V1().Pods()
			podSchedulingInformer := informerFactory.Resource().V1alpha2().PodSchedulingContexts()
			// Order is important: on function exit, we first must
			// cancel, then wait (last-in-first-out).
			defer informerFactory.Shutdown()
			defer cancel()

			for _, obj := range initialObjects {
				switch obj.(type) {
				case *resourcev1alpha2.ResourceClass:
					require.NoError(t, rcInformer.Informer().GetStore().Add(obj), "add resource class")
				case *resourcev1alpha2.ResourceClaim:
					require.NoError(t, claimInformer.Informer().GetStore().Add(obj), "add resource claim")
				case *corev1.Pod:
					require.NoError(t, podInformer.Informer().GetStore().Add(obj), "add pod")
				case *resourcev1alpha2.PodSchedulingContext:
					require.NoError(t, podSchedulingInformer.Informer().GetStore().Add(obj), "add pod scheduling")
				default:
					t.Fatalf("unknown initialObject type: %+v", obj)
				}
			}

			driver := test.driver
			driver.t = t

			ctrl := New(ctx, driverName, driver, kubeClient, informerFactory)
			informerFactory.Start(ctx.Done())
			if !cache.WaitForCacheSync(ctx.Done(),
				informerFactory.Resource().V1alpha2().ResourceClasses().Informer().HasSynced,
				informerFactory.Resource().V1alpha2().ResourceClaims().Informer().HasSynced,
				informerFactory.Resource().V1alpha2().PodSchedulingContexts().Informer().HasSynced,
			) {
				t.Fatal("could not sync caches")
			}
			_, err := ctrl.(*controller).syncKey(ctx, test.key)
			if err != nil && test.expectedError == "" {
				t.Fatalf("unexpected error: %v", err)
			}
			if err == nil && test.expectedError != "" {
				t.Fatalf("did not get expected error %q", test.expectedError)
			}
			if err != nil && err.Error() != test.expectedError {
				t.Fatalf("expected error %q, got %q", test.expectedError, err.Error())
			}
			claims, err := kubeClient.ResourceV1alpha2().ResourceClaims("").List(ctx, metav1.ListOptions{})
			require.NoError(t, err, "list claims")
			var expectedClaims []resourcev1alpha2.ResourceClaim
			if test.expectedClaim != nil {
				expectedClaims = append(expectedClaims, *test.expectedClaim)
			}
			assert.Equal(t, expectedClaims, claims.Items)

			podSchedulings, err := kubeClient.ResourceV1alpha2().PodSchedulingContexts("").List(ctx, metav1.ListOptions{})
			require.NoError(t, err, "list pod schedulings")
			var expectedPodSchedulings []resourcev1alpha2.PodSchedulingContext
			if test.expectedSchedulingCtx != nil {
				expectedPodSchedulings = append(expectedPodSchedulings, *test.expectedSchedulingCtx)
			}
			assert.Equal(t, expectedPodSchedulings, podSchedulings.Items)

			// TODO: add testing of events.
			// Right now, client-go/tools/record/event.go:267 fails during unit testing with
			// request namespace does not match object namespace, request: "" object: "default",
		})
	}
}

type mockDriver struct {
	t *testing.T

	// TODO: change this so that the mock driver expects calls in a certain order
	// and fails when the next call isn't the expected one or calls didn't happen
	classParameters      map[string]interface{}
	claimParameters      map[string]interface{}
	allocate             map[string]allocate
	deallocate           map[string]error
	unsuitableNodes      map[string][]string
	unsuitableNodesError error
}

type allocate struct {
	selectedNode string
	allocResult  *resourcev1alpha2.AllocationResult
	allocErr     error
}

func (m mockDriver) expectClassParameters(expected map[string]interface{}) mockDriver {
	m.classParameters = expected
	return m
}

func (m mockDriver) expectClaimParameters(expected map[string]interface{}) mockDriver {
	m.claimParameters = expected
	return m
}

func (m mockDriver) expectAllocate(expected map[string]allocate) mockDriver {
	m.allocate = expected
	return m
}

func (m mockDriver) expectDeallocate(expected map[string]error) mockDriver {
	m.deallocate = expected
	return m
}

func (m mockDriver) expectUnsuitableNodes(expected map[string][]string, err error) mockDriver {
	m.unsuitableNodes = expected
	m.unsuitableNodesError = err
	return m
}

func (m mockDriver) GetClassParameters(ctx context.Context, class *resourcev1alpha2.ResourceClass) (interface{}, error) {
	m.t.Logf("GetClassParameters(%s)", class)
	result, ok := m.classParameters[class.Name]
	if !ok {
		m.t.Fatal("unexpected GetClassParameters call")
	}
	if err, ok := result.(error); ok {
		return nil, err
	}
	return result, nil
}

func (m mockDriver) GetClaimParameters(ctx context.Context, claim *resourcev1alpha2.ResourceClaim, class *resourcev1alpha2.ResourceClass, classParameters interface{}) (interface{}, error) {
	m.t.Logf("GetClaimParameters(%s)", claim)
	result, ok := m.claimParameters[claim.Name]
	if !ok {
		m.t.Fatal("unexpected GetClaimParameters call")
	}
	if err, ok := result.(error); ok {
		return nil, err
	}
	return result, nil
}

func (m mockDriver) Allocate(ctx context.Context, claims []*ClaimAllocation, selectedNode string) {
	m.t.Logf("Allocate(number of claims %d)", len(claims))
	for _, claimAllocation := range claims {
		m.t.Logf("Allocate(%s)", claimAllocation.Claim.Name)
		allocate, ok := m.allocate[claimAllocation.Claim.Name]
		if !ok {
			m.t.Fatalf("unexpected Allocate call for claim %s", claimAllocation.Claim.Name)
		}
		assert.Equal(m.t, allocate.selectedNode, selectedNode, "selected node")
		claimAllocation.Error = allocate.allocErr
		claimAllocation.Allocation = allocate.allocResult
	}
	return
}

func (m mockDriver) Deallocate(ctx context.Context, claim *resourcev1alpha2.ResourceClaim) error {
	m.t.Logf("Deallocate(%s)", claim)
	err, ok := m.deallocate[claim.Name]
	if !ok {
		m.t.Fatal("unexpected Deallocate call")
	}
	return err
}

func (m mockDriver) UnsuitableNodes(ctx context.Context, pod *corev1.Pod, claims []*ClaimAllocation, potentialNodes []string) error {
	m.t.Logf("UnsuitableNodes(%s, %v, %v)", pod, claims, potentialNodes)
	if len(m.unsuitableNodes) == 0 {
		m.t.Fatal("unexpected UnsuitableNodes call")
	}
	if m.unsuitableNodesError != nil {
		return m.unsuitableNodesError
	}
	found := map[string]bool{}
	for _, delayed := range claims {
		unsuitableNodes, ok := m.unsuitableNodes[delayed.PodClaimName]
		if !ok {
			m.t.Errorf("unexpected pod claim: %s", delayed.PodClaimName)
		}
		delayed.UnsuitableNodes = unsuitableNodes
		found[delayed.PodClaimName] = true
	}
	for expectedName := range m.unsuitableNodes {
		if !found[expectedName] {
			m.t.Errorf("pod claim %s not in actual claims list", expectedName)
		}
	}
	return nil
}

func createClass(className, driverName string) *resourcev1alpha2.ResourceClass {
	return &resourcev1alpha2.ResourceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className,
		},
		DriverName: driverName,
	}
}

func createClaim(claimName, claimNamespace, className string) *resourcev1alpha2.ResourceClaim {
	return &resourcev1alpha2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      claimName,
			Namespace: claimNamespace,
		},
		Spec: resourcev1alpha2.ResourceClaimSpec{
			ResourceClassName: className,
			AllocationMode:    resourcev1alpha2.AllocationModeImmediate,
		},
	}
}

func createPod(podName, podNamespace string, claims map[string]string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: podNamespace,
			UID:       "1234",
		},
	}
	for podClaimName, claimName := range claims {
		pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims,
			corev1.PodResourceClaim{
				Name:              podClaimName,
				ResourceClaimName: &claimName,
			},
		)
	}
	return pod
}

func createPodSchedulingContexts(pod *corev1.Pod) *resourcev1alpha2.PodSchedulingContext {
	controller := true
	return &resourcev1alpha2.PodSchedulingContext{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				{
					Name:       pod.Name,
					Controller: &controller,
					UID:        pod.UID,
				},
			},
		},
	}
}

func fakeK8s(objs []runtime.Object) (kubernetes.Interface, informers.SharedInformerFactory) {
	// This is a very simple replacement for a real apiserver. For example,
	// it doesn't do defaulting and accepts updates to the status in normal
	// Update calls. Therefore this test does not catch when we use Update
	// instead of UpdateStatus. Reactors could be used to catch that, but
	// that seems overkill because E2E tests will find that.
	//
	// Interactions with the fake apiserver also never fail. TODO:
	// simulate update errors.
	client := fake.NewSimpleClientset(objs...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	return client, informerFactory
}
