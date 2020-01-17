/*
Copyright 2019 The Kubernetes Authors.

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

package endpointslice

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1alpha1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	corelisters "k8s.io/client-go/listers/core/v1"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controller"
	utilpointer "k8s.io/utils/pointer"
)

var defaultMaxEndpointsPerSlice = int32(100)

// Even when there are no pods, we want to have a placeholder slice for each service
func TestReconcileEmpty(t *testing.T) {
	client := newClientset()
	namespace := "test"
	svc, _ := newServiceAndendpointMeta("foo", namespace)

	r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, []*corev1.Pod{}, []*discovery.EndpointSlice{}, time.Now())
	expectActions(t, client.Actions(), 1, "create", "endpointslices")

	slices := fetchEndpointSlices(t, client, namespace)
	assert.Len(t, slices, 1, "Expected 1 endpoint slices")

	assert.Regexp(t, "^"+svc.Name, slices[0].Name)
	assert.Equal(t, svc.Name, slices[0].Labels[discovery.LabelServiceName])
	assert.EqualValues(t, []discovery.EndpointPort{}, slices[0].Ports)
	assert.EqualValues(t, []discovery.Endpoint{}, slices[0].Endpoints)
}

// Given a single pod matching a service selector and no existing endpoint slices,
// a slice should be created
func TestReconcile1Pod(t *testing.T) {
	client := newClientset()
	namespace := "test"
	svc, _ := newServiceAndendpointMeta("foo", namespace)
	pod1 := newPod(1, namespace, true, 1)
	pod1.Spec.Hostname = "example-hostname"
	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: pod1.Spec.NodeName,
			Labels: map[string]string{
				"topology.kubernetes.io/zone":   "us-central1-a",
				"topology.kubernetes.io/region": "us-central1",
			},
		},
	}

	triggerTime := time.Now()
	r := newReconciler(client, []*corev1.Node{node1}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, []*corev1.Pod{pod1}, []*discovery.EndpointSlice{}, triggerTime)
	assert.Len(t, client.Actions(), 1, "Expected 1 additional clientset action")

	slices := fetchEndpointSlices(t, client, namespace)
	assert.Len(t, slices, 1, "Expected 1 endpoint slices")
	assert.Regexp(t, "^"+svc.Name, slices[0].Name)
	assert.Equal(t, svc.Name, slices[0].Labels[discovery.LabelServiceName])
	assert.Equal(t, slices[0].Annotations, map[string]string{
		"endpoints.kubernetes.io/last-change-trigger-time": triggerTime.Format(time.RFC3339Nano),
	})
	assert.EqualValues(t, []discovery.Endpoint{{
		Addresses:  []string{"1.2.3.5"},
		Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
		Topology: map[string]string{
			"kubernetes.io/hostname":        "node-1",
			"topology.kubernetes.io/zone":   "us-central1-a",
			"topology.kubernetes.io/region": "us-central1",
		},
		TargetRef: &corev1.ObjectReference{
			Kind:      "Pod",
			Namespace: namespace,
			Name:      "pod1",
		},
	}}, slices[0].Endpoints)
}

// given an existing endpoint slice and no pods matching the service, the existing
// slice should be updated to a placeholder (not deleted)
func TestReconcile1EndpointSlice(t *testing.T) {
	client := newClientset()
	namespace := "test"
	svc, endpointMeta := newServiceAndendpointMeta("foo", namespace)
	endpointSlice1 := newEmptyEndpointSlice(1, namespace, endpointMeta, svc)

	_, createErr := client.DiscoveryV1alpha1().EndpointSlices(namespace).Create(endpointSlice1)
	assert.Nil(t, createErr, "Expected no error creating endpoint slice")

	numActionsBefore := len(client.Actions())
	r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, []*corev1.Pod{}, []*discovery.EndpointSlice{endpointSlice1}, time.Now())
	assert.Len(t, client.Actions(), numActionsBefore+1, "Expected 1 additional clientset action")
	actions := client.Actions()
	assert.True(t, actions[numActionsBefore].Matches("update", "endpointslices"), "Action should be update endpoint slice")

	slices := fetchEndpointSlices(t, client, namespace)
	assert.Len(t, slices, 1, "Expected 1 endpoint slices")

	assert.Regexp(t, "^"+svc.Name, slices[0].Name)
	assert.Equal(t, svc.Name, slices[0].Labels[discovery.LabelServiceName])
	assert.EqualValues(t, []discovery.EndpointPort{}, slices[0].Ports)
	assert.EqualValues(t, []discovery.Endpoint{}, slices[0].Endpoints)
}

// a simple use case with 250 pods matching a service and no existing slices
// reconcile should create 3 slices, completely filling 2 of them
func TestReconcileManyPods(t *testing.T) {
	client := newClientset()
	namespace := "test"
	svc, _ := newServiceAndendpointMeta("foo", namespace)

	// start with 250 pods
	pods := []*corev1.Pod{}
	for i := 0; i < 250; i++ {
		ready := !(i%3 == 0)
		pods = append(pods, newPod(i, namespace, ready, 1))
	}

	r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, pods, []*discovery.EndpointSlice{}, time.Now())

	// This is an ideal scenario where only 3 actions are required, and they're all creates
	assert.Len(t, client.Actions(), 3, "Expected 3 additional clientset actions")
	expectActions(t, client.Actions(), 3, "create", "endpointslices")

	// Two endpoint slices should be completely full, the remainder should be in another one
	expectUnorderedSlicesWithLengths(t, fetchEndpointSlices(t, client, namespace), []int{100, 100, 50})
}

// now with preexisting slices, we have 250 pods matching a service
// the first endpoint slice contains 62 endpoints, all desired
// the second endpoint slice contains 61 endpoints, all desired
// that leaves 127 to add
// to minimize writes, our strategy is to create new slices for multiples of 100
// that leaves 27 to drop in an existing slice
// dropping them in the first slice will result in the slice being closest to full
// this approach requires 1 update + 1 create instead of 2 updates + 1 create
func TestReconcileEndpointSlicesSomePreexisting(t *testing.T) {
	client := newClientset()
	namespace := "test"
	svc, endpointMeta := newServiceAndendpointMeta("foo", namespace)

	// start with 250 pods
	pods := []*corev1.Pod{}
	for i := 0; i < 250; i++ {
		ready := !(i%3 == 0)
		pods = append(pods, newPod(i, namespace, ready, 1))
	}

	// have approximately 1/4 in first slice
	endpointSlice1 := newEmptyEndpointSlice(1, namespace, endpointMeta, svc)
	for i := 1; i < len(pods)-4; i += 4 {
		endpointSlice1.Endpoints = append(endpointSlice1.Endpoints, podToEndpoint(pods[i], &corev1.Node{}))
	}

	// have approximately 1/4 in second slice
	endpointSlice2 := newEmptyEndpointSlice(2, namespace, endpointMeta, svc)
	for i := 3; i < len(pods)-4; i += 4 {
		endpointSlice2.Endpoints = append(endpointSlice2.Endpoints, podToEndpoint(pods[i], &corev1.Node{}))
	}

	existingSlices := []*discovery.EndpointSlice{endpointSlice1, endpointSlice2}
	createEndpointSlices(t, client, namespace, existingSlices)

	numActionsBefore := len(client.Actions())
	r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, pods, existingSlices, time.Now())

	actions := client.Actions()
	assert.Equal(t, numActionsBefore+2, len(actions), "Expected 2 additional client actions as part of reconcile")
	assert.True(t, actions[numActionsBefore].Matches("create", "endpointslices"), "First action should be create endpoint slice")
	assert.True(t, actions[numActionsBefore+1].Matches("update", "endpointslices"), "Second action should be update endpoint slice")

	// 1 new slice (0->100) + 1 updated slice (62->89)
	expectUnorderedSlicesWithLengths(t, fetchEndpointSlices(t, client, namespace), []int{89, 61, 100})
}

// now with preexisting slices, we have 300 pods matching a service
// this scenario will show some less ideal allocation
// the first endpoint slice contains 74 endpoints, all desired
// the second endpoint slice contains 74 endpoints, all desired
// that leaves 152 to add
// to minimize writes, our strategy is to create new slices for multiples of 100
// that leaves 52 to drop in an existing slice
// that capacity could fit if split in the 2 existing slices
// to minimize writes though, reconcile create a new slice with those 52 endpoints
// this approach requires 2 creates instead of 2 updates + 1 create
func TestReconcileEndpointSlicesSomePreexistingWorseAllocation(t *testing.T) {
	client := newClientset()
	namespace := "test"
	svc, endpointMeta := newServiceAndendpointMeta("foo", namespace)

	// start with 300 pods
	pods := []*corev1.Pod{}
	for i := 0; i < 300; i++ {
		ready := !(i%3 == 0)
		pods = append(pods, newPod(i, namespace, ready, 1))
	}

	// have approximately 1/4 in first slice
	endpointSlice1 := newEmptyEndpointSlice(1, namespace, endpointMeta, svc)
	for i := 1; i < len(pods)-4; i += 4 {
		endpointSlice1.Endpoints = append(endpointSlice1.Endpoints, podToEndpoint(pods[i], &corev1.Node{}))
	}

	// have approximately 1/4 in second slice
	endpointSlice2 := newEmptyEndpointSlice(2, namespace, endpointMeta, svc)
	for i := 3; i < len(pods)-4; i += 4 {
		endpointSlice2.Endpoints = append(endpointSlice2.Endpoints, podToEndpoint(pods[i], &corev1.Node{}))
	}

	existingSlices := []*discovery.EndpointSlice{endpointSlice1, endpointSlice2}
	createEndpointSlices(t, client, namespace, existingSlices)

	numActionsBefore := len(client.Actions())
	r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, pods, existingSlices, time.Now())

	actions := client.Actions()
	assert.Equal(t, numActionsBefore+2, len(actions), "Expected 2 additional client actions as part of reconcile")
	expectActions(t, client.Actions(), 2, "create", "endpointslices")

	// 2 new slices (100, 52) in addition to existing slices (74, 74)
	expectUnorderedSlicesWithLengths(t, fetchEndpointSlices(t, client, namespace), []int{74, 74, 100, 52})
}

// In some cases, such as a service port change, all slices for that service will require a change
// This test ensures that we are updating those slices and not calling create + delete for each
func TestReconcileEndpointSlicesUpdating(t *testing.T) {
	client := newClientset()
	namespace := "test"
	svc, _ := newServiceAndendpointMeta("foo", namespace)

	// start with 250 pods
	pods := []*corev1.Pod{}
	for i := 0; i < 250; i++ {
		ready := !(i%3 == 0)
		pods = append(pods, newPod(i, namespace, ready, 1))
	}

	r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, pods, []*discovery.EndpointSlice{}, time.Now())
	numActionsExpected := 3
	assert.Len(t, client.Actions(), numActionsExpected, "Expected 3 additional clientset actions")

	slices := fetchEndpointSlices(t, client, namespace)
	numActionsExpected++
	expectUnorderedSlicesWithLengths(t, slices, []int{100, 100, 50})

	svc.Spec.Ports[0].TargetPort.IntVal = 81
	reconcileHelper(t, r, &svc, pods, []*discovery.EndpointSlice{&slices[0], &slices[1], &slices[2]}, time.Now())

	numActionsExpected += 3
	assert.Len(t, client.Actions(), numActionsExpected, "Expected 3 additional clientset actions")
	expectActions(t, client.Actions(), 3, "update", "endpointslices")

	expectUnorderedSlicesWithLengths(t, fetchEndpointSlices(t, client, namespace), []int{100, 100, 50})
}

// In this test, we start with 10 slices that only have 30 endpoints each
// An initial reconcile makes no changes (as desired to limit writes)
// When we change a service port, all slices will need to be updated in some way
// reconcile repacks the endpoints into 3 slices, and deletes the extras
func TestReconcileEndpointSlicesRecycling(t *testing.T) {
	client := newClientset()
	namespace := "test"
	svc, endpointMeta := newServiceAndendpointMeta("foo", namespace)

	// start with 300 pods
	pods := []*corev1.Pod{}
	for i := 0; i < 300; i++ {
		ready := !(i%3 == 0)
		pods = append(pods, newPod(i, namespace, ready, 1))
	}

	// generate 10 existing slices with 30 pods/endpoints each
	existingSlices := []*discovery.EndpointSlice{}
	for i, pod := range pods {
		sliceNum := i / 30
		if i%30 == 0 {
			existingSlices = append(existingSlices, newEmptyEndpointSlice(sliceNum, namespace, endpointMeta, svc))
		}
		existingSlices[sliceNum].Endpoints = append(existingSlices[sliceNum].Endpoints, podToEndpoint(pod, &corev1.Node{}))
	}

	createEndpointSlices(t, client, namespace, existingSlices)

	numActionsBefore := len(client.Actions())
	r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, pods, existingSlices, time.Now())
	// initial reconcile should be a no op, all pods are accounted for in slices, no repacking should be done
	assert.Equal(t, numActionsBefore+0, len(client.Actions()), "Expected 0 additional client actions as part of reconcile")

	// changing a service port should require all slices to be updated, time for a repack
	svc.Spec.Ports[0].TargetPort.IntVal = 81
	reconcileHelper(t, r, &svc, pods, existingSlices, time.Now())

	// this should reflect 3 updates + 7 deletes
	assert.Equal(t, numActionsBefore+10, len(client.Actions()), "Expected 10 additional client actions as part of reconcile")

	// thanks to recycling, we get a free repack of endpoints, resulting in 3 full slices instead of 10 mostly empty slices
	expectUnorderedSlicesWithLengths(t, fetchEndpointSlices(t, client, namespace), []int{100, 100, 100})
}

// In this test, we want to verify that endpoints are added to a slice that will
// be closest to full after the operation, even when slices are already marked
// for update.
func TestReconcileEndpointSlicesUpdatePacking(t *testing.T) {
	client := newClientset()
	namespace := "test"
	svc, endpointMeta := newServiceAndendpointMeta("foo", namespace)

	existingSlices := []*discovery.EndpointSlice{}
	pods := []*corev1.Pod{}

	slice1 := newEmptyEndpointSlice(1, namespace, endpointMeta, svc)
	for i := 0; i < 80; i++ {
		pod := newPod(i, namespace, true, 1)
		slice1.Endpoints = append(slice1.Endpoints, podToEndpoint(pod, &corev1.Node{}))
		pods = append(pods, pod)
	}
	existingSlices = append(existingSlices, slice1)

	slice2 := newEmptyEndpointSlice(2, namespace, endpointMeta, svc)
	for i := 100; i < 120; i++ {
		pod := newPod(i, namespace, true, 1)
		slice2.Endpoints = append(slice2.Endpoints, podToEndpoint(pod, &corev1.Node{}))
		pods = append(pods, pod)
	}
	existingSlices = append(existingSlices, slice2)

	createEndpointSlices(t, client, namespace, existingSlices)

	// ensure that endpoints in each slice will be marked for update.
	for i, pod := range pods {
		if i%10 == 0 {
			pod.Status.Conditions = []corev1.PodCondition{{
				Type:   corev1.PodReady,
				Status: corev1.ConditionFalse,
			}}
		}
	}

	// add a few additional endpoints - no more than could fit in either slice.
	for i := 200; i < 215; i++ {
		pods = append(pods, newPod(i, namespace, true, 1))
	}

	r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, pods, existingSlices, time.Now())

	// ensure that both endpoint slices have been updated
	expectActions(t, client.Actions(), 2, "update", "endpointslices")

	// additional pods should get added to fuller slice
	expectUnorderedSlicesWithLengths(t, fetchEndpointSlices(t, client, namespace), []int{95, 20})
}

// Named ports can map to different port numbers on different pods.
// This test ensures that EndpointSlices are grouped correctly in that case.
func TestReconcileEndpointSlicesNamedPorts(t *testing.T) {
	client := newClientset()
	namespace := "test"

	portNameIntStr := intstr.IntOrString{
		Type:   intstr.String,
		StrVal: "http",
	}

	svc := corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "named-port-example", Namespace: namespace},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{{
				TargetPort: portNameIntStr,
				Protocol:   corev1.ProtocolTCP,
			}},
			Selector: map[string]string{"foo": "bar"},
		},
	}

	// start with 300 pods
	pods := []*corev1.Pod{}
	for i := 0; i < 300; i++ {
		ready := !(i%3 == 0)
		portOffset := i % 5
		pod := newPod(i, namespace, ready, 1)
		pod.Spec.Containers[0].Ports = []corev1.ContainerPort{{
			Name:          portNameIntStr.StrVal,
			ContainerPort: int32(8080 + portOffset),
			Protocol:      corev1.ProtocolTCP,
		}}
		pods = append(pods, pod)
	}

	r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, defaultMaxEndpointsPerSlice)
	reconcileHelper(t, r, &svc, pods, []*discovery.EndpointSlice{}, time.Now())

	// reconcile should create 5 endpoint slices
	assert.Equal(t, 5, len(client.Actions()), "Expected 5 client actions as part of reconcile")
	expectActions(t, client.Actions(), 5, "create", "endpointslices")

	fetchedSlices := fetchEndpointSlices(t, client, namespace)

	// each slice should have 60 endpoints to match 5 unique variations of named port mapping
	expectUnorderedSlicesWithLengths(t, fetchedSlices, []int{60, 60, 60, 60, 60})

	// generate data structures for expected slice ports and address types
	protoTCP := corev1.ProtocolTCP
	ipAddressType := discovery.AddressTypeIP
	expectedSlices := []discovery.EndpointSlice{}
	for i := range fetchedSlices {
		expectedSlices = append(expectedSlices, discovery.EndpointSlice{
			Ports: []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(""),
				Protocol: &protoTCP,
				Port:     utilpointer.Int32Ptr(int32(8080 + i)),
			}},
			AddressType: &ipAddressType,
		})
	}

	// slices fetched should match expected address type and ports
	expectUnorderedSlicesWithTopLevelAttrs(t, fetchedSlices, expectedSlices)
}

// This test ensures that maxEndpointsPerSlice configuration results in
// appropriate endpoints distribution among slices
func TestReconcileMaxEndpointsPerSlice(t *testing.T) {
	namespace := "test"
	svc, _ := newServiceAndendpointMeta("foo", namespace)

	// start with 250 pods
	pods := []*corev1.Pod{}
	for i := 0; i < 250; i++ {
		ready := !(i%3 == 0)
		pods = append(pods, newPod(i, namespace, ready, 1))
	}

	testCases := []struct {
		maxEndpointsPerSlice int32
		expectedSliceLengths []int
	}{
		{
			maxEndpointsPerSlice: int32(50),
			expectedSliceLengths: []int{50, 50, 50, 50, 50},
		}, {
			maxEndpointsPerSlice: int32(80),
			expectedSliceLengths: []int{80, 80, 80, 10},
		}, {
			maxEndpointsPerSlice: int32(150),
			expectedSliceLengths: []int{150, 100},
		}, {
			maxEndpointsPerSlice: int32(250),
			expectedSliceLengths: []int{250},
		}, {
			maxEndpointsPerSlice: int32(500),
			expectedSliceLengths: []int{250},
		},
	}

	for _, testCase := range testCases {
		client := newClientset()
		r := newReconciler(client, []*corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}}, testCase.maxEndpointsPerSlice)
		reconcileHelper(t, r, &svc, pods, []*discovery.EndpointSlice{}, time.Now())
		expectUnorderedSlicesWithLengths(t, fetchEndpointSlices(t, client, namespace), testCase.expectedSliceLengths)
	}
}

// Test Helpers

func newReconciler(client *fake.Clientset, nodes []*corev1.Node, maxEndpointsPerSlice int32) *reconciler {
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	nodeInformer := informerFactory.Core().V1().Nodes()
	indexer := nodeInformer.Informer().GetIndexer()
	for _, node := range nodes {
		indexer.Add(node)
	}

	return &reconciler{
		client:               client,
		nodeLister:           corelisters.NewNodeLister(indexer),
		maxEndpointsPerSlice: maxEndpointsPerSlice,
	}
}

// ensures endpoint slices exist with the desired set of lengths
func expectUnorderedSlicesWithLengths(t *testing.T, endpointSlices []discovery.EndpointSlice, expectedLengths []int) {
	assert.Len(t, endpointSlices, len(expectedLengths), "Expected %d endpoint slices", len(expectedLengths))

	lengthsWithNoMatch := []int{}
	desiredLengths := expectedLengths
	actualLengths := []int{}
	for _, endpointSlice := range endpointSlices {
		actualLen := len(endpointSlice.Endpoints)
		actualLengths = append(actualLengths, actualLen)
		matchFound := false
		for i := 0; i < len(desiredLengths); i++ {
			if desiredLengths[i] == actualLen {
				matchFound = true
				desiredLengths = append(desiredLengths[:i], desiredLengths[i+1:]...)
				break
			}
		}

		if !matchFound {
			lengthsWithNoMatch = append(lengthsWithNoMatch, actualLen)
		}
	}

	if len(lengthsWithNoMatch) > 0 || len(desiredLengths) > 0 {
		t.Errorf("Actual slice lengths (%v) don't match expected (%v)", actualLengths, expectedLengths)
	}
}

// ensures endpoint slices exist with the desired set of ports and address types
func expectUnorderedSlicesWithTopLevelAttrs(t *testing.T, endpointSlices []discovery.EndpointSlice, expectedSlices []discovery.EndpointSlice) {
	t.Helper()
	assert.Len(t, endpointSlices, len(expectedSlices), "Expected %d endpoint slices", len(expectedSlices))

	slicesWithNoMatch := []discovery.EndpointSlice{}
	for _, endpointSlice := range endpointSlices {
		matchFound := false
		for i := 0; i < len(expectedSlices); i++ {
			if portsAndAddressTypeEqual(expectedSlices[i], endpointSlice) {
				matchFound = true
				expectedSlices = append(expectedSlices[:i], expectedSlices[i+1:]...)
				break
			}
		}

		if !matchFound {
			slicesWithNoMatch = append(slicesWithNoMatch, endpointSlice)
		}
	}

	assert.Len(t, slicesWithNoMatch, 0, "EndpointSlice(s) found without matching attributes")
	assert.Len(t, expectedSlices, 0, "Expected slices(s) not found in EndpointSlices")
}

func expectActions(t *testing.T, actions []k8stesting.Action, num int, verb, resource string) {
	t.Helper()
	for i := 0; i < num; i++ {
		relativePos := len(actions) - i - 1
		assert.Equal(t, verb, actions[relativePos].GetVerb(), "Expected action -%d verb to be %s", i, verb)
		assert.Equal(t, resource, actions[relativePos].GetResource().Resource, "Expected action -%d resource to be %s", i, resource)
	}
}

func portsAndAddressTypeEqual(slice1, slice2 discovery.EndpointSlice) bool {
	return apiequality.Semantic.DeepEqual(slice1.Ports, slice2.Ports) && apiequality.Semantic.DeepEqual(slice1.AddressType, slice2.AddressType)
}

func createEndpointSlices(t *testing.T, client *fake.Clientset, namespace string, endpointSlices []*discovery.EndpointSlice) {
	t.Helper()
	for _, endpointSlice := range endpointSlices {
		_, err := client.DiscoveryV1alpha1().EndpointSlices(namespace).Create(endpointSlice)
		if err != nil {
			t.Fatalf("Expected no error creating Endpoint Slice, got: %v", err)
		}
	}
}

func fetchEndpointSlices(t *testing.T, client *fake.Clientset, namespace string) []discovery.EndpointSlice {
	t.Helper()
	fetchedSlices, err := client.DiscoveryV1alpha1().EndpointSlices(namespace).List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Expected no error fetching Endpoint Slices, got: %v", err)
		return []discovery.EndpointSlice{}
	}
	return fetchedSlices.Items
}

func reconcileHelper(t *testing.T, r *reconciler, service *corev1.Service, pods []*corev1.Pod, existingSlices []*discovery.EndpointSlice, triggerTime time.Time) {
	t.Helper()
	err := r.reconcile(service, pods, existingSlices, triggerTime)
	if err != nil {
		t.Fatalf("Expected no error reconciling Endpoint Slices, got: %v", err)
	}
}
