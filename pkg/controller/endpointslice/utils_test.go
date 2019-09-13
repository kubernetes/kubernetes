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
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
	utilpointer "k8s.io/utils/pointer"
)

func TestNewEndpointSlice(t *testing.T) {
	ipAddressType := discovery.AddressTypeIP
	portName := "foo"
	protocol := v1.ProtocolTCP
	endpointMeta := endpointMeta{
		Ports:       []discovery.EndpointPort{{Name: &portName, Protocol: &protocol}},
		AddressType: &ipAddressType,
	}
	service := v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec: v1.ServiceSpec{
			Ports:    []v1.ServicePort{{Port: 80}},
			Selector: map[string]string{"foo": "bar"},
		},
	}

	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Service"}
	ownerRef := metav1.NewControllerRef(&service, gvk)

	expectedSlice := discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Labels:          map[string]string{discovery.LabelServiceName: service.Name},
			GenerateName:    fmt.Sprintf("%s-", service.Name),
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
			Namespace:       service.Namespace,
		},
		Ports:       endpointMeta.Ports,
		AddressType: endpointMeta.AddressType,
		Endpoints:   []discovery.Endpoint{},
	}
	generatedSlice := newEndpointSlice(&service, &endpointMeta)

	assert.EqualValues(t, expectedSlice, *generatedSlice)
}

func TestPodToEndpoint(t *testing.T) {
	ns := "test"

	readyPod := newPod(1, ns, true, 1)
	unreadyPod := newPod(1, ns, false, 1)
	multiIPPod := newPod(1, ns, true, 1)

	multiIPPod.Status.PodIPs = []v1.PodIP{{IP: "1.2.3.4"}, {IP: "1234::5678:0000:0000:9abc:def0"}}

	node1 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: readyPod.Spec.NodeName,
			Labels: map[string]string{
				"topology.kubernetes.io/zone":   "us-central1-a",
				"topology.kubernetes.io/region": "us-central1",
			},
		},
	}

	testCases := []struct {
		name             string
		pod              *v1.Pod
		node             *v1.Node
		expectedEndpoint discovery.Endpoint
	}{
		{
			name: "Ready pod",
			pod:  readyPod,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
				Topology:   map[string]string{"kubernetes.io/hostname": "node-1"},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
		{
			name: "Unready pod",
			pod:  unreadyPod,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(false)},
				Topology:   map[string]string{"kubernetes.io/hostname": "node-1"},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
		{
			name: "Ready pod + node labels",
			pod:  readyPod,
			node: node1,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
				Topology: map[string]string{
					"kubernetes.io/hostname":        "node-1",
					"topology.kubernetes.io/zone":   "us-central1-a",
					"topology.kubernetes.io/region": "us-central1",
				},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
		{
			name: "Multi IP Ready pod + node labels",
			pod:  multiIPPod,
			node: node1,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.4", "1234::5678:0000:0000:9abc:def0"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
				Topology: map[string]string{
					"kubernetes.io/hostname":        "node-1",
					"topology.kubernetes.io/zone":   "us-central1-a",
					"topology.kubernetes.io/region": "us-central1",
				},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			endpoint := podToEndpoint(testCase.pod, testCase.node)
			assert.EqualValues(t, testCase.expectedEndpoint, endpoint, "Test case failed: %s", testCase.name)
		})
	}
}

func TestPodChangedWithpodEndpointChanged(t *testing.T) {
	podStore := cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc)
	ns := "test"
	podStore.Add(newPod(1, ns, true, 1))
	pods := podStore.List()
	if len(pods) != 1 {
		t.Errorf("podStore size: expected: %d, got: %d", 1, len(pods))
		return
	}
	oldPod := pods[0].(*v1.Pod)
	newPod := oldPod.DeepCopy()

	if podChangedHelper(oldPod, newPod, podEndpointChanged) {
		t.Errorf("Expected pod to be unchanged for copied pod")
	}

	newPod.Spec.NodeName = "changed"
	if !podChangedHelper(oldPod, newPod, podEndpointChanged) {
		t.Errorf("Expected pod to be changed for pod with NodeName changed")
	}
	newPod.Spec.NodeName = oldPod.Spec.NodeName

	newPod.ObjectMeta.ResourceVersion = "changed"
	if podChangedHelper(oldPod, newPod, podEndpointChanged) {
		t.Errorf("Expected pod to be unchanged for pod with only ResourceVersion changed")
	}
	newPod.ObjectMeta.ResourceVersion = oldPod.ObjectMeta.ResourceVersion

	newPod.Status.PodIP = "1.2.3.1"
	if !podChangedHelper(oldPod, newPod, podEndpointChanged) {
		t.Errorf("Expected pod to be changed with pod IP address change")
	}
	newPod.Status.PodIP = oldPod.Status.PodIP

	newPod.ObjectMeta.Name = "wrong-name"
	if !podChangedHelper(oldPod, newPod, podEndpointChanged) {
		t.Errorf("Expected pod to be changed with pod name change")
	}
	newPod.ObjectMeta.Name = oldPod.ObjectMeta.Name

	saveConditions := oldPod.Status.Conditions
	oldPod.Status.Conditions = nil
	if !podChangedHelper(oldPod, newPod, podEndpointChanged) {
		t.Errorf("Expected pod to be changed with pod readiness change")
	}
	oldPod.Status.Conditions = saveConditions

	now := metav1.NewTime(time.Now().UTC())
	newPod.ObjectMeta.DeletionTimestamp = &now
	if !podChangedHelper(oldPod, newPod, podEndpointChanged) {
		t.Errorf("Expected pod to be changed with DeletionTimestamp change")
	}
	newPod.ObjectMeta.DeletionTimestamp = oldPod.ObjectMeta.DeletionTimestamp.DeepCopy()
}

// Test helpers

func newPod(n int, namespace string, ready bool, nPorts int) *v1.Pod {
	status := v1.ConditionTrue
	if !ready {
		status = v1.ConditionFalse
	}

	p := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      fmt.Sprintf("pod%d", n),
			Labels:    map[string]string{"foo": "bar"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name: "container-1",
			}},
			NodeName: "node-1",
		},
		Status: v1.PodStatus{
			PodIP: fmt.Sprintf("1.2.3.%d", 4+n),
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodReady,
					Status: status,
				},
			},
		},
	}

	return p
}

func newClientset() *fake.Clientset {
	client := fake.NewSimpleClientset()

	client.PrependReactor("create", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)

		if endpointSlice.ObjectMeta.GenerateName != "" {
			endpointSlice.ObjectMeta.Name = fmt.Sprintf("%s-%s", endpointSlice.ObjectMeta.GenerateName, rand.String(8))
			endpointSlice.ObjectMeta.GenerateName = ""
		}

		return false, endpointSlice, nil
	}))

	return client
}

func newServiceAndendpointMeta(name, namespace string) (v1.Service, endpointMeta) {
	portNum := int32(80)
	portNameIntStr := intstr.IntOrString{
		Type:   intstr.Int,
		IntVal: portNum,
	}

	svc := v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				TargetPort: portNameIntStr,
				Protocol:   v1.ProtocolTCP,
				Name:       name,
			}},
			Selector: map[string]string{"foo": "bar"},
		},
	}

	ipAddressType := discovery.AddressTypeIP
	protocol := v1.ProtocolTCP
	endpointMeta := endpointMeta{
		AddressType: &ipAddressType,
		Ports:       []discovery.EndpointPort{{Name: &name, Port: &portNum, Protocol: &protocol}},
	}

	return svc, endpointMeta
}

func newEmptyEndpointSlice(n int, namespace string, endpointMeta endpointMeta, svc v1.Service) *discovery.EndpointSlice {
	return &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s.%d", svc.Name, n),
			Namespace: namespace,
		},
		Ports:       endpointMeta.Ports,
		AddressType: endpointMeta.AddressType,
		Endpoints:   []discovery.Endpoint{},
	}
}

func podChangedHelper(oldPod, newPod *v1.Pod, endpointChanged endpointutil.EndpointsMatch) bool {
	podChanged, _ := endpointutil.PodChanged(oldPod, newPod, podEndpointChanged)
	return podChanged
}
