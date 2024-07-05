/*
Copyright 2024 The Kubernetes Authors.

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
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/klog/v2/ktesting"
)

func TestNewEndpointSlice(t *testing.T) {
	addressType := discovery.AddressTypeIPv4
	portName := "foo"
	protocol := v1.ProtocolTCP
	controllerName := "endpointslice-controller.k8s.io"
	ports := []discovery.EndpointPort{{Name: &portName, Protocol: &protocol}}
	service := v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec: v1.ServiceSpec{
			ClusterIP: "1.1.1.1",
			Ports:     []v1.ServicePort{{Port: 80}},
			Selector:  map[string]string{"foo": "bar"},
		},
	}

	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Service"}
	ownerRef := metav1.NewControllerRef(&service, gvk)

	testCases := []struct {
		name          string
		updateSvc     func(svc v1.Service) v1.Service // given basic valid services, each test case can customize them
		expectedSlice *discovery.EndpointSlice
	}{
		{
			name: "Service without labels",
			updateSvc: func(svc v1.Service) v1.Service {
				return svc
			},
			expectedSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					GenerateName:    fmt.Sprintf("%s-", service.Name),
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
					Namespace:       service.Namespace,
				},
				Ports:       ports,
				AddressType: addressType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "Service with labels",
			updateSvc: func(svc v1.Service) v1.Service {
				labels := map[string]string{"foo": "bar"}
				svc.Labels = labels
				return svc
			},
			expectedSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
						"foo":                      "bar",
					},
					GenerateName:    fmt.Sprintf("%s-", service.Name),
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
					Namespace:       service.Namespace,
				},
				Ports:       ports,
				AddressType: addressType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "Headless Service with labels",
			updateSvc: func(svc v1.Service) v1.Service {
				labels := map[string]string{"foo": "bar"}
				svc.Labels = labels
				svc.Spec.ClusterIP = v1.ClusterIPNone
				return svc
			},
			expectedSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
						v1.IsHeadlessService:       "",
						"foo":                      "bar",
					},
					GenerateName:    fmt.Sprintf("%s-", service.Name),
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
					Namespace:       service.Namespace,
				},
				Ports:       ports,
				AddressType: addressType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "Service with multiple labels",
			updateSvc: func(svc v1.Service) v1.Service {
				labels := map[string]string{"foo": "bar", "foo2": "bar2"}
				svc.Labels = labels
				return svc
			},
			expectedSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
						"foo":                      "bar",
						"foo2":                     "bar2",
					},
					GenerateName:    fmt.Sprintf("%s-", service.Name),
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
					Namespace:       service.Namespace,
				},
				Ports:       ports,
				AddressType: addressType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "Evil service hijacking endpoint slices labels",
			updateSvc: func(svc v1.Service) v1.Service {
				labels := map[string]string{
					discovery.LabelServiceName: "bad",
					discovery.LabelManagedBy:   "actor",
					"foo":                      "bar",
				}
				svc.Labels = labels
				return svc
			},
			expectedSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
						"foo":                      "bar",
					},
					GenerateName:    fmt.Sprintf("%s-", service.Name),
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
					Namespace:       service.Namespace,
				},
				Ports:       ports,
				AddressType: addressType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "Service with annotations",
			updateSvc: func(svc v1.Service) v1.Service {
				annotations := map[string]string{"foo": "bar"}
				svc.Annotations = annotations
				return svc
			},
			expectedSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					GenerateName:    fmt.Sprintf("%s-", service.Name),
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
					Namespace:       service.Namespace,
				},
				Ports:       ports,
				AddressType: addressType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			svc := tc.updateSvc(service)
			lfs := &LabelsFromService{Service: &svc}
			generatedSlice := newEndpointSlice(
				logger,
				controllerName,
				schema.GroupVersionKind{Version: "v1", Kind: "Service"},
				&svc,
				ports,
				addressType,
				lfs.SetLabels,
			)

			assert.EqualValues(t, tc.expectedSlice, generatedSlice)
		})
	}
}

// Test helpers

func newPod(n int, namespace string, ready bool, nPorts int, terminating bool) *v1.Pod {
	status := v1.ConditionTrue
	if !ready {
		status = v1.ConditionFalse
	}

	var deletionTimestamp *metav1.Time
	if terminating {
		deletionTimestamp = &metav1.Time{
			Time: time.Now(),
		}
	}

	p := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         namespace,
			Name:              fmt.Sprintf("pod%d", n),
			Labels:            map[string]string{"foo": "bar"},
			DeletionTimestamp: deletionTimestamp,
			ResourceVersion:   fmt.Sprint(n),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name: "container-1",
			}},
			NodeName: "node-1",
		},
		Status: v1.PodStatus{
			PodIP: fmt.Sprintf("1.2.3.%d", 4+n),
			PodIPs: []v1.PodIP{{
				IP: fmt.Sprintf("1.2.3.%d", 4+n),
			}},
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
		endpointSlice.Generation = 1

		return false, endpointSlice, nil
	}))
	client.PrependReactor("update", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)
		endpointSlice.Generation++
		return false, endpointSlice, nil
	}))

	return client
}

func newServicePortsAddressType(name, namespace string) (v1.Service, []discovery.EndpointPort, discovery.AddressType) {
	portNum := int32(80)
	portNameIntStr := intstr.IntOrString{
		Type:   intstr.Int,
		IntVal: portNum,
	}

	svc := v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			UID:       types.UID(namespace + "-" + name),
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				TargetPort: portNameIntStr,
				Protocol:   v1.ProtocolTCP,
				Name:       name,
			}},
			Selector:   map[string]string{"foo": "bar"},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	}

	addressType := discovery.AddressTypeIPv4
	protocol := v1.ProtocolTCP

	return svc, []discovery.EndpointPort{{Name: &name, Port: &portNum, Protocol: &protocol}}, addressType
}

func newEmptyEndpointSlice(n int, namespace string, ports []discovery.EndpointPort, addressType discovery.AddressType, svc v1.Service) *discovery.EndpointSlice {
	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Service"}
	ownerRef := metav1.NewControllerRef(&svc, gvk)

	return &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:            fmt.Sprintf("%s-%d", svc.Name, n),
			Namespace:       namespace,
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
		},
		Ports:       ports,
		AddressType: addressType,
		Endpoints:   []discovery.Endpoint{},
	}
}

func Test_hintsEnabled(t *testing.T) {
	testCases := []struct {
		name          string
		annotations   map[string]string
		expectEnabled bool
	}{{
		name:          "empty annotations",
		expectEnabled: false,
	}, {
		name:          "different annotations",
		annotations:   map[string]string{"topology-hints": "enabled"},
		expectEnabled: false,
	}, {
		name:          "hints annotation == enabled",
		annotations:   map[string]string{v1.DeprecatedAnnotationTopologyAwareHints: "enabled"},
		expectEnabled: false,
	}, {
		name:          "hints annotation == aUto",
		annotations:   map[string]string{v1.DeprecatedAnnotationTopologyAwareHints: "aUto"},
		expectEnabled: false,
	}, {
		name:          "hints annotation == auto",
		annotations:   map[string]string{v1.DeprecatedAnnotationTopologyAwareHints: "auto"},
		expectEnabled: true,
	}, {
		name:          "hints annotation == Auto",
		annotations:   map[string]string{v1.DeprecatedAnnotationTopologyAwareHints: "Auto"},
		expectEnabled: true,
	}, {
		name:          "hints annotation == disabled",
		annotations:   map[string]string{v1.DeprecatedAnnotationTopologyAwareHints: "disabled"},
		expectEnabled: false,
	}, {
		name:          "mode annotation == enabled",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "enabled"},
		expectEnabled: false,
	}, {
		name:          "mode annotation == aUto",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "aUto"},
		expectEnabled: false,
	}, {
		name:          "mode annotation == auto",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "auto"},
		expectEnabled: true,
	}, {
		name:          "mode annotation == Auto",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "Auto"},
		expectEnabled: true,
	}, {
		name:          "mode annotation == disabled",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "disabled"},
		expectEnabled: false,
	}, {
		name:          "mode annotation == enabled",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "enabled"},
		expectEnabled: false,
	}, {
		name:          "mode annotation == aUto",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "aUto"},
		expectEnabled: false,
	}, {
		name:          "mode annotation == auto",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "auto"},
		expectEnabled: true,
	}, {
		name:          "mode annotation == Auto",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "Auto"},
		expectEnabled: true,
	}, {
		name:          "mode annotation == disabled",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "disabled"},
		expectEnabled: false,
	}, {
		name:          "mode annotation == disabled, hints annotation == auto",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "disabled", v1.DeprecatedAnnotationTopologyAwareHints: "auto"},
		expectEnabled: true,
	}, {
		name:          "mode annotation == auto, hints annotation == disabled",
		annotations:   map[string]string{v1.AnnotationTopologyMode: "auto", v1.DeprecatedAnnotationTopologyAwareHints: "disabled"},
		expectEnabled: false,
	}}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualEnabled := hintsEnabled(tc.annotations)
			if actualEnabled != tc.expectEnabled {
				t.Errorf("Expected %t, got %t", tc.expectEnabled, actualEnabled)
			}
		})
	}
}
