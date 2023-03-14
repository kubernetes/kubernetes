/*
Copyright 2020 The Kubernetes Authors.

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

package endpointslicemirroring

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/utils/pointer"
)

func TestNewEndpointSlice(t *testing.T) {
	portName := "foo"
	protocol := corev1.ProtocolTCP

	ports := []discovery.EndpointPort{{Name: &portName, Protocol: &protocol}}
	addrType := discovery.AddressTypeIPv4
	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Endpoints"}

	endpoints := corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "test",
		},
		Subsets: []corev1.EndpointSubset{{
			Ports: []corev1.EndpointPort{{Port: 80}},
		}},
	}
	ownerRef := metav1.NewControllerRef(&endpoints, gvk)

	testCases := []struct {
		name          string
		tweakEndpoint func(ep *corev1.Endpoints)
		expectedSlice discovery.EndpointSlice
	}{
		{
			name: "create slice from endpoints",
			tweakEndpoint: func(ep *corev1.Endpoints) {
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "create slice from endpoints with annotations",
			tweakEndpoint: func(ep *corev1.Endpoints) {
				annotations := map[string]string{"foo": "bar"}
				ep.Annotations = annotations
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{"foo": "bar"},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "create slice from endpoints with labels",
			tweakEndpoint: func(ep *corev1.Endpoints) {
				labels := map[string]string{"foo": "bar"}
				ep.Labels = labels
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo":                      "bar",
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "create slice from endpoints with labels and annotations",
			tweakEndpoint: func(ep *corev1.Endpoints) {
				labels := map[string]string{"foo": "bar"}
				ep.Labels = labels
				annotations := map[string]string{"foo2": "bar2"}
				ep.Annotations = annotations
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo":                      "bar",
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{"foo2": "bar2"},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "create slice from endpoints with labels and annotations triggertime",
			tweakEndpoint: func(ep *corev1.Endpoints) {
				labels := map[string]string{"foo": "bar"}
				ep.Labels = labels
				annotations := map[string]string{
					"foo2":                                "bar2",
					corev1.EndpointsLastChangeTriggerTime: "date",
				}
				ep.Annotations = annotations
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo":                      "bar",
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{"foo2": "bar2"},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ep := endpoints.DeepCopy()
			tc.tweakEndpoint(ep)
			generatedSlice := newEndpointSlice(ep, ports, addrType, "")
			assert.EqualValues(t, tc.expectedSlice, *generatedSlice)
			if len(endpoints.Labels) > 1 {
				t.Errorf("Expected Endpoints labels to not be modified, got %+v", endpoints.Labels)
			}
		})
	}
}

func TestAddressToEndpoint(t *testing.T) {
	//name: "simple + gate enabled",
	epAddress := corev1.EndpointAddress{
		IP:       "10.1.2.3",
		Hostname: "foo",
		NodeName: pointer.String("node-abc"),
		TargetRef: &corev1.ObjectReference{
			APIVersion: "v1",
			Kind:       "Pod",
			Namespace:  "default",
			Name:       "foo",
		},
	}
	ready := true
	expectedEndpoint := discovery.Endpoint{
		Addresses: []string{"10.1.2.3"},
		Hostname:  pointer.String("foo"),
		Conditions: discovery.EndpointConditions{
			Ready: pointer.BoolPtr(true),
		},
		TargetRef: &corev1.ObjectReference{
			APIVersion: "v1",
			Kind:       "Pod",
			Namespace:  "default",
			Name:       "foo",
		},
		NodeName: pointer.String("node-abc"),
	}

	ep := addressToEndpoint(epAddress, ready)
	assert.EqualValues(t, expectedEndpoint, *ep)
}

// Test helpers

func newClientset() *fake.Clientset {
	client := fake.NewSimpleClientset()

	client.PrependReactor("create", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)

		if endpointSlice.ObjectMeta.GenerateName != "" {
			endpointSlice.ObjectMeta.Name = fmt.Sprintf("%s-%s", endpointSlice.ObjectMeta.GenerateName, rand.String(8))
			endpointSlice.ObjectMeta.GenerateName = ""
		}
		endpointSlice.ObjectMeta.Generation = 1

		return false, endpointSlice, nil
	}))
	client.PrependReactor("update", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)
		endpointSlice.ObjectMeta.Generation++
		return false, endpointSlice, nil
	}))

	return client
}

func Test_getServiceFromDeleteAction(t *testing.T) {
	tests := []struct {
		name string
		obj  interface{}
		want *corev1.Service
	}{
		{
			name: "obj type is Service",
			obj:  &corev1.Service{},
			want: &corev1.Service{},
		},
		{
			name: "obj type is DeletedFinalStateUnknown and content is empty",
			obj:  cache.DeletedFinalStateUnknown{},
			want: nil,
		},
		{
			name: "obj type is DeletedFinalStateUnknown and the Obj's type is Service",
			obj: cache.DeletedFinalStateUnknown{
				Key: "test",
				Obj: &corev1.Service{},
			},
			want: &corev1.Service{},
		},
		{
			name: "obj type is DeletedFinalStateUnknown and the Obj's type is not Service",
			obj: cache.DeletedFinalStateUnknown{
				Key: "test",
				Obj: &discovery.Endpoint{},
			},
			want: nil,
		},
		{
			name: "obj is not EndpointSlice or DeletedFinalStateUnknown",
			obj: &discovery.Endpoint{
				Addresses: []string{"127.0.0.1:80", "127.0.0.2:80"},
			},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getServiceFromDeleteAction(tt.obj); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getServiceFromDeleteAction() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_getEndpointsFromDeleteAction(t *testing.T) {
	tests := []struct {
		name string
		obj  interface{}
		want *corev1.Endpoints
	}{
		{
			name: "obj type is Service",
			obj:  &corev1.Endpoints{},
			want: &corev1.Endpoints{},
		},
		{
			name: "obj type is DeletedFinalStateUnknown and content is empty",
			obj:  cache.DeletedFinalStateUnknown{},
			want: nil,
		},
		{
			name: "obj type is DeletedFinalStateUnknown and the Obj's type is Endpoints",
			obj: cache.DeletedFinalStateUnknown{
				Key: "test",
				Obj: &corev1.Endpoints{},
			},
			want: &corev1.Endpoints{},
		},
		{
			name: "obj type is DeletedFinalStateUnknown and the Obj's type is not Endpoints",
			obj: cache.DeletedFinalStateUnknown{
				Key: "test",
				Obj: &corev1.Service{},
			},
			want: nil,
		},
		{
			name: "obj is not EndpointSlice or DeletedFinalStateUnknown",
			obj:  &corev1.Service{},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getEndpointsFromDeleteAction(tt.obj); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getEndpointsFromDeleteAction() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_getEndpointSliceFromDeleteAction(t *testing.T) {
	tests := []struct {
		name string
		obj  interface{}
		want *discovery.EndpointSlice
	}{
		{
			name: "obj type is EndpointSlice",
			obj:  &discovery.EndpointSlice{},
			want: &discovery.EndpointSlice{},
		},
		{
			name: "obj type is DeletedFinalStateUnknown and content is nil",
			obj:  cache.DeletedFinalStateUnknown{},
			want: nil,
		},
		{
			name: "obj type is DeletedFinalStateUnknown and the Obj's type is EndpointSlice",
			obj: cache.DeletedFinalStateUnknown{
				Key: "test",
				Obj: &discovery.EndpointSlice{},
			},
			want: &discovery.EndpointSlice{},
		},
		{
			name: "obj type is DeletedFinalStateUnknown and the Obj's type is not EndpointSlice",
			obj: cache.DeletedFinalStateUnknown{
				Key: "test",
				Obj: &discovery.Endpoint{},
			},
			want: nil,
		},
		{
			name: "obj is not EndpointSlice or DeletedFinalStateUnknown",
			obj: &discovery.Endpoint{
				Addresses: []string{"127.0.0.1:80", "127.0.0.2:80"},
			},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getEndpointSliceFromDeleteAction(tt.obj); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getEndpointSliceFromDeleteAction() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_endpointsControllerKey(t *testing.T) {
	tests := []struct {
		name          string
		endpointSlice *discovery.EndpointSlice
		want          string
		wantErr       bool
	}{
		{
			name:          "endpointSlice is nil",
			endpointSlice: nil,
			want:          "",
			wantErr:       true,
		},
		{
			name: "have service name",
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
					Labels: map[string]string{
						"kubernetes.io/service-name": "foo",
					},
				},
			},
			want:    "bar/foo",
			wantErr: false,
		},
		{
			name: "have no service name",
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
					Labels: map[string]string{
						"app": "foo",
					},
				},
			},
			want:    "",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := endpointsControllerKey(tt.endpointSlice)
			if (err != nil) != tt.wantErr {
				t.Errorf("endpointsControllerKey() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("endpointsControllerKey() = %v, want %v", got, tt.want)
			}
		})
	}
}
