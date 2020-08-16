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
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

func TestNewEndpointSlice(t *testing.T) {
	portName := "foo"
	protocol := v1.ProtocolTCP

	ports := []discovery.EndpointPort{{Name: &portName, Protocol: &protocol}}
	addrType := discovery.AddressTypeIPv4

	endpoints := v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "test",
			Labels:    map[string]string{"foo": "bar"},
		},
		Subsets: []v1.EndpointSubset{{
			Ports: []v1.EndpointPort{{Port: 80}},
		}},
	}

	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Endpoints"}
	ownerRef := metav1.NewControllerRef(&endpoints, gvk)

	expectedSlice := discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				"foo":                      "bar",
				discovery.LabelServiceName: endpoints.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
			Namespace:       endpoints.Namespace,
		},
		Ports:       ports,
		AddressType: addrType,
		Endpoints:   []discovery.Endpoint{},
	}
	generatedSlice := newEndpointSlice(&endpoints, ports, addrType, "")

	assert.EqualValues(t, expectedSlice, *generatedSlice)

	if len(endpoints.Labels) > 1 {
		t.Errorf("Expected Endpoints labels to not be modified, got %+v", endpoints.Labels)
	}
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
		endpointSlice.ObjectMeta.ResourceVersion = "100"

		return false, endpointSlice, nil
	}))
	client.PrependReactor("update", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)
		endpointSlice.ObjectMeta.ResourceVersion = "200"
		return false, endpointSlice, nil
	}))

	return client
}
