/*
Copyright 2025 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/discovery"
)

var apiVersions = []string{"v1", "v1beta1"}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "discovery.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "endpointslices",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        discovery.EndpointSlice
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidEndpointSlice(),
		},
		"invalid missing endpoint addresses": {
			input: mkValidEndpointSlice(func(obj *discovery.EndpointSlice) {
				obj.Endpoints[0].Addresses = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("endpoints").Index(0).Child("addresses"), ""),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       discovery.EndpointSlice
		updateObj    discovery.EndpointSlice
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj: mkValidEndpointSlice(func(obj *discovery.EndpointSlice) {
				obj.ResourceVersion = "1"
			}),
			updateObj: mkValidEndpointSlice(func(obj *discovery.EndpointSlice) {
				obj.ResourceVersion = "1"
			}),
		},
		"invalid update missing addresses": {
			oldObj: mkValidEndpointSlice(func(obj *discovery.EndpointSlice) {
				obj.ResourceVersion = "1"
			}),
			updateObj: mkValidEndpointSlice(func(obj *discovery.EndpointSlice) {
				obj.ResourceVersion = "1"
				obj.Endpoints[0].Addresses = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("endpoints").Index(0).Child("addresses"), ""),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "discovery.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "endpointslices",
				Name:              "valid-endpointslice",
				IsResourceRequest: true,
				Verb:              "update",
			})

			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidEndpointSlice(tweaks ...func(obj *discovery.EndpointSlice)) discovery.EndpointSlice {
	obj := discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-endpointslice",
			Namespace: metav1.NamespaceDefault,
		},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{
			{
				Addresses: []string{"10.1.2.1"},
			},
		},
	}

	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
