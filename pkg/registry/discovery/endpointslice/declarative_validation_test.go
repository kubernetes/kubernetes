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
	"fmt"
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
		"valid at limit endpoint addresses": {
			input: mkValidEndpointSlice(tweakAddresses(100)),
		},
		"invalid missing endpoint addresses": {
			input: mkValidEndpointSlice(func(obj *discovery.EndpointSlice) {
				obj.Endpoints[0].Addresses = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("endpoints").Index(0).Child("addresses"), ""),
			},
		},
		"invalid too many endpoint addresses": {
			input: mkValidEndpointSlice(tweakAddresses(101)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("endpoints").Index(0).Child("addresses"), 101, 100).WithOrigin("maxItems"),
			},
		},
		"invalid missing addressType": {
			input: mkValidEndpointSlice(tweakAddressType("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("addressType"), ""),
			},
		},
		"invalid addressType not supported": {
			input: mkValidEndpointSlice(tweakAddressType("invalid")),
			expectedErrs: field.ErrorList{
				field.NotSupported(field.NewPath("addressType"), discovery.AddressType("invalid"), []string{string(discovery.AddressTypeIPv4), string(discovery.AddressTypeIPv6), string(discovery.AddressTypeFQDN)}),
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
			oldObj:    mkValidEndpointSlice(),
			updateObj: mkValidEndpointSlice(),
		},
		"valid update at limit endpoint addresses": {
			oldObj:    mkValidEndpointSlice(),
			updateObj: mkValidEndpointSlice(tweakAddresses(100)),
		},
		"invalid update missing addresses": {
			oldObj: mkValidEndpointSlice(),
			updateObj: mkValidEndpointSlice(func(obj *discovery.EndpointSlice) {
				obj.Endpoints[0].Addresses = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("endpoints").Index(0).Child("addresses"), ""),
			},
		},
		"invalid update too many addresses": {
			oldObj:    mkValidEndpointSlice(),
			updateObj: mkValidEndpointSlice(tweakAddresses(101)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("endpoints").Index(0).Child("addresses"), 101, 100).WithOrigin("maxItems"),
			},
		},
		"invalid update addressType immutable": {
			oldObj:    mkValidEndpointSlice(),
			updateObj: mkValidEndpointSlice(tweakAddressType(discovery.AddressTypeIPv6)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("addressType"), discovery.AddressTypeIPv6, "field is immutable").WithOrigin("immutable"),
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
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
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

func tweakAddresses(count int) func(*discovery.EndpointSlice) {
	return func(obj *discovery.EndpointSlice) {
		addrs := make([]string, count)
		for i := range addrs {
			addrs[i] = fmt.Sprintf("10.0.0.%d", i%255)
		}
		obj.Endpoints[0].Addresses = addrs
	}
}

func tweakAddressType(addrType discovery.AddressType) func(*discovery.EndpointSlice) {
	return func(obj *discovery.EndpointSlice) {
		obj.AddressType = addrType
	}
}
