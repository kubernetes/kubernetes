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

package limitrange

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestDeclarativeValidate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "",
		APIVersion:        "v1",
		Resource:          "limitranges",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        api.LimitRange
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidLimitRange(),
		},
		"name: empty": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata.name"), ""),
			},
		},
		"name: invalid characters": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = "^Invalid"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: label format": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = "this-is-a-label"
			}),
		},
		"name: subdomain format": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = "this.is.a.subdomain"
			}),
		},
		"name: invalid label format": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = "-this-is-not-a-label"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: invalid subdomain format": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = ".this.is.not.a.subdomain"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: label format with trailing dash": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = "this-is-a-label-"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: subdomain format with trailing dash": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = "this.is.a.subdomain-"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: long label format": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = strings.Repeat("x", 253)
			}),
		},
		"name: long subdomain format": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = strings.Repeat("x.", 126) + "x"
			}),
		},
		"name: too long label format": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = strings.Repeat("x", 254)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: too long subdomain format": {
			input: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = strings.Repeat("x.", 126) + "xx"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		// TODO: Add more test cases as validation tags are migrated
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "",
		APIVersion:        "v1",
		Resource:          "limitranges",
		Name:              "valid-limit-range",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		oldObj       api.LimitRange
		updateObj    api.LimitRange
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidLimitRange(func(obj *api.LimitRange) { obj.ResourceVersion = "1" }),
			updateObj: mkValidLimitRange(func(obj *api.LimitRange) { obj.ResourceVersion = "1" }),
		},
		// Note: LimitRange's ValidateUpdate doesn't call ValidateObjectMetaUpdate, so
		// name immutability isn't explicitly checked - it just re-validates the format.
		// This test exists to verify the DV framework is correctly wired up on
		// the update path and produces the same error as the hand-written validation.
		"name: invalid format on update": {
			oldObj: mkValidLimitRange(func(obj *api.LimitRange) { obj.ResourceVersion = "1" }),
			updateObj: mkValidLimitRange(func(obj *api.LimitRange) {
				obj.Name = "-invalid-name"
				obj.ResourceVersion = "1"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		// TODO: Add more test cases as validation tags are migrated
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidLimitRange(tweaks ...func(obj *api.LimitRange)) api.LimitRange {
	obj := api.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-limit-range",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
