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

package pod

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/pod"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "api",
			APIGroup:          "",
			APIVersion:        apiVersion,
			IsResourceRequest: true,
			Verb:              "create",
		})
		testCases := map[string]struct {
			input        *api.Pod
			expectedErrs field.ErrorList
		}{
			"valid": {
				input: podtest.MakePod("foo"),
			},
			"valid toleration key": {
				input: podtest.MakePod("foo", podtest.SetTolerations(
					api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists},
				)),
			},
			"valid toleration key without prefix": {
				input: podtest.MakePod("foo", podtest.SetTolerations(
					api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists},
				)),
			},
			"empty toleration key (match all, skipped)": {
				input: podtest.MakePod("foo", podtest.SetTolerations(
					api.Toleration{Operator: api.TolerationOpExists},
				)),
			},
			"invalid toleration key format": {
				input: podtest.MakePod("foo", podtest.SetTolerations(
					api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists},
				)),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
				},
			},
		}
		for k, tc := range testCases {
			t.Run(k, func(t *testing.T) {
				apitesting.VerifyValidationEquivalence(t, ctx, tc.input, registry.Strategy, tc.expectedErrs)
			})
		}
		obj := *podtest.MakePod("foo")
		meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "api",
			APIGroup:          "",
			APIVersion:        apiVersion,
			Name:              "valid-obj",
			IsResourceRequest: true,
			Verb:              "update",
		})
		updateObj := *podtest.MakePod("foo")
		meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
	}
}
