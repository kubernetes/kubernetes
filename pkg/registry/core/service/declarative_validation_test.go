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

package service

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		input        api.Service
		expectedErrs field.ErrorList
	}{
		// baseline
		"default resource": {
			input: *makeValidService(),
		},
		// TODO: Add more tests.
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
            apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          api.Service
		update       api.Service
		expectedErrs field.ErrorList
	}{
		// baseline
		"no change": {
			old:    *mkValidService(),
			update: *mkValidService(),
		},
		// TODO: Add more tests.
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateStatusForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          api.Service
		update       api.Service
		expectedErrs field.ErrorList
	}{
		// baseline
		"no change": {
			old:    *mkValidService(),
			update: *mkValidService(),
		},
		// TODO: Add more tests.
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, StatusStrategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// mkValidService produces a service with a set of tweaks to test validation. 
func mkValidService(tweaks ...func(svc *api.Service)) *api.Service {
    svc := makeValidService()

    for _, tweak := range tweaks {
        tweak(svc)
    }

    return svc
}