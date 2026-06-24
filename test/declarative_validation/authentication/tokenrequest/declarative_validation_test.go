/*
Copyright The Kubernetes Authors.

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

// Package tokenrequest exercises declarative validation for the
// authentication.k8s.io TokenRequest resource.
package tokenrequest

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/authentication"
	authenticationvalidation "k8s.io/kubernetes/pkg/apis/authentication/validation"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			testDeclarativeValidate(t, v)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{APIGroup: "authentication.k8s.io", APIVersion: apiVersion, Resource: "tokenrequests", Namespace: "default"})
	ctx = genericapirequest.WithNamespace(ctx, "default")

	testCases := map[string]struct {
		obj          authentication.TokenRequest
		expectedErrs field.ErrorList
	}{
		"valid": {
			obj: mkTokenRequest(setExpirationSeconds(1000)),
		},
		"expiration_too_small": {
			obj:          mkTokenRequest(setExpirationSeconds(599)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec.expirationSeconds"), int64(599), "").WithOrigin("minimum").MarkAlpha()},
		},
		"expiration_too_large": {
			obj:          mkTokenRequest(setExpirationSeconds(1<<32 + 1)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec.expirationSeconds"), int64(1<<32+1), "").WithOrigin("maximum").MarkAlpha()},
		},
		"expiration_zero": {
			obj:          mkTokenRequest(setExpirationSeconds(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec.expirationSeconds"), int64(0), "").WithOrigin("minimum").MarkAlpha()},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalenceFunc(t, ctx, &tc.obj, func(ctx context.Context, obj runtime.Object) field.ErrorList {
				tr := obj.(*authentication.TokenRequest)
				return authenticationvalidation.ValidateTokenRequestCreate(ctx, legacyscheme.Scheme, tr)
			}, tc.expectedErrs)
		})
	}
}

func mkTokenRequest(tweaks ...func(*authentication.TokenRequest)) authentication.TokenRequest {
	tr := authentication.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default"},
		Spec: authentication.TokenRequestSpec{
			Audiences: []string{"api"},
		},
	}
	for _, tweak := range tweaks {
		tweak(&tr)
	}
	return tr
}

func setExpirationSeconds(s int64) func(*authentication.TokenRequest) {
	return func(tr *authentication.TokenRequest) {
		tr.Spec.ExpirationSeconds = s
	}
}
