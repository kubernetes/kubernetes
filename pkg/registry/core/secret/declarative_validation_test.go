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

package secret

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestDeclarativeValidate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		input        api.Secret
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidSecret(),
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		oldObj       api.Secret
		updateObj    api.Secret
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidSecret(),
			updateObj: mkValidSecret(),
		},
		"immutable type": {
			oldObj: mkValidSecret(),
			updateObj: mkValidSecret(func(s *api.Secret) {
				s.Type = api.SecretTypeTLS
				s.Data = map[string][]byte{
					api.TLSCertKey:       []byte("cert"),
					api.TLSPrivateKeyKey: []byte("key"),
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("type"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.oldObj.ObjectMeta.ResourceVersion = "1"
			tc.updateObj.ObjectMeta.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidSecret(tweaks ...func(*api.Secret)) api.Secret {
	s := api.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "valid-secret", Namespace: metav1.NamespaceDefault},
		Type:       api.SecretTypeOpaque,
	}
	for _, fn := range tweaks {
		fn(&s)
	}
	return s
}
