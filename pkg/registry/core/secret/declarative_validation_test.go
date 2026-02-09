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

package secret

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestDeclarativeValidateUpdate(t *testing.T) {
	testCases := map[string]struct {
		oldObj       api.Secret
		updateObj    api.Secret
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    makeValidSecret(),
			updateObj: makeValidSecret(),
		},
		"invalid update: type changed": {
			oldObj:    makeValidSecret(),
			updateObj: makeValidSecret(tweakType(api.SecretType("custom-type"))),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("type"), api.SecretType("custom-type"), "field is immutable").WithOrigin("immutable"),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "",
				APIVersion:        "v1",
				Resource:          "secrets",
				Name:              "test-secret",
				IsResourceRequest: true,
				Verb:              "update",
			})
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func makeValidSecret(mutators ...func(*api.Secret)) api.Secret {
	secret := api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "default",
		},
		Type: api.SecretTypeOpaque,
		Data: map[string][]byte{
			"key": []byte("value"),
		},
	}
	for _, mutate := range mutators {
		mutate(&secret)
	}
	return secret
}

func tweakType(secretType api.SecretType) func(*api.Secret) {
	return func(s *api.Secret) {
		s.Type = secretType
	}
}
