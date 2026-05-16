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
	registry "k8s.io/kubernetes/pkg/registry/core/secret"
)

func TestDeclarativeValidateUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          api.Secret
		update       api.Secret
		expectedErrs field.ErrorList
	}{
		"no change": {
			old:    mkValidSecret(),
			update: mkValidSecret(),
		},
		"type: changed": {
			old:    mkValidSecret(),
			update: mkValidSecret(tweakType(api.SecretType("custom-type"))),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("type"), api.SecretType("custom-type"), "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, registry.Strategy, tc.expectedErrs)
		})
	}
}

func mkValidSecret(tweaks ...func(*api.Secret)) api.Secret {
	s := api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: metav1.NamespaceDefault,
		},
		Type: api.SecretTypeOpaque,
		Data: map[string][]byte{
			"key": []byte("value"),
		},
	}
	for _, tweak := range tweaks {
		tweak(&s)
	}
	return s
}

func tweakType(t api.SecretType) func(*api.Secret) {
	return func(s *api.Secret) {
		s.Type = t
	}
}
