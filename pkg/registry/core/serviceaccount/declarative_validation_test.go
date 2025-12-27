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

package serviceaccount

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
		input        api.ServiceAccount
		expectedErrs field.ErrorList
	}{
		// baseline
		"empty resource": {
			input: mkValidServiceAccount(),
		},
		// metadata.name
		"name: empty": {
			input: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata.name"), ""),
			},
		},
		"name: label format": {
			input: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Name = "this-is-a-label"
			}),
		},
		"name: subdomain format": {
			input: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Name = "this.is.a.subdomain"
			}),
		},
		"name: invalid label format": {
			input: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Name = "-this-is-not-a-label"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, ""),
			},
		},
		"name: invalid subdomain format": {
			input: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Name = ".this.is.not.a.subdomain"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, ""),
			},
		},
		"name: label format with trailing dash": {
			input: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Name = "this-is-a-label-"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, ""),
			},
		},
		"name: too long": {
			input: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Name = strings.Repeat("x", 254)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, ""),
			},
		},
		// metadata.namespace
		"namespace: empty": {
			input: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Namespace = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata.namespace"), ""),
			},
		},
		"namespace: valid": {
			input: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Namespace = "valid-namespace"
			}),
		},
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
		old          api.ServiceAccount
		update       api.ServiceAccount
		expectedErrs field.ErrorList
	}{
		// baseline
		"no change": {
			old:    mkValidServiceAccount(),
			update: mkValidServiceAccount(),
		},
		// metadata.name
		"name: changed": {
			old: mkValidServiceAccount(),
			update: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Name += "x"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, ""),
			},
		},
		// metadata.namespace
		"namespace: changed": {
			old: mkValidServiceAccount(),
			update: mkValidServiceAccount(func(sa *api.ServiceAccount) {
				sa.Namespace = "other-namespace"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.namespace"), nil, ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ObjectMeta.ResourceVersion = "1"
			tc.update.ObjectMeta.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// mkValidServiceAccount produces a ServiceAccount which passes
// validation with no tweaks.
func mkValidServiceAccount(tweaks ...func(sa *api.ServiceAccount)) api.ServiceAccount {
	sa := api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "default",
			Namespace: metav1.NamespaceDefault,
		},
	}
	for _, tweak := range tweaks {
		tweak(&sa)
	}
	return sa
}
