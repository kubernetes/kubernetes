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

package configmap

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	testDeclarativeValidateForDeclarative(t, "v1")
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: apiVersion,
	})
	testCases := map[string]struct {
		input        api.ConfigMap
		expectedErrs field.ErrorList
	}{
		"valid configmap": {
			input:        mkValidConfigMap(),
			expectedErrs: field.ErrorList{},
		},
		"invalid name with underscore": {
			input: mkValidConfigMap(func(cm *api.ConfigMap) {
				cm.ObjectMeta.Name = "invalid_name"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "invalid_name", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"invalid name with uppercase": {
			input: mkValidConfigMap(func(cm *api.ConfigMap) {
				cm.ObjectMeta.Name = "InvalidName"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "InvalidName", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"invalid name with special character": {
			input: mkValidConfigMap(func(cm *api.ConfigMap) {
				cm.ObjectMeta.Name = "invalid@name"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "invalid@name", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	testValidateUpdateForDeclarative(t, "v1")
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: apiVersion,
	})
	testCases := map[string]struct {
		old          api.ConfigMap
		update       api.ConfigMap
		expectedErrs field.ErrorList
	}{
		"valid update": {
			old:          mkValidConfigMap(),
			update:       mkValidConfigMap(),
			expectedErrs: field.ErrorList{},
		},
		"invalid name format in update": {
			old: mkValidConfigMap(),
			update: mkValidConfigMap(func(cm *api.ConfigMap) {
				cm.ObjectMeta.Name = "invalid_name"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "invalid_name", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"invalid name with uppercase in update": {
			old: mkValidConfigMap(),
			update: mkValidConfigMap(func(cm *api.ConfigMap) {
				cm.ObjectMeta.Name = "InvalidName"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "InvalidName", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"invalid name with special character in update": {
			old: mkValidConfigMap(),
			update: mkValidConfigMap(func(cm *api.ConfigMap) {
				cm.ObjectMeta.Name = "invalid@name"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "invalid@name", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ObjectMeta.ResourceVersion = "1"
			tc.update.ObjectMeta.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// Helper function to create a valid ConfigMap
func mkValidConfigMap(tweaks ...func(cm *api.ConfigMap)) api.ConfigMap {
	cm := api.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-configmap",
			Namespace: "default",
		},
		Data: map[string]string{
			"key1": "value1",
		},
	}
	for _, tweak := range tweaks {
		tweak(&cm)
	}
	return cm
}
