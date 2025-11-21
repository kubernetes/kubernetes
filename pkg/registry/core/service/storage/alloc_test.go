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

package storage

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestPreAllocIPs(t *testing.T) {
	testCases := []struct {
		name            string
		meta            metav1.ObjectMeta
		featureDisabled bool
		wantErr         error
	}{
		{
			name: "missing name",
			meta: metav1.ObjectMeta{
				Name:      "",
				Namespace: "default",
			},
			wantErr: errors.NewInvalid(api.Kind("Service"), "", field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")}),
		},
		{
			name: "FG enabled, name is relaxed",
			meta: metav1.ObjectMeta{
				Name:      "1-test",
				Namespace: "default",
			},
		},
		{
			name: "FG disabled, name is invalid",
			meta: metav1.ObjectMeta{
				Name:      "1-test",
				Namespace: "default",
			},
			featureDisabled: true,
			wantErr:         errors.NewInvalid(api.Kind("Service"), "1-test", field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), "1-test", "a DNS-1035 label must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123', regex used for validation is '[a-z]([-a-z0-9]*[a-z0-9])?')")}),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RelaxedServiceNameValidation, !tc.featureDisabled)

			err := preAllocIPs(&api.Service{ObjectMeta: tc.meta})
			if !reflect.DeepEqual(err, tc.wantErr) {
				t.Fatalf("expected error: %v, got: %v", tc.wantErr, err)
			}
		})
	}
}
