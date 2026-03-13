/*
Copyright 2016 The Kubernetes Authors.

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

package create

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCreateServiceAccount(t *testing.T) {
	tests := map[string]struct {
		options  *ServiceAccountOpts
		expected *corev1.ServiceAccount
	}{
		"service account": {
			options: &ServiceAccountOpts{
				Name: "my-service-account",
			},
			expected: &corev1.ServiceAccount{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ServiceAccount",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-service-account",
				},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			serviceAccount, err := tc.options.createServiceAccount()
			if err != nil {
				t.Errorf("unexpected error:\n%#v\n", err)
				return
			}
			if !apiequality.Semantic.DeepEqual(serviceAccount, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, serviceAccount)
			}
		})
	}
}
