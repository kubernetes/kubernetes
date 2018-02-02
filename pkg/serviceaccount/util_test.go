/*
Copyright 2017 The Kubernetes Authors.

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
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsServiceAccountToken(t *testing.T) {
	testCases := map[string]struct {
		inputSecret *v1.Secret
		inputSa     *v1.ServiceAccount
		expected    bool
	}{
		"secret type not match": {
			inputSecret: &v1.Secret{
				Type: v1.SecretTypeOpaque,
			},
			inputSa:  &v1.ServiceAccount{},
			expected: false,
		},
		"names not match": {
			inputSecret: &v1.Secret{
				Type: v1.SecretTypeServiceAccountToken,
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.ServiceAccountNameKey: "secret-name-key",
					},
				},
			},
			inputSa: &v1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name: "serviceAccount-name-key",
				},
			},
			expected: false,
		},
		"UID not match": {
			inputSecret: &v1.Secret{
				Type: v1.SecretTypeServiceAccountToken,
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.ServiceAccountNameKey: "account-name-key",
						v1.ServiceAccountUIDKey:  "secret-UID-key",
					},
				},
			},
			inputSa: &v1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name: "account-name-key",
					UID:  "serviceAccount-UID-key",
				},
			},
			expected: false,
		},
		"all match": {
			inputSecret: &v1.Secret{
				Type: v1.SecretTypeServiceAccountToken,
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.ServiceAccountNameKey: "account-name-key-1",
						v1.ServiceAccountUIDKey:  "account-UID-key-1",
					},
				},
			},
			inputSa: &v1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name: "account-name-key-1",
					UID:  "account-UID-key-1",
				},
			},
			expected: true,
		},
	}

	for testCaseName, testCase := range testCases {
		result := IsServiceAccountToken(testCase.inputSecret, testCase.inputSa)
		if result != testCase.expected {
			t.Errorf("unexpected result in testcase: %s, expected: %t, actual: %t", testCaseName, testCase.expected, result)
		}
	}
}
