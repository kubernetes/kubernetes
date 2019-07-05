/*
Copyright 2018 The Kubernetes Authors.

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

	secretIns := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "token-secret-1",
			Namespace:       "default",
			UID:             "23456",
			ResourceVersion: "1",
			Annotations: map[string]string{
				v1.ServiceAccountNameKey: "default",
				v1.ServiceAccountUIDKey:  "12345",
			},
		},
		Type: v1.SecretTypeServiceAccountToken,
		Data: map[string][]byte{
			"token":     []byte("ABC"),
			"ca.crt":    []byte("CA Data"),
			"namespace": []byte("default"),
		},
	}

	secretTypeMistmatch := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "token-secret-2",
			Namespace:       "default",
			UID:             "23456",
			ResourceVersion: "1",
			Annotations: map[string]string{
				v1.ServiceAccountNameKey: "default",
				v1.ServiceAccountUIDKey:  "12345",
			},
		},
		Type: v1.SecretTypeOpaque,
	}

	saIns := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "default",
			UID:             "12345",
			Namespace:       "default",
			ResourceVersion: "1",
		},
	}

	saInsNameNotEqual := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "non-default",
			UID:             "12345",
			Namespace:       "default",
			ResourceVersion: "1",
		},
	}

	saInsUIDNotEqual := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "default",
			UID:             "67890",
			Namespace:       "default",
			ResourceVersion: "1",
		},
	}

	tests := map[string]struct {
		secret *v1.Secret
		sa     *v1.ServiceAccount
		expect bool
	}{
		"correct service account": {
			secret: secretIns,
			sa:     saIns,
			expect: true,
		},
		"service account name not equal": {
			secret: secretIns,
			sa:     saInsNameNotEqual,
			expect: false,
		},
		"service account uid not equal": {
			secret: secretIns,
			sa:     saInsUIDNotEqual,
			expect: false,
		},
		"service account type not equal": {
			secret: secretTypeMistmatch,
			sa:     saIns,
			expect: false,
		},
	}

	for k, v := range tests {
		actual := IsServiceAccountToken(v.secret, v.sa)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}

}
