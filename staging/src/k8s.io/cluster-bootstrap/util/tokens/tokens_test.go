/*
Copyright 2019 The Kubernetes Authors.

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

package tokens

import (
	"reflect"
	"testing"

	bootstraptokenapi "k8s.io/cluster-bootstrap/token/api"
)

func TestNewBootstrapTokenStringFromIDAndSecret(t *testing.T) {
	var tests = []struct {
		id, secret    string
		expectedError bool
		bts           *bootstraptokenapi.BootstrapTokenString
	}{
		{id: "", secret: "", expectedError: true, bts: nil},
		{id: "1234567890123456789012", secret: "", expectedError: true, bts: nil}, // invalid parcel size
		{id: "12345", secret: "1234567890123456", expectedError: true, bts: nil},  // invalid parcel size
		{id: "", secret: "1234567890123456", expectedError: true, bts: nil},       // invalid parcel size
		{id: "123456", secret: "", expectedError: true, bts: nil},                 // invalid parcel size
		{id: "Abcdef", secret: "1234567890123456", expectedError: true, bts: nil}, // invalid token id
		{id: "123456", secret: "AABBCCDDEEFFGGHH", expectedError: true, bts: nil}, // invalid token secret
		{id: "123456", secret: "AABBCCD-EEFFGGHH", expectedError: true, bts: nil}, // invalid character
		{id: "abc*ef", secret: "1234567890123456", expectedError: true, bts: nil}, // invalid character
		{id: "abcdef", secret: "1234567890123456", expectedError: false, bts: &bootstraptokenapi.BootstrapTokenString{ID: "abcdef", Secret: "1234567890123456"}},
		{id: "123456", secret: "aabbccddeeffgghh", expectedError: false, bts: &bootstraptokenapi.BootstrapTokenString{ID: "123456", Secret: "aabbccddeeffgghh"}},
		{id: "abcdef", secret: "abcdef0123456789", expectedError: false, bts: &bootstraptokenapi.BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"}},
		{id: "123456", secret: "1234560123456789", expectedError: false, bts: &bootstraptokenapi.BootstrapTokenString{ID: "123456", Secret: "1234560123456789"}},
	}
	for _, rt := range tests {
		t.Run(rt.id, func(t *testing.T) {
			actual, err := NewBootstrapTokenStringFromIDAndSecret(rt.id, rt.secret)
			if (err != nil) != rt.expectedError {
				t.Errorf(
					"failed NewBootstrapTokenStringFromIDAndSecret for the token with id %q and secret %q\n\texpected error: %t\n\t  actual error: %v",
					rt.id,
					rt.secret,
					rt.expectedError,
					err,
				)
			} else if !reflect.DeepEqual(actual, rt.bts) {
				t.Errorf(
					"failed NewBootstrapTokenStringFromIDAndSecret for the token with id %q and secret %q\n\texpected: %v\n\t  actual: %v",
					rt.id,
					rt.secret,
					rt.bts,
					actual,
				)
			}
		})
	}
}
