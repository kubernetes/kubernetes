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

package authority

import (
	"crypto/x509"
	"fmt"
	"reflect"
	"testing"

	capi "k8s.io/api/certificates/v1"
)

func TestKeyUsagesFromStrings(t *testing.T) {
	testcases := []struct {
		usages              []capi.KeyUsage
		expectedKeyUsage    x509.KeyUsage
		expectedExtKeyUsage []x509.ExtKeyUsage
		expectErr           bool
	}{
		{
			usages:              []capi.KeyUsage{"signing"},
			expectedKeyUsage:    x509.KeyUsageDigitalSignature,
			expectedExtKeyUsage: nil,
			expectErr:           false,
		},
		{
			usages:              []capi.KeyUsage{"client auth"},
			expectedKeyUsage:    0,
			expectedExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			expectErr:           false,
		},
		{
			usages:              []capi.KeyUsage{"client auth", "client auth"},
			expectedKeyUsage:    0,
			expectedExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			expectErr:           false,
		},
		{
			usages:              []capi.KeyUsage{"cert sign", "encipher only"},
			expectedKeyUsage:    x509.KeyUsageCertSign | x509.KeyUsageEncipherOnly,
			expectedExtKeyUsage: nil,
			expectErr:           false,
		},
		{
			usages:              []capi.KeyUsage{"ocsp signing", "crl sign", "s/mime", "content commitment"},
			expectedKeyUsage:    x509.KeyUsageCRLSign | x509.KeyUsageContentCommitment,
			expectedExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageEmailProtection, x509.ExtKeyUsageOCSPSigning},
			expectErr:           false,
		},
		{
			usages:              []capi.KeyUsage{"unsupported string"},
			expectedKeyUsage:    0,
			expectedExtKeyUsage: nil,
			expectErr:           true,
		},
	}

	for _, tc := range testcases {
		t.Run(fmt.Sprint(tc.usages), func(t *testing.T) {
			ku, eku, err := keyUsagesFromStrings(tc.usages)

			if tc.expectErr {
				if err == nil {
					t.Errorf("did not return an error, but expected one")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if ku != tc.expectedKeyUsage || !reflect.DeepEqual(eku, tc.expectedExtKeyUsage) {
				t.Errorf("got=(%v, %v), want=(%v, %v)", ku, eku, tc.expectedKeyUsage, tc.expectedExtKeyUsage)
			}
		})
	}
}
