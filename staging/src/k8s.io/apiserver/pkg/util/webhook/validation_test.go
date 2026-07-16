/*
Copyright 2024 The Kubernetes Authors.

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

package webhook

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

// exampleCert was generated from crypto/tls/generate_cert.go with the following command:
//
//	go run generate_cert.go  --rsa-bits 2048 --host example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var exampleCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDADCCAeigAwIBAgIQVHG3Fn9SdWayyLOZKCW1vzANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMCAXDTcwMDEwMTAwMDAwMFoYDzIwODQwMTI5MTYw
MDAwWjASMRAwDgYDVQQKEwdBY21lIENvMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A
MIIBCgKCAQEArTCu9fiIclNgDdWHphewM+JW55dCb5yYGlJgCBvwbOx547M9p+tn
zm9QOhsdZDHDZsG9tqnWxE2Nc1HpIJyOlfYsOoonpEoG/Ep6nnK91ngj0bn/JlNy
+i/bwU4r97MOukvnOIQez9/D9jAJaOX2+b8/d4lRz9BsqiwJyg+ynZ5tVVYj7aMi
vXnd6HOnJmtqutOtr3beucJnkd6XbwRkLUcAYATT+ZihOWRbTuKqhCg6zGkJOoUG
f8sX61JjoilxiURA//ftGVbdTCU3DrmGmardp5NNOHbumMYU8Vhmqgx1Bqxb+9he
7G42uW5YWYK/GqJzgVPjjlB2dOGj9KrEWQIDAQABo1AwTjAOBgNVHQ8BAf8EBAMC
AqQwEwYDVR0lBAwwCgYIKwYBBQUHAwEwDwYDVR0TAQH/BAUwAwEB/zAWBgNVHREE
DzANggtleGFtcGxlLmNvbTANBgkqhkiG9w0BAQsFAAOCAQEAig4AIi9xWs1+pLES
eeGGdSDoclplFpcbXANnsYYFyLf+8pcWgVi2bOmb2gXMbHFkB07MA82wRJAUTaA+
2iNXVQMhPCoA7J6ADUbww9doJX2S9HGyArhiV/MhHtE8txzMn2EKNLdhhk3N9rmV
x/qRbWAY1U2z4BpdrAR87Fe81Nlj7h45csW9K+eS+NgXipiNTIfEShKgCFM8EdxL
1WXg7r9AvYV3TNDPWTjLsm1rQzzZQ7Uvcf6deWiNodZd8MOT/BFLclDPTK6cF2Hr
UU4dq6G4kCwMSxWE4cM3HlZ4u1dyIt47VbkP0rtvkBCXx36y+NXYA5lzntchNFZP
uvEQdw==
-----END CERTIFICATE-----`)

func TestValidateCABundle(t *testing.T) {
	tests := []struct {
		name      string
		caBundle  []byte
		expectErr bool
	}{
		{
			name:      "nil caBundle is valid",
			caBundle:  nil,
			expectErr: false,
		}, {
			name:      "empty caBundle is valid",
			caBundle:  []byte(""),
			expectErr: false,
		}, {
			name:      "non empty caBundle with invalid certificate should not validate",
			caBundle:  []byte("bogus"),
			expectErr: true,
		}, {
			name:      "non empty caBundle with no certificate should not validate",
			caBundle:  []byte("Cg=="),
			expectErr: true,
		}, {
			name:      "non empty caBundle with valid certificate should validate",
			caBundle:  exampleCert,
			expectErr: false,
		},
	}

	for _, tc := range tests {
		errList := ValidateCABundle(field.NewPath(""), tc.caBundle)
		if len(errList) > 0 && !tc.expectErr {
			t.Errorf("Expected no error for test %s, got %v", tc.name, errList)
		} else if len(errList) == 0 && tc.expectErr {
			t.Errorf("Expected error for test %s, received no error", tc.name)
		}
	}
}
