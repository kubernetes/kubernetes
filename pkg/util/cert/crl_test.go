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

package cert

import (
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"testing"
)

func TestLoadCRLFile(t *testing.T) {
	loadCerts := func(ca, serialNum int) []*x509.Certificate {
		p := fmt.Sprintf("testdata/client_ca%d_id%d.crt", ca, serialNum)
		data, err := ioutil.ReadFile(p)
		if err != nil {
			t.Fatalf("read file %s: %v", p, err)
		}
		block, _ := pem.Decode(data)
		if block == nil {
			t.Fatalf("no pem data in file %s", p)
		}
		certs, err := x509.ParseCertificates(block.Bytes)
		if err != nil {
			t.Fatalf("parse certs from file %s: %v", p, err)
		}
		return certs
	}

	loadCRL := func(ca, serialNum int) RevocationPolicy {
		p := fmt.Sprintf("testdata/crl_ca%d_id%d.crl", ca, serialNum)
		policy, err := LoadCRLFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return policy
	}

	tests := []struct {
		name      string
		certs     []*x509.Certificate
		policy    RevocationPolicy
		wantValid bool
	}{
		{
			name:      "revoked cert ca #0",
			certs:     loadCerts(0, 0),
			policy:    loadCRL(0, 0),
			wantValid: false,
		},
		{
			name:      "revoked cert ca #1",
			certs:     loadCerts(1, 0),
			policy:    loadCRL(1, 0),
			wantValid: false,
		},
		{
			name:      "unrevoked cert",
			certs:     loadCerts(0, 1),
			policy:    loadCRL(0, 0),
			wantValid: true,
		},
		{
			name:      "same serial number, different ca",
			certs:     loadCerts(1, 0),
			policy:    loadCRL(0, 0),
			wantValid: true,
		},
	}

	for _, tc := range tests {
		gotValid, err := func() (valid bool, err error) {
			for _, cert := range tc.certs {
				valid, err = tc.policy.VerifyCertificate(cert)
				if !valid || err != nil {
					return
				}
			}
			return
		}()
		if err != nil {
			t.Errorf("case %q failed to verify cert: %v", tc.name, err)
			continue
		}

		if gotValid != tc.wantValid {
			t.Errorf("case %q expected valid=%t got=%t", tc.name, tc.wantValid, gotValid)
		}
	}
}
