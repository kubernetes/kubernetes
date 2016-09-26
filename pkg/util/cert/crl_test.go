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
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"testing"
)

func TestLoadCRLFile(t *testing.T) {
	n := 3
	clientCerts := make([][]*x509.Certificate, n)
	for i := 0; i < n; i++ {
		keyfile := fmt.Sprintf("testdata/client_%d.key", i)
		crtfile := fmt.Sprintf("testdata/client_%d.crt", i)
		tlsCert, err := tls.LoadX509KeyPair(crtfile, keyfile)
		if err != nil {
			t.Fatal(err)
		}

		for _, rawCert := range tlsCert.Certificate {
			cert, err := x509.ParseCertificate(rawCert)
			if err != nil {
				t.Fatal(err)
			}
			clientCerts[i] = append(clientCerts[i], cert)
		}
	}

	for i := 0; i < n; i++ {
		crl, err := LoadCRLFile(fmt.Sprintf("testdata/crl_%d.crl", i))
		if err != nil {
			t.Fatal(err)
		}
		for j, clientCert := range clientCerts {
			// Each CRL rejects the certificate with the same ID. "crl_1.crl" will
			// only reject "client_1.crt".
			wantReject := i == j

			gotReject := func() error {
				for _, cert := range clientCert {
					if err := crl.VerifyCertificate(cert); err != nil {
						return err
					}
				}
				return nil
			}() != nil

			if wantReject != gotReject {
				t.Errorf("case crl_%d.crt client_%d.crt: expected reject=%t got=%t", i, j, wantReject, gotReject)
			}
		}
	}
}
