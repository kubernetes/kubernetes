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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"math/big"
	"net/url"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	capi "k8s.io/api/certificates/v1beta1"
	"k8s.io/apimachinery/pkg/util/diff"
)

func TestCertificateAuthority(t *testing.T) {
	caKey, err := ecdsa.GenerateKey(elliptic.P224(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(42),
		Subject: pkix.Name{
			CommonName: "test-ca",
		},
		NotBefore:             now.Add(-24 * time.Hour),
		NotAfter:              now.Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, caKey.Public(), caKey)
	if err != nil {
		t.Fatal(err)
	}
	caCert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}

	uri, err := url.Parse("help://me@what:8080/where/when?why=true")
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name     string
		cr       x509.CertificateRequest
		backdate time.Duration
		policy   SigningPolicy

		want    x509.Certificate
		wantErr bool
	}{
		{
			name:   "ca info",
			policy: PermissiveSigningPolicy{TTL: time.Hour},
			want: x509.Certificate{
				Issuer:                caCert.Subject,
				AuthorityKeyId:        caCert.SubjectKeyId,
				NotBefore:             now,
				NotAfter:              now.Add(1 * time.Hour),
				BasicConstraintsValid: true,
			},
		},
		{
			name:   "key usage",
			policy: PermissiveSigningPolicy{TTL: time.Hour, Usages: []capi.KeyUsage{"signing"}},
			want: x509.Certificate{
				NotBefore:             now,
				NotAfter:              now.Add(1 * time.Hour),
				BasicConstraintsValid: true,
				KeyUsage:              x509.KeyUsageDigitalSignature,
			},
		},
		{
			name:   "ext key usage",
			policy: PermissiveSigningPolicy{TTL: time.Hour, Usages: []capi.KeyUsage{"client auth"}},
			want: x509.Certificate{
				NotBefore:             now,
				NotAfter:              now.Add(1 * time.Hour),
				BasicConstraintsValid: true,
				ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
		{
			name:     "backdate",
			policy:   PermissiveSigningPolicy{TTL: time.Hour},
			backdate: 5 * time.Minute,
			want: x509.Certificate{
				NotBefore:             now.Add(-5 * time.Minute),
				NotAfter:              now.Add(55 * time.Minute),
				BasicConstraintsValid: true,
			},
		},
		{
			name:   "truncate expiration",
			policy: PermissiveSigningPolicy{TTL: 48 * time.Hour},
			want: x509.Certificate{
				NotBefore:             now,
				NotAfter:              now.Add(24 * time.Hour),
				BasicConstraintsValid: true,
			},
		},
		{
			name:   "uri sans",
			policy: PermissiveSigningPolicy{TTL: time.Hour},
			cr: x509.CertificateRequest{
				URIs: []*url.URL{uri},
			},
			want: x509.Certificate{
				URIs:                  []*url.URL{uri},
				NotBefore:             now,
				NotAfter:              now.Add(1 * time.Hour),
				BasicConstraintsValid: true,
			},
		},
	}

	crKey, err := ecdsa.GenerateKey(elliptic.P224(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ca := &CertificateAuthority{
				Certificate: caCert,
				PrivateKey:  caKey,
				Now: func() time.Time {
					return now
				},
				Backdate: test.backdate,
			}

			csr, err := x509.CreateCertificateRequest(rand.Reader, &test.cr, crKey)
			if err != nil {
				t.Fatal(err)
			}

			certDER, err := ca.Sign(csr, test.policy)
			if err != nil {
				t.Fatal(err)
			}
			if test.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}

			cert, err := x509.ParseCertificate(certDER)
			if err != nil {
				t.Fatal(err)
			}

			opts := cmp.Options{
				cmpopts.IgnoreFields(x509.Certificate{},
					"SignatureAlgorithm",
					"PublicKeyAlgorithm",
					"Version",
					"MaxPathLen",
				),
				diff.IgnoreUnset(),
				cmp.Transformer("RoundTime", func(x time.Time) time.Time {
					return x.Truncate(time.Second)
				}),
				cmp.Comparer(func(x, y *url.URL) bool {
					return ((x == nil) && (y == nil)) || x.String() == y.String()
				}),
			}
			if !cmp.Equal(*cert, test.want, opts) {
				t.Errorf("unexpected diff: %v", cmp.Diff(*cert, test.want, opts))
			}
		})
	}
}
