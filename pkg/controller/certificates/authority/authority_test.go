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
	"fmt"
	"math/big"
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	capi "k8s.io/api/certificates/v1"
)

func TestCertificateAuthority(t *testing.T) {
	caKey, err := ecdsa.GenerateKey(elliptic.P224(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	nowFunc := func() time.Time { return now }
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
		policy   SigningPolicy
		mutateCA func(ca *CertificateAuthority)

		want    x509.Certificate
		wantErr string
	}{
		{
			name:   "ca info",
			policy: PermissiveSigningPolicy{TTL: time.Hour, Now: nowFunc},
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
			policy: PermissiveSigningPolicy{TTL: time.Hour, Usages: []capi.KeyUsage{"signing"}, Now: nowFunc},
			want: x509.Certificate{
				NotBefore:             now,
				NotAfter:              now.Add(1 * time.Hour),
				BasicConstraintsValid: true,
				KeyUsage:              x509.KeyUsageDigitalSignature,
			},
		},
		{
			name:   "ext key usage",
			policy: PermissiveSigningPolicy{TTL: time.Hour, Usages: []capi.KeyUsage{"client auth"}, Now: nowFunc},
			want: x509.Certificate{
				NotBefore:             now,
				NotAfter:              now.Add(1 * time.Hour),
				BasicConstraintsValid: true,
				ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
		{
			name:   "backdate without short",
			policy: PermissiveSigningPolicy{TTL: time.Hour, Backdate: 5 * time.Minute, Now: nowFunc},
			want: x509.Certificate{
				NotBefore:             now.Add(-5 * time.Minute),
				NotAfter:              now.Add(55 * time.Minute),
				BasicConstraintsValid: true,
			},
		},
		{
			name:   "backdate without short and super small ttl",
			policy: PermissiveSigningPolicy{TTL: time.Minute, Backdate: 5 * time.Minute, Now: nowFunc},
			want: x509.Certificate{
				NotBefore:             now.Add(-5 * time.Minute),
				NotAfter:              now.Add(-4 * time.Minute),
				BasicConstraintsValid: true,
			},
		},
		{
			name:   "backdate with short",
			policy: PermissiveSigningPolicy{TTL: time.Hour, Backdate: 5 * time.Minute, Short: 8 * time.Hour, Now: nowFunc},
			want: x509.Certificate{
				NotBefore:             now.Add(-5 * time.Minute),
				NotAfter:              now.Add(time.Hour),
				BasicConstraintsValid: true,
			},
		},
		{
			name:   "backdate with short and super small ttl",
			policy: PermissiveSigningPolicy{TTL: time.Minute, Backdate: 5 * time.Minute, Short: 8 * time.Hour, Now: nowFunc},
			want: x509.Certificate{
				NotBefore:             now.Add(-5 * time.Minute),
				NotAfter:              now.Add(time.Minute),
				BasicConstraintsValid: true,
			},
		},
		{
			name:   "backdate with short but longer ttl",
			policy: PermissiveSigningPolicy{TTL: 24 * time.Hour, Backdate: 5 * time.Minute, Short: 8 * time.Hour, Now: nowFunc},
			want: x509.Certificate{
				NotBefore:             now.Add(-5 * time.Minute),
				NotAfter:              now.Add(24*time.Hour - 5*time.Minute),
				BasicConstraintsValid: true,
			},
		},
		{
			name:   "truncate expiration",
			policy: PermissiveSigningPolicy{TTL: 48 * time.Hour, Now: nowFunc},
			want: x509.Certificate{
				NotBefore:             now,
				NotAfter:              now.Add(24 * time.Hour),
				BasicConstraintsValid: true,
			},
		},
		{
			name:   "uri sans",
			policy: PermissiveSigningPolicy{TTL: time.Hour, Now: nowFunc},
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
		{
			name:   "expired ca",
			policy: PermissiveSigningPolicy{TTL: time.Hour, Now: nowFunc},
			mutateCA: func(ca *CertificateAuthority) {
				ca.Certificate.NotAfter = now // pretend that the CA has expired
			},
			wantErr: "the signer has expired: NotAfter=" + now.String(),
		},
		{
			name:   "expired ca with backdate",
			policy: PermissiveSigningPolicy{TTL: time.Hour, Backdate: 5 * time.Minute, Now: nowFunc},
			mutateCA: func(ca *CertificateAuthority) {
				ca.Certificate.NotAfter = now // pretend that the CA has expired
			},
			wantErr: "refusing to sign a certificate that expired in the past: NotAfter=" + now.String(),
		},
	}

	crKey, err := ecdsa.GenerateKey(elliptic.P224(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			caCertShallowCopy := *caCert

			ca := &CertificateAuthority{
				Certificate: &caCertShallowCopy,
				PrivateKey:  caKey,
			}

			if test.mutateCA != nil {
				test.mutateCA(ca)
			}

			csr, err := x509.CreateCertificateRequest(rand.Reader, &test.cr, crKey)
			if err != nil {
				t.Fatal(err)
			}

			certDER, err := ca.Sign(csr, test.policy)
			if len(test.wantErr) > 0 {
				if errStr := errString(err); test.wantErr != errStr {
					t.Fatalf("expected error %s but got %s", test.wantErr, errStr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
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
				ignoreUnset(),
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

// ignoreUnset is an option that ignores fields that are unset on the right
// hand side of a comparison. This is useful in testing to assert that an
// object is a derivative.
func ignoreUnset() cmp.Option {
	return cmp.Options{
		// ignore unset fields in v2
		cmp.FilterPath(func(path cmp.Path) bool {
			_, v2 := path.Last().Values()
			switch v2.Kind() {
			case reflect.Slice, reflect.Map:
				if v2.IsNil() || v2.Len() == 0 {
					return true
				}
			case reflect.String:
				if v2.Len() == 0 {
					return true
				}
			case reflect.Interface, reflect.Pointer:
				if v2.IsNil() {
					return true
				}
			}
			return false
		}, cmp.Ignore()),
		// ignore map entries that aren't set in v2
		cmp.FilterPath(func(path cmp.Path) bool {
			switch i := path.Last().(type) {
			case cmp.MapIndex:
				if _, v2 := i.Values(); !v2.IsValid() {
					fmt.Println("E")
					return true
				}
			}
			return false
		}, cmp.Ignore()),
	}
}

func errString(err error) string {
	if err == nil {
		return ""
	}

	return err.Error()
}
