/*
Copyright 2023 The Kubernetes Authors.

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

package dynamiccertificates

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"math/big"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestServerCertValidator_Validate(t *testing.T) {
	rsaKey, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)

	tests := []struct {
		name     string
		template *x509.Certificate
		want     []error
	}{
		{
			name: "no ServerAuth EKU",
			template: &x509.Certificate{
				SerialNumber: big.NewInt(1),
				NotBefore:    time.Now().Add(-10 * time.Second),
				NotAfter:     time.Now().Add(10 * time.Minute),
				Subject:      pkix.Name{CommonName: "test-server"},
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
				DNSNames:     []string{"dns.name.one"},
			},
			want: []error{fmt.Errorf("missing ServerAuth extended key usage extension")},
		},
		{
			name: "no SAN",
			template: &x509.Certificate{
				SerialNumber: big.NewInt(1),
				NotBefore:    time.Now().Add(-10 * time.Second),
				NotAfter:     time.Now().Add(10 * time.Minute),
				Subject:      pkix.Name{CommonName: "test-server"},
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
			},
			want: []error{fmt.Errorf("missing the Subject Alternative Name extension")},
		},
		{
			name: "expired",
			template: &x509.Certificate{
				SerialNumber: big.NewInt(1),
				NotBefore:    time.Now().Add(-10 * time.Minute),
				NotAfter:     time.Now().Add(-10 * time.Second),
				Subject:      pkix.Name{CommonName: "test-server"},
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
				DNSNames:     []string{"dns.name.one"},
			},
			want: []error{fmt.Errorf("expired")},
		},
		{
			name: "not yet valid",
			template: &x509.Certificate{
				SerialNumber: big.NewInt(1),
				NotBefore:    time.Now().Add(10 * time.Second),
				NotAfter:     time.Now().Add(10 * time.Minute),
				Subject:      pkix.Name{CommonName: "test-server"},
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
				DNSNames:     []string{"dns.name.one"},
			},
			want: []error{fmt.Errorf("not yet valid")},
		},
		{
			name: "proper server cert",
			template: &x509.Certificate{
				SerialNumber: big.NewInt(1),
				NotBefore:    time.Now().Add(-10 * time.Second),
				NotAfter:     time.Now().Add(10 * time.Minute),
				Subject:      pkix.Name{CommonName: "test-server"},
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
				DNSNames:     []string{"dns.name.one"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			certDER, err := x509.CreateCertificate(rand.Reader, tt.template, tt.template, rsaKey.Public(), rsaKey)
			require.NoError(t, err)

			cert, err := x509.ParseCertificate(certDER)
			require.NoError(t, err)

			v := &ServerCertValidator{}
			if got := v.Validate(cert); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ServerCertValidator.Validate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestClientCertValidator_Validate(t *testing.T) {
	rsaKey, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)

	tests := []struct {
		name     string
		template *x509.Certificate
		want     []error
	}{
		{
			name: "no ClientAuth EKU",
			template: &x509.Certificate{
				SerialNumber: big.NewInt(1),
				NotBefore:    time.Now().Add(-10 * time.Second),
				NotAfter:     time.Now().Add(10 * time.Minute),
				Subject:      pkix.Name{CommonName: "test-client"},
				DNSNames:     []string{"dns.name.one"},
			},
			want: []error{fmt.Errorf("missing ClientAuth extended key usage extension")},
		},
		{
			name: "expired",
			template: &x509.Certificate{
				SerialNumber: big.NewInt(1),
				NotBefore:    time.Now().Add(-10 * time.Minute),
				NotAfter:     time.Now().Add(-10 * time.Second),
				Subject:      pkix.Name{CommonName: "test-client"},
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
			want: []error{fmt.Errorf("expired")},
		},
		{
			name: "not yet valid",
			template: &x509.Certificate{
				SerialNumber: big.NewInt(1),
				NotBefore:    time.Now().Add(10 * time.Second),
				NotAfter:     time.Now().Add(10 * time.Minute),
				Subject:      pkix.Name{CommonName: "test-client"},
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
			want: []error{fmt.Errorf("not yet valid")},
		},
		{
			name: "proper client cert",
			template: &x509.Certificate{
				SerialNumber: big.NewInt(1),
				NotBefore:    time.Now().Add(-10 * time.Second),
				NotAfter:     time.Now().Add(10 * time.Minute),
				Subject:      pkix.Name{CommonName: "test-client"},
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
				DNSNames:     []string{"client.tld"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			certDER, err := x509.CreateCertificate(rand.Reader, tt.template, tt.template, rsaKey.Public(), rsaKey)
			require.NoError(t, err)

			cert, err := x509.ParseCertificate(certDER)
			require.NoError(t, err)

			v := &ClientCertValidator{}
			if got := v.Validate(cert); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ClientCertValidator.Validate() = %v, want %v", got, tt.want)
			}
		})
	}
}
