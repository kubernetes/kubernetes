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

package proxy

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	utilnet "k8s.io/apimachinery/pkg/util/net"
)

func TestDialURL(t *testing.T) {
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("error setting up localhostCert pool")
	}

	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatal(err)
	}
	var d net.Dialer

	testcases := map[string]struct {
		TLSConfig   *tls.Config
		Dial        utilnet.DialFunc
		ExpectError string
	}{
		"insecure": {
			TLSConfig: &tls.Config{InsecureSkipVerify: true},
		},
		"secure, no roots": {
			TLSConfig:   &tls.Config{InsecureSkipVerify: false},
			ExpectError: "unknown authority",
		},
		"secure with roots": {
			TLSConfig: &tls.Config{InsecureSkipVerify: false, RootCAs: roots},
		},
		"secure with mismatched server": {
			TLSConfig:   &tls.Config{InsecureSkipVerify: false, RootCAs: roots, ServerName: "bogus.com"},
			ExpectError: "not bogus.com",
		},
		"secure with matched server": {
			TLSConfig: &tls.Config{InsecureSkipVerify: false, RootCAs: roots, ServerName: "example.com"},
		},

		"insecure, custom dial": {
			TLSConfig: &tls.Config{InsecureSkipVerify: true},
			Dial:      d.DialContext,
		},
		"secure, no roots, custom dial": {
			TLSConfig:   &tls.Config{InsecureSkipVerify: false},
			Dial:        d.DialContext,
			ExpectError: "unknown authority",
		},
		"secure with roots, custom dial": {
			TLSConfig: &tls.Config{InsecureSkipVerify: false, RootCAs: roots},
			Dial:      d.DialContext,
		},
		"secure with mismatched server, custom dial": {
			TLSConfig:   &tls.Config{InsecureSkipVerify: false, RootCAs: roots, ServerName: "bogus.com"},
			Dial:        d.DialContext,
			ExpectError: "not bogus.com",
		},
		"secure with matched server, custom dial": {
			TLSConfig: &tls.Config{InsecureSkipVerify: false, RootCAs: roots, ServerName: "example.com"},
			Dial:      d.DialContext,
		},
	}

	for k, tc := range testcases {
		func() {
			ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {}))
			defer ts.Close()
			ts.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
			ts.StartTLS()

			// Make a copy of the config
			tlsConfigCopy := tc.TLSConfig.Clone()
			// Clone() mutates the receiver (!), so also call it on the copy
			tlsConfigCopy.Clone()
			transport := &http.Transport{
				DialContext:     tc.Dial,
				TLSClientConfig: tlsConfigCopy,
			}

			extractedDial, err := utilnet.DialerFor(transport)
			if err != nil {
				t.Fatal(err)
			}
			if fmt.Sprintf("%p", extractedDial) != fmt.Sprintf("%p", tc.Dial) {
				t.Fatalf("%s: Unexpected dial", k)
			}

			extractedTLSConfig, err := utilnet.TLSClientConfig(transport)
			if err != nil {
				t.Fatal(err)
			}
			if extractedTLSConfig == nil {
				t.Fatalf("%s: Expected tlsConfig", k)
			}

			u, _ := url.Parse(ts.URL)
			_, p, _ := net.SplitHostPort(u.Host)
			u.Host = net.JoinHostPort("127.0.0.1", p)
			conn, err := DialURL(context.Background(), u, transport)

			// Make sure dialing doesn't mutate the transport's TLSConfig
			if !reflect.DeepEqual(tc.TLSConfig, tlsConfigCopy) {
				t.Errorf("%s: transport's copy of TLSConfig was mutated\n%s", k, diff.ObjectReflectDiff(tc.TLSConfig, tlsConfigCopy))
			}

			if err != nil {
				if tc.ExpectError == "" {
					t.Errorf("%s: expected no error, got %q", k, err.Error())
				}
				if !strings.Contains(err.Error(), tc.ExpectError) {
					t.Errorf("%s: expected error containing %q, got %q", k, tc.ExpectError, err.Error())
				}
				return
			}
			conn.Close()
			if tc.ExpectError != "" {
				t.Errorf("%s: expected error %q, got none", k, tc.ExpectError)
			}
		}()
	}

}

// localhostCert was generated from crypto/tls/generate_cert.go with the following command:
//     go run generate_cert.go  --rsa-bits 1024 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIICEzCCAXygAwIBAgIQRWyrLzhq/urpj7m6uPiMgjANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMCAXDTcwMDEwMTAwMDAwMFoYDzIwODQwMTI5MTYw
MDAwWjASMRAwDgYDVQQKEwdBY21lIENvMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCB
iQKBgQDjSYON17r13esbCFoS9l4xTBjqCqw7O4QWuTi7jBJHhU7wJ2TxCHuMO/3L
s8PE700nz5ryfnIu/5P/8wGVYOj27ixAWTNFgAyHW62q5i4uCD2VlOQrCZoEOsw6
a0hiDsnam63yW1nc/UK96Y3Yvmb7B6t34tAQ2MigoUeYwoKsPwIDAQABo2gwZjAO
BgNVHQ8BAf8EBAMCAqQwEwYDVR0lBAwwCgYIKwYBBQUHAwEwDwYDVR0TAQH/BAUw
AwEB/zAuBgNVHREEJzAlggtleGFtcGxlLmNvbYcEfwAAAYcQAAAAAAAAAAAAAAAA
AAAAATANBgkqhkiG9w0BAQsFAAOBgQAyyXwM2Up1i7/pLB+crSnH/TJnwhfwSVMZ
vAlDgYkGEb8YLc2K+sYqRRiwLuKivDck1xRH6vx3ENxmoX+SOIWVG8amXmqqFifh
G+i1AqOdHggw/UCu0uog8OZablbKxnbkBYlnnaOpNC492nnniIqm1ztVygKprMu3
7YCl3ybB5Q==
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIICdgIBADANBgkqhkiG9w0BAQEFAASCAmAwggJcAgEAAoGBAONJg43XuvXd6xsI
WhL2XjFMGOoKrDs7hBa5OLuMEkeFTvAnZPEIe4w7/cuzw8TvTSfPmvJ+ci7/k//z
AZVg6PbuLEBZM0WADIdbrarmLi4IPZWU5CsJmgQ6zDprSGIOydqbrfJbWdz9Qr3p
jdi+ZvsHq3fi0BDYyKChR5jCgqw/AgMBAAECgYEA1qHVWV0fcI7gNebtKHr++A6k
eF8bxdOuKMdAi9r6aA+7O434BKW+Be+g+3wGozJX6gBikhxWN4uid1FDbYzWcJFB
i6RHGnHkxm7DifKIXF+cHUAiQhE1W5nwy5aays8B5Kc9eC+a/m9bpxWGRY00tq6x
+WhWEUF3fPbGOqnktgECQQD3RmqraDbhvMo3CcghB63TQncafTIiNPmJPXK1uZcy
CtGRdb1cF2TJXPO+ukUYQEltG2MP+m7Ds0XL1SsPtGd1AkEA606M/BPdaAs0MZIt
u0eH+9Q3Pxp0UqX7Ro2Q4NDWmj6wcqY1E0zeWR4V8XSwbLoiw7GJdqRrL0GSgHQT
wPjCYwJAbtCV2T8Y6U0r6kJt969zTOvKaIqWvxGyiriJAbuscHa8uE1lkTHCryMC
8QSVFmso/MZ7PJvkq7tZmiFr7NvSSQJATEwCBtJiHhRT7ibZ0TnWa99ZsopfYVUU
bsIEUgElNIpTKDmgSAvKpNbOgqY1dmu8TfvI+MFDR+VZHXGF3jJKxQJAOoMB6VH/
SDNYVyHKU57OA5F8qgnIr+4OWPLtK3khbplpc4kkdBE5OJTDRKXJr+oSZDDe2elI
wsDf21paAlthnA==
-----END RSA PRIVATE KEY-----`)
