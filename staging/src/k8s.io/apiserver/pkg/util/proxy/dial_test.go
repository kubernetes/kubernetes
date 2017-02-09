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
			Dial:      net.Dial,
		},
		"secure, no roots, custom dial": {
			TLSConfig:   &tls.Config{InsecureSkipVerify: false},
			Dial:        net.Dial,
			ExpectError: "unknown authority",
		},
		"secure with roots, custom dial": {
			TLSConfig: &tls.Config{InsecureSkipVerify: false, RootCAs: roots},
			Dial:      net.Dial,
		},
		"secure with mismatched server, custom dial": {
			TLSConfig:   &tls.Config{InsecureSkipVerify: false, RootCAs: roots, ServerName: "bogus.com"},
			Dial:        net.Dial,
			ExpectError: "not bogus.com",
		},
		"secure with matched server, custom dial": {
			TLSConfig: &tls.Config{InsecureSkipVerify: false, RootCAs: roots, ServerName: "example.com"},
			Dial:      net.Dial,
		},
	}

	for k, tc := range testcases {
		func() {
			ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {}))
			defer ts.Close()
			ts.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
			ts.StartTLS()

			tlsConfigCopy := utilnet.CloneTLSConfig(tc.TLSConfig)
			transport := &http.Transport{
				Dial:            tc.Dial,
				TLSClientConfig: tlsConfigCopy,
			}

			extractedDial, err := utilnet.Dialer(transport)
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
			conn, err := DialURL(u, transport)

			// Make sure dialing doesn't mutate the transport's TLSConfig
			if !reflect.DeepEqual(tc.TLSConfig, tlsConfigCopy) {
				t.Errorf("%s: transport's copy of TLSConfig was mutated\n%#v\n\n%#v", k, tc.TLSConfig, tlsConfigCopy)
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
//     go run generate_cert.go  --rsa-bits 512 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBdzCCASOgAwIBAgIBADALBgkqhkiG9w0BAQUwEjEQMA4GA1UEChMHQWNtZSBD
bzAeFw03MDAxMDEwMDAwMDBaFw00OTEyMzEyMzU5NTlaMBIxEDAOBgNVBAoTB0Fj
bWUgQ28wWjALBgkqhkiG9w0BAQEDSwAwSAJBAN55NcYKZeInyTuhcCwFMhDHCmwa
IUSdtXdcbItRB/yfXGBhiex00IaLXQnSU+QZPRZWYqeTEbFSgihqi1PUDy8CAwEA
AaNoMGYwDgYDVR0PAQH/BAQDAgCkMBMGA1UdJQQMMAoGCCsGAQUFBwMBMA8GA1Ud
EwEB/wQFMAMBAf8wLgYDVR0RBCcwJYILZXhhbXBsZS5jb22HBH8AAAGHEAAAAAAA
AAAAAAAAAAAAAAEwCwYJKoZIhvcNAQEFA0EAAoQn/ytgqpiLcZu9XKbCJsJcvkgk
Se6AbGXgSlq+ZCEVo0qIwSgeBqmsJxUu7NCSOwVJLYNEBO2DtIxoYVk+MA==
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBPAIBAAJBAN55NcYKZeInyTuhcCwFMhDHCmwaIUSdtXdcbItRB/yfXGBhiex0
0IaLXQnSU+QZPRZWYqeTEbFSgihqi1PUDy8CAwEAAQJBAQdUx66rfh8sYsgfdcvV
NoafYpnEcB5s4m/vSVe6SU7dCK6eYec9f9wpT353ljhDUHq3EbmE4foNzJngh35d
AekCIQDhRQG5Li0Wj8TM4obOnnXUXf1jRv0UkzE9AHWLG5q3AwIhAPzSjpYUDjVW
MCUXgckTpKCuGwbJk7424Nb8bLzf3kllAiA5mUBgjfr/WtFSJdWcPQ4Zt9KTMNKD
EUO0ukpTwEIl6wIhAMbGqZK3zAAFdq8DD2jPx+UJXnh0rnOkZBzDtJ6/iN69AiEA
1Aq8MJgTaYsDQWyU/hDq5YkDJc9e9DSCvUIzqxQWMQE=
-----END RSA PRIVATE KEY-----`)
