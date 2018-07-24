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
//     go run generate_cert.go  --rsa-bits 512 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBjzCCATmgAwIBAgIRAKpi2WmTcFrVjxrl5n5YDUEwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2
MDAwMFowEjEQMA4GA1UEChMHQWNtZSBDbzBcMA0GCSqGSIb3DQEBAQUAA0sAMEgC
QQC9fEbRszP3t14Gr4oahV7zFObBI4TfA5i7YnlMXeLinb7MnvT4bkfOJzE6zktn
59zP7UiHs3l4YOuqrjiwM413AgMBAAGjaDBmMA4GA1UdDwEB/wQEAwICpDATBgNV
HSUEDDAKBggrBgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MC4GA1UdEQQnMCWCC2V4
YW1wbGUuY29thwR/AAABhxAAAAAAAAAAAAAAAAAAAAABMA0GCSqGSIb3DQEBCwUA
A0EAUsVE6KMnza/ZbodLlyeMzdo7EM/5nb5ywyOxgIOCf0OOLHsPS9ueGLQX9HEG
//yjTXuhNcUugExIjM/AIwAZPQ==
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBOwIBAAJBAL18RtGzM/e3XgavihqFXvMU5sEjhN8DmLtieUxd4uKdvsye9Phu
R84nMTrOS2fn3M/tSIezeXhg66quOLAzjXcCAwEAAQJBAKcRxH9wuglYLBdI/0OT
BLzfWPZCEw1vZmMR2FF1Fm8nkNOVDPleeVGTWoOEcYYlQbpTmkGSxJ6ya+hqRi6x
goECIQDx3+X49fwpL6B5qpJIJMyZBSCuMhH4B7JevhGGFENi3wIhAMiNJN5Q3UkL
IuSvv03kaPR5XVQ99/UeEetUgGvBcABpAiBJSBzVITIVCGkGc7d+RCf49KTCIklv
bGWObufAR8Ni4QIgWpILjW8dkGg8GOUZ0zaNA6Nvt6TIv2UWGJ4v5PoV98kCIQDx
rIiZs5QbKdycsv9gQJzwQAogC8o04X3Zz3dsoX+h4A==
-----END RSA PRIVATE KEY-----`)
