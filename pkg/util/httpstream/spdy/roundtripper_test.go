/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package spdy

import (
	"crypto/tls"
	"crypto/x509"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/util/httpstream"
)

func TestRoundTripAndNewConnection(t *testing.T) {

	localhostPool := x509.NewCertPool()
	if !localhostPool.AppendCertsFromPEM(localhostCert) {
		t.Errorf("error setting up localhostCert pool")
	}

	testCases := map[string]struct {
		serverFunc             func(http.Handler) *httptest.Server
		clientTLS              *tls.Config
		serverConnectionHeader string
		serverUpgradeHeader    string
		shouldError            bool
	}{
		"no headers": {
			serverFunc:             httptest.NewServer,
			serverConnectionHeader: "",
			serverUpgradeHeader:    "",
			shouldError:            true,
		},
		"no upgrade header": {
			serverFunc:             httptest.NewServer,
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "",
			shouldError:            true,
		},
		"no connection header": {
			serverFunc:             httptest.NewServer,
			serverConnectionHeader: "",
			serverUpgradeHeader:    "SPDY/3.1",
			shouldError:            true,
		},
		"http": {
			serverFunc:             httptest.NewServer,
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			shouldError:            false,
		},
		"https (invalid hostname + InsecureSkipVerify)": {
			serverFunc: func(h http.Handler) *httptest.Server {
				cert, err := tls.X509KeyPair(exampleCert, exampleKey)
				if err != nil {
					t.Errorf("https (invalid hostname): proxy_test: %v", err)
				}
				ts := httptest.NewUnstartedServer(h)
				ts.TLS = &tls.Config{
					Certificates: []tls.Certificate{cert},
				}
				ts.StartTLS()
				return ts
			},
			clientTLS:              &tls.Config{InsecureSkipVerify: true},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			shouldError:            false,
		},
		"https (valid hostname + RootCAs)": {
			serverFunc: func(h http.Handler) *httptest.Server {
				cert, err := tls.X509KeyPair(localhostCert, localhostKey)
				if err != nil {
					t.Errorf("https (valid hostname): proxy_test: %v", err)
				}
				ts := httptest.NewUnstartedServer(h)
				ts.TLS = &tls.Config{
					Certificates: []tls.Certificate{cert},
				}
				ts.StartTLS()
				return ts
			},
			clientTLS:              &tls.Config{RootCAs: localhostPool},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			shouldError:            false,
		},
	}

	for k, testCase := range testCases {
		server := testCase.serverFunc(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if testCase.shouldError {
				if e, a := httpstream.HeaderUpgrade, req.Header.Get(httpstream.HeaderConnection); e != a {
					t.Fatalf("%s: Expected connection=upgrade header, got '%s", k, a)
				}

				w.Header().Set(httpstream.HeaderConnection, testCase.serverConnectionHeader)
				w.Header().Set(httpstream.HeaderUpgrade, testCase.serverUpgradeHeader)
				w.WriteHeader(http.StatusSwitchingProtocols)

				return
			}

			streamCh := make(chan httpstream.Stream)

			responseUpgrader := NewResponseUpgrader()
			spdyConn := responseUpgrader.UpgradeResponse(w, req, func(s httpstream.Stream) error {
				streamCh <- s
				return nil
			})
			if spdyConn == nil {
				t.Fatalf("%s: unexpected nil spdyConn", k)
			}
			defer spdyConn.Close()

			stream := <-streamCh
			io.Copy(stream, stream)
		}))
		defer server.Close()

		req, err := http.NewRequest("GET", server.URL, nil)
		if err != nil {
			t.Fatalf("%s: Error creating request: %s", k, err)
		}

		spdyTransport := NewRoundTripper(testCase.clientTLS)
		client := &http.Client{Transport: spdyTransport}

		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("%s: unexpected error from client.Do: %s", k, err)
		}

		conn, err := spdyTransport.NewConnection(resp)
		haveErr := err != nil
		if e, a := testCase.shouldError, haveErr; e != a {
			t.Fatalf("%s: shouldError=%t, got %t: %v", k, e, a, err)
		}
		if testCase.shouldError {
			continue
		}
		defer conn.Close()

		if resp.StatusCode != http.StatusSwitchingProtocols {
			t.Fatalf("%s: expected http 101 switching protocols, got %d", k, resp.StatusCode)
		}

		stream, err := conn.CreateStream(http.Header{})
		if err != nil {
			t.Fatalf("%s: error creating client stream: %s", k, err)
		}

		n, err := stream.Write([]byte("hello"))
		if err != nil {
			t.Fatalf("%s: error writing to stream: %s", k, err)
		}
		if n != 5 {
			t.Fatalf("%s: Expected to write 5 bytes, but actually wrote %d", k, n)
		}

		b := make([]byte, 5)
		n, err = stream.Read(b)
		if err != nil {
			t.Fatalf("%s: error reading from stream: %s", k, err)
		}
		if n != 5 {
			t.Fatalf("%s: Expected to read 5 bytes, but actually read %d", k, n)
		}
		if e, a := "hello", string(b[0:n]); e != a {
			t.Fatalf("%s: expected '%s', got '%s'", k, e, a)
		}
	}
}

// exampleCert was generated from crypto/tls/generate_cert.go with the following command:
//    go run generate_cert.go  --rsa-bits 512 --host example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var exampleCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBcjCCAR6gAwIBAgIQBOUTYowZaENkZi0faI9DgTALBgkqhkiG9w0BAQswEjEQ
MA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2MDAw
MFowEjEQMA4GA1UEChMHQWNtZSBDbzBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQCZ
xfR3sgeHBraGFfF/24tTn4PRVAHOf2UOOxSQRs+aYjNqimFqf/SRIblQgeXdBJDR
gVK5F1Js2zwlehw0bHxRAgMBAAGjUDBOMA4GA1UdDwEB/wQEAwIApDATBgNVHSUE
DDAKBggrBgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MBYGA1UdEQQPMA2CC2V4YW1w
bGUuY29tMAsGCSqGSIb3DQEBCwNBAI/mfBB8dm33IpUl+acSyWfL6gX5Wc0FFyVj
dKeesE1XBuPX1My/rzU6Oy/YwX7LOL4FaeNUS6bbL4axSLPKYSs=
-----END CERTIFICATE-----`)

var exampleKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBOgIBAAJBAJnF9HeyB4cGtoYV8X/bi1Ofg9FUAc5/ZQ47FJBGz5piM2qKYWp/
9JEhuVCB5d0EkNGBUrkXUmzbPCV6HDRsfFECAwEAAQJBAJLH9yPuButniACTn5L5
IJQw1mWQt6zBw9eCo41YWkA0866EgjC53aPZaRjXMp0uNJGdIsys2V5rCOOLWN2C
ODECIQDICHsi8QQQ9wpuJy8X5l8MAfxHL+DIqI84wQTeVM91FQIhAMTME8A18/7h
1Ad6drdnxAkuC0tX6Sx0LDozrmen+HFNAiAlcEDrt0RVkIcpOrg7tuhPLQf0oudl
Zvb3Xlj069awSQIgcT15E/43w2+RASifzVNhQ2MCTr1sSA8lL+xzK+REmnUCIBhQ
j4139pf8Re1J50zBxS/JlQfgDQi9sO9pYeiHIxNs
-----END RSA PRIVATE KEY-----`)

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
