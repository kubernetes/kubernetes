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
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util/httpstream"
)

func TestRoundTripAndNewConnection(t *testing.T) {
	testCases := []struct {
		serverConnectionHeader string
		serverUpgradeHeader    string
		useTLS                 bool
		shouldError            bool
	}{
		{
			serverConnectionHeader: "",
			serverUpgradeHeader:    "",
			shouldError:            true,
		},
		{
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "",
			shouldError:            true,
		},
		{
			serverConnectionHeader: "",
			serverUpgradeHeader:    "SPDY/3.1",
			shouldError:            true,
		},
		{
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			shouldError:            false,
		},
		{
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			useTLS:                 true,
			shouldError:            false,
		},
	}

	for i, testCase := range testCases {
		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if testCase.shouldError {
				if e, a := httpstream.HeaderUpgrade, req.Header.Get(httpstream.HeaderConnection); e != a {
					t.Fatalf("%d: Expected connection=upgrade header, got '%s", i, a)
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
				t.Fatalf("%d: unexpected nil spdyConn", i)
			}
			defer spdyConn.Close()

			stream := <-streamCh
			io.Copy(stream, stream)
		}))

		clientTLS := &tls.Config{}

		if testCase.useTLS {
			privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
			if err != nil {
				t.Fatalf("%d: error generating keypair: %s", i, err)
			}

			notBefore := time.Now()
			notAfter := notBefore.Add(1 * time.Hour)

			template := x509.Certificate{
				SerialNumber: big.NewInt(1),
				Subject: pkix.Name{
					Organization: []string{"Localhost Co"},
				},
				NotBefore:             notBefore,
				NotAfter:              notAfter,
				KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
				ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
				BasicConstraintsValid: true,
				IsCA: true,
			}

			host := "127.0.0.1"
			if ip := net.ParseIP(host); ip != nil {
				template.IPAddresses = append(template.IPAddresses, ip)
			}
			template.DNSNames = append(template.DNSNames, host)

			derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
			if err != nil {
				t.Fatalf("%d: error creating cert: %s", i, err)
			}

			cert, err := x509.ParseCertificate(derBytes)
			if err != nil {
				t.Fatalf("%d: error parsing cert: %s", i, err)
			}

			roots := x509.NewCertPool()
			roots.AddCert(cert)
			server.TLS = &tls.Config{
				RootCAs: roots,
			}
			clientTLS.RootCAs = roots

			certBuf := bytes.Buffer{}
			err = pem.Encode(&certBuf, &pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw})
			if err != nil {
				t.Fatalf("%d: error encoding cert: %s", i, err)
			}

			keyBuf := bytes.Buffer{}
			err = pem.Encode(&keyBuf, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)})
			if err != nil {
				t.Fatalf("%d: error encoding key: %s", i, err)
			}

			tlsCert, err := tls.X509KeyPair(certBuf.Bytes(), keyBuf.Bytes())
			if err != nil {
				t.Fatalf("%d: error calling tls.X509KeyPair: %s", i, err)
			}
			server.TLS.Certificates = []tls.Certificate{tlsCert}
			clientTLS.Certificates = []tls.Certificate{tlsCert}
			server.StartTLS()
		} else {
			server.Start()
		}
		defer server.Close()

		req, err := http.NewRequest("GET", server.URL, nil)
		if err != nil {
			t.Fatalf("%d: Error creating request: %s", i, err)
		}

		spdyTransport := NewRoundTripper(clientTLS)
		client := &http.Client{Transport: spdyTransport}

		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("%d: unexpected error from client.Do: %s", i, err)
		}

		conn, err := spdyTransport.NewConnection(resp)
		haveErr := err != nil
		if e, a := testCase.shouldError, haveErr; e != a {
			t.Fatalf("%d: shouldError=%t, got %t: %v", i, e, a, err)
		}
		if testCase.shouldError {
			continue
		}
		defer conn.Close()

		if resp.StatusCode != http.StatusSwitchingProtocols {
			t.Fatalf("%d: expected http 101 switching protocols, got %d", i, resp.StatusCode)
		}

		stream, err := conn.CreateStream(http.Header{})
		if err != nil {
			t.Fatalf("%d: error creating client stream: %s", i, err)
		}

		n, err := stream.Write([]byte("hello"))
		if err != nil {
			t.Fatalf("%d: error writing to stream: %s", i, err)
		}
		if n != 5 {
			t.Fatalf("%d: Expected to write 5 bytes, but actually wrote %d", i, n)
		}

		b := make([]byte, 5)
		n, err = stream.Read(b)
		if err != nil {
			t.Fatalf("%d: error reading from stream: %s", i, err)
		}
		if n != 5 {
			t.Fatalf("%d: Expected to read 5 bytes, but actually read %d", i, n)
		}
		if e, a := "hello", string(b[0:n]); e != a {
			t.Fatalf("%d: expected '%s', got '%s'", i, e, a)
		}
	}
}
