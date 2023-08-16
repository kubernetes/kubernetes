/*
Copyright 2021 The Kubernetes Authors.

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

package server

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/http2"
)

var (
	// startServerShutdown is a signal from the backend after receiving all (25) requests
	// after which the test shuts down the HTTP server
	startServerShutdown = make(chan struct{})

	// handlerLock used in the backendHTTPHandler to count the number of requests and signal the test that the termination can start
	handlerLock = sync.Mutex{}
)

var backendCrt = []byte(`-----BEGIN CERTIFICATE-----
MIIDTjCCAjagAwIBAgIJANYWBFaLyBC/MA0GCSqGSIb3DQEBCwUAMFAxCzAJBgNV
BAYTAlBMMQ8wDQYDVQQIDAZQb2xhbmQxDzANBgNVBAcMBkdkYW5zazELMAkGA1UE
CgwCU0sxEjAQBgNVBAMMCTEyNy4wLjAuMTAeFw0yMDEyMTExMDI0MzBaFw0zMDEy
MDkxMDI0MzBaMFAxCzAJBgNVBAYTAlBMMQ8wDQYDVQQIDAZQb2xhbmQxDzANBgNV
BAcMBkdkYW5zazELMAkGA1UECgwCU0sxEjAQBgNVBAMMCTEyNy4wLjAuMTCCASIw
DQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMYax2q/m/N237UFMFKZsox4EyKq
De+mbaRGeKqnI7Gi9Ai3b7BPCIa7RFJ2ntpGUd5GyL+HCQHG8/f6DjsbUuhZnmn7
F7ZJeih2DP2acKkODdGbXA52kABCMdDs2DMYhR2UwECY2t+DLpxqJqE2ab8pI9Xd
BZ3pCNodS03yHXzfeJV44lCjxoDOi9ynXLjd3w3+FowomHMEBunTepiqnbgoYtnn
RW9tQyQQK5g6+/j/O1M8o71s/0loBT3vKSqNSrdlMOEGrj4yyL/Cw1NmQf1V1sGf
w1QAW5xk7Br5oh8h1D+oflGWV3Y3zluuZQnA9D+vFpjL0969oFedsgr4UU8CAwEA
AaMrMCkwCQYDVR0TBAIwADALBgNVHQ8EBAMCBaAwDwYDVR0RBAgwBocEfwAAATAN
BgkqhkiG9w0BAQsFAAOCAQEAWbOF7TOfGiC59S50okfcS7M4gwz2kcbqOftWzcA1
lT1qX6TWj7A4bVIOMAFK2tWNd4Omk6bnIAxTJdHB7b1hrBjkpt2krEGH1S8xeRRz
Gs62KQwehM3fMhLvYSEqOQMETZn9AjEigYm6ohCO5obG9Gkfz7uvuv9rbIetbAmm
YE9HdDv6qhCqtynpP2yad3v53idlrDnCIe9e4eKUD5uR/MIp9mEFgnMXR1m43/ya
DnmddSsjtzamVvI/+2Cqjb8qT8dMHZrCBK64UwSaJsUKzSeF6yNvZKQ1yfA/NrfV
P6gNULDOqtPgXFP4j+Z402gjYox1bGHjeDHh1OVSnr9jVw==
-----END CERTIFICATE-----`)

var backendKey = []byte(`-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDGGsdqv5vzdt+1
BTBSmbKMeBMiqg3vpm2kRniqpyOxovQIt2+wTwiGu0RSdp7aRlHeRsi/hwkBxvP3
+g47G1LoWZ5p+xe2SXoodgz9mnCpDg3Rm1wOdpAAQjHQ7NgzGIUdlMBAmNrfgy6c
aiahNmm/KSPV3QWd6QjaHUtN8h1833iVeOJQo8aAzovcp1y43d8N/haMKJhzBAbp
03qYqp24KGLZ50VvbUMkECuYOvv4/ztTPKO9bP9JaAU97ykqjUq3ZTDhBq4+Msi/
wsNTZkH9VdbBn8NUAFucZOwa+aIfIdQ/qH5Rlld2N85brmUJwPQ/rxaYy9PevaBX
nbIK+FFPAgMBAAECggEAKmdZABR7gSWUxN6TdVrIySB6mBTmXsG0/lDHS1/zV/aV
XbhGA+sm3BABk9UoM3iR1Y45MiXpW6QGXLH9kdFLccidC/pfHPmlWDvMlAwWyVjk
xFUI41+leyiwGRRZQrag57ALZshRMT6XH4vpMODAydY4gXKJ3T8gUe+rSsfkX/Hl
Ce59c8pDsV3NDy4WKy00lYZfTqBqHu10qy9W8/eVYf+RUt53nrygCesnFfmJx/P8
GnHnN06QbZdpgVgbU49u+BujkjFgKH/60Ct9A19o34upXvkPOaKbABZ4dL1lUrbo
e3L3vnSdgXh1oOsy/JyICmDG5M2b68h33YNa+qUEgQKBgQDs1rf1+hw75o7iDlnx
E46CPC+9DkDuisWLgbUyW5KHPgropPl80uqnRxmaWpYGU/Fgyml08orpduHIWxtU
0tMRKm2HoFRM010fAp3xWc/B4pt2pdRMMSjMle//4FmoNlcJ8+owmD+2eook9Qjm
qN1UsQllkSoH4zx4iI+HhDJnHwKBgQDWIdGmlZqaYGhsndkco9yK+gve6W80ik4J
qnjnv9ux28SBrlORn2zzfGcu5LkJw8Dp9yjZzVUiFT8VFsWVNNuJyFba227Qxrwz
Hb/qvd5l2DfXHk4poyMZThzg7cxkxlVaWUIBMoGynDxQZIOypc6WmTeEG5+9W4+w
NCuTKt6/0QKBgQCOgALftUUXpXmC+i+TpbixE5WFovXekRCbB8gGLKLVTLczk0+p
kx4s19LH1Ik/9XHeUutwuh5qqmTfMDIZr1/fjC+q0wTl1KbK6cAuX2NpvPbdRJmf
3lQ2BGELC+nmFAv6qQ/XfUOYf9JuuiBI6IGDW6HTwqwPYuIXg9MYLqpE8QKBgA/2
2YCH6szTnzVp10PxW4Ho/nWSBb5vCT5jPTxZ63EpJ09bxdM3hZHplm/CkaEOvRU0
XhFO46f02Y0i83waQrvU+dS7Q1nBV0qgTyybFzeUlSUulzk3dmhukGycjf59YuOn
f+pC77R3PW/o7oClJ+/GYIMy5AfkCaRjX1RLf+vhAoGBANJBi0ARkhwOWbnD2urA
0tPMURSYIZ+JW7ghMspbm1XV1NTreCB/llLNqUGQ7zLAmH+KyqJK8O37/oh3VHrV
6jp9pqrqmibtGEIpQi4D9IM8Zo9mc8GexCf0x+11mamC+ZXjT+bvLQzbcJGnG5CL
W+S7SneWTL09leh5ATNhog6s
-----END PRIVATE KEY-----`)

// TestGracefulShutdownForActiveHTTP2Streams checks if graceful shut down of HTTP2 server works.
// It expects that all active connections will be finished (without any errors) before the server exits.
//
// The test sends 25 requests to the target server in parallel. Each request is held by the target server for 60s.
// As soon as the target server receives the last request the test calls backendServer.Config.Shutdown which gracefully shuts down the server without interrupting any active connections.
//
// See more at: https://github.com/golang/go/issues/39776
//
// Note this test will fail on upstream golang 1.15
func TestGracefulShutdownForActiveHTTP2Streams(t *testing.T) {
	// set up the backend server
	backendHandler := &backendHTTPHandler{}
	backendServer := httptest.NewUnstartedServer(backendHandler)
	backendCert, err := tls.X509KeyPair(backendCrt, backendKey)
	if err != nil {
		t.Fatalf("backend: invalid x509/key pair: %v", err)
	}
	backendServer.TLS = &tls.Config{
		Certificates: []tls.Certificate{backendCert},
		NextProtos:   []string{http2.NextProtoTLS},
	}
	backendServer.StartTLS()
	defer backendServer.Close()

	// set up the client
	clientCACertPool := x509.NewCertPool()
	clientCACertPool.AppendCertsFromPEM(backendCrt)
	clientTLSConfig := &tls.Config{
		RootCAs:    clientCACertPool,
		NextProtos: []string{http2.NextProtoTLS},
	}
	client := &http.Client{}
	client.Transport = &http2.Transport{
		TLSClientConfig: clientTLSConfig,
	}

	// client request
	sendRequest := func(wg *sync.WaitGroup) {
		defer func() {
			wg.Done()
		}()

		// act
		resp, err := client.Get(fmt.Sprintf("https://127.0.0.1:%d", backendServer.Listener.Addr().(*net.TCPAddr).Port))
		if err != nil {
			t.Errorf("%v", err)
			return
		}

		// validate
		defer resp.Body.Close()
		_, err = io.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("%v", err)
		}
		if resp.StatusCode != 200 {
			t.Errorf("unexpected HTTP staus: %v, expected: 200", resp.StatusCode)
		}
		expectedProto := "HTTP/2.0"
		if resp.Proto != expectedProto {
			t.Errorf("unexpected response proto: %v, expected: %v", resp.Proto, expectedProto)
		}
	}

	// this function starts the graceful shutdown
	go func() {
		<-startServerShutdown // signal from the backend after receiving all (25) requests

		backendServer.Config.Shutdown(context.Background())
	}()

	wg := sync.WaitGroup{}
	wg.Add(25)
	for i := 0; i < 25; i++ {
		go sendRequest(&wg)
	}
	wg.Wait()

	// validate backendHandler
	if backendHandler.counter != 25 {
		t.Errorf("the target server haven't received all expected requests, expected 25, it got %d", backendHandler.counter)
	}
}

type backendHTTPHandler struct {
	counter int
}

func (b *backendHTTPHandler) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	handlerLock.Lock()
	b.counter++
	if b.counter == 25 {
		startServerShutdown <- struct{}{}
	}
	handlerLock.Unlock()

	time.Sleep(60 * time.Second)

	w.Write([]byte("hello from the backend"))
	w.WriteHeader(http.StatusOK)
}
