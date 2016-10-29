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

package init

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

const (
	testNamespace    = "test-ns"
	testSvcName      = "test-service"
	testCertValidity = 1 * time.Hour

	helloMsg = "Hello, certificate test!"
)

type clientServerTLSConfigs struct {
	server *tls.Config
	client *tls.Config
}

type certParams struct {
	cAddr     string
	ips       []string
	hostnames []string
}

// TestCertsTLS tests TLS handshake with client authentication for any server
// name. There is a separate test below to test the certificate generation
// end-to-end over HTTPS.
// TODO(madhusudancs): Consider using a deterministic random number generator
// for generating certificates in tests.
func TestCertsTLS(t *testing.T) {
	params := []certParams{
		{
			cAddr:     "10.1.2.3",
			ips:       []string{"10.1.2.3", "10.2.3.4"},
			hostnames: []string{"federation.test", "federation2.test"},
		},
		{
			cAddr:     "10.10.20.30",
			ips:       []string{"10.20.30.40", "10.64.128.4"},
			hostnames: []string{"tls.federation.test"},
		},
	}

	tlsCfgs, err := tlsConfigs(params)
	if err != nil {
		t.Errorf("failed to generate tls configs: %v", err)
		// No point in proceeding further
		return
	}

	testCases := []struct {
		serverName string
		sCfg       *tls.Config
		cCfg       *tls.Config
		failType   string
	}{
		{
			serverName: "10.1.2.3",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
		},
		{
			serverName: "10.2.3.4",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
		},
		{
			serverName: "federation.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
		},
		{
			serverName: "federation2.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
		},
		{
			serverName: "10.20.30.40",
			sCfg:       tlsCfgs[1].server,
			cCfg:       tlsCfgs[1].client,
		},
		{
			serverName: "tls.federation.test",
			sCfg:       tlsCfgs[1].server,
			cCfg:       tlsCfgs[1].client,
		},
		{
			serverName: "10.100.200.50",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "noexist.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "10.64.128.4",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "tls.federation.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "10.1.2.3",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[1].client,
			failType:   "UnknownAuthorityError",
		},
		{
			serverName: "federation2.test",
			sCfg:       tlsCfgs[0].server,
			cCfg:       tlsCfgs[1].client,
			failType:   "UnknownAuthorityError",
		},
		{
			serverName: "10.1.2.3",
			sCfg:       tlsCfgs[1].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
		{
			serverName: "federation2.test",
			sCfg:       tlsCfgs[1].server,
			cCfg:       tlsCfgs[0].client,
			failType:   "HostnameError",
		},
	}

	for i, tc := range testCases {
		// Make a copy of the client config before modifying it.
		// We can't do a regular pointer deref shallow copy because
		// tls.Config contains an unexported sync.Once field which
		// must not be copied. This was pointed out by go vet.
		cCfg := copyTLSConfig(tc.cCfg)
		cCfg.ServerName = tc.serverName
		cCfg.BuildNameToCertificate()

		err := tlsHandshake(t, tc.sCfg, cCfg)
		if len(tc.failType) > 0 {
			switch tc.failType {
			case "HostnameError":
				if _, ok := err.(x509.HostnameError); !ok {
					t.Errorf("[%d] unexpected error: want x509.HostnameError, got: %T", i, err)
				}
			case "UnknownAuthorityError":
				if _, ok := err.(x509.UnknownAuthorityError); !ok {
					t.Errorf("[%d] unexpected error: want x509.UnknownAuthorityError, got: %T", i, err)
				}
			default:
				t.Errorf("cannot handle error type: %s", tc.failType)

			}
		} else if err != nil {
			t.Errorf("[%d] unexpected error: %v", i, err)
		}
	}
}

// TestCertsHTTPS cannot test client authentication for non-localhost server
// names, but it tests TLS handshake end-to-end over HTTPS.
func TestCertsHTTPS(t *testing.T) {
	params := []certParams{
		{
			// Unfortunately, due to the limitation in the way Go
			// net/http/httptest package sets up the test HTTPS/TLS server,
			// 127.0.0.1 is the only accepted server address. So, we need to
			// generate certificates for this address.
			cAddr:     "127.0.0.1",
			ips:       []string{"127.0.0.1"},
			hostnames: []string{},
		},
		{
			// Unfortunately, due to the limitation in the way Go
			// net/http/httptest package sets up the test HTTPS/TLS server,
			// 127.0.0.1 is the only accepted server address. So, we need to
			// generate certificates for this address.
			cAddr:     "localhost",
			ips:       []string{"127.0.0.1"},
			hostnames: []string{"localhost"},
		},
	}

	tlsCfgs, err := tlsConfigs(params)
	if err != nil {
		t.Errorf("failed to generate tls configs: %v", err)
		// No point in proceeding further
		return
	}

	testCases := []struct {
		sCfg *tls.Config
		cCfg *tls.Config
		fail bool
	}{
		{
			sCfg: tlsCfgs[0].server,
			cCfg: tlsCfgs[0].client,
			fail: false,
		},
		{
			sCfg: tlsCfgs[0].server,
			cCfg: tlsCfgs[1].client,
			fail: true,
		},
		{
			sCfg: tlsCfgs[1].server,
			cCfg: tlsCfgs[0].client,
			fail: true,
		},
	}

	for i, tc := range testCases {
		// Make a copy of the client config before modifying it.
		// We can't do a regular pointer deref shallow copy because
		// tls.Config contains an unexported sync.Once field which
		// must not be copied. This was pointed out by go vet.
		cCfg := copyTLSConfig(tc.cCfg)
		cCfg.BuildNameToCertificate()

		s, err := fakeHTTPSServer(tc.sCfg)
		if err != nil {
			t.Errorf("[%d] unexpected error starting TLS server: %v", i, err)
			// No point in proceeding
			continue
		}
		defer s.Close()

		tr := &http.Transport{
			TLSClientConfig: cCfg,
		}
		client := &http.Client{Transport: tr}
		resp, err := client.Get(s.URL)
		if tc.fail {
			_, ok := err.(*url.Error)
			if !ok || !strings.HasSuffix(err.Error(), "x509: certificate signed by unknown authority") {
				t.Errorf("[%d] unexpected error: want x509.HostnameError, got: %T", i, err)
			}
			// We are done for this test.
			continue
		} else if err != nil {
			t.Errorf("[%d] unexpected error while sending GET request to the server: %T", i, err)
			// No point in proceeding
			continue
		}
		defer resp.Body.Close()

		got, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("[%d] unexpected error reading server response: %v", i, err)
		} else if string(got) != helloMsg {
			t.Errorf("[%d] want %q, got %q", i, helloMsg, got)
		}
	}
}

func tlsHandshake(t *testing.T, sCfg, cCfg *tls.Config) error {
	// Tried to use net.Pipe() instead of TCP. But the connections returned by
	// net.Pipe() do a fully-synchronous reads and writes on both the ends.
	// So if a TLS handshake fails, they can't return the error until the
	// other side reads the message which it did not expect. Since the other
	// side does not read the message it did not expect, the server and
	// clients hang. Since TCP is non-blocking we use that as transport
	// instead. One could have as well used a Unix Domain Socket, but TCP is
	// more portable.
	s, err := tls.Listen("tcp", "", sCfg)
	if err != nil {
		return fmt.Errorf("failed to create a test TLS server: %v", err)
	}
	defer s.Close()

	errCh := make(chan error)
	go func() {
		for {
			conn, err := s.Accept()
			if err != nil {
				errCh <- fmt.Errorf("failed to accept a TLS connection: %v", err)
				return
			}
			gotByte := make([]byte, len(helloMsg))
			_, err = conn.Read(gotByte)
			if err != nil && err != io.EOF {
				errCh <- fmt.Errorf("failed to read input: %v", err)
			} else if got := string(gotByte); got != helloMsg {
				errCh <- fmt.Errorf("got %q, want %q", got, helloMsg)
			}
			errCh <- nil
			return
		}
	}()

	c, err := tls.Dial("tcp", s.Addr().String(), cCfg)
	if err != nil {
		// Intentionally not serializing the error received because we want to
		// test for the failure case in the caller test function.
		return err
	}
	defer c.Close()
	if _, err := c.Write([]byte(helloMsg)); err != nil {
		return fmt.Errorf("failed to write to server: %v", err)
	}

	return <-errCh
}

func fakeHTTPSServer(sCfg *tls.Config) (*httptest.Server, error) {
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, helloMsg)
	}))

	s.TLS.Certificates = sCfg.Certificates
	s.TLS.RootCAs = sCfg.RootCAs
	s.TLS.ClientAuth = sCfg.ClientAuth
	s.TLS.ClientCAs = sCfg.ClientCAs
	s.TLS.InsecureSkipVerify = sCfg.InsecureSkipVerify
	return s, nil
}

func tlsConfigs(params []certParams) ([]clientServerTLSConfigs, error) {
	tlsCfgs := []clientServerTLSConfigs{}
	for i, p := range params {
		sCfg, cCfg, err := genServerClientTLSConfigs(testNamespace, p.cAddr, testSvcName, HostClusterLocalDNSZoneName, p.ips, p.hostnames)
		if err != nil {
			return nil, fmt.Errorf("[%d] failed to generate tls configs: %v", i, err)
		}
		tlsCfgs = append(tlsCfgs, clientServerTLSConfigs{sCfg, cCfg})
	}
	return tlsCfgs, nil
}

func genServerClientTLSConfigs(namespace, name, svcName, localDNSZoneName string, ips, hostnames []string) (*tls.Config, *tls.Config, error) {
	entKeyPairs, err := genCerts(namespace, name, svcName, localDNSZoneName, ips, hostnames)
	if err != nil {
		return nil, nil, fmt.Errorf("unexpected error generating certs: %v", err)
	}

	roots := x509.NewCertPool()
	roots.AddCert(entKeyPairs.ca.Cert)

	serverCert := tls.Certificate{
		Certificate: [][]byte{
			entKeyPairs.server.Cert.Raw,
		},
		PrivateKey: entKeyPairs.server.Key,
	}

	cmCert := tls.Certificate{
		Certificate: [][]byte{
			entKeyPairs.controllerManager.Cert.Raw,
		},
		PrivateKey: entKeyPairs.controllerManager.Key,
	}

	sCfg := &tls.Config{
		Certificates:       []tls.Certificate{serverCert},
		RootCAs:            roots,
		ClientAuth:         tls.RequireAndVerifyClientCert,
		ClientCAs:          roots,
		InsecureSkipVerify: false,
	}

	cCfg := &tls.Config{
		Certificates: []tls.Certificate{cmCert},
		RootCAs:      roots,
	}

	return sCfg, cCfg, nil
}

func copyTLSConfig(cfg *tls.Config) *tls.Config {
	// We are copying only the required fields.
	return &tls.Config{
		Certificates:       cfg.Certificates,
		RootCAs:            cfg.RootCAs,
		ClientAuth:         cfg.ClientAuth,
		ClientCAs:          cfg.ClientCAs,
		InsecureSkipVerify: cfg.InsecureSkipVerify,
	}
}
