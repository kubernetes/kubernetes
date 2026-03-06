/*
Copyright The Kubernetes Authors.

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

package oidc

import (
	"crypto/rand"
	"crypto/rsa"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	certutil "k8s.io/client-go/util/cert"
	utilsnet "k8s.io/utils/net"
)

const (
	rsaKeyBitSize = 2048
)

// RSAGenerateKey generates an RSA key pair for testing.
func RSAGenerateKey(t *testing.T) (*rsa.PrivateKey, *rsa.PublicKey) {
	t.Helper()

	privateKey, err := rsa.GenerateKey(rand.Reader, rsaKeyBitSize)
	require.NoError(t, err)

	return privateKey, &privateKey.PublicKey
}

// GenerateCert generates a self-signed certificate for localhost/127.0.0.1
// and returns the cert bytes, key bytes, and file paths where they are stored.
func GenerateCert(t *testing.T) (cert, key []byte, certFilePath, keyFilePath string) {
	t.Helper()

	tempDir := t.TempDir()
	certFilePath = filepath.Join(tempDir, "localhost_127.0.0.1_.crt")
	keyFilePath = filepath.Join(tempDir, "localhost_127.0.0.1_.key")

	cert, key, err := certutil.GenerateSelfSignedCertKeyWithFixtures("localhost", []net.IP{utilsnet.ParseIPSloppy("127.0.0.1")}, nil, tempDir)
	require.NoError(t, err)

	return cert, key, certFilePath, keyFilePath
}

// WriteTempFile writes content to a temporary file and returns its path.
// The file is automatically cleaned up when the test completes.
func WriteTempFile(t *testing.T, content string) string {
	t.Helper()
	file, err := os.CreateTemp("", "oidc-test")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Remove(file.Name()); err != nil {
			t.Fatal(err)
		}
	})
	if err := os.WriteFile(file.Name(), []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	return file.Name()
}

// IndentCertificateAuthority indents the certificate authority to match
// the format of the generated authentication config.
func IndentCertificateAuthority(caCert string) string {
	return strings.ReplaceAll(caCert, "\n", "\n        ")
}

// NewHTTPConnectProxyHandler returns an http.Handler that implements an HTTP CONNECT proxy.
// When a CONNECT request is received, it dials the target, hijacks the client connection,
// and bidirectionally copies data between them. The called flag is set to true when a
// non-ready request is received.
func NewHTTPConnectProxyHandler(t testing.TB, called *atomic.Bool) http.Handler {
	t.Helper()

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/ready" {
			t.Log("egress proxy ready")
			w.WriteHeader(http.StatusOK)
			return
		}

		called.Store(true)

		if r.Method != http.MethodConnect {
			http.Error(w, "this proxy only supports CONNECT passthrough", http.StatusMethodNotAllowed)
			return
		}

		backendConn, err := (&net.Dialer{}).DialContext(r.Context(), "tcp", r.Host)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer func() { _ = backendConn.Close() }()

		hijacker, ok := w.(http.Hijacker)
		if !ok {
			http.Error(w, "hijacking not supported", http.StatusInternalServerError)
			return
		}

		clientConn, _, err := hijacker.Hijack()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer func() { _ = clientConn.Close() }()

		// use t.Errorf for all errors after this Write since the client may think the connection is good
		_, err = clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
		if err != nil {
			t.Errorf("unexpected established error: %v", err)
			return
		}

		writerComplete := make(chan struct{})
		readerComplete := make(chan struct{})

		go func() {
			_, err := io.Copy(backendConn, clientConn)
			if err != nil && !utilnet.IsProbableEOF(err) {
				t.Logf("writer error: %v", err)
			}
			close(writerComplete)
		}()

		go func() {
			_, err := io.Copy(clientConn, backendConn)
			if err != nil && !utilnet.IsProbableEOF(err) {
				t.Logf("reader error: %v", err)
			}
			close(readerComplete)
		}()

		// Wait for one half the connection to exit. Once it does,
		// the defer will clean up the other half of the connection.
		select {
		case <-writerComplete:
		case <-readerComplete:
		}
	})
}
