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

package oidc

import (
	"crypto/rand"
	"crypto/rsa"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
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
