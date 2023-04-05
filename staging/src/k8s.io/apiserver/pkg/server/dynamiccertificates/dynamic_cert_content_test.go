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
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	certutil "k8s.io/client-go/util/cert"
)

func TestDynamicCertKeyPairContent(t *testing.T) {
	tempDir := t.TempDir()

	cert, key, err := certutil.GenerateSelfSignedCertKey("test.host", parseIPList([]string{"127.0.0.1", "172.0.1.1"}), []string{"test_alias.host"})
	require.NoError(t, err)
	cert2, key2, err := certutil.GenerateSelfSignedCertKey("test.host2", parseIPList([]string{"172.0.1.2"}), []string{"test_alias.host2"})
	require.NoError(t, err)

	certPath := filepath.Join(tempDir, "server.crt")
	keyPath := filepath.Join(tempDir, "server.key")
	cert2Path := filepath.Join(tempDir, "server_two.crt")
	key2Path := filepath.Join(tempDir, "server_two.key")

	require.NoError(t, certutil.WriteCert(certPath, cert))
	require.NoError(t, certutil.WriteCert(cert2Path, cert2))
	require.NoError(t, os.WriteFile(keyPath, key, 0600))
	require.NoError(t, os.WriteFile(key2Path, key2, 0600))

	tests := []struct {
		name                  string
		certPath, keyPath     string
		updateCert, updateKey []byte
		expectUpdate          bool
		expectListenerUpdate  bool
		wantInitError         bool
	}{
		{
			name:          "missing cert path",
			keyPath:       keyPath,
			wantInitError: true,
		},
		{
			name:          "missing key path",
			certPath:      certPath,
			wantInitError: true,
		},
		{
			name:          "missing cert content",
			certPath:      "/thisisnot/the/file/yourelookingfor",
			keyPath:       keyPath,
			wantInitError: true,
		},
		{
			name:          "missing key content",
			certPath:      certPath,
			keyPath:       "/thisisnot/the/file/yourelookingfor",
			wantInitError: true,
		},
		{
			name:          "key does not match the cert",
			certPath:      certPath,
			keyPath:       key2Path,
			wantInitError: true,
		},
		{
			name:          "no cert in the file",
			certPath:      keyPath,
			keyPath:       keyPath,
			wantInitError: true,
		},
		{
			name:          "no key in the file",
			certPath:      certPath,
			keyPath:       certPath,
			wantInitError: true,
		},
		{
			name:     "all good",
			certPath: certPath,
			keyPath:  keyPath,
		},
		{
			name:       "update cert to invalid cert",
			certPath:   certPath,
			keyPath:    keyPath,
			updateCert: cert2,
		},
		{
			name:      "update key to invalid key",
			certPath:  certPath,
			keyPath:   keyPath,
			updateKey: key2,
		},
		{
			name:                 "update both to a valid combination",
			certPath:             certPath,
			keyPath:              keyPath,
			updateCert:           cert2,
			updateKey:            key2,
			expectUpdate:         true,
			expectListenerUpdate: true,
		},
	}
	for _, tt := range tests {
		require.NoError(t, certutil.WriteCert(certPath, cert))
		require.NoError(t, os.WriteFile(keyPath, key, 0600))

		t.Run(tt.name, func(t *testing.T) {
			listener := make(testListener)

			c, err := NewDynamicCertKeyPairContentFromFiles("test server auth", tt.certPath, tt.keyPath)
			if (err != nil) != tt.wantInitError {
				t.Errorf("DynamicCertKeyPairContent init error = %v, wantErr %v", err, tt.wantInitError)
			}
			if c == nil {
				return
			}

			c.AddListener(&listener)

			certPre, keyPre := c.CurrentCertKeyContent()

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			go c.Run(ctx, 1)

			if len(tt.updateCert) > 0 {
				require.NoError(t, certutil.WriteCert(tt.certPath, tt.updateCert))
			}

			if len(tt.updateKey) > 0 {
				require.NoError(t, os.WriteFile(tt.keyPath, tt.updateKey, 0600))
			}

			if tt.expectListenerUpdate {
				<-listener
			} else {
				select {
				case <-listener:
					t.Errorf("did not expect any update in the listener")
				case <-time.After(500 * time.Millisecond):
				}
			}

			certPast, keyPast := c.CurrentCertKeyContent()

			certsUpdated := !bytes.Equal(certPre, certPast)
			keysUpdated := !bytes.Equal(keyPre, keyPast)
			if tt.expectUpdate != (certsUpdated || keysUpdated) {
				t.Errorf("expected cert key update: %t, but got certs updated %t, keys updated %t", tt.expectUpdate, certsUpdated, keysUpdated)
			}
		})
	}
}

type testListener chan struct{}

func (l testListener) Enqueue() {
	l <- struct{}{}
}
