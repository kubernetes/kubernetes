/*
Copyright 2026 The Kubernetes Authors.

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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

func mustCreateCA(t *testing.T, cn string) []byte {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: cn},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
}

func waitForCABundle(t *testing.T, c *DynamicFileCAContent, want []byte, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var last []byte
	for time.Now().Before(deadline) {
		last = c.CurrentCABundleContent()
		if bytes.Equal(last, want) {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for CA bundle update: got %d bytes, want %d bytes", len(last), len(want))
}

// TestFileRefreshPollReloadsWithoutFsnotify starts only the worker + FileRefreshDuration
// enqueue loop from Run (not watchCAFile). A rewrite of the CA file then cannot be
// attributed to inotify; the poll must pick it up. Run() must keep that loop.
func TestFileRefreshPollReloadsWithoutFsnotify(t *testing.T) {
	orig := FileRefreshDuration
	FileRefreshDuration = 50 * time.Millisecond
	t.Cleanup(func() { FileRefreshDuration = orig })

	dir := t.TempDir()
	filename := filepath.Join(dir, "ca.crt")
	ca1 := mustCreateCA(t, "ca-one")
	ca2 := mustCreateCA(t, "ca-two")
	if err := os.WriteFile(filename, ca1, 0644); err != nil {
		t.Fatal(err)
	}

	c, err := NewDynamicCAContentFromFile("test", filename)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go wait.Until(c.runWorker, time.Second, ctx.Done())
	go wait.Until(func() { c.queue.Add(workItemKey) }, FileRefreshDuration, ctx.Done())

	if err := os.WriteFile(filename, ca2, 0644); err != nil {
		t.Fatal(err)
	}
	waitForCABundle(t, c, ca2, 2*time.Second)
}
