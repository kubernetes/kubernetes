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

package app

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

const fakeKubeconfig = `
apiVersion: v1
kind: Config
clusters:
- cluster:
    insecure-skip-tls-verify: true
    server: %s
  name: default
contexts:
- context:
    cluster: default
    user: default
  name: default
current-context: default
users:
- name: default
  user:
    username: config
`

// TestHollowNode is a naive test that attempts to start hollow node and checks if it's not crashing.
// Such test is sufficient to detect e.g. missing kubelet dependencies that are not added in
// pkg/kubemark/hollow_kubelet.go.
func TestHollowNode(t *testing.T) {
	// temp dir
	tmpDir, err := os.MkdirTemp("", "hollow-node")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// https server
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`ok`))
	}))
	defer server.Close()

	kubeconfigPath := filepath.Join(tmpDir, "config.kubeconfig")
	if err := os.WriteFile(kubeconfigPath, []byte(fmt.Sprintf(fakeKubeconfig, server.URL)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	for morph := range knownMorphs {
		morph := morph
		t.Run(morph, func(t *testing.T) {
			s := &hollowNodeConfig{
				KubeconfigPath: kubeconfigPath,
				Morph:          morph,
			}
			errCh := make(chan error)
			go func() {
				data, err := os.ReadFile(kubeconfigPath)
				t.Logf("read %d, err=%v\n", len(data), err)
				errCh <- run(s)
			}()

			select {
			case err := <-errCh:
				t.Fatalf("Run finished unexpectedly with error: %v", err)
			case <-time.After(3 * time.Second):
				t.Logf("Morph %q hasn't crashed for 3s. Calling success.", morph)
			}
		})
	}
}
