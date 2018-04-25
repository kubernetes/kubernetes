/*
Copyright 2018 The Kubernetes Authors.

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

package cmd

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

const (
	ServerAddr = "localhost:9008"
	TestConfig = `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data:
    server: ` + ServerAddr + `
  name: prod
contexts:
- context:
    cluster: prod
    namespace: default
    user: default-service-account
  name: default
current-context: default
kind: Config
preferences: {}
users:
- name: kubernetes-admin
  user:
    client-certificate-data:
    client-key-data:
`
	ValidResponseYAML = `api:
  advertiseAddress: 127.0.0.1
  bindPort: 9008
`
)

func TestNewCmdConfigView(t *testing.T) {
	var buf bytes.Buffer
	testConfigFile := "test-config-file"

	tmpDir, err := ioutil.TempDir("", "kubeadm-token-test")
	if err != nil {
		t.Errorf("Unable to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	fullPath := filepath.Join(tmpDir, testConfigFile)

	f, err := os.Create(fullPath)
	if err != nil {
		t.Errorf("Unable to create test file %q: %v", fullPath, err)
	}
	defer f.Close()

	if _, err = f.WriteString(TestConfig); err != nil {
		t.Errorf("Unable to write test file %q: %v", fullPath, err)
	}

	http.HandleFunc("/", httpHandler)
	httpServer := &http.Server{Addr: ServerAddr}
	go func() {
		err := httpServer.ListenAndServe()
		if err != nil {
			t.Errorf("Failed to start dummy API server: %s\n", ServerAddr)
		}
	}()

	cmd := NewCmdConfigView(&buf, &fullPath)
	if err := cmd.Execute(); err != nil {
		t.Errorf("Failed to execute NewCmdConfigView for config file: %v; %v", fullPath, err)
	}
}

func httpHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/yaml")
	w.WriteHeader(200)
	w.Write([]byte(ValidResponseYAML))
}
