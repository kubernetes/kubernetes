/*
Copyright 2017 The Kubernetes Authors.

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

package upgrade

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestMoveFiles(t *testing.T) {
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)
	os.Chmod(tmpdir, 0766)

	certPath := filepath.Join(tmpdir, constants.APIServerCertName)
	certFile, err := os.OpenFile(certPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create cert file %s: %v", certPath, err)
	}
	defer certFile.Close()

	keyPath := filepath.Join(tmpdir, constants.APIServerKeyName)
	keyFile, err := os.OpenFile(keyPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create key file %s: %v", keyPath, err)
	}
	defer keyFile.Close()

	subDir := filepath.Join(tmpdir, "expired")
	if err := os.Mkdir(subDir, 0766); err != nil {
		t.Fatalf("Failed to create backup directory %s: %v", subDir, err)
	}

	filesToMove := map[string]string{
		filepath.Join(tmpdir, constants.APIServerCertName): filepath.Join(subDir, constants.APIServerCertName),
		filepath.Join(tmpdir, constants.APIServerKeyName):  filepath.Join(subDir, constants.APIServerKeyName),
	}

	if err := moveFiles(filesToMove); err != nil {
		t.Fatalf("Failed to move files %v: %v", filesToMove, err)
	}
}

func TestRollbackFiles(t *testing.T) {
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)
	os.Chmod(tmpdir, 0766)

	subDir := filepath.Join(tmpdir, "expired")
	if err := os.Mkdir(subDir, 0766); err != nil {
		t.Fatalf("Failed to create backup directory %s: %v", subDir, err)
	}

	certPath := filepath.Join(subDir, constants.APIServerCertName)
	certFile, err := os.OpenFile(certPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create cert file %s: %v", certPath, err)
	}
	defer certFile.Close()

	keyPath := filepath.Join(subDir, constants.APIServerKeyName)
	keyFile, err := os.OpenFile(keyPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create key file %s: %v", keyPath, err)
	}
	defer keyFile.Close()

	filesToRollBack := map[string]string{
		filepath.Join(subDir, constants.APIServerCertName): filepath.Join(tmpdir, constants.APIServerCertName),
		filepath.Join(subDir, constants.APIServerKeyName):  filepath.Join(tmpdir, constants.APIServerKeyName),
	}

	errString := "there are files need roll back"
	originalErr := errors.New(errString)
	err = rollbackFiles(filesToRollBack, originalErr)
	if err == nil {
		t.Fatalf("Expected error contains %q, got nil", errString)
	}
	if !strings.Contains(err.Error(), errString) {
		t.Fatalf("Expected error contains %q, got %v", errString, err)
	}
}

func TestCleanupKubeletDynamicEnvFileContainerRuntime(t *testing.T) {
	tcases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "common flag",
			input:    fmt.Sprintf("%s=\"--container-runtime=remote --container-runtime-endpoint=unix:///var/run/containerd/containerd.sock --pod-infra-container-image=registry.k8s.io/pause:3.8\"", constants.KubeletEnvFileVariableName),
			expected: fmt.Sprintf("%s=\"--container-runtime-endpoint=unix:///var/run/containerd/containerd.sock --pod-infra-container-image=registry.k8s.io/pause:3.8\"", constants.KubeletEnvFileVariableName),
		},
		{
			name:     "missing flag of interest",
			input:    fmt.Sprintf("%s=\"--foo=abc --bar=def --container-runtime-endpoint=unix:///var/run/containerd/containerd.sock\"", constants.KubeletEnvFileVariableName),
			expected: fmt.Sprintf("%s=\"--foo=abc --bar=def --container-runtime-endpoint=unix:///var/run/containerd/containerd.sock\"", constants.KubeletEnvFileVariableName),
		},
		{
			name:     "add missing URL scheme",
			input:    fmt.Sprintf("%s=\"--foo=abc --container-runtime=remote --bar=def\"", constants.KubeletEnvFileVariableName),
			expected: fmt.Sprintf("%s=\"--foo=abc --bar=def\"", constants.KubeletEnvFileVariableName),
		},
		{
			name:     "add missing URL scheme if there is no '=' after the flag name",
			input:    fmt.Sprintf("%s=\"--foo=abc --container-runtime remote --container-runtime-endpoint=unix:///var/run/containerd/containerd.sock --bar=def\"", constants.KubeletEnvFileVariableName),
			expected: fmt.Sprintf("%s=\"--foo=abc --container-runtime-endpoint=unix:///var/run/containerd/containerd.sock --bar=def\"", constants.KubeletEnvFileVariableName),
		},
		{
			name:     "empty flag of interest value following '='",
			input:    fmt.Sprintf("%s=\"--foo=abc --container-runtime= --container-runtime-endpoint unix:///var/run/containerd/containerd.sock --bar=def\"", constants.KubeletEnvFileVariableName),
			expected: fmt.Sprintf("%s=\"--foo=abc --container-runtime-endpoint unix:///var/run/containerd/containerd.sock --bar=def\"", constants.KubeletEnvFileVariableName),
		},
		{
			name:     "empty flag of interest value without '='",
			input:    fmt.Sprintf("%s=\"--foo=abc --container-runtime --bar=def\"", constants.KubeletEnvFileVariableName),
			expected: fmt.Sprintf("%s=\"--foo=abc --bar=def\"", constants.KubeletEnvFileVariableName),
		},
	}
	for _, tt := range tcases {
		t.Run(tt.name, func(t *testing.T) {
			output := cleanupKubeletDynamicEnvFileContainerRuntime(tt.input)
			if output != tt.expected {
				t.Errorf("expected output: %q, got: %q", tt.expected, output)
			}
		})
	}
}
