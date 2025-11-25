/*
Copyright 2024 The Kubernetes Authors.

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

package helmapplyset

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
)

func TestIsHelmReleaseSecret(t *testing.T) {
	tests := []struct {
		name     string
		secret   *v1.Secret
		expected bool
	}{
		{
			name: "valid Helm release Secret",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "sh.helm.release.v1.my-app.1",
				},
				Type: HelmReleaseSecretType,
			},
			expected: true,
		},
		{
			name: "wrong Secret type",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "sh.helm.release.v1.my-app.1",
				},
				Type: v1.SecretTypeOpaque,
			},
			expected: false,
		},
		{
			name: "wrong name prefix",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-secret",
				},
				Type: HelmReleaseSecretType,
			},
			expected: false,
		},
		{
			name: "non-Helm Secret",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "regular-secret",
				},
				Type: v1.SecretTypeOpaque,
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsHelmReleaseSecret(tt.secret)
			if result != tt.expected {
				t.Errorf("IsHelmReleaseSecret() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestExtractReleaseNameFromSecretName(t *testing.T) {
	tests := []struct {
		name        string
		secretName  string
		expected    string
		expectError bool
	}{
		{
			name:       "valid Secret name",
			secretName: "sh.helm.release.v1.my-app.1",
			expected:   "my-app",
		},
		{
			name:       "valid Secret name with longer release name",
			secretName: "sh.helm.release.v1.my-long-release-name.5",
			expected:   "my-long-release-name",
		},
		{
			name:        "invalid prefix",
			secretName:  "my-secret",
			expectError: true,
		},
		{
			name:        "too few parts",
			secretName:  "sh.helm.release",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ExtractReleaseNameFromSecretName(tt.secretName)
			if tt.expectError {
				if err == nil {
					t.Errorf("ExtractReleaseNameFromSecretName() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("ExtractReleaseNameFromSecretName() error = %v", err)
				return
			}
			if result != tt.expected {
				t.Errorf("ExtractReleaseNameFromSecretName() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestExtractReleaseVersionFromSecretName(t *testing.T) {
	tests := []struct {
		name        string
		secretName  string
		expected    int
		expectError bool
	}{
		{
			name:       "valid Secret name",
			secretName: "sh.helm.release.v1.my-app.1",
			expected:   1,
		},
		{
			name:       "higher version",
			secretName: "sh.helm.release.v1.my-app.42",
			expected:   42,
		},
		{
			name:        "invalid prefix",
			secretName:  "my-secret",
			expectError: true,
		},
		{
			name:        "non-numeric version",
			secretName:  "sh.helm.release.v1.my-app.abc",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ExtractReleaseVersionFromSecretName(tt.secretName)
			if tt.expectError {
				if err == nil {
					t.Errorf("ExtractReleaseVersionFromSecretName() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("ExtractReleaseVersionFromSecretName() error = %v", err)
				return
			}
			if result != tt.expected {
				t.Errorf("ExtractReleaseVersionFromSecretName() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestDecodeHelmRelease(t *testing.T) {
	// Create a test Helm release JSON
	releaseJSON := `{
		"name": "test-release",
		"namespace": "default",
		"version": 1,
		"status": "deployed",
		"chart": {
			"metadata": {
				"name": "test-chart",
				"version": "1.0.0"
			}
		},
		"manifest": "apiVersion: v1\nkind: Service\nmetadata:\n  name: test-service"
	}`

	// Compress and encode
	var buf bytes.Buffer
	gzWriter := gzip.NewWriter(&buf)
	_, err := gzWriter.Write([]byte(releaseJSON))
	if err != nil {
		t.Fatalf("Failed to write gzip: %v", err)
	}
	if err := gzWriter.Close(); err != nil {
		t.Fatalf("Failed to close gzip: %v", err)
	}

	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())

	// Test decoding
	decoded, err := decodeHelmRelease([]byte(encoded))
	if err != nil {
		t.Fatalf("decodeHelmRelease() error = %v", err)
	}

	var decodedRelease map[string]interface{}
	if err := json.Unmarshal(decoded, &decodedRelease); err != nil {
		t.Fatalf("Failed to unmarshal decoded release: %v", err)
	}

	if decodedRelease["name"] != "test-release" {
		t.Errorf("decoded release name = %v, want test-release", decodedRelease["name"])
	}
}

func TestParseHelmReleaseSecret(t *testing.T) {
	// Create a test Helm release JSON
	releaseJSON := HelmReleaseJSON{
		Name:      "test-release",
		Namespace: "default",
		Version:   1,
		Status:    "deployed",
		Chart: struct {
			Metadata struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"metadata"`
		}{
			Metadata: struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			}{
				Name:    "test-chart",
				Version: "1.0.0",
			},
		},
		Manifest: `apiVersion: v1
kind: Service
metadata:
  name: test-service
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
`,
	}

	jsonBytes, err := json.Marshal(releaseJSON)
	if err != nil {
		t.Fatalf("Failed to marshal release JSON: %v", err)
	}

	// Compress and encode
	var buf bytes.Buffer
	gzWriter := gzip.NewWriter(&buf)
	_, err = gzWriter.Write(jsonBytes)
	if err != nil {
		t.Fatalf("Failed to write gzip: %v", err)
	}
	if err := gzWriter.Close(); err != nil {
		t.Fatalf("Failed to close gzip: %v", err)
	}

	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())

	// Create Secret
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sh.helm.release.v1.test-release.1",
			Namespace: "default",
		},
		Type: HelmReleaseSecretType,
		Data: map[string][]byte{
			HelmReleaseDataKey: []byte(encoded),
		},
	}

	// Parse
	info, err := ParseHelmReleaseSecret(secret)
	if err != nil {
		t.Fatalf("ParseHelmReleaseSecret() error = %v", err)
	}

	// Verify parsed information
	if info.Name != "test-release" {
		t.Errorf("Name = %v, want test-release", info.Name)
	}
	if info.Namespace != "default" {
		t.Errorf("Namespace = %v, want default", info.Namespace)
	}
	if info.Version != 1 {
		t.Errorf("Version = %v, want 1", info.Version)
	}
	if info.Status != "deployed" {
		t.Errorf("Status = %v, want deployed", info.Status)
	}
	if info.Chart != "test-chart-1.0.0" {
		t.Errorf("Chart = %v, want test-chart-1.0.0", info.Chart)
	}

	// Verify GroupKinds
	expectedGroupKinds := sets.New[schema.GroupKind](
		schema.GroupKind{Group: "", Kind: "Service"},
		schema.GroupKind{Group: "apps", Kind: "Deployment"},
	)
	if !info.GroupKinds.Equal(expectedGroupKinds) {
		t.Errorf("GroupKinds = %v, want %v", info.GroupKinds, expectedGroupKinds)
	}
}

func TestParseHelmReleaseSecret_InvalidSecret(t *testing.T) {
	tests := []struct {
		name        string
		secret      *v1.Secret
		expectError bool
	}{
		{
			name: "wrong Secret type",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "sh.helm.release.v1.test.1",
				},
				Type: v1.SecretTypeOpaque,
			},
			expectError: true,
		},
		{
			name: "missing release data",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "sh.helm.release.v1.test.1",
				},
				Type: HelmReleaseSecretType,
				Data: map[string][]byte{},
			},
			expectError: true,
		},
		{
			name: "invalid base64 data",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "sh.helm.release.v1.test.1",
				},
				Type: HelmReleaseSecretType,
				Data: map[string][]byte{
					HelmReleaseDataKey: []byte("invalid-base64!!!"),
				},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseHelmReleaseSecret(tt.secret)
			if tt.expectError {
				if err == nil {
					t.Errorf("ParseHelmReleaseSecret() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("ParseHelmReleaseSecret() unexpected error = %v", err)
			}
		})
	}
}

func TestWatcher_EventHandlers(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	secretInformer := informerFactory.Core().V1().Secrets()

	enqueued := []interface{}{}
	enqueueFunc := func(obj interface{}) {
		enqueued = append(enqueued, obj)
	}

	watcher := NewWatcher(secretInformer, enqueueFunc, logger)

	// Create a Helm release Secret
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sh.helm.release.v1.test.1",
			Namespace: "default",
		},
		Type: HelmReleaseSecretType,
		Data: map[string][]byte{
			HelmReleaseDataKey: []byte("test-data"),
		},
	}

	// Test Add handler
	watcher.handleAdd(secret)
	if len(enqueued) != 1 {
		t.Errorf("Expected 1 enqueued item, got %d", len(enqueued))
	}

	// Test Update handler
	enqueued = []interface{}{}
	oldSecret := secret.DeepCopy()
	newSecret := secret.DeepCopy()
	newSecret.ResourceVersion = "2"
	watcher.handleUpdate(oldSecret, newSecret)
	if len(enqueued) != 1 {
		t.Errorf("Expected 1 enqueued item, got %d", len(enqueued))
	}

	// Test Update handler with same ResourceVersion (should skip)
	enqueued = []interface{}{}
	watcher.handleUpdate(oldSecret, oldSecret)
	if len(enqueued) != 0 {
		t.Errorf("Expected 0 enqueued items, got %d", len(enqueued))
	}

	// Test Delete handler
	enqueued = []interface{}{}
	watcher.handleDelete(secret)
	if len(enqueued) != 1 {
		t.Errorf("Expected 1 enqueued item, got %d", len(enqueued))
	}
}

func TestExtractGroupKindsFromManifest(t *testing.T) {
	manifest := `apiVersion: v1
kind: Service
metadata:
  name: test-service
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
`

	groupKinds, err := extractGroupKindsFromManifest(manifest)
	if err != nil {
		t.Fatalf("extractGroupKindsFromManifest() error = %v", err)
	}

	expected := sets.New[schema.GroupKind](
		schema.GroupKind{Group: "", Kind: "Service"},
		schema.GroupKind{Group: "apps", Kind: "Deployment"},
		schema.GroupKind{Group: "", Kind: "ConfigMap"},
	)

	if !groupKinds.Equal(expected) {
		t.Errorf("GroupKinds = %v, want %v", groupKinds, expected)
	}
}

func TestExtractGroupKindsFromManifest_EmptyManifest(t *testing.T) {
	groupKinds, err := extractGroupKindsFromManifest("")
	if err != nil {
		t.Fatalf("extractGroupKindsFromManifest() error = %v", err)
	}
	if groupKinds.Len() != 0 {
		t.Errorf("Expected empty GroupKinds set, got %v", groupKinds)
	}
}
