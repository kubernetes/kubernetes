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

package loader

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	admissionregistrationv1scheme "k8s.io/client-go/kubernetes/scheme"
)

func TestValidateManifestName(t *testing.T) {
	tests := []struct {
		name        string
		objName     string
		filePath    string
		seenNames   map[string]string
		wantErr     bool
		errContains string
	}{
		{
			name:      "valid name",
			objName:   "test.static.k8s.io",
			filePath:  "test.yaml",
			seenNames: map[string]string{},
		},
		{
			name:        "empty name",
			objName:     "",
			filePath:    "test.yaml",
			seenNames:   map[string]string{},
			wantErr:     true,
			errContains: "must have a name",
		},
		{
			name:        "missing suffix",
			objName:     "no-suffix",
			filePath:    "test.yaml",
			seenNames:   map[string]string{},
			wantErr:     true,
			errContains: "must have a name ending with",
		},
		{
			name:        "duplicate name",
			objName:     "dup.static.k8s.io",
			filePath:    "second.yaml",
			seenNames:   map[string]string{"dup.static.k8s.io": "first.yaml"},
			wantErr:     true,
			errContains: "duplicate",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateManifestName(tt.objName, tt.filePath, tt.seenNames)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errContains)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestValidateWebhookClientConfig(t *testing.T) {
	url := "https://example.com"
	tests := []struct {
		name        string
		cc          admissionregistrationv1.WebhookClientConfig
		wantErr     bool
		errContains string
	}{
		{
			name: "valid URL config",
			cc:   admissionregistrationv1.WebhookClientConfig{URL: &url},
		},
		{
			name:        "service ref",
			cc:          admissionregistrationv1.WebhookClientConfig{Service: &admissionregistrationv1.ServiceReference{Name: "svc", Namespace: "ns"}},
			wantErr:     true,
			errContains: "clientConfig.service is not supported",
		},
		{
			name:        "no URL",
			cc:          admissionregistrationv1.WebhookClientConfig{},
			wantErr:     true,
			errContains: "clientConfig.url is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateWebhookClientConfig("wh", "cfg", "test.yaml", tt.cc)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errContains)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestLoadManifests(t *testing.T) {
	// Build a minimal decoder using client-go's scheme (v1 types only, no internal types)
	s := runtime.NewScheme()
	if err := admissionregistrationv1scheme.AddToScheme(s); err != nil {
		t.Fatalf("failed to add scheme: %v", err)
	}
	s.AddUnversionedTypes(metav1.SchemeGroupVersion, &metav1.List{}, &metav1.Status{})
	decoder := serializer.NewCodecFactory(s).UniversalDeserializer()

	// Simple accept function that extracts the typed object (no kube-apiserver defaulting)
	accept := func(obj runtime.Object) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, bool, error) {
		vwc, ok := obj.(*admissionregistrationv1.ValidatingWebhookConfiguration)
		if !ok {
			return nil, false, nil
		}
		return []*admissionregistrationv1.ValidatingWebhookConfiguration{vwc}, true, nil
	}

	dir := t.TempDir()
	content := `apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: test.static.k8s.io
webhooks:
- name: test.webhook.io
  admissionReviewVersions: ["v1"]
  clientConfig:
    url: "https://example.com"
  sideEffects: None
`
	if err := os.WriteFile(filepath.Join(dir, "wh.yaml"), []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	configs, hash, err := LoadManifests(dir, decoder, accept)
	if err != nil {
		t.Fatalf("LoadManifests() error: %v", err)
	}
	if len(configs) != 1 {
		t.Fatalf("expected 1 config, got %d", len(configs))
	}
	if configs[0].Name != "test.static.k8s.io" {
		t.Errorf("config name = %q, want %q", configs[0].Name, "test.static.k8s.io")
	}
	if hash == "" {
		t.Error("expected non-empty hash")
	}
}

func TestLoadManifests_EmptyDir(t *testing.T) {
	s := runtime.NewScheme()
	if err := admissionregistrationv1scheme.AddToScheme(s); err != nil {
		t.Fatalf("failed to add scheme: %v", err)
	}
	decoder := serializer.NewCodecFactory(s).UniversalDeserializer()

	accept := func(obj runtime.Object) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, bool, error) {
		return nil, false, nil
	}

	dir := t.TempDir()
	configs, _, err := LoadManifests(dir, decoder, accept)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(configs) != 0 {
		t.Errorf("expected 0 configs for empty dir, got %d", len(configs))
	}
}

func TestValidatingLoadResult_GetWebhookAccessors(t *testing.T) {
	url := "https://example.com"
	result := &ValidatingLoadResult{
		Configurations: []*admissionregistrationv1.ValidatingWebhookConfiguration{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "test.static.k8s.io"},
				Webhooks: []admissionregistrationv1.ValidatingWebhook{
					{
						Name:                    "first.webhook.io",
						ClientConfig:            admissionregistrationv1.WebhookClientConfig{URL: &url},
						AdmissionReviewVersions: []string{"v1"},
					},
					{
						Name:                    "second.webhook.io",
						ClientConfig:            admissionregistrationv1.WebhookClientConfig{URL: &url},
						AdmissionReviewVersions: []string{"v1"},
					},
				},
			},
		},
	}
	accessors := result.GetWebhookAccessors()
	if len(accessors) != 2 {
		t.Fatalf("expected 2 accessors, got %d", len(accessors))
	}
}

func TestMutatingLoadResult_GetWebhookAccessors(t *testing.T) {
	url := "https://example.com"
	result := &MutatingLoadResult{
		Configurations: []*admissionregistrationv1.MutatingWebhookConfiguration{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "test.static.k8s.io"},
				Webhooks: []admissionregistrationv1.MutatingWebhook{
					{
						Name:                    "mutate.webhook.io",
						ClientConfig:            admissionregistrationv1.WebhookClientConfig{URL: &url},
						AdmissionReviewVersions: []string{"v1"},
					},
				},
			},
		},
	}
	accessors := result.GetWebhookAccessors()
	if len(accessors) != 1 {
		t.Fatalf("expected 1 accessor, got %d", len(accessors))
	}
}
