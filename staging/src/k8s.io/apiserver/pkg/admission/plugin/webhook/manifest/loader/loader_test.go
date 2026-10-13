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
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

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
	// Build a minimal decoder (admissionregistration types only, no full client-go scheme)
	s := runtime.NewScheme()
	utilruntime.Must(admissionregistrationv1.AddToScheme(s))
	s.AddUnversionedTypes(metav1.SchemeGroupVersion, &metav1.List{}, &metav1.Status{})
	decoder := serializer.NewCodecFactory(s).UniversalDeserializer()

	// Simple accept function that extracts the typed object (no kube-apiserver defaulting)
	accept := func(obj runtime.Object) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
		vwc, ok := obj.(*admissionregistrationv1.ValidatingWebhookConfiguration)
		if !ok {
			return nil, fmt.Errorf("unsupported type %T", obj)
		}
		return []*admissionregistrationv1.ValidatingWebhookConfiguration{vwc}, nil
	}

	tests := []struct {
		name           string
		files          map[string]string
		wantConfigs    int
		wantErr        bool
		wantErrContain string
	}{
		{
			name: "single validating webhook",
			files: map[string]string{
				"wh.yaml": `apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: test.static.k8s.io
webhooks:
- name: test.webhook.io
  admissionReviewVersions: ["v1"]
  clientConfig:
    url: "https://example.com"
  sideEffects: None
`,
			},
			wantConfigs: 1,
		},
		{
			name:        "empty directory",
			files:       map[string]string{},
			wantConfigs: 0,
		},
		{
			name: "unsupported resource type",
			files: map[string]string{
				"wrong.yaml": `apiVersion: v1
kind: ConfigMap
metadata:
  name: not-a-webhook
`,
			},
			wantErr:        true,
			wantErrContain: "error loading",
		},
		{
			name: "v1.List with validating webhook",
			files: map[string]string{
				"list.yaml": `apiVersion: v1
kind: List
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingWebhookConfiguration
  metadata:
    name: v1list-wh.static.k8s.io
  webhooks:
  - name: list.webhook.io
    admissionReviewVersions: ["v1"]
    clientConfig:
      url: "https://example.com"
    sideEffects: None
`,
			},
			wantConfigs: 1,
		},
		{
			name: "duplicate webhook config names",
			files: map[string]string{
				"01-wh.yaml": `apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: dup.static.k8s.io
webhooks:
- name: first.webhook.io
  admissionReviewVersions: ["v1"]
  clientConfig:
    url: "https://example.com"
  sideEffects: None
`,
				"02-wh.yaml": `apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: dup.static.k8s.io
webhooks:
- name: second.webhook.io
  admissionReviewVersions: ["v1"]
  clientConfig:
    url: "https://example.com"
  sideEffects: None
`,
			},
			wantErr:        true,
			wantErrContain: "duplicate",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			for name, content := range tt.files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0644); err != nil {
					t.Fatalf("failed to write file %s: %v", name, err)
				}
			}

			configs, hash, err := LoadManifests(dir, decoder, accept)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error but got none")
				} else if tt.wantErrContain != "" && !strings.Contains(err.Error(), tt.wantErrContain) {
					t.Errorf("expected error containing %q, got %q", tt.wantErrContain, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(configs) != tt.wantConfigs {
				t.Errorf("expected %d configs, got %d", tt.wantConfigs, len(configs))
			}
			if tt.wantConfigs > 0 && hash == "" {
				t.Error("expected non-empty hash")
			}
		})
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
