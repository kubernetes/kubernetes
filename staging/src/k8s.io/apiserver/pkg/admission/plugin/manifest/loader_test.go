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

package manifest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func Test_splitYAMLDocuments(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantCount int
		wantErr   bool
		// wantContains checks that each returned doc contains the expected substring
		wantContains []string
	}{
		{
			name:      "empty input",
			input:     "",
			wantCount: 0,
		},
		{
			name:      "only whitespace",
			input:     "   \n\n  \n",
			wantCount: 0,
		},
		{
			name: "single YAML webhook configuration",
			input: `apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: test-webhook.static.k8s.io
webhooks:
- name: test.example.com
  clientConfig:
    url: "https://example.com/validate"
`,
			wantCount:    1,
			wantContains: []string{"ValidatingWebhookConfiguration"},
		},
		{
			name: "multi-document YAML with policy and binding",
			input: `apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: deny-privileged.static.k8s.io
spec:
  failurePolicy: Fail
---
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: deny-privileged-binding.static.k8s.io
spec:
  policyName: deny-privileged.static.k8s.io
`,
			wantCount:    2,
			wantContains: []string{"ValidatingAdmissionPolicy", "ValidatingAdmissionPolicyBinding"},
		},
		{
			name:         "single JSON document",
			input:        `{"apiVersion":"admissionregistration.k8s.io/v1","kind":"ValidatingWebhookConfiguration","metadata":{"name":"test.static.k8s.io"}}`,
			wantCount:    1,
			wantContains: []string{"ValidatingWebhookConfiguration"},
		},
		{
			name: "empty document between separators is skipped",
			input: `a: 1
---
---
b: 2
`,
			wantCount: 2,
		},
		{
			name: "leading separator",
			input: `---
apiVersion: v1
kind: ConfigMap
`,
			wantCount: 1,
		},
		{
			name: "trailing separator",
			input: `apiVersion: v1
kind: ConfigMap
---
`,
			wantCount: 1,
		},
		{
			name: "three documents",
			input: `kind: A
---
kind: B
---
kind: C
`,
			wantCount:    3,
			wantContains: []string{"A", "B", "C"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			docs, err := splitYAMLDocuments([]byte(tc.input))
			if (err != nil) != tc.wantErr {
				t.Fatalf("splitYAMLDocuments() error = %v, wantErr %v", err, tc.wantErr)
			}
			if len(docs) != tc.wantCount {
				t.Fatalf("splitYAMLDocuments() returned %d documents, want %d", len(docs), tc.wantCount)
			}
			// Verify no empty documents were returned
			for i, doc := range docs {
				trimmed := strings.TrimSpace(string(doc))
				if trimmed == "" {
					t.Errorf("document %d is empty after trim", i)
				}
			}
			// Verify expected content
			for i, want := range tc.wantContains {
				if i >= len(docs) {
					break
				}
				if !strings.Contains(string(docs[i]), want) {
					t.Errorf("document %d: expected to contain %q, got %q", i, want, string(docs[i]))
				}
			}
		})
	}
}

func TestLoadFiles(t *testing.T) {
	t.Run("empty dir returns no docs", func(t *testing.T) {
		dir := t.TempDir()
		docs, hash, err := LoadFiles(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(docs) != 0 {
			t.Errorf("expected 0 docs, got %d", len(docs))
		}
		if hash != "" {
			t.Errorf("expected empty hash for empty dir, got %q", hash)
		}
	})

	t.Run("reads yaml and json files", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "a.yaml"), []byte("kind: A"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "b.json"), []byte(`{"kind":"B"}`), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "c.yml"), []byte("kind: C"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "d.txt"), []byte("ignored"), 0644); err != nil {
			t.Fatal(err)
		}

		docs, hash, err := LoadFiles(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(docs) != 3 {
			t.Fatalf("expected 3 docs (.yaml, .json, .yml), got %d", len(docs))
		}
		if len(hash) == 0 {
			t.Error("expected non-empty hash")
		}
	})

	t.Run("splits multi-document yaml", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "multi.yaml"), []byte("a: 1\n---\nb: 2\n"), 0644); err != nil {
			t.Fatal(err)
		}

		docs, _, err := LoadFiles(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(docs) != 2 {
			t.Fatalf("expected 2 docs, got %d", len(docs))
		}
	})

	t.Run("alphabetical order", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "02-second.yaml"), []byte("name: second"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "01-first.yaml"), []byte("name: first"), 0644); err != nil {
			t.Fatal(err)
		}

		docs, _, err := LoadFiles(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(docs) != 2 {
			t.Fatalf("expected 2 docs, got %d", len(docs))
		}
		if !strings.Contains(string(docs[0].Doc), "first") {
			t.Errorf("first doc should be from 01-first.yaml, got %q", string(docs[0].Doc))
		}
	})

	t.Run("empty path returns error", func(t *testing.T) {
		_, _, err := LoadFiles("")
		if err == nil {
			t.Error("expected error for empty path")
		}
	})

	t.Run("nonexistent dir returns error", func(t *testing.T) {
		_, _, err := LoadFiles("/nonexistent/path")
		if err == nil {
			t.Error("expected error for nonexistent dir")
		}
	})

	t.Run("skips subdirectories", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "a.yaml"), []byte("kind: A"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Join(dir, "subdir"), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "subdir", "b.yaml"), []byte("kind: B"), 0644); err != nil {
			t.Fatal(err)
		}

		docs, _, err := LoadFiles(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(docs) != 1 {
			t.Fatalf("expected 1 doc (subdir skipped), got %d", len(docs))
		}
	})

	t.Run("skips empty files", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "empty.yaml"), []byte(""), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "valid.yaml"), []byte("kind: A"), 0644); err != nil {
			t.Fatal(err)
		}

		docs, _, err := LoadFiles(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(docs) != 1 {
			t.Fatalf("expected 1 doc (empty skipped), got %d", len(docs))
		}
	})
}
