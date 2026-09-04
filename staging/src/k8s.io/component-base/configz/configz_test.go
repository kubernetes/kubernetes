/*
Copyright 2015 The Kubernetes Authors.

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

package configz

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// secretConfig is a minimal runtime.Object carrying a datapolicy-tagged field,
// used to verify that MarshalJSON redacts sensitive values.
type secretConfig struct {
	metav1.TypeMeta `json:",inline"`
	Public          string              `json:"public"`
	Token           string              `json:"token" datapolicy:"token"`
	Headers         map[string][]string `json:"headers" datapolicy:"token"`
}

func (c *secretConfig) GetObjectKind() schema.ObjectKind { return &c.TypeMeta }

func (c *secretConfig) DeepCopyObject() runtime.Object {
	cp := &secretConfig{
		TypeMeta: c.TypeMeta,
		Public:   c.Public,
		Token:    c.Token,
	}
	if c.Headers != nil {
		cp.Headers = make(map[string][]string, len(c.Headers))
		for k, v := range c.Headers {
			vals := make([]string, len(v))
			copy(vals, v)
			cp.Headers[k] = vals
		}
	}
	return cp
}

// unredactableConfig carries a field of a kind datapol.Redact does not know how
// to traverse (a channel), so redaction fails closed. It is used to verify that
// Set surfaces that error rather than storing an un-redacted value.
type unredactableConfig struct {
	metav1.TypeMeta `json:",inline"`
	Ch              chan int `json:"-"`
}

func (c *unredactableConfig) GetObjectKind() schema.ObjectKind { return &c.TypeMeta }

func (c *unredactableConfig) DeepCopyObject() runtime.Object {
	return &unredactableConfig{TypeMeta: c.TypeMeta, Ch: c.Ch}
}

func TestConfigz(t *testing.T) {
	v, err := New("testing")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := v.Set(&unstructured.Unstructured{Object: map[string]any{"apiVersion": "example.com/v1", "kind": "Blah"}}); err != nil {
		t.Fatal(err)
	}

	s := httptest.NewServer(http.HandlerFunc(handle))
	defer s.Close()

	resp, err := http.Get(s.URL + "/configz")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if string(body) != `{"testing":{"apiVersion":"example.com/v1","kind":"Blah"}}` {
		t.Fatalf("unexpected output: %s", body)
	}

	if err := v.Set(&unstructured.Unstructured{Object: map[string]any{"apiVersion": "example.com/v1", "kind": "Bing"}}); err != nil {
		t.Fatal(err)
	}
	resp, err = http.Get(s.URL + "/configz")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	body, err = io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if string(body) != `{"testing":{"apiVersion":"example.com/v1","kind":"Bing"}}` {
		t.Fatalf("unexpected output: %s", body)
	}

	Delete("testing")
	resp, err = http.Get(s.URL + "/configz")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	body, err = io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if string(body) != `{}` {
		t.Fatalf("unexpected output: %s", body)
	}
	if resp.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("unexpected Content-Type: %s", resp.Header.Get("Content-Type"))
	}
}

func TestConfigzWithAPIVersionAndKind(t *testing.T) {
	cfg := &unstructured.Unstructured{
		Object: map[string]any{
			"apiVersion": "test.k8s.io/v1",
			"kind":       "TestConfig",
			"value":      "test-value",
		},
	}

	v, err := New("testobj")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := v.Set(cfg); err != nil {
		t.Fatalf("err: %v", err)
	}

	s := httptest.NewServer(http.HandlerFunc(handle))
	defer s.Close()

	resp, err := http.Get(s.URL + "/configz")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	expected := `{"testobj":{"apiVersion":"test.k8s.io/v1","kind":"TestConfig","value":"test-value"}}`
	if string(body) != expected {
		t.Fatalf("unexpected output: %s, expected: %s", body, expected)
	}

	Delete("testobj")
}

func TestConfigzRedactsDatapolicyFields(t *testing.T) {
	cfg := &secretConfig{
		TypeMeta: metav1.TypeMeta{APIVersion: "test.k8s.io/v1", Kind: "SecretConfig"},
		Public:   "visible",
		Token:    "super-secret-token",
		Headers: map[string][]string{
			"Authorization": {"Bearer abc123"},
		},
	}

	v, err := New("secret")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer Delete("secret")

	if err := v.Set(cfg); err != nil {
		t.Fatalf("err: %v", err)
	}

	s := httptest.NewServer(http.HandlerFunc(handle))
	defer s.Close()

	resp, err := http.Get(s.URL + "/configz")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	out := string(body)
	if strings.Contains(out, "super-secret-token") {
		t.Errorf("expected token to be redacted, got: %s", out)
	}
	if strings.Contains(out, "Bearer abc123") {
		t.Errorf("expected header value to be redacted, got: %s", out)
	}
	if !strings.Contains(out, "CLASSIFIED") {
		t.Errorf("expected redacted CLASSIFIED value, got: %s", out)
	}
	if !strings.Contains(out, "visible") {
		t.Errorf("expected untagged field to be preserved, got: %s", out)
	}

	// The registered config must not be mutated by serialization.
	if cfg.Token != "super-secret-token" {
		t.Errorf("registered config was mutated: token = %q", cfg.Token)
	}
	if got := cfg.Headers["Authorization"]; len(got) != 1 || got[0] != "Bearer abc123" {
		t.Errorf("registered config was mutated: headers = %v", cfg.Headers)
	}
}

func TestConfigzSetRedactionError(t *testing.T) {
	v, err := New("redaction-error")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer Delete("redaction-error")

	cfg := &unredactableConfig{
		TypeMeta: metav1.TypeMeta{APIVersion: "test.k8s.io/v1", Kind: "Unredactable"},
		Ch:       make(chan int),
	}

	err = v.Set(cfg)
	if err == nil {
		t.Fatalf("expected error from Set, got nil")
	}
	if !strings.Contains(err.Error(), "failed to redact sensitive fields") {
		t.Fatalf("unexpected error: %v", err)
	}
	// A failed Set must not store the un-redacted value.
	if v.val != nil {
		t.Fatalf("expected val to remain unset after failed Set, got %v", v.val)
	}
}

func TestConfigzSetRedactsOnce(t *testing.T) {
	cfg := &secretConfig{
		TypeMeta: metav1.TypeMeta{APIVersion: "test.k8s.io/v1", Kind: "SecretConfig"},
		Public:   "visible",
		Token:    "super-secret-token",
	}

	v, err := New("redact-once")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer Delete("redact-once")

	if err := v.Set(cfg); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Redaction happens once in Set: the stored value is already redacted, so
	// mutating the caller's object afterwards must not affect the served output.
	stored, ok := v.val.(*secretConfig)
	if !ok {
		t.Fatalf("expected stored value of type *secretConfig, got %T", v.val)
	}
	if stored == cfg {
		t.Fatalf("expected Set to store a copy, not the caller's object")
	}
	if stored.Token != "CLASSIFIED" {
		t.Fatalf("expected stored token to be redacted at Set time, got %q", stored.Token)
	}

	cfg.Token = "mutated-after-set"
	b, err := v.MarshalJSON()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	out := string(b)
	if strings.Contains(out, "mutated-after-set") || strings.Contains(out, "super-secret-token") {
		t.Fatalf("expected served output to reflect the redacted copy, got: %s", out)
	}
	if !strings.Contains(out, "CLASSIFIED") {
		t.Fatalf("expected redacted CLASSIFIED value, got: %s", out)
	}
}

func TestConfigzErrors(t *testing.T) {
	v, err := New("errors")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer Delete("errors")

	tests := []struct {
		name     string
		obj      runtime.Object
		expected string
	}{
		{
			name:     "nil",
			obj:      nil,
			expected: "val may not be nil",
		},
		{
			name:     "empty kind",
			obj:      &unstructured.Unstructured{Object: map[string]any{"apiVersion": "example.com/v1"}},
			expected: "val must specify a kind",
		},
		{
			name:     "empty version",
			obj:      &unstructured.Unstructured{Object: map[string]any{"kind": "Blah"}},
			expected: "val must specify a group/version",
		},
		{
			name:     "internal version",
			obj:      &unstructured.Unstructured{Object: map[string]any{"apiVersion": "__internal", "kind": "Blah"}},
			expected: "val must specify an external version",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := v.Set(test.obj)
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			if err.Error() != test.expected {
				t.Fatalf("expected error %q, got %q", test.expected, err.Error())
			}
		})
	}
}

func TestDuplicateNew(t *testing.T) {
	name := "duplicate"
	v1, err := New(name)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if v1 == nil {
		t.Fatalf("expected non-nil config")
	}
	defer Delete(name)

	v2, err := New(name)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if v2 != nil {
		t.Fatalf("expected nil config, got %v", v2)
	}
	if !strings.Contains(err.Error(), "register config \"duplicate\" twice") {
		t.Fatalf("unexpected error message: %v", err)
	}
}
