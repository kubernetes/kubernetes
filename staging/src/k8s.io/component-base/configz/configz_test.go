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

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

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
