/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/resource"
)

func TestValidateDuplicateLabelsFailCases(t *testing.T) {
	strs := []string{
		`{
	"metadata": {
		"labels": {
			"foo": "bar",
			"foo": "baz"
		}
	}
}`,
		`{
	"metadata": {
		"annotations": {
			"foo": "bar",
			"foo": "baz"
		}
	}
}`,
		`{
	"metadata": {
		"labels": {
			"foo": "blah"
		},
		"annotations": {
			"foo": "bar",
			"foo": "baz"
		}
	}
}`,
	}
	schema := NoDoubleKeySchema{}
	for _, str := range strs {
		err := schema.ValidateBytes([]byte(str))
		if err == nil {
			t.Errorf("Unexpected non-error %s", str)
		}
	}
}

func TestValidateDuplicateLabelsPassCases(t *testing.T) {
	strs := []string{
		`{
	"metadata": {
		"labels": {
			"foo": "bar"
		},
		"annotations": {
			"foo": "baz"
		}
	}
}`,
		`{
	"metadata": {}
}`,
		`{
	"metadata": {
		"labels": {}
	}
}`,
	}
	schema := NoDoubleKeySchema{}
	for _, str := range strs {
		err := schema.ValidateBytes([]byte(str))
		if err != nil {
			t.Errorf("Unexpected error: %v %s", err, str)
		}
	}
}

// AlwaysInvalidSchema is always invalid.
type AlwaysInvalidSchema struct{}

// ValidateBytes always fails to validate.
func (AlwaysInvalidSchema) ValidateBytes([]byte) error {
	return fmt.Errorf("always invalid")
}

func TestConjunctiveSchema(t *testing.T) {
	tests := []struct {
		schemas    []Schema
		shouldPass bool
		name       string
	}{
		{
			schemas:    []Schema{NullSchema{}, NullSchema{}},
			shouldPass: true,
			name:       "all pass",
		},
		{
			schemas:    []Schema{NullSchema{}, AlwaysInvalidSchema{}},
			shouldPass: false,
			name:       "one fail",
		},
		{
			schemas:    []Schema{AlwaysInvalidSchema{}, AlwaysInvalidSchema{}},
			shouldPass: false,
			name:       "all fail",
		},
		{
			schemas:    []Schema{},
			shouldPass: true,
			name:       "empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema := ConjunctiveSchema(tt.schemas)
			err := schema.ValidateBytes([]byte{})
			if err != nil && tt.shouldPass {
				t.Errorf("Unexpected error: %v in %s", err, tt.name)
			}
			if err == nil && !tt.shouldPass {
				t.Errorf("Unexpected non-error: %s", tt.name)
			}
		})
	}
}

type mockVerifier struct {
	supported bool
}

func (v *mockVerifier) HasSupport(gvk schema.GroupVersionKind) error {
	if !v.supported {
		return resource.NewParamUnsupportedError(gvk, resource.QueryParamFieldValidation)
	}
	return nil
}

// gvkVerifier reports fieldValidation support per kind and records
// the kinds it was asked about.
type gvkVerifier struct {
	supported map[string]bool
	errs      map[string]error
	asked     []string
}

func (v *gvkVerifier) HasSupport(gvk schema.GroupVersionKind) error {
	v.asked = append(v.asked, gvk.Kind)
	if err := v.errs[gvk.Kind]; err != nil {
		return err
	}
	if !v.supported[gvk.Kind] {
		return resource.NewParamUnsupportedError(gvk, resource.QueryParamFieldValidation)
	}
	return nil
}

// countingSchema counts how often client-side validation ran.
type countingSchema struct {
	calls int
}

func (s *countingSchema) ValidateBytes([]byte) error {
	s.calls++
	return nil
}

// TestParamVerifyingSchemaList tests that a list is verified per item, so
// that a list of kinds the server validates is not validated client-side.
func TestParamVerifyingSchemaList(t *testing.T) {
	pod := `{"apiVersion":"v1","kind":"Pod","metadata":{"name":"a"}}`
	service := `{"apiVersion":"v1","kind":"Service","metadata":{"name":"b"}}`
	foo := `{"apiVersion":"example.com/v1","kind":"Foo","metadata":{"name":"c"}}`
	bar := `{"apiVersion":"example.com/v1","kind":"Bar","metadata":{"name":"d"}}`
	tests := []struct {
		name          string
		data          string
		directive     string
		verifierErrs  map[string]error
		wantAsked     []string
		wantValidated int
		wantErr       string
	}{
		{
			name:      "v1 List of supported items is not validated client-side",
			data:      `{"apiVersion":"v1","kind":"List","items":[` + pod + `,` + service + `]}`,
			directive: "Strict",
			wantAsked: []string{"Pod", "Service"},
		},
		{
			name:      "typed list of supported items is not validated client-side",
			data:      `{"apiVersion":"v1","kind":"ServiceList","items":[` + service + `]}`,
			directive: "Strict",
			wantAsked: []string{"Service"},
		},
		{
			name:      "nested lists are verified per leaf item",
			data:      `{"apiVersion":"v1","kind":"List","items":[{"apiVersion":"v1","kind":"List","items":[` + pod + `]},` + service + `]}`,
			directive: "Strict",
			wantAsked: []string{"Pod", "Service"},
		},
		{
			name:      "empty list is not validated client-side",
			data:      `{"apiVersion":"v1","kind":"List","items":[]}`,
			directive: "Strict",
		},
		{
			name:          "list with an unsupported item is validated client-side under strict",
			data:          `{"apiVersion":"v1","kind":"List","items":[` + pod + `,` + foo + `]}`,
			directive:     "Strict",
			wantAsked:     []string{"Pod", "Foo"},
			wantValidated: 1,
		},
		{
			name:      "list with an unsupported item is skipped under warn",
			data:      `{"apiVersion":"v1","kind":"List","items":[` + foo + `]}`,
			directive: "Warn",
			wantAsked: []string{"Foo"},
		},
		{
			name:         "verifier error on a later item is not hidden by an earlier unsupported item",
			data:         `{"apiVersion":"v1","kind":"List","items":[` + foo + `,` + bar + `]}`,
			directive:    "Strict",
			verifierErrs: map[string]error{"Bar": errors.New("openapi v3 error")},
			wantAsked:    []string{"Foo", "Bar"},
			wantErr:      "openapi v3 error",
		},
		{
			name:      "item without a kind is an error",
			data:      `{"apiVersion":"v1","kind":"List","items":[{"apiVersion":"v1","metadata":{"name":"a"}}]}`,
			directive: "Strict",
			wantErr:   "kind not set",
		},
		{
			name:      "non-list object is verified by its own kind",
			data:      pod,
			directive: "Strict",
			wantAsked: []string{"Pod"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			verifier := &gvkVerifier{
				supported: map[string]bool{"Pod": true, "Service": true},
				errs:      tt.verifierErrs,
			}
			validated := &countingSchema{}
			schema := NewParamVerifyingSchema(validated, verifier, tt.directive)
			err := schema.ValidateBytes([]byte(tt.data))
			if tt.wantErr == "" && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
			if !reflect.DeepEqual(verifier.asked, tt.wantAsked) {
				t.Errorf("verifier asked about %v, want %v", verifier.asked, tt.wantAsked)
			}
			if validated.calls != tt.wantValidated {
				t.Errorf("client-side validation ran %d times, want %d", validated.calls, tt.wantValidated)
			}
		})
	}
}

// TestParamVerifyingSchema tests that client-side schema validation
// should be bypassed (and therefore validation succeeds) in all cases
// except when the field validation is "Strict" and server-side validation is
// unsupported.
func TestParamVerifyingSchema(t *testing.T) {
	bytes := []byte(`
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "name",
    "labels": {
      "name": "redis-master"
    }
  },
  "spec": {
    "containers": [
      {
        "name": "master",
	"image": "gcr.io/fake_project/fake_image:fake_tag",
        "args": "this is a bad command"
      }
    ]
  }
}
`)
	supportedVerifier := &mockVerifier{true}
	unsupportedVerifier := &mockVerifier{false}
	tests := []struct {
		name       string
		supported  bool
		schema     Schema
		verifier   resource.Verifier
		directive  string
		shouldPass bool
	}{
		{
			name:       "supported, strict",
			schema:     NullSchema{},
			verifier:   supportedVerifier,
			directive:  "Strict",
			shouldPass: true,
		},
		{
			name:       "supported, warn",
			schema:     NullSchema{},
			verifier:   supportedVerifier,
			directive:  "Warn",
			shouldPass: true,
		},
		{
			name:       "unsupported, strict",
			schema:     NullSchema{},
			verifier:   unsupportedVerifier,
			directive:  "Strict",
			shouldPass: true,
		},
		{
			name:       "unsupported, warn",
			schema:     NullSchema{},
			verifier:   unsupportedVerifier,
			directive:  "Warn",
			shouldPass: true,
		},
		{
			name:       "supported, strict, invalid schema",
			schema:     AlwaysInvalidSchema{},
			verifier:   supportedVerifier,
			directive:  "Strict",
			shouldPass: true,
		},
		{
			name:       "supported, warn, invalid schema",
			schema:     AlwaysInvalidSchema{},
			verifier:   supportedVerifier,
			directive:  "Warn",
			shouldPass: true,
		},
		{
			name:       "unsupported, strict, invalid schema",
			schema:     AlwaysInvalidSchema{},
			verifier:   unsupportedVerifier,
			directive:  "Strict",
			shouldPass: false,
		},
		{
			name:       "unsupported, warn, invalid schema",
			schema:     AlwaysInvalidSchema{},
			verifier:   unsupportedVerifier,
			directive:  "Warn",
			shouldPass: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema := NewParamVerifyingSchema(tt.schema, tt.verifier, tt.directive)
			err := schema.ValidateBytes(bytes)
			if err != nil && tt.shouldPass {
				t.Errorf("Unexpected error: %v in %s", err, tt.name)
			}
			if err == nil && !tt.shouldPass {
				t.Errorf("Unexpected non-error: %s", tt.name)
			}
		})
	}
}
