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
	"fmt"
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
