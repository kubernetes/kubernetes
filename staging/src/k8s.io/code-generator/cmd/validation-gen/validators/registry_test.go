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

package validators

import (
	"sync/atomic"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

func TestExtractValidations_TagValidator(t *testing.T) {
	tests := []struct {
		name    string
		tags    []codetags.Tag
		setup   func(*registry)
		wantErr string
	}{
		{
			name: "known tag",
			tags: []codetags.Tag{{Name: "known:tag"}},
			setup: func(r *registry) {
				r.tagValidators["known:tag"] = &mockTagValidator{tagName: "known:tag", scopes: sets.New(ScopeType)}
			},
			wantErr: "",
		},
		{
			name:    "unknown tag",
			tags:    []codetags.Tag{{Name: "unknown:tag"}},
			setup:   func(r *registry) {},
			wantErr: "tag processing errors: unknown tag \"unknown:tag\"",
		},
		{
			name:    "nil validator",
			tags:    []codetags.Tag{{Name: "nil:tag"}},
			setup:   func(r *registry) { r.tagValidators["nil:tag"] = nil },
			wantErr: "tag processing errors: nil validator for tag \"nil:tag\"",
		},
		{
			name: "multiple errors",
			tags: []codetags.Tag{
				{Name: "valid:tag1"}, {Name: "valid:tag2"},
				{Name: "nil:tag1"}, {Name: "nil:tag2"},
				{Name: "unknown:tag1"}, {Name: "unknown:tag2"}, {Name: "unknown:tag3"},
			},
			setup: func(r *registry) {
				r.tagValidators["valid:tag1"] = &mockTagValidator{tagName: "valid:tag1", scopes: sets.New(ScopeType)}
				r.tagValidators["valid:tag2"] = &mockTagValidator{tagName: "valid:tag2", scopes: sets.New(ScopeType)}
				r.tagValidators["nil:tag1"], r.tagValidators["nil:tag2"] = nil, nil
			},
			wantErr: "tag processing errors: nil validator for tag \"nil:tag1\"; nil validator for tag \"nil:tag2\"; unknown tag \"unknown:tag1\"; unknown tag \"unknown:tag2\"; unknown tag \"unknown:tag3\"",
		},
		{
			name: "chained tag errors",
			tags: []codetags.Tag{
				{
					Name:     "valid:tag",
					ValueTag: &codetags.Tag{Name: "unknown:chained"},
				},
			},
			setup: func(r *registry) {
				r.tagValidators["valid:tag"] = &mockTagValidator{tagName: "valid:tag", scopes: sets.New(ScopeType)}
			},
			wantErr: "tag processing errors: unknown tag \"unknown:chained\"",
		},
		{
			name: "deeply chained tags",
			tags: []codetags.Tag{
				{
					Name: "valid:tag1",
					ValueTag: &codetags.Tag{
						Name: "valid:tag2",
						ValueTag: &codetags.Tag{
							Name: "unknown:deeply:chained",
						},
					},
				},
			},
			setup: func(r *registry) {
				r.tagValidators["valid:tag1"] = &mockTagValidator{tagName: "valid:tag1", scopes: sets.New(ScopeType)}
				r.tagValidators["valid:tag2"] = &mockTagValidator{tagName: "valid:tag2", scopes: sets.New(ScopeType)}
			},
			wantErr: "tag processing errors: unknown tag \"unknown:deeply:chained\"",
		},
		{
			name: "mixed chained tag errors",
			tags: []codetags.Tag{
				{Name: "unknown:top"},
				{
					Name:     "valid:tag",
					ValueTag: &codetags.Tag{Name: "nil:validator"},
				},
			},
			setup: func(r *registry) {
				r.tagValidators["valid:tag"] = &mockTagValidator{tagName: "valid:tag", scopes: sets.New(ScopeType)}
				r.tagValidators["nil:validator"] = nil
			},
			wantErr: "tag processing errors: unknown tag \"unknown:top\"; nil validator for tag \"nil:validator\"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &registry{tagValidators: map[string]TagValidator{}, initialized: atomic.Bool{}}
			r.initialized.Store(true)
			tt.setup(r)

			_, err := r.ExtractValidations(Context{Scope: ScopeType}, tt.tags...)

			if tt.wantErr == "" {
				if err != nil {
					t.Errorf("Expected no error, got %v", err)
				}
			} else if err == nil {
				t.Fatal("Expected error, got nil")
			} else if err.Error() != tt.wantErr {
				t.Errorf("Expected error %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}

// Mock implementations for testing
type mockTagValidator struct {
	tagName string
	scopes  sets.Set[Scope]
}

func (m *mockTagValidator) Init(cfg Config) {}
func (m *mockTagValidator) TagName() string { return m.tagName }
func (m *mockTagValidator) ValidScopes() sets.Set[Scope] {
	if m.scopes == nil {
		return sets.New(ScopeAny)
	}
	return m.scopes
}
func (m *mockTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	result := Validations{}
	// Create a simple validation function
	fn := Function(m.tagName, DefaultFlags, types.Name{Package: "test", Name: "MockValidator"})
	result.AddFunction(fn)
	return result, nil
}
func (m *mockTagValidator) Docs() TagDoc {
	return TagDoc{Tag: m.tagName}
}
