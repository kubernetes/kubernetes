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

package generic

import (
	"context"
	"testing"
)

// testHook is a minimal Hook implementation for testing.
type testHook struct {
	name string
}

// mockPolicySource implements Source[testHook] for testing.
type mockPolicySource struct {
	hooks     []testHook
	hasSynced bool
}

func (m *mockPolicySource) Hooks() []testHook {
	return m.hooks
}

func (m *mockPolicySource) Run(_ context.Context) error {
	return nil
}

func (m *mockPolicySource) HasSynced() bool {
	return m.hasSynced
}

var _ Source[testHook] = &mockPolicySource{}

func TestCompositePolicySource_Hooks(t *testing.T) {
	staticHook := testHook{name: "static-1"}
	apiHook := testHook{name: "api-1"}

	tests := []struct {
		name         string
		staticSource Source[testHook]
		apiSource    Source[testHook]
		wantNames    []string
	}{
		{
			name:         "only static source",
			staticSource: &mockPolicySource{hooks: []testHook{staticHook}, hasSynced: true},
			apiSource:    &mockPolicySource{hooks: nil, hasSynced: true},
			wantNames:    []string{"static-1"},
		},
		{
			name:         "only api source",
			staticSource: &mockPolicySource{hooks: nil, hasSynced: true},
			apiSource:    &mockPolicySource{hooks: []testHook{apiHook}, hasSynced: true},
			wantNames:    []string{"api-1"},
		},
		{
			name:         "both sources - static first",
			staticSource: &mockPolicySource{hooks: []testHook{staticHook}, hasSynced: true},
			apiSource:    &mockPolicySource{hooks: []testHook{apiHook}, hasSynced: true},
			wantNames:    []string{"static-1", "api-1"},
		},
		{
			name:         "nil static source",
			staticSource: nil,
			apiSource:    &mockPolicySource{hooks: []testHook{apiHook}, hasSynced: true},
			wantNames:    []string{"api-1"},
		},
		{
			name:         "both empty",
			staticSource: &mockPolicySource{hooks: nil, hasSynced: true},
			apiSource:    &mockPolicySource{hooks: nil, hasSynced: true},
			wantNames:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := NewCompositePolicySource[testHook](tt.staticSource, tt.apiSource)
			hooks := source.Hooks()

			if len(hooks) != len(tt.wantNames) {
				t.Errorf("Hooks() returned %d hooks, want %d", len(hooks), len(tt.wantNames))
				return
			}

			for i, h := range hooks {
				if h.name != tt.wantNames[i] {
					t.Errorf("Hooks()[%d].name = %s, want %s", i, h.name, tt.wantNames[i])
				}
			}
		})
	}
}

func TestCompositePolicySource_Caching(t *testing.T) {
	staticSource := &mockPolicySource{hooks: []testHook{{name: "static-1"}}, hasSynced: true}
	apiSource := &mockPolicySource{hooks: []testHook{{name: "api-1"}}, hasSynced: true}

	source := NewCompositePolicySource[testHook](staticSource, apiSource)

	// First call should create the combined slice.
	hooks1 := source.Hooks()
	if len(hooks1) != 2 {
		t.Fatalf("Expected 2 hooks, got %d", len(hooks1))
	}

	// Second call with same underlying slices should return the cached combined slice.
	hooks2 := source.Hooks()
	if &hooks1[0] != &hooks2[0] {
		t.Error("Expected cached slice to be returned when underlying slices haven't changed")
	}

	// Changing the underlying slice should produce a new combined slice.
	staticSource.hooks = []testHook{{name: "static-2"}}
	hooks3 := source.Hooks()
	if hooks3[0].name != "static-2" {
		t.Errorf("Expected first hook to be static-2 after update, got %s", hooks3[0].name)
	}
	if &hooks2[0] == &hooks3[0] {
		t.Error("Expected new slice after underlying source changed")
	}
}

func TestCompositePolicySource_HasSynced(t *testing.T) {
	tests := []struct {
		name         string
		staticSource Source[testHook]
		apiSource    Source[testHook]
		want         bool
	}{
		{
			name:         "both synced",
			staticSource: &mockPolicySource{hasSynced: true},
			apiSource:    &mockPolicySource{hasSynced: true},
			want:         true,
		},
		{
			name:         "static not synced",
			staticSource: &mockPolicySource{hasSynced: false},
			apiSource:    &mockPolicySource{hasSynced: true},
			want:         false,
		},
		{
			name:         "api not synced",
			staticSource: &mockPolicySource{hasSynced: true},
			apiSource:    &mockPolicySource{hasSynced: false},
			want:         false,
		},
		{
			name:         "neither synced",
			staticSource: &mockPolicySource{hasSynced: false},
			apiSource:    &mockPolicySource{hasSynced: false},
			want:         false,
		},
		{
			name:         "nil static source, api synced",
			staticSource: nil,
			apiSource:    &mockPolicySource{hasSynced: true},
			want:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := NewCompositePolicySource[testHook](tt.staticSource, tt.apiSource)
			if got := source.HasSynced(); got != tt.want {
				t.Errorf("HasSynced() = %v, want %v", got, tt.want)
			}
		})
	}
}
