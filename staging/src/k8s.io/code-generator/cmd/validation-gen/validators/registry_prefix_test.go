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

package validators

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

// fakeTagValidator records the Config it was initialized with and emits one
// function call per use.
type fakeTagValidator struct {
	name string
	cfg  Config
}

func (f *fakeTagValidator) Init(cfg Config) { f.cfg = cfg }
func (f *fakeTagValidator) TagName() string { return f.name }
func (f *fakeTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeField)
}
func (f *fakeTagValidator) GetValidations(_ Context, tag codetags.Tag) (Validations, error) {
	fn := Function(f.name, DefaultFlags, types.Name{Package: "example.com/validate", Name: "Fake"}, tag.Value)
	return Validations{Functions: []FunctionGen{fn}}, nil
}
func (f *fakeTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            f.name,
		StabilityLevel: TagStabilityLevelStable,
		PayloadsType:   codetags.ValueTypeString,
	}
}

func TestRegistryTagPrefix(t *testing.T) {
	cases := []struct {
		prefix string
		tag    string
	}{
		{prefix: "k8s:", tag: "k8s:fake"},
		{prefix: "xyz:", tag: "xyz:fake"},
		{prefix: "a:b:", tag: "a:b:fake"},
		{prefix: "", tag: "fake"},
	}
	for _, tc := range cases {
		t.Run(tc.prefix, func(t *testing.T) {
			tv := &fakeTagValidator{name: "fake"}
			reg := &registry{}
			reg.addTagValidator(tv)
			reg.init(&generator.Context{}, nil, tc.prefix)

			if got := tv.cfg.TagPrefix; got != tc.prefix {
				t.Errorf("Config.TagPrefix = %q, want %q", got, tc.prefix)
			}
			if !reg.IsKnownTag(tc.tag) {
				t.Errorf("IsKnownTag(%q) = false, want true", tc.tag)
			}
			for _, other := range []string{"fake", "k8s:fake", "xyz:fake", "a:b:fake"} {
				if other != tc.tag && reg.IsKnownTag(other) {
					t.Errorf("IsKnownTag(%q) = true, want false", other)
				}
			}
			if got, err := reg.Stability(tc.tag); err != nil || got != TagStabilityLevelStable {
				t.Errorf("Stability(%q) = %q, %v; want Stable, nil", tc.tag, got, err)
			}
			docs := reg.Docs()
			if len(docs) != 1 || docs[0].Tag != tc.tag {
				t.Errorf("Docs() = %+v, want one doc for %q", docs, tc.tag)
			}

			ctx := Context{Scope: ScopeField}
			comments := []string{
				"+" + tc.tag + `="mine"`,
				`+other:fake="not mine"`,
				`+fake:="not a tag"`,
			}
			tags, err := reg.ExtractTags(ctx, comments)
			if err != nil {
				t.Fatalf("ExtractTags() error: %v", err)
			}
			if len(tags) != 1 || tags[0].Name != tc.tag || tags[0].Value != "mine" {
				t.Fatalf("ExtractTags() = %+v, want one %q tag with value \"mine\"", tags, tc.tag)
			}
			validations, err := reg.ExtractValidations(ctx, tags...)
			if err != nil {
				t.Fatalf("ExtractValidations() error: %v", err)
			}
			if len(validations.Functions) != 1 || validations.Functions[0].Args[0] != "mine" {
				t.Errorf("ExtractValidations() = %+v, want one function with arg \"mine\"", validations)
			}
		})
	}
}

func TestRegistryUnknownNestedTag(t *testing.T) {
	reg := &registry{}
	reg.addTagValidator(&fakeTagValidator{name: "fake"})
	reg.init(&generator.Context{}, nil, "xyz:")

	// A nested tag is only checked when its validations are extracted.
	nested := codetags.Tag{Name: "xyz:missing", ValueType: codetags.ValueTypeNone}
	_, err := reg.ExtractTagValidations(Context{Scope: ScopeField}, nested)
	if err == nil || !strings.Contains(err.Error(), `unknown validation tag "xyz:missing"`) {
		t.Errorf("ExtractTagValidations() error = %v, want unknown validation tag", err)
	}
}

func TestRegistryDuplicateTag(t *testing.T) {
	reg := &registry{}
	reg.addTagValidator(&fakeTagValidator{name: "fake"})
	reg.addTagValidator(&fakeTagValidator{name: "fake"})
	defer func() {
		if r := recover(); r == nil || !strings.Contains(r.(string), `"xyz:fake" was registered twice`) {
			t.Errorf("init() panic = %v, want registered twice", r)
		}
	}()
	reg.init(&generator.Context{}, nil, "xyz:")
}
