/*
Copyright 2026 The Kubernetes Authors.

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

package generators

import (
	"bytes"
	"strings"
	"testing"

	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestGenerateWithFuncsShadowedField(t *testing.T) {
	const pkg = "example.com/api/v1"

	base := &types.Type{
		Name: types.Name{Package: pkg, Name: "Base"},
		Kind: types.Struct,
		Members: []types.Member{{
			Name: "Enabled",
			Type: types.Bool,
			Tags: `json:"enabled,omitempty"`,
		}},
	}
	embeddedBase := types.Member{
		Name:     "Base",
		Embedded: true,
		Type:     base,
		Tags:     `json:",inline"`,
	}
	directEnabled := types.Member{
		Name: "Enabled",
		Type: types.Bool,
		Tags: `json:"enabled,omitempty"`,
	}

	// Middle promotes Base.Enabled without declaring it, so a direct field on
	// an outer type must still shadow the field promoted through Middle.
	middle := &types.Type{
		Name:    types.Name{Package: pkg, Name: "Middle"},
		Kind:    types.Struct,
		Members: []types.Member{embeddedBase},
	}
	embeddedMiddle := types.Member{
		Name:     "Middle",
		Embedded: true,
		Type:     middle,
		Tags:     `json:",inline"`,
	}

	// MiddleOverride declares Enabled itself and promotes Base.Enabled, so its
	// direct field shadows Base.Enabled even when an outer type embeds it.
	middleOverride := &types.Type{
		Name:    types.Name{Package: pkg, Name: "MiddleOverride"},
		Kind:    types.Struct,
		Members: []types.Member{embeddedBase, directEnabled},
	}
	embeddedMiddleOverride := types.Member{
		Name:     "MiddleOverride",
		Embedded: true,
		Type:     middleOverride,
		Tags:     `json:",inline"`,
	}
	ignoredEnabled := types.Member{
		Name: "Enabled",
		Type: types.Bool,
		Tags: `json:"-"`,
	}

	tests := []struct {
		name            string
		members         []types.Member
		wantContains    string
		wantNotContains string
	}{
		{
			name:            "embedded field before direct field",
			members:         []types.Member{embeddedBase, directEnabled},
			wantContains:    "b.Enabled = &value",
			wantNotContains: "b.BaseApplyConfiguration.Enabled = &value",
		},
		{
			name:            "direct field before embedded field",
			members:         []types.Member{directEnabled, embeddedBase},
			wantContains:    "b.Enabled = &value",
			wantNotContains: "b.BaseApplyConfiguration.Enabled = &value",
		},
		{
			name:            "embedded field without a conflicting direct field",
			members:         []types.Member{embeddedBase},
			wantContains:    "b.BaseApplyConfiguration.Enabled = &value",
			wantNotContains: "b.Enabled = &value",
		},
		{
			name:            "ignored direct field does not shadow the embedded field",
			members:         []types.Member{embeddedBase, ignoredEnabled},
			wantContains:    "b.BaseApplyConfiguration.Enabled = &value",
			wantNotContains: "b.Enabled = &value",
		},
		{
			name:            "nested embedded field without a conflicting direct field",
			members:         []types.Member{embeddedMiddle},
			wantContains:    "b.BaseApplyConfiguration.Enabled = &value",
			wantNotContains: "b.Enabled = &value",
		},
		{
			name:            "outer direct field shadows a nested promoted field",
			members:         []types.Member{embeddedMiddle, directEnabled},
			wantContains:    "b.Enabled = &value",
			wantNotContains: "b.BaseApplyConfiguration.Enabled = &value",
		},
		{
			name:            "intermediate direct field shadows a nested promoted field",
			members:         []types.Member{embeddedMiddleOverride},
			wantContains:    "b.MiddleOverrideApplyConfiguration.Enabled = &value",
			wantNotContains: "b.BaseApplyConfiguration.Enabled = &value",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			outer := &types.Type{
				Name:    types.Name{Package: pkg, Name: "Override"},
				Kind:    types.Struct,
				Members: tc.members,
			}
			applyConfigType := &types.Type{
				Name: types.Name{Package: pkg, Name: "OverrideApplyConfiguration"},
				Kind: types.Struct,
			}
			g := &applyConfigurationGenerator{
				refGraph: refGraph{
					base.Name:           pkg,
					middle.Name:         pkg,
					middleOverride.Name: pkg,
				},
			}
			c := &generator.Context{Namers: namer.NameSystems{
				"public": namer.NewPublicNamer(0),
				"raw":    namer.NewRawNamer(pkg, nil),
			}}
			var out bytes.Buffer
			sw := generator.NewSnippetWriter(&out, c, "$", "$")
			typeParams := TypeParams{
				Struct: outer,
				ApplyConfig: applyConfig{
					Type:               outer,
					ApplyConfiguration: applyConfigType,
				},
			}
			g.generateWithFuncs(outer, typeParams, sw, nil, &[]string{}, sets.New[string]())
			if err := sw.Error(); err != nil {
				t.Fatal(err)
			}
			got := out.String()
			if !strings.Contains(got, tc.wantContains) {
				t.Errorf("generated code does not contain %q:\n%s", tc.wantContains, got)
			}
			if strings.Contains(got, tc.wantNotContains) {
				t.Errorf("generated code contains %q:\n%s", tc.wantNotContains, got)
			}
		})
	}
}
