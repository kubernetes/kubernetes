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

package main

import (
	"errors"
	"testing"

	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

func ruleAlwaysPass(container *types.Type, t *types.Type, tags []codetags.Tag) (string, error) {
	return "", nil
}

func ruleAlwaysFail(container *types.Type, t *types.Type, tags []codetags.Tag) (string, error) {
	return "lintfail", nil
}

func ruleAlwaysErr(container *types.Type, t *types.Type, tags []codetags.Tag) (string, error) {
	return "", errors.New("linterr")
}

func mkCountRule(counter *int, realRule lintRule) lintRule {
	return func(container *types.Type, t *types.Type, tags []codetags.Tag) (string, error) {
		(*counter)++
		return realRule(container, t, tags)
	}
}

var validator = validators.InitGlobalValidator(&generator.Context{})

func TestLintCommentsRuleInvocation(t *testing.T) {
	tests := []struct {
		name              string
		rules             []lintRule
		commentLineGroups [][]string
		wantErr           bool
		wantCount         int
	}{
		{
			name:              "0 rules, 0 comments",
			rules:             []lintRule{},
			commentLineGroups: [][]string{},
			wantErr:           false,
			wantCount:         0,
		},
		{
			name:              "1 rule, 1 comment",
			rules:             []lintRule{ruleAlwaysPass},
			commentLineGroups: [][]string{{"comment"}},
			wantErr:           false,
			wantCount:         1,
		},
		{
			name:              "3 rules, 3 comments",
			rules:             []lintRule{ruleAlwaysPass, ruleAlwaysFail, ruleAlwaysErr},
			commentLineGroups: [][]string{{"comment1"}, {"comment2"}, {"comment3"}},
			wantErr:           true,
			wantCount:         9,
		},
		{
			name:              "1 rule, 1 comment, rule fails",
			rules:             []lintRule{ruleAlwaysFail},
			commentLineGroups: [][]string{{"comment"}},
			wantErr:           false,
			wantCount:         1,
		},
		{
			name:              "1 rule, 1 comment, rule errors",
			rules:             []lintRule{ruleAlwaysErr},
			commentLineGroups: [][]string{{"comment"}},
			wantErr:           true,
			wantCount:         1,
		},
		{
			name:              "3 rules, 1 comment, rule errors in the middle",
			rules:             []lintRule{ruleAlwaysPass, ruleAlwaysErr, ruleAlwaysFail},
			commentLineGroups: [][]string{{"comment"}},
			wantErr:           true,
			wantCount:         2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			counter := 0
			rules := make([]lintRule, len(tt.rules))
			for i, rule := range tt.rules {
				rules[i] = mkCountRule(&counter, rule)
			}
			l := newLinter(rules...)
			for _, commentLines := range tt.commentLineGroups {
				_, err := l.lintComments(nil, nil, commentLines)
				gotErr := err != nil
				if gotErr != tt.wantErr {
					t.Errorf("lintComments() error = %v, wantErr %v", err, tt.wantErr)
				}
			}
			if counter != tt.wantCount {
				t.Errorf("expected %d rule invocations, got %d", tt.wantCount, counter)
			}
		})
	}
}

func TestRuleAlphaBetaPrefix(t *testing.T) {
	tests := []struct {
		name     string
		comments []string
		wantMsg  string
	}{
		{
			name:     "valid alpha prefix",
			comments: []string{"+k8s:alpha=+k8s:required"},
			wantMsg:  "",
		},
		{
			name:     "valid beta prefix",
			comments: []string{"+k8s:beta=+k8s:required"},
			wantMsg:  "",
		},
		{
			name:     "invalid alpha prefix (no value)",
			comments: []string{"+k8s:alpha"},
			wantMsg:  `tag "k8s:alpha" requires a validation tag as its value payload`,
		},
		{
			name:     "invalid beta prefix (no value)",
			comments: []string{"+k8s:beta"},
			wantMsg:  `tag "k8s:beta" requires a validation tag as its value payload`,
		},
		{
			name:     "invalid alpha prefix (value not tag)",
			comments: []string{"+k8s:alpha=foo"},
			wantMsg:  `tag "k8s:alpha" requires a validation tag as its value payload`,
		},
		{
			name:     "invalid usage of alpha prefix",
			comments: []string{`+k8s:item(type: "Approved")=+k8s:alpha=+k8s:zeroOrOneOfMember`},
			wantMsg:  `tag "k8s:alpha" can't be used in between`,
		},
		{
			name:     "nested alpha in item",
			comments: []string{`+k8s:item=+k8s:alpha=+k8s:required`},
			wantMsg:  `tag "k8s:alpha" can't be used in between`,
		},
		{
			name:     "alpha nested in listType",
			comments: []string{`+k8s:listType=+k8s:alpha=+k8s:required`},
			wantMsg:  `tag "k8s:alpha" can't be used in between`,
		},
		{
			name:     "deeply nested alpha",
			comments: []string{`+k8s:item=+k8s:listType=+k8s:alpha=+k8s:required`},
			wantMsg:  `tag "k8s:alpha" can't be used in between`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tags, _ := validator.ExtractTags(validators.Context{}, tt.comments)
			msg, err := alphaBetaPrefix()(nil, nil, tags)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			} else if msg != tt.wantMsg {
				t.Errorf("got %q, want %q", msg, tt.wantMsg)
			}
		})
	}
}

func TestRuleStability(t *testing.T) {
	tests := []struct {
		name     string
		comments []string
		wantMsg  string
	}{
		{
			name:     "stable context, stable tag",
			comments: []string{"+k8s:required"}, // Stable
			wantMsg:  "",
		},
		{
			name:     "beta context, stable tag",
			comments: []string{"+k8s:beta=+k8s:required"}, // Beta context, Stable tag
			wantMsg:  "",
		},
		{
			name:     "alpha context, stable tag",
			comments: []string{"+k8s:alpha=+k8s:required"}, // Alpha context, Stable tag
			wantMsg:  "",
		},
		{
			name:     "alpha context, alpha tag",
			comments: []string{"+k8s:alpha=+k8s:forbidden"}, // Alpha context, Alpha tag
			wantMsg:  "",
		},
		{
			name:     "stable context, alpha tag",
			comments: []string{"+k8s:forbidden"}, // Stable context, Alpha tag
			wantMsg:  `tag "k8s:forbidden" with stability level "Alpha" cannot be used in Stable validation`,
		},
		{
			name:     "beta context, alpha tag",
			comments: []string{"+k8s:beta=+k8s:forbidden"}, // Beta context, Alpha tag
			wantMsg:  `tag "k8s:forbidden" with stability level "Alpha" cannot be used in Beta validation`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dummyType := &types.Type{Name: types.Name{Name: "Dummy"}}
			rule := validationStability()
			tags, _ := validator.ExtractTags(validators.Context{}, tt.comments)
			msg, err := rule(nil, dummyType, tags)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			} else if msg != tt.wantMsg {
				t.Errorf("got %q, want %q", msg, tt.wantMsg)
			}
		})
	}
}

func TestLintType(t *testing.T) {
	tests := []struct {
		name        string
		typeToLint  *types.Type
		wantCount   int
		expectError bool
	}{
		{
			name: "No comments",
			typeToLint: &types.Type{
				Name:         types.Name{Package: "testpkg", Name: "TestType"},
				CommentLines: nil,
			},
			wantCount:   0,
			expectError: false,
		},
		{
			name: "Valid comments",
			typeToLint: &types.Type{
				Name:         types.Name{Package: "testpkg", Name: "TestType"},
				CommentLines: []string{"+k8s:optional"},
			},
			wantCount:   1,
			expectError: false,
		},
		{
			name: "Pointer type",
			typeToLint: &types.Type{
				Name:         types.Name{Package: "testpkg", Name: "TestPointer"},
				Kind:         types.Pointer,
				Elem:         &types.Type{Name: types.Name{Package: "testpkg", Name: "ElemType"}, CommentLines: []string{"+k8s:optional"}},
				CommentLines: []string{"+k8s:optional"},
			},
			wantCount:   2,
			expectError: false,
		},
		{
			name: "Slice of pointers",
			typeToLint: &types.Type{
				Name: types.Name{Package: "testpkg", Name: "TestSlice"},
				Kind: types.Slice,
				Elem: &types.Type{
					Name:         types.Name{Package: "testpkg", Name: "PointerElem"},
					Kind:         types.Pointer,
					Elem:         &types.Type{Name: types.Name{Package: "testpkg", Name: "ElemType"}, CommentLines: []string{"+k8s:optional"}},
					CommentLines: []string{"+k8s:optional"},
				},
				CommentLines: []string{"+k8s:optional"},
			},
			wantCount:   3,
			expectError: false,
		},
		{
			name: "Map to pointers",
			typeToLint: &types.Type{
				Name: types.Name{Package: "testpkg", Name: "TestMap"},
				Kind: types.Map,
				Key:  &types.Type{Name: types.Name{Package: "testpkg", Name: "KeyType"}, CommentLines: []string{"+k8s:required"}},
				Elem: &types.Type{
					Name:         types.Name{Package: "testpkg", Name: "PointerElem"},
					Kind:         types.Pointer,
					Elem:         &types.Type{Name: types.Name{Package: "testpkg", Name: "ElemType"}, CommentLines: []string{"+k8s:optional"}},
					CommentLines: []string{"+k8s:optional"},
				},
				CommentLines: []string{"+k8s:optional"},
			},
			wantCount:   4,
			expectError: false,
		},
		{
			name: "Alias to pointers",
			typeToLint: &types.Type{
				Name: types.Name{Package: "testpkg", Name: "TestAlias"},
				Kind: types.Alias,
				Underlying: &types.Type{
					Name:         types.Name{Package: "testpkg", Name: "PointerElem"},
					Kind:         types.Pointer,
					Elem:         &types.Type{Name: types.Name{Package: "testpkg", Name: "ElemType"}, CommentLines: []string{"+k8s:optional"}},
					CommentLines: []string{"+k8s:optional"},
				},
				CommentLines: []string{"+k8s:optional"},
			},
			wantCount:   3,
			expectError: false,
		},
		{
			name: "Struct with members",
			typeToLint: &types.Type{
				Name: types.Name{Package: "testpkg", Name: "TestStruct"},
				Kind: types.Struct,
				Members: []types.Member{
					{
						Name:         "Field1",
						Type:         &types.Type{Name: types.Name{Package: "testpkg", Name: "FieldType"}},
						CommentLines: []string{"+k8s:optional"},
					},
					{
						Name:         "Field2",
						Type:         &types.Type{Name: types.Name{Package: "testpkg", Name: "FieldType"}},
						CommentLines: []string{"+k8s:required"},
					},
				},
			},
			wantCount:   2,
			expectError: false,
		},
		{
			name: "Nested types",
			typeToLint: &types.Type{
				Name: types.Name{Package: "testpkg", Name: "TestStruct"},
				Kind: types.Struct,
				Members: []types.Member{
					{
						Name: "Field1",
						Type: &types.Type{
							Name:         types.Name{Package: "testpkg", Name: "NestedStruct"},
							Kind:         types.Struct,
							CommentLines: []string{"+k8s:optional"},
							Members: []types.Member{
								{
									Name:         "NestedField1",
									Type:         &types.Type{Name: types.Name{Package: "testpkg", Name: "NestedFieldType"}},
									CommentLines: []string{"+k8s:required"},
								},
							},
						},
					},
				},
			},
			wantCount:   3,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			counter := 0
			rules := []lintRule{mkCountRule(&counter, ruleAlwaysPass)}
			l := newLinter(rules...)
			if err := l.lintType(tt.typeToLint); err != nil {
				t.Fatal(err)
			}
			gotErr := len(l.lintErrors) > 0
			if gotErr != tt.expectError {
				t.Errorf("LintType() errors = %v, expectError %v", l.lintErrors, tt.expectError)
			}
			if counter != tt.wantCount {
				t.Errorf("expected %d rule invocations, got %d", tt.wantCount, counter)
			}
		})
	}
}

func TestHasAnyValidationTag(t *testing.T) {
	tests := []struct {
		name     string
		comments []string
		want     bool
	}{
		{
			name:     "empty",
			comments: []string{},
			want:     false,
		},
		{
			name:     "no k8s tags",
			comments: []string{"just a comment"},
			want:     false,
		},
		{
			name:     "optional only",
			comments: []string{"+k8s:optional"},
			want:     false,
		},
		{
			name:     "required only",
			comments: []string{"+k8s:required"},
			want:     true,
		},
		{
			name:     "forbidden only",
			comments: []string{"+k8s:forbidden"},
			want:     true,
		},
		{
			name:     "unrecognized k8s tag",
			comments: []string{"+k8s:openapi-gen=true"},
			want:     false,
		},
		{
			name:     "minimum tag",
			comments: []string{"+k8s:minimum=0"},
			want:     true,
		},
		{
			name:     "enum tag",
			comments: []string{"+k8s:enum"},
			want:     true,
		},
		{
			name:     "mixed with optional",
			comments: []string{"+k8s:optional", "+k8s:minimum=0"},
			want:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tags, _ := validator.ExtractTags(validators.Context{}, tt.comments)
			if got := hasAnyValidationTag(tags); got != tt.want {
				t.Errorf("hasAnyValidationTag() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHasRequirednessTag(t *testing.T) {
	tests := []struct {
		name     string
		comments []string
		want     bool
	}{
		{
			name:     "empty",
			comments: []string{},
			want:     false,
		},
		{
			name:     "no requireness",
			comments: []string{"+k8s:minimum=0"},
			want:     false,
		},
		{
			name:     "optional",
			comments: []string{"+k8s:optional"},
			want:     true,
		},
		{
			name:     "required",
			comments: []string{"+k8s:required"},
			want:     true,
		},
		{
			name:     "optional with value",
			comments: []string{"+k8s:optional=true"},
			want:     true,
		},
		{
			name:     "conditional optional",
			comments: []string{`+k8s:alpha(since:"1.35")=+k8s:optional`},
			want:     true,
		},
		{
			name:     "conditional required",
			comments: []string{`+k8s:alpha(since:"1.35")=+k8s:required`},
			want:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tags, _ := validator.ExtractTags(validators.Context{}, tt.comments)
			if got := hasRequirednessTag(tags); got != tt.want {
				t.Errorf("hasRequirednessTag() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLintRequiredness(t *testing.T) {
	tests := []struct {
		name       string
		typeToLint *types.Type
		wantError  string
	}{
		{
			name: "pointer field without validation - no error",
			typeToLint: &types.Type{
				Name: types.Name{Package: "pkg", Name: "T"},
				Kind: types.Struct,
				Members: []types.Member{{
					Name: "Foo",
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "Bar"},
						Kind: types.Pointer,
						Elem: &types.Type{Name: types.Name{Package: "", Name: "string"}},
					},
				}},
			},
			wantError: "",
		},
		{
			name: "pointer field with direct validation, no requireness - error",
			typeToLint: &types.Type{
				Name: types.Name{Package: "pkg", Name: "T"},
				Kind: types.Struct,
				Members: []types.Member{{
					Name:         "Foo",
					CommentLines: []string{"+k8s:minimum=0"},
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "Bar"},
						Kind: types.Pointer,
						Elem: &types.Type{Name: types.Name{Package: "", Name: "int"}},
					},
				}},
			},
			wantError: "field Foo: field with validation must have +k8s:optional, +k8s:required or +k8s:forbidden",
		},
		{
			name: "pointer field with transitive validation, no requireness - error",
			typeToLint: &types.Type{
				Name: types.Name{Package: "pkg", Name: "T"},
				Kind: types.Struct,
				Members: []types.Member{{
					Name: "Foo",
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "Nested"},
						Kind: types.Pointer,
						Elem: &types.Type{
							Name: types.Name{Package: "pkg", Name: "Inner"},
							Kind: types.Struct,
							Members: []types.Member{{
								Name:         "Bar",
								CommentLines: []string{"+k8s:minimum=0"},
								Type:         &types.Type{Name: types.Name{Package: "", Name: "int"}},
							}},
						},
					},
				}},
			},
			wantError: "field Foo: field with validation must have +k8s:optional, +k8s:required or +k8s:forbidden",
		},
		{
			name: "pointer field with validation and +k8s:optional - no error",
			typeToLint: &types.Type{
				Name: types.Name{Package: "pkg", Name: "T"},
				Kind: types.Struct,
				Members: []types.Member{{
					Name:         "Foo",
					CommentLines: []string{"+k8s:optional", "+k8s:minimum=0"},
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "Bar"},
						Kind: types.Pointer,
						Elem: &types.Type{Name: types.Name{Package: "", Name: "int"}},
					},
				}},
			},
			wantError: "",
		},
		{
			name: "slice field with validation, no requireness - error",
			typeToLint: &types.Type{
				Name: types.Name{Package: "pkg", Name: "T"},
				Kind: types.Struct,
				Members: []types.Member{{
					Name:         "Items",
					CommentLines: []string{"+k8s:maxItems=10"},
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "List"},
						Kind: types.Slice,
						Elem: &types.Type{Name: types.Name{Package: "", Name: "string"}},
					},
				}},
			},
			wantError: "field Items: field with validation must have +k8s:optional, +k8s:required or +k8s:forbidden",
		},
		{
			name: "map field with validation, no requireness - error",
			typeToLint: &types.Type{
				Name: types.Name{Package: "pkg", Name: "T"},
				Kind: types.Struct,
				Members: []types.Member{{
					Name:         "Data",
					CommentLines: []string{"+k8s:maxItems=5"},
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "M"},
						Kind: types.Map,
						Key:  &types.Type{Name: types.Name{Package: "", Name: "string"}},
						Elem: &types.Type{Name: types.Name{Package: "", Name: "string"}},
					},
				}},
			},
			wantError: "field Data: field with validation must have +k8s:optional, +k8s:required or +k8s:forbidden",
		},
		{
			name: "non-pointer struct field with validation - no error (exempt)",
			typeToLint: &types.Type{
				Name: types.Name{Package: "pkg", Name: "T"},
				Kind: types.Struct,
				Members: []types.Member{{
					Name: "Nested",
					Type: &types.Type{
						Name:         types.Name{Package: "pkg", Name: "Inner"},
						Kind:         types.Struct,
						CommentLines: []string{"+k8s:minimum=0"},
					},
				}},
			},
			wantError: "",
		},
		{
			name: "recursive type with pointer to self - no infinite loop",
			typeToLint: func() *types.Type {
				t := &types.Type{
					Name: types.Name{Package: "pkg", Name: "Node"},
					Kind: types.Struct,
				}
				t.Members = []types.Member{{
					Name:         "Next",
					CommentLines: []string{"+k8s:optional"},
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "NodePtr"},
						Kind: types.Pointer,
						Elem: t, // cycle
					},
				}}
				return t
			}(),
			wantError: "",
		},
		{
			name: "recursive type with pointer to self - no infinite loop, missing required validation",
			typeToLint: func() *types.Type {
				t := &types.Type{
					Name: types.Name{Package: "pkg", Name: "Node"},
					Kind: types.Struct,
				}
				t.Members = []types.Member{{
					Name:         "Next",
					CommentLines: []string{"+k8s:immutable"},
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "NodePtr"},
						Kind: types.Pointer,
						Elem: t, // cycle
					},
				}}
				return t
			}(),
			wantError: "field Next: field with validation must have +k8s:optional, +k8s:required or +k8s:forbidden",
		},
		{
			name: "recursive type with validation - detects validation on first visit",
			typeToLint: func() *types.Type {
				t := &types.Type{
					Name:         types.Name{Package: "pkg", Name: "Node"},
					Kind:         types.Struct,
					CommentLines: []string{"+k8s:immutable"},
				}
				t.Members = []types.Member{{
					Name:         "Next",
					CommentLines: []string{"+k8s:optional"},
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "NodePtr"},
						Kind: types.Pointer,
						Elem: t, // cycle
					},
				}}
				return t
			}(),
			wantError: "",
		},
		{
			name: "array field with transitive validation, no requiredNess - error",
			typeToLint: &types.Type{
				Name: types.Name{Package: "pkg", Name: "T"},
				Kind: types.Struct,
				Members: []types.Member{{
					Name: "Arr",
					Type: &types.Type{
						Name: types.Name{Package: "pkg", Name: "ArrType"},
						Kind: types.Array,
						Elem: &types.Type{
							Name:         types.Name{Package: "pkg", Name: "Inner"},
							Kind:         types.Struct,
							CommentLines: []string{"+k8s:enum"},
						},
					},
				}},
			},
			wantError: "field Arr: field with validation must have +k8s:optional, +k8s:required or +k8s:forbidden",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := newLinter(requiredAndOptional(validator))
			if err := l.lintType(tt.typeToLint); err != nil {
				t.Fatalf("lintType() unexpected error: %v", err)
			}
			errs := l.lintErrors[tt.typeToLint]
			if len(errs) > 1 {
				t.Fatalf("got %d errors, but expected 0 or 1 error: %v", len(errs), errs)
			}
			var gotError string
			if len(errs) == 1 {
				gotError = errs[0].Error()
			}
			if gotError != tt.wantError {
				t.Errorf("lintRequiredness() error = %q, want %q", gotError, tt.wantError)
			}
		})
	}
}
