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

	"k8s.io/gengo/v2/types"
)

func ruleAlwaysPass(comments []string) (string, error) {
	return "", nil
}

func ruleAlwaysFail(comments []string) (string, error) {
	return "lintfail", nil
}

func ruleAlwaysErr(comments []string) (string, error) {
	return "", errors.New("linterr")
}

func mkCountRule(counter *int, realRule lintRule) lintRule {
	return func(comments []string) (string, error) {
		(*counter)++
		return realRule(comments)
	}
}

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
				_, err := l.lintComments(commentLines)
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

func TestRuleOptionalAndRequired(t *testing.T) {
	tests := []struct {
		name     string
		comments []string
		wantMsg  string
		wantErr  bool
	}{
		{
			name:     "no comments",
			comments: []string{},
			wantMsg:  "",
		},
		{
			name:     "only optional",
			comments: []string{"+k8s:optional"},
			wantMsg:  "",
		},
		{
			name:     "only required",
			comments: []string{"+k8s:required"},
			wantMsg:  "",
		},
		{
			name:     "optional and required",
			comments: []string{"+k8s:optional", "+k8s:required"},
			wantMsg:  "conflicting tags: {+k8s:optional, +k8s:required}",
		},
		{
			name:     "optional, empty, required",
			comments: []string{"+k8s:optional", "", "+k8s:required"},
			wantMsg:  "conflicting tags: {+k8s:optional, +k8s:required}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg, _ := ruleOptionalAndRequired(tt.comments)
			if msg != tt.wantMsg {
				t.Errorf("ruleOptionalAndRequired() msg = %v, wantMsg %v", msg, tt.wantMsg)
			}
		})
	}
}

func TestRuleRequiredAndDefault(t *testing.T) {
	tests := []struct {
		name     string
		comments []string
		wantMsg  string
	}{
		{
			name:     "no comments",
			comments: []string{},
			wantMsg:  "",
		},
		{
			name:     "only required",
			comments: []string{"+k8s:required"},
			wantMsg:  "",
		},
		{
			name:     "only default",
			comments: []string{"+default=somevalue"},
			wantMsg:  "",
		},
		{
			name:     "required and default",
			comments: []string{"+k8s:required", "+default=somevalue"},
			wantMsg:  "conflicting tags: {+k8s:required, +default}",
		},
		{
			name:     "required, empty, default",
			comments: []string{"+k8s:required", "", "+default=somevalue"},
			wantMsg:  "conflicting tags: {+k8s:required, +default}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg, _ := ruleRequiredAndDefault(tt.comments)
			if msg != tt.wantMsg {
				t.Errorf("ruleRequiredAndDefault() msg = %v, wantMsg %v", msg, tt.wantMsg)
			}
		})
	}
}

func TestConflictingTagsRule(t *testing.T) {
	tests := []struct {
		name     string
		comments []string
		tags     []string
		wantMsg  string
		wantErr  bool
	}{
		{
			name:     "no comments",
			comments: []string{},
			tags:     []string{"+tag1", "+tag2"},
			wantMsg:  "",
		},
		{
			name:     "only tag1",
			comments: []string{"+tag1"},
			tags:     []string{"+tag1", "+tag2"},
			wantMsg:  "",
		},
		{
			name:     "tag1, empty, tag2",
			comments: []string{"+tag1", "", "+tag2"},
			tags:     []string{"+tag1", "+tag2"},
			wantMsg:  "conflicting tags: {+tag1, +tag2}",
		},
		{
			name:     "3 tags",
			comments: []string{"tag1", "+tag2", "+tag3=value"},
			tags:     []string{"+tag1", "+tag2", "+tag3"},
			wantMsg:  "conflicting tags: {+tag1, +tag2, +tag3}",
		},
		{
			name:     "less than 2 tags",
			comments: []string{"+tag1"},
			tags:     []string{"+tag1"},
			wantMsg:  "",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg, err := conflictingTagsRule(tt.comments, tt.tags...)
			if (err != nil) != tt.wantErr {
				t.Errorf("conflictingTagsRule() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if msg != tt.wantMsg {
				t.Errorf("conflictingTagsRule() msg = %v, wantMsg %v", msg, tt.wantMsg)
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
