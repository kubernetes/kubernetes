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

package apidefinitions

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/gengo/v2/types"
)

func TestLintRules(t *testing.T) {
	pkg := func(path string, comments ...string) *types.Package {
		return &types.Package{Path: path, Dir: "/tmp/" + path, Comments: comments}
	}
	pkgWithTypeMeta := func(path, jsonTag string, comments ...string) *types.Package {
		return &types.Package{
			Path: path, Dir: "/tmp/" + path, Comments: comments,
			Types: map[string]*types.Type{
				"X": {
					Kind: types.Struct,
					Members: []types.Member{
						{Name: "TypeMeta", Embedded: true, Tags: `json:"` + jsonTag + `"`},
					},
				},
			},
		}
	}

	allRecognized := []string{
		"+k8s:conversion-gen=k8s.io/api/foo",
		"+k8s:conversion-gen-external-types=k8s.io/api/foo/v1",
		"+k8s:defaulter-gen=TypeMeta",
		"+k8s:defaulter-gen-input=k8s.io/api/foo/v1",
		"+k8s:validation-gen=TypeMeta",
		"+k8s:validation-gen-input=k8s.io/api/foo/v1",
		"+k8s:validation-gen-nolint",
		"+k8s:validation-gen-scheme-registry=foo",
		"+k8s:validation-gen-test-fixture=bar",
		"+k8s:prerelease-lifecycle-gen=true",
		"+k8s:deepcopy-gen=package",
		"+k8s:register-gen=true",
		"+k8s:openapi-gen=true",
		"+k8s:openapi-model-package=foo",
		"+k8s:protobuf-gen=package",
	}

	cases := []struct {
		name    string
		pkg     *types.Package
		rules   []string
		wantErr string
	}{
		{
			name: "linting off: typo'd -gen passes",
			pkg:  pkg("test/pkg", "+k8s:conversion-x-gen=value"),
		},
		{
			name: "linting off: foreign +k8s:*-gen alongside ours passes",
			pkg:  pkg("test/pkg", "+k8s:my-custom-gen=value", "+k8s:conversion-x-gen=k8s.io/api/foo"),
		},
		{
			name: "linting off: missing universals on InternalVersion passes",
			pkg: pkg("k8s.io/kubernetes/pkg/apis/apps/v1",
				"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/apps"),
		},
		{
			name: "linting off: missing deepcopy on ExternalVersion passes",
			pkg:  pkgWithTypeMeta("k8s.io/api/apps/v1", ",inline", "+groupName=apps"),
		},

		// LintRuleKnownTags
		{
			name:  "known-tags: all Recognized tags coexist",
			pkg:   pkg("test/pkg", allRecognized...),
			rules: []string{LintRuleKnownTagsOnly},
		},
		{
			name:  "known-tags: subtagged generator recognized by base",
			pkg:   pkg("test/pkg", "+k8s:prerelease-lifecycle-gen=true", "+k8s:prerelease-lifecycle-gen:introduced=1.30"),
			rules: []string{LintRuleKnownTagsOnly},
		},
		{
			name:  "known-tags: non-generator +k8s: tag passes",
			pkg:   pkg("test/pkg", "+k8s:conversion-gen=k8s.io/api/foo", "+k8s:validateFalse=field"),
			rules: []string{LintRuleKnownTagsOnly},
		},
		{
			name:    "known-tags: unknown -gen fails",
			pkg:     pkg("test/pkg", "+k8s:bogus-gen=value"),
			rules:   []string{LintRuleKnownTagsOnly},
			wantErr: "+k8s:bogus-gen",
		},
		{
			name:    "known-tags: typo'd generator fails",
			pkg:     pkg("test/pkg", "+k8s:conversion-x-gen=value"),
			rules:   []string{LintRuleKnownTagsOnly},
			wantErr: "+k8s:conversion-x-gen",
		},
		{
			name:    "known-tags: foreign +k8s:*-gen fails",
			pkg:     pkg("test/pkg", "+k8s:my-custom-gen=value", "+k8s:conversion-gen=k8s.io/api/foo"),
			rules:   []string{LintRuleKnownTagsOnly},
			wantErr: "+k8s:my-custom-gen",
		},
		{
			name:  "explicit-disablement: NotAPIPackage skips",
			pkg:   pkg("k8s.io/util/foo"),
			rules: []string{LintRuleExplicitDisablement},
		},
		{
			name:  "explicit-disablement: ExternalVersion deepcopy=value passes",
			pkg:   pkgWithTypeMeta("k8s.io/api/apps/v1", ",inline", "+k8s:deepcopy-gen=package"),
			rules: []string{LintRuleExplicitDisablement},
		},
		{
			name:  "explicit-disablement: ExternalVersion deepcopy=false counts as explicit",
			pkg:   pkgWithTypeMeta("k8s.io/api/apps/v1", ",inline", "+k8s:deepcopy-gen=false"),
			rules: []string{LintRuleExplicitDisablement},
		},
		{
			name:    "explicit-disablement: ExternalVersion missing deepcopy fails",
			pkg:     pkgWithTypeMeta("k8s.io/api/apps/v1", ",inline", "+groupName=apps"),
			rules:   []string{LintRuleExplicitDisablement},
			wantErr: "+k8s:deepcopy-gen",
		},
		{
			name: "explicit-disablement: InternalVersion all universals tagged passes",
			pkg: pkg("k8s.io/kubernetes/pkg/apis/apps/v1",
				"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/apps",
				"+k8s:defaulter-gen=TypeMeta",
				"+k8s:validation-gen=TypeMeta"),
			rules: []string{LintRuleExplicitDisablement},
		},
		{
			name: "explicit-disablement: InternalVersion missing validation fails",
			pkg: pkg("k8s.io/kubernetes/pkg/apis/apps/v1",
				"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/apps",
				"+k8s:defaulter-gen=TypeMeta"),
			rules:   []string{LintRuleExplicitDisablement},
			wantErr: "+k8s:validation-gen",
		},
		{
			name: "explicit-disablement: conversion-only shim does not classify, no lint",
			pkg: pkg("k8s.io/api/foo/v1beta1",
				"+k8s:conversion-gen=k8s.io/api/foo/v1",
				"+k8s:conversion-gen-external-types=k8s.io/api/foo/v1"),
			rules: []string{LintRuleExplicitDisablement},
		},
		{
			name: "explicit-disablement: InternalVersion with conversion+validation but missing defaulter fails",
			pkg: pkg("k8s.io/kubernetes/pkg/apis/apps/v1",
				"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/apps",
				"+k8s:validation-gen=TypeMeta"),
			rules:   []string{LintRuleExplicitDisablement},
			wantErr: "+k8s:defaulter-gen",
		},
		{
			name: "explicit-disablement: defaulter alone is enough to classify as InternalVersion",
			pkg: pkg("k8s.io/kubernetes/pkg/apis/apps/v1",
				"+k8s:defaulter-gen=TypeMeta"),
			rules:   []string{LintRuleExplicitDisablement},
			wantErr: "+k8s:conversion-gen",
		},
		{
			name:  "explicit-disablement: InternalGroup no convention passes",
			pkg:   pkgWithTypeMeta("k8s.io/kubernetes/pkg/apis/apps", "", "+groupName=apps"),
			rules: []string{LintRuleExplicitDisablement},
		},
		{
			name:  "explicit-disablement: config-style version (no shim signal) skips",
			pkg:   pkg("k8s.io/component-base/metrics/api/v1", "+k8s:deepcopy-gen=package"),
			rules: []string{LintRuleExplicitDisablement},
		},
		{
			name:  "composed: both rules satisfied passes",
			pkg:   pkgWithTypeMeta("k8s.io/api/apps/v1", ",inline", "+k8s:deepcopy-gen=package"),
			rules: []string{LintRuleKnownTagsOnly, LintRuleExplicitDisablement},
		},
		{
			name: "composed: known-tags fires on typo even with explicit-disablement on",
			pkg: pkgWithTypeMeta("k8s.io/api/apps/v1", ",inline",
				"+k8s:deepcopy-gen=package", "+k8s:conversion-x-gen=value"),
			rules:   []string{LintRuleKnownTagsOnly, LintRuleExplicitDisablement},
			wantErr: "+k8s:conversion-x-gen",
		},
		{
			name:    "unknown rule name fails",
			pkg:     pkg("test/pkg"),
			rules:   []string{"future-rule"},
			wantErr: "unrecognized lint-rule: future-rule",
		},
		{
			name:  "empty rule list passes",
			pkg:   pkg("test/pkg", "+k8s:bogus-gen=value"),
			rules: []string{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Identify(tc.pkg, Conversion, WithLintRules(tc.rules...))
			if tc.wantErr == "" {
				if err != nil {
					t.Errorf("err = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("err = nil, want substring %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("err = %q, want substring %q", err, tc.wantErr)
			}
		})
	}
}

func withTypeMeta(jsonTag *string) *types.Type {
	tags := ""
	if jsonTag != nil {
		tags = `json:"` + *jsonTag + `"`
	}
	return &types.Type{
		Kind: types.Struct,
		Members: []types.Member{
			{Name: "TypeMeta", Embedded: true, Tags: tags},
		},
	}
}

func TestClassifyPackage(t *testing.T) {
	cases := []struct {
		name string
		pkg  *types.Package
		want packageRoles
	}{
		{
			name: "nil package",
			pkg:  nil,
			want: packageRoles{},
		},
		{
			name: "no codegen signal",
			pkg: &types.Package{
				Path:     "k8s.io/util/foo",
				Comments: []string{"package foo"},
			},
			want: packageRoles{},
		},
		{
			name: "external version: staging-style API package with TypeMeta json inline",
			pkg: &types.Package{
				Path:     "k8s.io/api/apps/v1",
				Comments: []string{"+groupName=apps"},
				Types: map[string]*types.Type{
					"Deployment": withTypeMeta(new(",inline")),
				},
			},
			want: packageRoles{isExternalVersion: true},
		},
		{
			name: "internal version: server-side conversion shim, no own TypeMeta types",
			pkg: &types.Package{
				Path: "k8s.io/kubernetes/pkg/apis/apps/v1",
				Comments: []string{
					"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/apps",
					"+k8s:defaulter-gen=TypeMeta",
				},
			},
			want: packageRoles{isInternalVersion: true},
		},
		{
			name: "internal group: pkg/apis/<group>/ with internal types (no json inline)",
			pkg: &types.Package{
				Path:     "k8s.io/kubernetes/pkg/apis/apps",
				Comments: []string{"+groupName=apps", "+k8s:deepcopy-gen=package"},
				Types: map[string]*types.Type{
					"Deployment": withTypeMeta(nil),
				},
			},
			want: packageRoles{isInternalGroup: true},
		},
		{
			name: "external group: API group package with no TypeMeta types and version-less path",
			pkg: &types.Package{
				Path:     "k8s.io/api/apps",
				Comments: []string{"+groupName=apps"},
				Types: map[string]*types.Type{
					"Deployment": withTypeMeta(new(",inline")),
				},
			},
			want: packageRoles{isExternalGroup: true},
		},
		{
			name: "version path with conversion+defaulter tags classifies as internal-version",
			pkg: &types.Package{
				Path: "example.com/internal/foo/v1alpha1",
				Comments: []string{
					"+groupName=foo",
					"+k8s:conversion-gen=example.com/internal/foo",
					"+k8s:defaulter-gen=TypeMeta",
				},
				Types: map[string]*types.Type{
					"Foo": withTypeMeta(nil),
				},
			},
			want: packageRoles{isInternalVersion: true},
		},
		{
			name: "version path with conversion-only is not classified (conversion-only shim)",
			pkg: &types.Package{
				Path: "example.com/api/foo/v1beta1",
				Comments: []string{
					"+k8s:conversion-gen=example.com/api/foo/v1",
					"+k8s:conversion-gen-external-types=example.com/api/foo/v1",
				},
			},
			want: packageRoles{},
		},
		{
			name: "version path with no internal-version tags and no inline TypeMeta is not an API package",
			pkg: &types.Package{
				Path:     "example.com/config/foo/v1",
				Comments: []string{"+k8s:deepcopy-gen=package"},
			},
			want: packageRoles{},
		},
		{
			name: "v1beta2 path is recognized as version",
			pkg: &types.Package{
				Path:     "k8s.io/api/foo/v1beta2",
				Comments: []string{"+groupName=foo"},
				Types: map[string]*types.Type{
					"Foo": withTypeMeta(new(",inline")),
				},
			},
			want: packageRoles{isExternalVersion: true},
		},
		{
			name: "non-version-looking path with TypeMeta json inline is ExternalGroup",
			pkg: &types.Package{
				Path:     "k8s.io/api/special",
				Comments: []string{"+groupName=special"},
				Types: map[string]*types.Type{
					"Special": withTypeMeta(new(",inline")),
				},
			},
			want: packageRoles{isExternalGroup: true},
		},
		{
			name: "package with only +k8s:*-gen tag and no +groupName is internal group",
			pkg: &types.Package{
				Path: "example.com/pkg/foo",
				Comments: []string{
					"+k8s:deepcopy-gen=package",
				},
			},
			want: packageRoles{isInternalGroup: true},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := classifyPackage(tc.pkg)
			if got != tc.want {
				t.Errorf("classifyPackage = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestExpectedGenerators(t *testing.T) {
	cases := []struct {
		name  string
		roles packageRoles
		want  []Spec
	}{
		{name: "no roles", roles: packageRoles{}, want: nil},
		{name: "external version", roles: packageRoles{isExternalVersion: true}, want: []Spec{Deepcopy}},
		{name: "internal version", roles: packageRoles{isInternalVersion: true}, want: []Spec{Conversion, Defaulter, Validation}},
		{name: "internal group", roles: packageRoles{isInternalGroup: true}, want: nil},
		{name: "external group", roles: packageRoles{isExternalGroup: true}, want: nil},
		{
			name:  "external+internal version: union",
			roles: packageRoles{isExternalVersion: true, isInternalVersion: true},
			want:  []Spec{Deepcopy, Conversion, Defaulter, Validation},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := expectedGenerators(tc.roles)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("expectedGenerators(%+v) = %v, want %v", tc.roles, got, tc.want)
			}
		})
	}
}
