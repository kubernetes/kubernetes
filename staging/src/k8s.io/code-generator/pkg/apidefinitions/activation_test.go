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

// TestIdentify code generator identification of API defining packages.
func TestIdentify(t *testing.T) {
	cases := []struct {
		name               string
		spec               Spec
		comments           []string
		wantShouldGenerate bool
		wantExplicitOnly   bool
		wantValues         []string // checked via spec-appropriate accessor when applicable
		wantErr            string
	}{
		{
			name: "no tag, target inactive",
			spec: Conversion,
		},
		{
			name:     "different tag present, this tag inactive",
			spec:     Conversion,
			comments: []string{"+k8s:defaulter-gen=TypeMeta"},
		},
		{
			name:               "single peer pkg activates conversion-gen",
			spec:               Conversion,
			comments:           []string{"+k8s:conversion-gen=k8s.io/api/foo"},
			wantShouldGenerate: true,
			wantValues:         []string{"k8s.io/api/foo"},
		},
		{
			name: "extensions/v1beta1 5-peer pattern",
			spec: Conversion,
			comments: []string{
				"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/apps",
				"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/policy",
				"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/networking",
				"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/extensions",
				"+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/autoscaling",
			},
			wantShouldGenerate: true,
			wantValues: []string{
				"k8s.io/kubernetes/pkg/apis/apps",
				"k8s.io/kubernetes/pkg/apis/policy",
				"k8s.io/kubernetes/pkg/apis/networking",
				"k8s.io/kubernetes/pkg/apis/extensions",
				"k8s.io/kubernetes/pkg/apis/autoscaling",
			},
		},
		{
			name:               "defaulter-gen=TypeMeta activates",
			spec:               Defaulter,
			comments:           []string{"+k8s:defaulter-gen=TypeMeta"},
			wantShouldGenerate: true,
			wantValues:         []string{"TypeMeta"},
		},
		{
			name:               "validation-gen=TypeMeta activates",
			spec:               Validation,
			comments:           []string{"+k8s:validation-gen=TypeMeta"},
			wantShouldGenerate: true,
			wantValues:         []string{"TypeMeta"},
		},
		{
			name:               "prerelease-lifecycle-gen=true activates",
			spec:               PrereleaseLifecycle,
			comments:           []string{"+k8s:prerelease-lifecycle-gen=true"},
			wantShouldGenerate: true,
		},
		{
			name:               "conversion-gen=false runs in explicit-only mode",
			spec:               Conversion,
			comments:           []string{"+k8s:conversion-gen=false"},
			wantShouldGenerate: true,
			wantExplicitOnly:   true,
		},
		{
			name:     "defaulter-gen=false opts out",
			spec:     Defaulter,
			comments: []string{"+k8s:defaulter-gen=false"},
		},
		{
			name:     "prerelease-lifecycle-gen=false opts out",
			spec:     PrereleaseLifecycle,
			comments: []string{"+k8s:prerelease-lifecycle-gen=false"},
		},
		{
			name: "=false mixed with peer pkg passes through",
			spec: Conversion,
			comments: []string{
				"+k8s:conversion-gen=k8s.io/api/extensions/v1beta1",
				"+k8s:conversion-gen=false",
			},
			wantShouldGenerate: true,
			wantValues:         []string{"k8s.io/api/extensions/v1beta1", "false"},
		},
		{
			name: "repeated identical peer is allowed and preserved",
			spec: Conversion,
			comments: []string{
				"+k8s:conversion-gen=k8s.io/api/foo",
				"+k8s:conversion-gen=k8s.io/api/foo",
			},
			wantShouldGenerate: true,
			wantValues:         []string{"k8s.io/api/foo", "k8s.io/api/foo"},
		},
		{
			name: "unknown -gen tag fails validation",
			spec: Conversion,
			comments: []string{
				"+k8s:conversion-gen=k8s.io/api/foo",
				"+k8s:bogus-gen=value",
			},
			wantErr: "+k8s:bogus-gen",
		},
		{
			name:     "relative peer package is rejected",
			spec:     Conversion,
			comments: []string{"+k8s:conversion-gen=./peer"},
			wantErr:  "relative path",
		},
		{
			name: "typo in known generator name fails validation",
			spec: Conversion,
			comments: []string{
				"+k8s:conversion-x-gen=k8s.io/api/foo",
			},
			wantErr: "+k8s:conversion--gen",
		},
		{
			name: "non-generator +k8s: tag passes validation untouched",
			spec: Validation,
			comments: []string{
				"+k8s:validation-gen=TypeMeta",
				"+k8s:validateFalse=field",
			},
			wantShouldGenerate: true,
			wantValues:         []string{"TypeMeta"},
		},
		{
			name: "subtagged generator tag is recognized by its base",
			spec: PrereleaseLifecycle,
			comments: []string{
				"+k8s:prerelease-lifecycle-gen=true",
				"+k8s:prerelease-lifecycle-gen:introduced=1.30",
			},
			wantShouldGenerate: true,
		},
		{
			name: "all recognized -gen tags coexist",
			spec: Conversion,
			comments: []string{
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
				"+k8s:protobuf-gen=package",
			},
			wantShouldGenerate: true,
			wantValues:         []string{"k8s.io/api/foo"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pkg := &types.Package{Path: "test/pkg", Dir: "/tmp/test", Comments: tc.comments}
			info, err := Identify(pkg, tc.spec, WithLintRules(LintRuleKnownTagsOnly))
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("err = nil, want substring %q", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("err = %q, want substring %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if got := info.ShouldGenerate(); got != tc.wantShouldGenerate {
				t.Errorf("ShouldGenerate = %v, want %v", got, tc.wantShouldGenerate)
			}
			if got := info.IsExplicitOnly(); got != tc.wantExplicitOnly {
				t.Errorf("IsExplicitOnly = %v, want %v", got, tc.wantExplicitOnly)
			}
			switch tc.spec.ValueMode {
			case ConversionPeerList:
				if got := info.PeerPackages(); !reflect.DeepEqual(got, tc.wantValues) {
					t.Errorf("PeerPackages = %v, want %v", got, tc.wantValues)
				}
			case TypeFilterList:
				if got := info.TypeFilters(); !reflect.DeepEqual(got, tc.wantValues) {
					t.Errorf("TypeFilters = %v, want %v", got, tc.wantValues)
				}
			}
		})
	}
}

func TestExternalTypes(t *testing.T) {
	const ownPath = "k8s.io/kubernetes/pkg/apis/apps/v1"
	cases := []struct {
		name     string
		spec     Spec
		comments []string
		want     string
		wantErr  string
	}{
		{
			name: "no input tag: own path",
			spec: Conversion,
			want: ownPath,
		},
		{
			name:     "conversion: external-types tag",
			spec:     Conversion,
			comments: []string{"+k8s:conversion-gen-external-types=k8s.io/api/apps/v1"},
			want:     "k8s.io/api/apps/v1",
		},
		{
			name:     "defaulter: input tag",
			spec:     Defaulter,
			comments: []string{"+k8s:defaulter-gen-input=k8s.io/api/apps/v1"},
			want:     "k8s.io/api/apps/v1",
		},
		{
			name:     "validation: input tag",
			spec:     Validation,
			comments: []string{"+k8s:validation-gen-input=k8s.io/api/apps/v1"},
			want:     "k8s.io/api/apps/v1",
		},
		{
			name: "conversion ignores other generators' input tags",
			spec: Conversion,
			comments: []string{
				"+k8s:defaulter-gen-input=k8s.io/api/apps/v2",
				"+k8s:validation-gen-input=k8s.io/api/apps/v3",
			},
			want: ownPath,
		},
		{
			name: "Boolean spec returns own path regardless of input tags",
			spec: PrereleaseLifecycle,
			comments: []string{
				"+k8s:defaulter-gen-input=k8s.io/api/apps/v2",
			},
			want: ownPath,
		},
		{
			name: "multiple values for same tag is an error",
			spec: Conversion,
			comments: []string{
				"+k8s:conversion-gen-external-types=k8s.io/api/apps/v1",
				"+k8s:conversion-gen-external-types=k8s.io/api/apps/v2",
			},
			wantErr: "expected at most one value",
		},
		{
			name:     "relative input path is rejected",
			spec:     Validation,
			comments: []string{"+k8s:validation-gen-input=./types"},
			wantErr:  "relative path",
		},
		{
			name:     "parent-relative input path is rejected",
			spec:     Defaulter,
			comments: []string{"+k8s:defaulter-gen-input=../foo/v1"},
			wantErr:  "relative path",
		},
		{
			name:     "relative external-types path is rejected",
			spec:     Conversion,
			comments: []string{"+k8s:conversion-gen-external-types=./types"},
			wantErr:  "relative path",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pkg := &types.Package{Path: ownPath, Comments: tc.comments}
			tgt, err := Identify(pkg, tc.spec)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("err = nil, want substring %q", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("err = %q, want substring %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if got := tgt.ExternalTypes(); got != tc.want {
				t.Errorf("ExternalTypes = %q, want %q", got, tc.want)
			}
		})
	}
}
