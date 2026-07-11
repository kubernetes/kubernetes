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

package localupcluster

import (
	"reflect"
	"testing"

	"k8s.io/component-base/featuregate"
)

func TestMergeFeatureGatesFlags(t *testing.T) {
	for name, tc := range map[string]struct {
		cmdLine      []string
		featureGates string
		want         []string
		wantErr      bool
	}{
		"nop": {
			// --feature-gates= (empty value) with no new gates: output must be unchanged.
			cmdLine:      []string{"--feature-gates="},
			featureGates: "",
			want:         []string{"--feature-gates="},
		},
		"append-when-no-existing-flag": {
			cmdLine:      []string{"--some-other-option=bla", "--some-option"},
			featureGates: "fg1=true,fg2=false",
			want:         []string{"--some-other-option=bla", "--some-option", "--feature-gates=fg1=true,fg2=false"},
		},
		"replace-existing-flag": {
			cmdLine:      []string{"--some-other-option=bla", "--feature-gates=fg1=false", "--some-option"},
			featureGates: "fg1=true",
			want:         []string{"--some-other-option=bla", "--feature-gates=fg1=true", "--some-option"},
		},
		"merge-with-existing-flag": {
			cmdLine:      []string{"--feature-gates=fg1=false,fg2=true", "--some-option"},
			featureGates: "fg3=true",
			want:         []string{"--feature-gates=fg1=false,fg2=true,fg3=true", "--some-option"},
		},
		"new-gates-override-existing": {
			cmdLine:      []string{"--feature-gates=fg1=false,fg2=true"},
			featureGates: "fg1=true",
			want:         []string{"--feature-gates=fg1=true,fg2=true"},
		},
		"collapse-multiple-flags": {
			cmdLine: []string{
				"--feature-gates=fg1=false",
				"--some-option",
				"--feature-gates=fg2=true",
				"--other=val",
				"--feature-gates=fg3=false",
			},
			featureGates: "fg4=true",
			want: []string{
				"--feature-gates=fg1=false,fg2=true,fg3=false,fg4=true",
				"--some-option",
				"--other=val",
			},
		},
		"collapse-multiple-flags-later-wins": {
			cmdLine:      []string{"--feature-gates=fg1=false", "--feature-gates=fg1=true"},
			featureGates: "",
			want:         []string{"--feature-gates=fg1=true"},
		},
		"empty-feature-gates-no-existing": {
			cmdLine:      []string{"--some-option"},
			featureGates: "",
			want:         []string{"--some-option"},
		},
		"empty-feature-gates-with-existing": {
			cmdLine:      []string{"--feature-gates=fg1=true", "--some-option"},
			featureGates: "",
			want:         []string{"--feature-gates=fg1=true", "--some-option"},
		},
		"nil-cmdline": {
			cmdLine:      nil,
			featureGates: "fg1=true",
			want:         []string{"--feature-gates=fg1=true"},
		},
		"output-keys-are-sorted": {
			cmdLine:      nil,
			featureGates: "zeta=true,alpha=false,mike=true",
			want:         []string{"--feature-gates=alpha=false,mike=true,zeta=true"},
		},
		"unparseable-existing-flag-returns-error": {
			cmdLine:      []string{"--feature-gates=not-a-pair"},
			featureGates: "fg1=true",
			wantErr:      true,
		},
		"unparseable-input-returns-error": {
			cmdLine:      nil,
			featureGates: "not-a-pair",
			wantErr:      true,
		},
		"existing-empty-flag-no-new-gates": {
			// --feature-gates= (empty value) is parseable but yields no gates;
			// featureGates is also empty. The placeholder must not be left as "".
			cmdLine:      []string{"--feature-gates=", "--some-option"},
			featureGates: "",
			want:         []string{"--feature-gates=", "--some-option"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			// Snapshot input to detect accidental mutation.
			cmdLineCopy := append([]string(nil), tc.cmdLine...)

			got, err := mergeFeatureGatesFlags(tc.cmdLine, tc.featureGates)
			switch {
			case tc.wantErr && err == nil:
				t.Fatalf("expected error, got nil; result=%q", got)
			case !tc.wantErr && err != nil:
				t.Fatalf("unexpected error: %v", err)
			}
			if !tc.wantErr && !reflect.DeepEqual(got, tc.want) {
				t.Errorf("mergeFeatureGatesFlags() mismatch:\n got:  %q\n want: %q", got, tc.want)
			}
			if !reflect.DeepEqual(tc.cmdLine, cmdLineCopy) {
				t.Errorf("input slice was mutated:\n got:  %q\n want: %q", tc.cmdLine, cmdLineCopy)
			}
		})
	}
}

func TestFilterLockedFeatureGates(t *testing.T) {
	allGates := map[featuregate.Feature]featuregate.FeatureSpec{
		"Unlocked":      {Default: true, LockToDefault: false},
		"LockedToTrue":  {Default: true, LockToDefault: true},
		"LockedToFalse": {Default: false, LockToDefault: true},
	}
	for name, tc := range map[string]struct {
		featureGates string
		want         string
	}{
		"no-locked-gates":                   {featureGates: "Unlocked=false", want: "Unlocked=false"},
		"locked-to-true-set-false-removed":  {featureGates: "LockedToTrue=false", want: ""},
		"locked-to-true-set-true-removed":   {featureGates: "LockedToTrue=true", want: ""},
		"locked-to-false-set-true-removed":  {featureGates: "LockedToFalse=true", want: ""},
		"locked-to-false-set-false-removed": {featureGates: "LockedToFalse=false", want: ""},
		"locked-gate-mixed":                 {featureGates: "Unlocked=false,LockedToTrue=false", want: "Unlocked=false"},
		"unknown-gate-kept":                 {featureGates: "Unknown=true", want: "Unknown=true"},
		"multiple-unlocked":                 {featureGates: "Unlocked=false,Unknown=true", want: "Unlocked=false,Unknown=true"},
	} {
		t.Run(name, func(t *testing.T) {
			got := filterLockedFeatureGates(tc.featureGates, allGates)
			if got != tc.want {
				t.Errorf("filterLockedFeatureGates(%q) = %q, want %q", tc.featureGates, got, tc.want)
			}
		})
	}
}
