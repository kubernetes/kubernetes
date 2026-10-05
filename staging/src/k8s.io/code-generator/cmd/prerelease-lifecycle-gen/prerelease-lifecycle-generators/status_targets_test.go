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

package prereleaselifecyclegenerators

import (
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/code-generator/cmd/prerelease-lifecycle-gen/args"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

func TestGetTargets(t *testing.T) {
	cases := []struct {
		name     string
		pkgs     map[string]*types.Package
		inputs   []string
		wantPkgs []string
	}{
		{
			name: "enabled package activates",
			pkgs: map[string]*types.Package{
				"example.com/api/v1": {
					Path:     "example.com/api/v1",
					Dir:      "/tmp/example/api/v1",
					Comments: []string{"+k8s:prerelease-lifecycle-gen=true"},
				},
			},
			inputs:   []string{"example.com/api/v1"},
			wantPkgs: []string{"example.com/api/v1"},
		},
		{
			name: "sole =false opts out",
			pkgs: map[string]*types.Package{
				"example.com/api/v1": {
					Path:     "example.com/api/v1",
					Dir:      "/tmp/example/api/v1",
					Comments: []string{"+k8s:prerelease-lifecycle-gen=false"},
				},
			},
			inputs:   []string{"example.com/api/v1"},
			wantPkgs: nil,
		},
		{
			name: "no relevant tag is skipped",
			pkgs: map[string]*types.Package{
				"example.com/api/v1": {
					Path:     "example.com/api/v1",
					Dir:      "/tmp/example/api/v1",
					Comments: []string{"+groupName=example.com"},
				},
			},
			inputs:   []string{"example.com/api/v1"},
			wantPkgs: nil,
		},
		{
			name: "enabled with introduced subtag activates",
			pkgs: map[string]*types.Package{
				"example.com/api/v1beta1": {
					Path: "example.com/api/v1beta1",
					Dir:  "/tmp/example/api/v1beta1",
					Comments: []string{
						"+k8s:prerelease-lifecycle-gen=true",
						"+k8s:prerelease-lifecycle-gen:introduced=1.30",
					},
				},
			},
			inputs:   []string{"example.com/api/v1beta1"},
			wantPkgs: []string{"example.com/api/v1beta1"},
		},
		// Ecosystem regression: a third-party generator's tag in the same
		// doc.go must NOT cause prerelease-gen to fail or skip.
		{
			name: "foreign third-party generator tag is ignored",
			pkgs: map[string]*types.Package{
				"example.com/api/v1": {
					Path: "example.com/api/v1",
					Dir:  "/tmp/example/api/v1",
					Comments: []string{
						"+k8s:my-custom-gen=value",
						"+k8s:prerelease-lifecycle-gen=true",
					},
				},
			},
			inputs:   []string{"example.com/api/v1"},
			wantPkgs: []string{"example.com/api/v1"},
		},
		{
			name: "mix of all cases",
			pkgs: map[string]*types.Package{
				"example.com/enabled/v1": {
					Path:     "example.com/enabled/v1",
					Dir:      "/tmp/enabled/v1",
					Comments: []string{"+k8s:prerelease-lifecycle-gen=true"},
				},
				"example.com/optedout/v1": {
					Path:     "example.com/optedout/v1",
					Dir:      "/tmp/optedout/v1",
					Comments: []string{"+k8s:prerelease-lifecycle-gen=false"},
				},
				"example.com/untagged/v1": {
					Path:     "example.com/untagged/v1",
					Dir:      "/tmp/untagged/v1",
					Comments: []string{"+groupName=example.com"},
				},
				"example.com/withsubtag/v1beta1": {
					Path: "example.com/withsubtag/v1beta1",
					Dir:  "/tmp/withsubtag/v1beta1",
					Comments: []string{
						"+k8s:prerelease-lifecycle-gen=true",
						"+k8s:prerelease-lifecycle-gen:introduced=1.30",
					},
				},
			},
			inputs: []string{
				"example.com/enabled/v1",
				"example.com/optedout/v1",
				"example.com/untagged/v1",
				"example.com/withsubtag/v1beta1",
			},
			wantPkgs: []string{
				"example.com/enabled/v1",
				"example.com/withsubtag/v1beta1",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &generator.Context{
				Universe: types.Universe(tc.pkgs),
				Inputs:   tc.inputs,
			}
			a := &args.Args{OutputFile: "zz_generated.prerelease_lifecycle.go"}

			got := GetTargets(ctx, a)

			var gotPkgs []string
			for _, tgt := range got {
				gotPkgs = append(gotPkgs, tgt.Path())
			}
			sort.Strings(gotPkgs)
			want := append([]string(nil), tc.wantPkgs...)
			sort.Strings(want)

			if diff := cmp.Diff(want, gotPkgs); diff != "" {
				t.Errorf("GetTargets package paths mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
