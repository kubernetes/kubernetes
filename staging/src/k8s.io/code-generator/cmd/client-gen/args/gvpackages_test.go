/*
Copyright 2017 The Kubernetes Authors.

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

package args

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/pflag"

	"k8s.io/code-generator/cmd/client-gen/types"
)

func TestGVPackageFlag(t *testing.T) {
	tests := []struct {
		args           []string
		def            []string
		importBasePath string
		expected       map[types.GroupVersion]string
		expectedGroups []types.GroupVersions
		parseError     string
	}{
		{
			args:           []string{},
			expected:       map[types.GroupVersion]string{},
			expectedGroups: []types.GroupVersions{},
		},
		{
			args: []string{"foo/bar/v1", "foo/bar/v2", "foo/bar/", "foo/v1"},
			expectedGroups: []types.GroupVersions{
				{PackageName: "bar", Group: types.Group("bar"), Versions: []types.PackageVersion{
					{Version: "v1", Package: "foo/bar/v1"},
					{Version: "v2", Package: "foo/bar/v2"},
					{Version: "", Package: "foo/bar"},
				}},
				{PackageName: "foo", Group: types.Group("foo"), Versions: []types.PackageVersion{
					{Version: "v1", Package: "foo/v1"},
				}},
			},
		},
		{
			args: []string{"foo/bar/v1", "foo/bar/v2", "foo/bar/", "foo/v1"},
			def:  []string{"foo/bar/v1alpha1", "foo/v1"},
			expectedGroups: []types.GroupVersions{
				{PackageName: "bar", Group: types.Group("bar"), Versions: []types.PackageVersion{
					{Version: "v1", Package: "foo/bar/v1"},
					{Version: "v2", Package: "foo/bar/v2"},
					{Version: "", Package: "foo/bar"},
				}},
				{PackageName: "foo", Group: types.Group("foo"), Versions: []types.PackageVersion{
					{Version: "v1", Package: "foo/v1"},
				}},
			},
		},
		{
			args: []string{"api/v1", "api"},
			expectedGroups: []types.GroupVersions{
				{PackageName: "api", Group: types.Group("api"), Versions: []types.PackageVersion{
					{Version: "v1", Package: "api/v1"},
					{Version: "", Package: "api"},
				}},
			},
		},
		{
			args:           []string{"foo/v1"},
			importBasePath: "k8s.io/api",
			expectedGroups: []types.GroupVersions{
				{PackageName: "foo", Group: types.Group("foo"), Versions: []types.PackageVersion{
					{Version: "v1", Package: "k8s.io/api/foo/v1"},
				}},
			},
		},
	}
	for i, test := range tests {
		fs := pflag.NewFlagSet("testGVPackage", pflag.ContinueOnError)
		groups := []types.GroupVersions{}
		builder := NewGroupVersionsBuilder(&groups)
		fs.Var(NewGVPackagesValue(builder, test.def), "input", "usage")
		fs.Var(NewInputBasePathValue(builder, test.importBasePath), "input-base-path", "usage")

		args := []string{}
		for _, a := range test.args {
			args = append(args, fmt.Sprintf("--input=%s", a))
		}

		err := fs.Parse(args)
		if test.parseError != "" {
			if err == nil {
				t.Errorf("%d: expected error %q, got nil", i, test.parseError)
			} else if !strings.Contains(err.Error(), test.parseError) {
				t.Errorf("%d: expected error %q, got %q", i, test.parseError, err)
			}
		} else if err != nil {
			t.Errorf("%d: expected nil error, got %v", i, err)
		}
		if !reflect.DeepEqual(groups, test.expectedGroups) {
			t.Errorf("%d: expected groups %+v, got groups %+v", i, test.expectedGroups, groups)
		}
	}
}
