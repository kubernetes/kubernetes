// Copyright 2014 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"testing"

	stage1commontypes "github.com/coreos/rkt/stage1/common/types"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

const tstprefix = "pod-test"

func TestQuoteExec(t *testing.T) {
	tests := []struct {
		input  []string
		output string
	}{
		{
			input:  []string{`path`, `"arg1"`, `"'arg2'"`, `'arg3'`},
			output: `"path" "\"arg1\"" "\"\'arg2\'\"" "\'arg3\'"`,
		}, {
			input:  []string{`path`},
			output: `"path"`,
		}, {
			input:  []string{`path`, ``, `arg2`},
			output: `"path" "" "arg2"`,
		}, {
			input:  []string{`path`, `"foo\bar"`, `\`},
			output: `"path" "\"foo\\bar\"" "\\"`,
		}, {
			input:  []string{`path with spaces`, `"foo\bar"`, `\`},
			output: `"path with spaces" "\"foo\\bar\"" "\\"`,
		}, {
			input:  []string{`path with "quo't'es" and \slashes`, `"arg"`, `\`},
			output: `"path with \"quo\'t\'es\" and \\slashes" "\"arg\"" "\\"`,
		}, {
			input:  []string{`$path$`, `$argument`},
			output: `"$path$" "$$argument"`,
		}, {
			input:  []string{`path`, `Args\nwith\nnewlines`},
			output: `"path" "Args\\nwith\\nnewlines"`,
		}, {
			input:  []string{`path`, "Args\nwith\nnewlines"},
			output: `"path" "Args\nwith\nnewlines"`,
		},
	}

	for i, tt := range tests {
		o := quoteExec(tt.input)
		if o != tt.output {
			t.Errorf("#%d: expected `%v` got `%v`", i, tt.output, o)
		}
	}
}

var (
	falseVar = false
	trueVar  = true
)

// TestAppToNspawnArgsOverridesImageManifestReadOnly tests
// that the ImageManifest's `readOnly` volume setting will be
// overrided by PodManifest.
func Disabled_TestAppToNspawnArgsOverridesImageManifestReadOnly(t *testing.T) {
	tests := []struct {
		imageManifestVolumeReadOnly bool
		podManifestVolumeReadOnly   *bool
		expectReadOnly              bool
	}{
		{
			false,
			nil,
			false,
		},
		{
			false,
			&falseVar,
			false,
		},
		{
			false,
			&trueVar,
			true,
		},
		{
			true,
			nil,
			true,
		},
		{
			true,
			&falseVar,
			false,
		},
		{
			true,
			&trueVar,
			true,
		},
	}

	for i, tt := range tests {
		podManifest := &schema.PodManifest{
			Volumes: []types.Volume{
				{
					Name:     *types.MustACName("foo-mount"),
					Kind:     "host",
					Source:   "/host/foo",
					ReadOnly: tt.podManifestVolumeReadOnly,
				},
			},
		}
		appManifest := &schema.RuntimeApp{
			Mounts: []schema.Mount{
				{
					Volume: *types.MustACName("foo-mount"),
					Path:   "/app/foo",
				},
			},
			App: &types.App{
				Exec:  []string{"/bin/foo"},
				User:  "0",
				Group: "0",
				MountPoints: []types.MountPoint{
					{
						Name:     *types.MustACName("foo-mount"),
						Path:     "/app/foo",
						ReadOnly: tt.imageManifestVolumeReadOnly,
					},
				},
			},
		}

		tmpDir, err := ioutil.TempDir("", tstprefix)
		if err != nil {
			t.Errorf("error creating tempdir: %v", err)
			continue
		}
		defer os.RemoveAll(tmpDir)

		if err := os.MkdirAll(filepath.Join(tmpDir, "/stage1/rootfs/opt/stage2/rootfs"), 0755); err != nil {
			t.Errorf("error stage2 rootfs in tmpdir: %v", err)
			continue
		}

		p := &stage1commontypes.Pod{Manifest: podManifest, Root: tmpDir}
		output, err := appToNspawnArgs(p, appManifest)
		if err != nil {
			t.Errorf("#%d: unexpected error: `%v`", i, err)
		}

		if ro := hasBindROArg(output); ro != tt.expectReadOnly {
			t.Errorf("#%d: expected: readOnly: %v, saw: %v \nOutput:\n%v", i, tt.expectReadOnly, ro, output)
		}
	}
}

func hasBindROArg(output []string) bool {
	roRegexp := regexp.MustCompile("^--bind-ro=/host/foo:.*/app/foo(:rbind)?$")
	for i := len(output) - 1; i >= 0; i-- {
		if roRegexp.MatchString(output[i]) {
			return true
		}
	}
	return false
}

func Disabled_TestAppToNspawnArgsRecursive(t *testing.T) {
	tests := []struct {
		podManifestVolumeRecursive *bool
		expectRecursive            bool
	}{
		{
			nil,
			true,
		},
		{
			&trueVar,
			true,
		},
		{
			&falseVar,
			false,
		},
	}

	for i, tt := range tests {
		podManifest := &schema.PodManifest{
			Volumes: []types.Volume{
				{
					Name:      *types.MustACName("foo-mount"),
					Kind:      "host",
					Source:    "/host/foo",
					Recursive: tt.podManifestVolumeRecursive,
				},
			},
		}
		appManifest := &schema.RuntimeApp{
			Mounts: []schema.Mount{
				{
					Volume: *types.MustACName("foo-mount"),
					Path:   "/app/foo",
				},
			},
			App: &types.App{
				Exec:  []string{"/bin/foo"},
				User:  "0",
				Group: "0",
				MountPoints: []types.MountPoint{
					{
						Name: *types.MustACName("foo-mount"),
						Path: "/app/foo",
					},
				},
			},
		}

		tmpDir, err := ioutil.TempDir("", tstprefix)
		if err != nil {
			t.Errorf("error creating tempdir: %v", err)
			continue
		}
		defer os.RemoveAll(tmpDir)

		if err := os.MkdirAll(filepath.Join(tmpDir, "/stage1/rootfs/opt/stage2/rootfs"), 0755); err != nil {
			t.Errorf("error stage2 rootfs in tmpdir: %v", err)
			continue
		}

		p := &stage1commontypes.Pod{Manifest: podManifest, Root: tmpDir}
		output, err := appToNspawnArgs(p, appManifest)
		if err != nil {
			t.Errorf("#%d: unexpected error: `%v`", i, err)
		}
		if rbind := hasRbindArg(output); rbind != tt.expectRecursive {
			t.Errorf("#%d: expected: recursive: %v, saw: %v \nOutput:\n%v", i, tt.expectRecursive, rbind, output)
		}
	}
}

func hasRbindArg(output []string) bool {
	rbindRegexp := regexp.MustCompile("^--bind(-ro)?=/host/foo:.*/app/foo:rbind$")
	for i := len(output) - 1; i >= 0; i-- {
		if rbindRegexp.MatchString(output[i]) {
			return true
		}
	}
	return false
}
