/*
Copyright 2023 The Kubernetes Authors.

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

package verflag

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/spf13/pflag"

	"k8s.io/component-base/version"
)

func TestVersionFlag(t *testing.T) {
	initialFlagValue := string(*versionFlag)
	initialVersion := version.Get()

	testcases := []struct {
		name               string
		flags              []string
		expectError        string
		expectExit         bool
		expectPrintVersion string
		expectGitVersion   string
	}{
		{
			name:             "no flag",
			flags:            []string{},
			expectGitVersion: initialVersion.GitVersion,
		},
		{
			name:             "false",
			flags:            []string{"--version=false"},
			expectGitVersion: initialVersion.GitVersion,
		},

		{
			name:               "valueless",
			flags:              []string{"--version"},
			expectGitVersion:   initialVersion.GitVersion,
			expectExit:         true,
			expectPrintVersion: "Kubernetes " + initialVersion.GitVersion,
		},
		{
			name:               "true",
			flags:              []string{"--version=true"},
			expectGitVersion:   initialVersion.GitVersion,
			expectExit:         true,
			expectPrintVersion: "Kubernetes " + initialVersion.GitVersion,
		},
		{
			name:               "raw",
			flags:              []string{"--version=raw"},
			expectGitVersion:   initialVersion.GitVersion,
			expectExit:         true,
			expectPrintVersion: fmt.Sprintf("%#v", initialVersion),
		},
		{
			name:               "truthy",
			flags:              []string{"--version=T"},
			expectGitVersion:   initialVersion.GitVersion,
			expectExit:         true,
			expectPrintVersion: "Kubernetes " + initialVersion.GitVersion,
		},

		{
			name:             "override",
			flags:            []string{"--version=v0.0.0-custom"},
			expectGitVersion: "v0.0.0-custom",
		},
		{
			name:        "invalid override semver",
			flags:       []string{"--version=vX"},
			expectError: `could not parse "vX"`,
		},
		{
			name:        "invalid override major",
			flags:       []string{"--version=v1.0.0"},
			expectError: `must match major/minor/patch`,
		},
		{
			name:        "invalid override minor",
			flags:       []string{"--version=v0.1.0"},
			expectError: `must match major/minor/patch`,
		},
		{
			name:        "invalid override patch",
			flags:       []string{"--version=v0.0.1"},
			expectError: `must match major/minor/patch`,
		},

		{
			name:               "override and exit",
			flags:              []string{"--version=v0.0.0-custom", "--version"},
			expectGitVersion:   "v0.0.0-custom",
			expectExit:         true,
			expectPrintVersion: "Kubernetes v0.0.0-custom",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {

			originalOutput := output
			originalExit := exit

			outputBuffer := &bytes.Buffer{}
			output = outputBuffer
			exitCalled := false
			exit = func(code int) { exitCalled = true }

			t.Cleanup(func() {
				output = originalOutput
				exit = originalExit
				*versionFlag = versionValue(initialFlagValue)
				err := version.SetDynamicVersion(initialVersion.GitVersion)
				if err != nil {
					t.Fatal(err)
				}
			})

			fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
			AddFlags(fs)
			err := fs.Parse(tc.flags)
			if tc.expectError != "" {
				if err == nil {
					t.Fatal("expected error, got none")
				}
				if !strings.Contains(err.Error(), tc.expectError) {
					t.Fatalf("expected error containing %q, got %q", tc.expectError, err.Error())
				}
				return
			} else if err != nil {
				t.Fatalf("unexpected parse error: %v", err)
			}

			if e, a := tc.expectGitVersion, version.Get().GitVersion; e != a {
				t.Fatalf("gitversion: expected %v, got %v", e, a)
			}

			PrintAndExitIfRequested()
			if e, a := tc.expectExit, exitCalled; e != a {
				t.Fatalf("exit(): expected %v, got %v", e, a)
			}
			if e, a := tc.expectPrintVersion, strings.TrimSpace(outputBuffer.String()); e != a {
				t.Fatalf("print version: expected %v, got %v", e, a)
			}
		})
	}
}
