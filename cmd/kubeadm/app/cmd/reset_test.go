/*
Copyright 2022 The Kubernetes Authors.

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

package cmd

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"k8s.io/apimachinery/pkg/util/sets"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
)

func TestNewResetData(t *testing.T) {
	testCases := []struct {
		name        string
		args        []string
		flags       map[string]string
		validate    func(*testing.T, *resetData)
		expectError string
		data        *resetData
	}{
		{
			name: "flags parsed correctly",
			flags: map[string]string{
				options.CertificatesDir:       "/tmp",
				options.NodeCRISocket:         "unix:///var/run/crio/crio.sock",
				options.IgnorePreflightErrors: "all",
				options.ForceReset:            "true",
				options.DryRun:                "true",
				options.CleanupTmpDir:         "true",
			},
			data: &resetData{
				certificatesDir:       "/tmp",
				criSocketPath:         "unix:///var/run/crio/crio.sock",
				ignorePreflightErrors: sets.New("all"),
				forceReset:            true,
				dryRun:                true,
				cleanupTmpDir:         true,
			},
		},
		{
			name: "fails if preflight ignores all but has individual check",
			flags: map[string]string{
				options.IgnorePreflightErrors: "all,something-else",
				options.NodeCRISocket:         "unix:///var/run/crio/crio.sock",
			},
			expectError: "don't specify individual checks if 'all' is used",
		},
		{
			name: "pre-flights errors from CLI args",
			flags: map[string]string{
				options.IgnorePreflightErrors: "a,b",
				options.NodeCRISocket:         "unix:///var/run/crio/crio.sock",
			},
			validate: expectedResetIgnorePreflightErrors(sets.New("a", "b")),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// initialize an external reset option and inject it to the reset cmd
			resetOptions := newResetOptions()
			cmd := newCmdReset(nil, nil, resetOptions)

			// sets cmd flags (that will be reflected on the reset options)
			for f, v := range tc.flags {
				cmd.Flags().Set(f, v)
			}

			// test newResetData method
			data, err := newResetData(cmd, resetOptions, nil, nil)
			if err != nil && !strings.Contains(err.Error(), tc.expectError) {
				t.Fatalf("newResetData returned unexpected error, expected: %s, got %v", tc.expectError, err)
			}
			if err == nil && len(tc.expectError) != 0 {
				t.Fatalf("newResetData didn't return error expected %s", tc.expectError)
			}

			if tc.data != nil {
				if diff := cmp.Diff(tc.data, data, cmp.AllowUnexported(resetData{}), cmpopts.IgnoreFields(resetData{}, "client", "cfg")); diff != "" {
					t.Fatalf("newResetData returned data (-want,+got):\n%s", diff)
				}
			}
			// exec additional validation on the returned value
			if tc.validate != nil {
				tc.validate(t, data)
			}
		})
	}
}

func expectedResetIgnorePreflightErrors(expected sets.Set[string]) func(t *testing.T, data *resetData) {
	return func(t *testing.T, data *resetData) {
		if !expected.Equal(data.ignorePreflightErrors) {
			t.Errorf("Invalid ignore preflight errors. Expected: %v. Actual: %v", sets.List(expected), sets.List(data.ignorePreflightErrors))
		}
		if data.cfg != nil && !expected.HasAll(data.cfg.NodeRegistration.IgnorePreflightErrors...) {
			t.Errorf("Invalid ignore preflight errors in InitConfiguration. Expected: %v. Actual: %v", sets.List(expected), data.cfg.NodeRegistration.IgnorePreflightErrors)
		}
	}
}
