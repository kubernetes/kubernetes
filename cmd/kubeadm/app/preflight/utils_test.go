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

package preflight

import (
	"testing"

	"github.com/pkg/errors"

	utilsexec "k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func TestGetKubeletVersion(t *testing.T) {
	cases := []struct {
		output   string
		expected string
		err      error
		valid    bool
	}{
		{"Kubernetes v1.7.0", "1.7.0", nil, true},
		{"Kubernetes v1.8.0-alpha.2.1231+afabd012389d53a", "1.8.0-alpha.2.1231+afabd012389d53a", nil, true},
		{"something-invalid", "", nil, false},
		{"command not found", "", errors.New("kubelet not found"), false},
		{"", "", nil, false},
	}

	for _, tc := range cases {
		t.Run(tc.output, func(t *testing.T) {
			fcmd := fakeexec.FakeCmd{
				CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
					func() ([]byte, error) { return []byte(tc.output), tc.err },
				},
			}
			fexec := &fakeexec.FakeExec{
				CommandScript: []fakeexec.FakeCommandAction{
					func(cmd string, args ...string) utilsexec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				},
			}
			ver, err := GetKubeletVersion(fexec)
			switch {
			case err != nil && tc.valid:
				t.Errorf("GetKubeletVersion: unexpected error for %q. Error: %v", tc.output, err)
			case err == nil && !tc.valid:
				t.Errorf("GetKubeletVersion: error expected for key %q, but result is %q", tc.output, ver)
			case ver != nil && ver.String() != tc.expected:
				t.Errorf("GetKubeletVersion: unexpected version result for key %q. Expected: %q Actual: %q", tc.output, tc.expected, ver)
			}
		})
	}
}
