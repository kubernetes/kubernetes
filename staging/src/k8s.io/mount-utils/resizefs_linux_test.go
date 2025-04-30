//go:build linux
// +build linux

/*
Copyright 2021 The Kubernetes Authors.

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

package mount

import (
	"testing"

	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func TestNeedResize(t *testing.T) {
	testcases := []struct {
		name            string
		devicePath      string
		deviceMountPath string
		readonly        string
		deviceSize      string
		extSize         string
		cmdOutputFsType string
		expectError     bool
		expectResult    bool
	}{
		{
			name:            "True",
			devicePath:      "/dev/test1",
			deviceMountPath: "/mnt/test1",
			readonly:        "0",
			deviceSize:      "2048",
			cmdOutputFsType: "TYPE=ext3",
			extSize:         "20",
			expectError:     false,
			expectResult:    true,
		},
		{
			name:            "False - needed by size but fs is readonly",
			devicePath:      "/dev/test1",
			deviceMountPath: "/mnt/test1",
			readonly:        "1",
			deviceSize:      "2048",
			cmdOutputFsType: "TYPE=ext3",
			extSize:         "20",
			expectError:     false,
			expectResult:    false,
		},
		{
			name:            "False - Unsupported fs type",
			devicePath:      "/dev/test1",
			deviceMountPath: "/mnt/test1",
			readonly:        "0",
			deviceSize:      "2048",
			extSize:         "1",
			cmdOutputFsType: "TYPE=ntfs",
			expectError:     true,
			expectResult:    false,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			fcmd := fakeexec.FakeCmd{
				CombinedOutputScript: []fakeexec.FakeAction{
					func() ([]byte, []byte, error) { return []byte(test.readonly), nil, nil },
					func() ([]byte, []byte, error) { return []byte(test.cmdOutputFsType), nil, nil },
				},
			}
			fexec := &fakeexec.FakeExec{
				CommandScript: []fakeexec.FakeCommandAction{
					func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
					func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				},
			}
			resizefs := ResizeFs{exec: fexec}

			needResize, err := resizefs.NeedResize(test.devicePath, test.deviceMountPath)
			if !test.expectError && err != nil {
				t.Fatalf("Expect no error but got %v", err)
			}
			if needResize != test.expectResult {
				t.Fatalf("Expect result is %v but got %v", test.expectResult, needResize)
			}
		})
	}
}
