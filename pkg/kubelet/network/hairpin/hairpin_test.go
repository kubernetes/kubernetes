/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package hairpin

import (
	"errors"
	"fmt"
	"net"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/util/exec"
)

func TestFindPairInterfaceOfContainerInterface(t *testing.T) {
	// there should be at least "lo" on any system
	interfaces, _ := net.Interfaces()
	validOutput := fmt.Sprintf("garbage\n   peer_ifindex: %d", interfaces[0].Index)
	invalidOutput := fmt.Sprintf("garbage\n   unknown: %d", interfaces[0].Index)

	tests := []struct {
		output       string
		err          error
		expectedName string
		expectErr    bool
	}{
		{
			output:       validOutput,
			expectedName: interfaces[0].Name,
		},
		{
			output:    invalidOutput,
			expectErr: true,
		},
		{
			output:    validOutput,
			err:       errors.New("error"),
			expectErr: true,
		},
	}
	for _, test := range tests {
		fcmd := exec.FakeCmd{
			CombinedOutputScript: []exec.FakeCombinedOutputAction{
				func() ([]byte, error) { return []byte(test.output), test.err },
			},
		}
		fexec := exec.FakeExec{
			CommandScript: []exec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return exec.InitFakeCmd(&fcmd, cmd, args...)
				},
			},
			LookPathFunc: func(file string) (string, error) {
				return fmt.Sprintf("/fake-bin/%s", file), nil
			},
		}
		name, err := findPairInterfaceOfContainerInterface(&fexec, 123, "eth0")
		if test.expectErr {
			if err == nil {
				t.Errorf("unexpected non-error")
			}
		} else {
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		}
		if name != test.expectedName {
			t.Errorf("unexpected name: %s (expected: %s)", name, test.expectedName)
		}
	}
}

func TestSetUpInterface(t *testing.T) {
	err := setUpInterface("non-existent")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	hairpinModeFile := fmt.Sprintf("%s/%s/%s", sysfsNetPath, "non-existent", hairpinModeRelativePath)
	if !strings.Contains(fmt.Sprintf("%v", err), hairpinModeFile) {
		t.Errorf("should have tried to open %s", hairpinModeFile)
	}
}
