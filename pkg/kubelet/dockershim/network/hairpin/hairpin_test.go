/*
Copyright 2015 The Kubernetes Authors.

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
	"os"
	"strings"
	"testing"

	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
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
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeAction{
				func() ([]byte, []byte, error) { return []byte(test.output), nil, test.err },
			},
		}
		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fcmd, cmd, args...)
				},
			},
			LookPathFunc: func(file string) (string, error) {
				return fmt.Sprintf("/fake-bin/%s", file), nil
			},
		}
		nsenterArgs := []string{"-t", "123", "-n"}
		name, err := findPairInterfaceOfContainerInterface(&fexec, "eth0", "123", nsenterArgs)
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

func TestSetUpInterfaceNonExistent(t *testing.T) {
	err := setUpInterface("non-existent")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	deviceDir := fmt.Sprintf("%s/%s", sysfsNetPath, "non-existent")
	if !strings.Contains(fmt.Sprintf("%v", err), deviceDir) {
		t.Errorf("should have tried to open %s", deviceDir)
	}
}

func TestSetUpInterfaceNotBridged(t *testing.T) {
	err := setUpInterface("lo")
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("'lo' device does not exist??? (%v)", err)
		}
		t.Errorf("unexpected error: %v", err)
	}
}
