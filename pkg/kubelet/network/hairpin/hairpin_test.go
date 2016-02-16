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
	"os"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/util/exec"
)

func TestFindPairInterfaceOfContainerInterface(t *testing.T) {
	// there should be at least "lo" on any system
	interfaces, _ := net.Interfaces()
	validEthtoolOutput := fmt.Sprintf("garbage\n   peer_ifindex: %d", interfaces[0].Index)
	invalidEthtoolOutput := fmt.Sprintf("garbage\n   unknown: %d", interfaces[0].Index)
	validSysfsOutput := fmt.Sprintf("%d", interfaces[0].Index)
	invalidSysfsOutput := fmt.Sprintf("adsfadsf")

	tests := []struct {
		sysfsOutput   string
		ethtoolOutput string
		err           error
		expectedName  string
		expectErr     bool
	}{
		{
			// cat exists; ethtool does not
			sysfsOutput:  validSysfsOutput,
			expectedName: interfaces[0].Name,
		},
		{
			// ethtool exists; cat does not
			ethtoolOutput: validEthtoolOutput,
			expectedName:  interfaces[0].Name,
		},
		{
			// neither binary exists
			expectedName: interfaces[0].Name,
			expectErr:    true,
		},

		{
			// valid sysfs output
			sysfsOutput:  validSysfsOutput,
			expectedName: interfaces[0].Name,
		},
		{
			// invalid sysfs output
			sysfsOutput: invalidSysfsOutput,
			expectErr:   true,
		},
		{
			// valid sysfs output, but error
			sysfsOutput: validSysfsOutput,
			err:         errors.New("error"),
			expectErr:   true,
		},

		{
			// valid ethtool output
			ethtoolOutput: validEthtoolOutput,
			expectedName:  interfaces[0].Name,
		},
		{
			// invalid ethtool output
			ethtoolOutput: invalidEthtoolOutput,
			expectErr:     true,
		},
		{
			// valid ethtool output, but error
			ethtoolOutput: validEthtoolOutput,
			err:           errors.New("error"),
			expectErr:     true,
		},
	}

	for _, test := range tests {
		fEthtoolCmd := exec.FakeCmd{
			CombinedOutputScript: []exec.FakeCombinedOutputAction{
				func() ([]byte, error) { return []byte(test.ethtoolOutput), test.err },
			},
		}
		fSysfsCmd := exec.FakeCmd{
			CombinedOutputScript: []exec.FakeCombinedOutputAction{
				func() ([]byte, error) { return []byte(test.sysfsOutput), test.err },
			},
		}

		fexec := exec.FakeExec{
			CommandScript: []exec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					allArgs := strings.Join(args, " ")
					if strings.Index(allArgs, "ethtool") >= 0 {
						return exec.InitFakeCmd(&fEthtoolCmd, cmd, args...)
					} else if strings.Index(allArgs, "cat") >= 0 {
						return exec.InitFakeCmd(&fSysfsCmd, cmd, args...)
					}
					panic("Invalid command")
				},
			},
			LookPathFunc: func(file string) (string, error) {
				if file == "cat" && test.sysfsOutput == "" {
					return "", fmt.Errorf("cat not found")
				} else if file == "ethtool" && test.ethtoolOutput == "" {
					return "", fmt.Errorf("ethtool not found")
				}
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
		if !test.expectErr && name != test.expectedName {
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
