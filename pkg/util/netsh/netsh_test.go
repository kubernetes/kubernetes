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

package netsh

import (
	"net"
	"os"
	"testing"

	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
)

func fakeCommonRunner() *runner {
	fakeCmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// Success
			func() ([]byte, []byte, error) {
				return []byte{}, nil, nil
			},
			// utilexec.ExitError exists, and status is not 0
			func() ([]byte, []byte, error) {
				return nil, nil, &fakeexec.FakeExitError{Status: 1}
			},
			// utilexec.ExitError exists, and status is 0
			func() ([]byte, []byte, error) {
				return nil, nil, &fakeexec.FakeExitError{Status: 0}
			},
			// other error exists
			func() ([]byte, []byte, error) {
				return nil, nil, errors.New("not ExitError")
			},
		},
	}

	return &runner{
		exec: &fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeCmd, cmd, args...)
				},
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeCmd, cmd, args...)
				},
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeCmd, cmd, args...)
				},
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeCmd, cmd, args...)
				},
			},
		},
	}
}

func TestEnsurePortProxyRule(t *testing.T) {
	runner := fakeCommonRunner()

	tests := []struct {
		name           string
		arguments      []string
		expectedResult bool
		expectedError  bool
	}{
		{"Success", []string{"ensure-port-proxy-rule"}, true, false},
		{"utilexec.ExitError exists, and status is not 0", []string{"ensure-port-proxy-rule"}, false, false},
		{"utilexec.ExitError exists, and status is 0", []string{"ensure-port-proxy-rule"}, false, true},
		{"other error exists", []string{"ensure-port-proxy-rule"}, false, true},
	}

	for _, test := range tests {
		result, err := runner.EnsurePortProxyRule(test.arguments)
		if test.expectedError {
			assert.Errorf(t, err, "Failed to test: %s", test.name)
		} else {
			if err != nil {
				assert.NoErrorf(t, err, "Failed to test: %s", test.name)
			} else {
				assert.EqualValuesf(t, test.expectedResult, result, "Failed to test: %s", test.name)
			}
		}
	}

}

func TestDeletePortProxyRule(t *testing.T) {
	runner := fakeCommonRunner()

	tests := []struct {
		name          string
		arguments     []string
		expectedError bool
	}{
		{"Success", []string{"delete-port-proxy-rule"}, false},
		{"utilexec.ExitError exists, and status is not 0", []string{"delete-port-proxy-rule"}, true},
		{"utilexec.ExitError exists, and status is 0", []string{"delete-port-proxy-rule"}, false},
		{"other error exists", []string{"delete-port-proxy-rule"}, true},
	}

	for _, test := range tests {
		err := runner.DeletePortProxyRule(test.arguments)
		if test.expectedError {
			assert.Errorf(t, err, "Failed to test: %s", test.name)
		} else {
			assert.NoErrorf(t, err, "Failed to test: %s", test.name)
		}
	}
}

func TestEnsureIPAddress(t *testing.T) {
	tests := []struct {
		name           string
		arguments      []string
		ip             net.IP
		fakeCmdAction  []fakeexec.FakeCommandAction
		expectedError  bool
		expectedResult bool
	}{
		{
			"IP address exists",
			[]string{"delete-port-proxy-rule"},
			net.IPv4(10, 10, 10, 20),
			[]fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// IP address exists
							func() ([]byte, []byte, error) {
								return []byte("IP Address:10.10.10.10\nIP Address:10.10.10.20"), nil, nil
							},
						},
					}, cmd, args...)
				},
			},
			false,
			true,
		},

		{
			"IP address not exists, but set successful(find it in the second time)",
			[]string{"ensure-ip-address"},
			net.IPv4(10, 10, 10, 20),
			[]fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// IP address not exists
							func() ([]byte, []byte, error) {
								return []byte("IP Address:10.10.10.10"), nil, nil
							},
						},
					}, cmd, args...)
				},
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// Success to set ip
							func() ([]byte, []byte, error) {
								return []byte(""), nil, nil
							},
						},
					}, cmd, args...)
				},
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// IP address still not exists
							func() ([]byte, []byte, error) {
								return []byte("IP Address:10.10.10.10"), nil, nil
							},
						},
					}, cmd, args...)
				},
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// IP address exists
							func() ([]byte, []byte, error) {
								return []byte("IP Address:10.10.10.10\nIP Address:10.10.10.20"), nil, nil
							},
						},
					}, cmd, args...)
				},
			},
			false,
			true,
		},
		{
			"IP address not exists, utilexec.ExitError exists, but status is not 0)",
			[]string{"ensure-ip-address"},
			net.IPv4(10, 10, 10, 20),
			[]fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// IP address not exists
							func() ([]byte, []byte, error) {
								return []byte("IP Address:10.10.10.10"), nil, nil
							},
						},
					}, cmd, args...)
				},
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// Failed to set ip, utilexec.ExitError exists, and status is not 0
							func() ([]byte, []byte, error) {
								return nil, nil, &fakeexec.FakeExitError{Status: 1}
							},
						},
					}, cmd, args...)
				},
			},
			false,
			false,
		},
		{
			"IP address not exists, utilexec.ExitError exists, and status is 0)",
			[]string{"ensure-ip-address"},
			net.IPv4(10, 10, 10, 20),
			[]fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// IP address not exists
							func() ([]byte, []byte, error) {
								return []byte("IP Address:10.10.10.10"), nil, nil
							},
						},
					}, cmd, args...)
				},
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// Failed to set ip, utilexec.ExitError exists, and status is 0
							func() ([]byte, []byte, error) {
								return nil, nil, &fakeexec.FakeExitError{Status: 0}
							},
						},
					}, cmd, args...)
				},
			},
			true,
			false,
		},
		{
			"IP address not exists, and error is not utilexec.ExitError)",
			[]string{"ensure-ip-address"},
			net.IPv4(10, 10, 10, 20),
			[]fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// IP address not exists
							func() ([]byte, []byte, error) {
								return []byte("IP Address:10.10.10.10"), nil, nil
							},
						},
					}, cmd, args...)
				},
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
						CombinedOutputScript: []fakeexec.FakeAction{
							// Failed to set ip, other error exists
							func() ([]byte, []byte, error) {
								return nil, nil, errors.New("not ExitError")
							},
						},
					}, cmd, args...)
				},
			},
			true,
			false,
		},
	}

	for _, test := range tests {
		runner := New(&fakeexec.FakeExec{CommandScript: test.fakeCmdAction})
		result, err := runner.EnsureIPAddress(test.arguments, test.ip)
		if test.expectedError {
			assert.Errorf(t, err, "Failed to test: %s", test.name)
		} else {
			if err != nil {
				assert.NoErrorf(t, err, "Failed to test: %s", test.name)
			} else {
				assert.EqualValuesf(t, test.expectedResult, result, "Failed to test: %s", test.name)
			}
		}
	}
}

func TestDeleteIPAddress(t *testing.T) {
	runner := fakeCommonRunner()

	tests := []struct {
		name          string
		arguments     []string
		expectedError bool
	}{
		{"Success", []string{"delete-ip-address"}, false},
		{"utilexec.ExitError exists, and status is not 0", []string{"delete-ip-address"}, true},
		{"utilexec.ExitError exists, and status is 0", []string{"delete-ip-address"}, false},
		{"other error exists", []string{"delete-ip-address"}, true},
	}

	for _, test := range tests {
		err := runner.DeleteIPAddress(test.arguments)
		if test.expectedError {
			assert.Errorf(t, err, "Failed to test: %s", test.name)
		} else {
			assert.NoErrorf(t, err, "Failed to test: %s", test.name)
		}
	}
}

func TestGetInterfaceToAddIP(t *testing.T) {
	// backup env 'INTERFACE_TO_ADD_SERVICE_IP'
	backupValue := os.Getenv("INTERFACE_TO_ADD_SERVICE_IP")
	// recover env
	defer os.Setenv("INTERFACE_TO_ADD_SERVICE_IP", backupValue)

	tests := []struct {
		name           string
		envToBeSet     string
		expectedResult string
	}{
		{"env_value_is_empty", "", "vEthernet (HNS Internal NIC)"},
		{"env_value_is_not_empty", "eth0", "eth0"},
	}

	fakeExec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{},
	}
	netsh := New(&fakeExec)

	for _, test := range tests {
		os.Setenv("INTERFACE_TO_ADD_SERVICE_IP", test.envToBeSet)
		result := netsh.GetInterfaceToAddIP()

		assert.EqualValuesf(t, test.expectedResult, result, "Failed to test: %s", test.name)
	}
}

func TestRestore(t *testing.T) {
	runner := New(&fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{},
	})

	result := runner.Restore([]string{})
	assert.NoErrorf(t, result, "The return value must be nil")
}

func TestCheckIPExists(t *testing.T) {
	fakeCmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// Error exists
			func() ([]byte, []byte, error) {
				return nil, nil, &fakeexec.FakeExitError{Status: 1}
			},
			// IP address string is empty
			func() ([]byte, []byte, error) {
				return []byte(""), nil, nil
			},
			// "IP Address:" field not exists
			func() ([]byte, []byte, error) {
				return []byte("10.10.10.10"), nil, nil
			},
			// IP not exists
			func() ([]byte, []byte, error) {
				return []byte("IP Address:10.10.10.10"), nil, nil
			},
			// IP exists
			func() ([]byte, []byte, error) {
				return []byte("IP Address:10.10.10.10\nIP Address:10.10.10.20"), nil, nil
			},
		},
	}
	fakeExec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd {
				return fakeexec.InitFakeCmd(&fakeCmd, cmd, args...)
			},
			func(cmd string, args ...string) exec.Cmd {
				return fakeexec.InitFakeCmd(&fakeCmd, cmd, args...)
			},
			func(cmd string, args ...string) exec.Cmd {
				return fakeexec.InitFakeCmd(&fakeCmd, cmd, args...)
			},
			func(cmd string, args ...string) exec.Cmd {
				return fakeexec.InitFakeCmd(&fakeCmd, cmd, args...)
			},
			func(cmd string, args ...string) exec.Cmd {
				return fakeexec.InitFakeCmd(&fakeCmd, cmd, args...)
			},
		},
	}
	fakeRunner := &runner{
		exec: &fakeExec,
	}

	tests := []struct {
		name           string
		ipToCheck      string
		arguments      []string
		expectedError  bool
		expectedResult bool
	}{
		{"Error exists", "10.10.10.20", []string{"check-IP-exists"}, true, false},
		{"IP address string is empty", "10.10.10.20", []string{"check-IP-exists"}, false, false},
		{"'IP Address:' field not exists", "10.10.10.20", []string{"check-IP-exists"}, false, false},
		{"IP not exists", "10.10.10.20", []string{"check-IP-exists"}, false, false},
		{"IP exists", "10.10.10.20", []string{"check-IP-exists"}, false, true},
	}

	for _, test := range tests {
		result, err := checkIPExists(test.ipToCheck, test.arguments, fakeRunner)
		if test.expectedError {
			assert.Errorf(t, err, "Failed to test: %s", test.name)
		} else {
			assert.EqualValuesf(t, test.expectedResult, result, "Failed to test: %s", test.name)
		}
	}
}

func TestGetIP(t *testing.T) {
	testcases := []struct {
		showAddress   string
		expectAddress string
	}{
		{
			showAddress:   "IP 地址:                           10.96.0.2",
			expectAddress: "10.96.0.2",
		},
		{
			showAddress:   "IP Address:                           10.96.0.3",
			expectAddress: "10.96.0.3",
		},
		{
			showAddress:   "IP Address:10.96.0.4",
			expectAddress: "10.96.0.4",
		},
	}

	for _, tc := range testcases {
		address := getIP(tc.showAddress)
		if address != tc.expectAddress {
			t.Errorf("expected address=%q, got %q", tc.expectAddress, address)
		}
	}
}
