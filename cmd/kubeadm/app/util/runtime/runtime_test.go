/*
Copyright 2018 The Kubernetes Authors.

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

package runtime

import (
	"net"
	"os"
	"reflect"
	"runtime"
	"testing"

	"github.com/pkg/errors"

	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestNewContainerRuntime(t *testing.T) {
	execLookPathOK := &fakeexec.FakeExec{
		LookPathFunc: func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}
	execLookPathErr := &fakeexec.FakeExec{
		LookPathFunc: func(cmd string) (string, error) { return "", errors.Errorf("%s not found", cmd) },
	}
	cases := []struct {
		name    string
		execer  *fakeexec.FakeExec
		isError bool
	}{
		{"valid: crictl present", execLookPathOK, false},
		{"invalid: no crictl", execLookPathErr, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewContainerRuntime(tc.execer, "unix:///some/socket.sock")
			if err != nil {
				if !tc.isError {
					t.Fatalf("unexpected NewContainerRuntime error. error: %v", err)
				}
				return // expected error occurs, impossible to test runtime further
			}
			if tc.isError && err == nil {
				t.Fatal("unexpected NewContainerRuntime success")
			}
		})
	}
}

func genFakeActions(fcmd *fakeexec.FakeCmd, num int) []fakeexec.FakeCommandAction {
	var actions []fakeexec.FakeCommandAction
	for i := 0; i < num; i++ {
		actions = append(actions, func(cmd string, args ...string) exec.Cmd {
			return fakeexec.InitFakeCmd(fcmd, cmd, args...)
		})
	}
	return actions
}

func TestIsRunning(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}

	criExecer := &fakeexec.FakeExec{
		CommandScript: genFakeActions(&fcmd, len(fcmd.CombinedOutputScript)),
		LookPathFunc:  func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}

	cases := []struct {
		name      string
		criSocket string
		execer    *fakeexec.FakeExec
		isError   bool
	}{
		{"valid: CRI-O is running", "unix:///var/run/crio/crio.sock", criExecer, false},
		{"invalid: CRI-O is not running", "unix:///var/run/crio/crio.sock", criExecer, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(tc.execer, tc.criSocket)
			if err != nil {
				t.Fatalf("unexpected NewContainerRuntime error: %v", err)
			}
			isRunning := runtime.IsRunning()
			if tc.isError && isRunning == nil {
				t.Error("unexpected IsRunning() success")
			}
			if !tc.isError && isRunning != nil {
				t.Error("unexpected IsRunning() error")
			}
		})
	}
}

func TestListKubeContainers(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte("k8s_p1\nk8s_p2"), nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return []byte("k8s_p1\nk8s_p2"), nil, nil },
		},
	}
	execer := &fakeexec.FakeExec{
		CommandScript: genFakeActions(&fcmd, len(fcmd.CombinedOutputScript)),
		LookPathFunc:  func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}

	cases := []struct {
		name      string
		criSocket string
		isError   bool
	}{
		{"valid: list containers using CRI socket url", "unix:///var/run/crio/crio.sock", false},
		{"invalid: list containers using CRI socket url", "unix:///var/run/crio/crio.sock", true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(execer, tc.criSocket)
			if err != nil {
				t.Fatalf("unexpected NewContainerRuntime error: %v", err)
			}

			containers, err := runtime.ListKubeContainers()
			if tc.isError {
				if err == nil {
					t.Errorf("unexpected ListKubeContainers success")
				}
				return
			} else if err != nil {
				t.Errorf("unexpected ListKubeContainers error: %v", err)
			}

			if !reflect.DeepEqual(containers, []string{"k8s_p1", "k8s_p2"}) {
				t.Errorf("unexpected ListKubeContainers output: %v", containers)
			}
		})
	}
}

func TestSandboxImage(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte("registry.k8s.io/pause:3.9"), nil, nil },
			func() ([]byte, []byte, error) { return []byte("registry.k8s.io/pause:3.9\n"), nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}

	execer := &fakeexec.FakeExec{
		CommandScript: genFakeActions(&fcmd, len(fcmd.CombinedOutputScript)),
		LookPathFunc:  func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}

	cases := []struct {
		name     string
		expected string
		isError  bool
	}{
		{"valid: read sandbox image normally", "registry.k8s.io/pause:3.9", false},
		{"valid: read sandbox image with leading/trailing white spaces", "registry.k8s.io/pause:3.9", false},
		{"invalid: read empty sandbox image", "", true},
		{"invalid: failed to read sandbox image", "", true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(execer, "unix:///some/socket.sock")
			if err != nil {
				t.Fatalf("unexpected NewContainerRuntime error: %v", err)
			}

			sandboxImage, err := runtime.SandboxImage()
			if tc.isError {
				if err == nil {
					t.Errorf("unexpected SandboxImage success")
				}
				return
			} else if err != nil {
				t.Errorf("unexpected SandboxImage error: %v", err)
			}

			if sandboxImage != tc.expected {
				t.Errorf("expected sandbox image %v, but got %v", tc.expected, sandboxImage)
			}
		})
	}
}

func TestRemoveContainers(t *testing.T) {
	fakeOK := func() ([]byte, []byte, error) { return nil, nil, nil }
	fakeErr := func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} }
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			fakeOK, fakeOK, fakeOK, fakeOK, fakeOK, fakeOK, // Test case 1
			fakeOK, fakeOK, fakeOK, fakeErr, fakeOK, fakeErr, fakeOK, fakeErr, fakeOK, fakeErr, fakeOK, fakeErr, fakeOK, fakeOK, // Test case 2
			fakeErr, fakeErr, fakeErr, fakeErr, fakeErr, fakeOK, fakeOK, fakeOK, fakeOK, // Test case 3
		},
	}
	execer := &fakeexec.FakeExec{
		CommandScript: genFakeActions(&fcmd, len(fcmd.CombinedOutputScript)),
		LookPathFunc:  func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}

	cases := []struct {
		name       string
		criSocket  string
		containers []string
		isError    bool
	}{
		{"valid: remove containers using CRI", "unix:///var/run/crio/crio.sock", []string{"k8s_p1", "k8s_p2", "k8s_p3"}, false}, // Test case 1
		{"invalid: CRI rmp failure", "unix:///var/run/crio/crio.sock", []string{"k8s_p1", "k8s_p2", "k8s_p3"}, true},            // Test case 2
		{"invalid: CRI stopp failure", "unix:///var/run/crio/crio.sock", []string{"k8s_p1", "k8s_p2", "k8s_p3"}, true},          // Test case 3
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(execer, tc.criSocket)
			if err != nil {
				t.Fatalf("unexpected NewContainerRuntime error: %v, criSocket: %s", err, tc.criSocket)
			}

			err = runtime.RemoveContainers(tc.containers)
			if !tc.isError && err != nil {
				t.Errorf("unexpected RemoveContainers errors: %v, criSocket: %s, containers: %v", err, tc.criSocket, tc.containers)
			}
			if tc.isError && err == nil {
				t.Errorf("unexpected RemoveContainers success, criSocket: %s, containers: %v", tc.criSocket, tc.containers)
			}
		})
	}
}

func TestPullImage(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return nil, nil, nil },
			// If the pull fails, it will be retried 5 times (see PullImageRetry in constants/constants.go)
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return nil, nil, nil },
			// If the pull fails, it will be retried 5 times (see PullImageRetry in constants/constants.go)
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	execer := &fakeexec.FakeExec{
		CommandScript: genFakeActions(&fcmd, len(fcmd.CombinedOutputScript)),
		LookPathFunc:  func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}

	cases := []struct {
		name      string
		criSocket string
		image     string
		isError   bool
	}{
		{"valid: pull image using CRI", "unix:///var/run/crio/crio.sock", "image1", false},
		{"invalid: CRI pull error", "unix:///var/run/crio/crio.sock", "image2", true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(execer, tc.criSocket)
			if err != nil {
				t.Fatalf("unexpected NewContainerRuntime error: %v, criSocket: %s", err, tc.criSocket)
			}

			err = runtime.PullImage(tc.image)
			if !tc.isError && err != nil {
				t.Errorf("unexpected PullImage error: %v, criSocket: %s, image: %s", err, tc.criSocket, tc.image)
			}
			if tc.isError && err == nil {
				t.Errorf("unexpected PullImage success, criSocket: %s, image: %s", tc.criSocket, tc.image)
			}
		})
	}
}

func TestImageExists(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		RunScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, []byte, error) { return nil, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	execer := &fakeexec.FakeExec{
		CommandScript: genFakeActions(&fcmd, len(fcmd.RunScript)),
		LookPathFunc:  func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}

	cases := []struct {
		name      string
		criSocket string
		image     string
		result    bool
	}{
		{"valid: test if image exists using CRI", "unix:///var/run/crio/crio.sock", "image1", false},
		{"invalid: CRI inspect failure", "unix:///var/run/crio/crio.sock", "image2", true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(execer, tc.criSocket)
			if err != nil {
				t.Fatalf("unexpected NewContainerRuntime error: %v, criSocket: %s", err, tc.criSocket)
			}

			result, err := runtime.ImageExists(tc.image)
			if !tc.result != result {
				t.Errorf("unexpected ImageExists result: %t, criSocket: %s, image: %s, expected result: %t", err, tc.criSocket, tc.image, tc.result)
			}
		})
	}
}

func TestIsExistingSocket(t *testing.T) {
	// this test is not expected to work on Windows
	if runtime.GOOS == "windows" {
		return
	}

	const tempPrefix = "test.kubeadm.runtime.isExistingSocket."
	tests := []struct {
		name string
		proc func(*testing.T)
	}{
		{
			name: "Valid domain socket is detected as such",
			proc: func(t *testing.T) {
				tmpFile, err := os.CreateTemp("", tempPrefix)
				if err != nil {
					t.Fatalf("unexpected error by TempFile: %v", err)
				}
				theSocket := tmpFile.Name()
				os.Remove(theSocket)
				tmpFile.Close()

				con, err := net.Listen("unix", theSocket)
				if err != nil {
					t.Fatalf("unexpected error while dialing a socket: %v", err)
				}
				defer con.Close()

				if !isExistingSocket("unix://" + theSocket) {
					t.Fatalf("isExistingSocket(%q) gave unexpected result. Should have been true, instead of false", theSocket)
				}
			},
		},
		{
			name: "Regular file is not a domain socket",
			proc: func(t *testing.T) {
				tmpFile, err := os.CreateTemp("", tempPrefix)
				if err != nil {
					t.Fatalf("unexpected error by TempFile: %v", err)
				}
				theSocket := tmpFile.Name()
				defer os.Remove(theSocket)
				tmpFile.Close()

				if isExistingSocket(theSocket) {
					t.Fatalf("isExistingSocket(%q) gave unexpected result. Should have been false, instead of true", theSocket)
				}
			},
		},
		{
			name: "Non existent socket is not a domain socket",
			proc: func(t *testing.T) {
				const theSocket = "/non/existent/socket"
				if isExistingSocket(theSocket) {
					t.Fatalf("isExistingSocket(%q) gave unexpected result. Should have been false, instead of true", theSocket)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, test.proc)
	}
}

func TestDetectCRISocketImpl(t *testing.T) {
	tests := []struct {
		name            string
		existingSockets []string
		expectedError   bool
		expectedSocket  string
	}{
		{
			name:            "No existing sockets, use default",
			existingSockets: []string{},
			expectedError:   false,
			expectedSocket:  constants.DefaultCRISocket,
		},
		{
			name:            "One valid CRI socket leads to success",
			existingSockets: []string{"unix:///foo/bar.sock"},
			expectedError:   false,
			expectedSocket:  "unix:///foo/bar.sock",
		},
		{
			name: "Multiple CRI sockets lead to an error",
			existingSockets: []string{
				"unix:///foo/bar.sock",
				"unix:///foo/baz.sock",
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			socket, err := detectCRISocketImpl(func(path string) bool {
				for _, existing := range test.existingSockets {
					if path == existing {
						return true
					}
				}
				return false
			}, test.existingSockets)

			if (err != nil) != test.expectedError {
				t.Fatalf("detectCRISocketImpl returned unexpected result\n\tExpected error: %t\n\tGot error: %t", test.expectedError, err != nil)
			}
			if !test.expectedError && socket != test.expectedSocket {
				t.Fatalf("detectCRISocketImpl returned unexpected CRI socket\n\tExpected socket: %s\n\tReturned socket: %s",
					test.expectedSocket, socket)
			}
		})
	}
}

func TestPullImagesInParallelImpl(t *testing.T) {
	testError := errors.New("error")

	tests := []struct {
		name            string
		images          []string
		ifNotPresent    bool
		imageExistsFunc func(string) (bool, error)
		pullImageFunc   func(string) error
		expectedErrors  int
	}{
		{
			name:         "all images exist, no errors",
			images:       []string{"foo", "bar", "baz"},
			ifNotPresent: true,
			imageExistsFunc: func(string) (bool, error) {
				return true, nil
			},
			pullImageFunc:  nil,
			expectedErrors: 0,
		},
		{
			name:         "cannot check if one image exists due to error",
			images:       []string{"foo", "bar", "baz"},
			ifNotPresent: true,
			imageExistsFunc: func(image string) (bool, error) {
				if image == "baz" {
					return false, testError
				}
				return true, nil
			},
			pullImageFunc:  nil,
			expectedErrors: 1,
		},
		{
			name:         "cannot pull two images",
			images:       []string{"foo", "bar", "baz"},
			ifNotPresent: true,
			imageExistsFunc: func(string) (bool, error) {
				return false, nil
			},
			pullImageFunc: func(image string) error {
				if image == "foo" {
					return nil
				}
				return testError
			},
			expectedErrors: 2,
		},
		{
			name:         "pull all images",
			images:       []string{"foo", "bar", "baz"},
			ifNotPresent: true,
			imageExistsFunc: func(string) (bool, error) {
				return false, nil
			},
			pullImageFunc: func(string) error {
				return nil
			},
			expectedErrors: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := pullImagesInParallelImpl(tc.images, tc.ifNotPresent,
				tc.imageExistsFunc, tc.pullImageFunc)
			if len(actual) != tc.expectedErrors {
				t.Fatalf("expected non-nil errors: %v, got: %v, full list of errors: %v",
					tc.expectedErrors, len(actual), actual)
			}
		})
	}
}
