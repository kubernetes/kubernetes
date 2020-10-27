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

package util

import (
	"io/ioutil"
	"net"
	"os"
	"reflect"
	"runtime"
	"testing"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func TestNewContainerRuntime(t *testing.T) {
	execLookPathOK := fakeexec.FakeExec{
		LookPathFunc: func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}
	execLookPathErr := fakeexec.FakeExec{
		LookPathFunc: func(cmd string) (string, error) { return "", errors.Errorf("%s not found", cmd) },
	}
	cases := []struct {
		name      string
		execer    fakeexec.FakeExec
		criSocket string
		isDocker  bool
		isError   bool
	}{
		{"valid: default cri socket", execLookPathOK, constants.DefaultDockerCRISocket, true, false},
		{"valid: cri-o socket url", execLookPathOK, "unix:///var/run/crio/crio.sock", false, false},
		{"valid: cri-o socket path", execLookPathOK, "/var/run/crio/crio.sock", false, false},
		{"invalid: no crictl", execLookPathErr, "unix:///var/run/crio/crio.sock", false, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(&tc.execer, tc.criSocket)
			if err != nil {
				if !tc.isError {
					t.Fatalf("unexpected NewContainerRuntime error. criSocket: %s, error: %v", tc.criSocket, err)
				}
				return // expected error occurs, impossible to test runtime further
			}
			if tc.isError && err == nil {
				t.Fatalf("unexpected NewContainerRuntime success. criSocket: %s", tc.criSocket)
			}
			isDocker := runtime.IsDocker()
			if tc.isDocker != isDocker {
				t.Fatalf("unexpected isDocker() result %v for the criSocket %s", isDocker, tc.criSocket)
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

	criExecer := fakeexec.FakeExec{
		CommandScript: genFakeActions(&fcmd, len(fcmd.CombinedOutputScript)),
		LookPathFunc:  func(cmd string) (string, error) { return "/usr/bin/crictl", nil },
	}

	dockerExecer := fakeexec.FakeExec{
		CommandScript: genFakeActions(&fcmd, len(fcmd.CombinedOutputScript)),
		LookPathFunc:  func(cmd string) (string, error) { return "/usr/bin/docker", nil },
	}

	cases := []struct {
		name      string
		criSocket string
		execer    fakeexec.FakeExec
		isError   bool
	}{
		{"valid: CRI-O is running", "unix:///var/run/crio/crio.sock", criExecer, false},
		{"invalid: CRI-O is not running", "unix:///var/run/crio/crio.sock", criExecer, true},
		{"valid: docker is running", constants.DefaultDockerCRISocket, dockerExecer, false},
		{"invalid: docker is not running", constants.DefaultDockerCRISocket, dockerExecer, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(&tc.execer, tc.criSocket)
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
	execer := fakeexec.FakeExec{
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
		{"valid: list containers using docker", constants.DefaultDockerCRISocket, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(&execer, tc.criSocket)
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

func TestRemoveContainers(t *testing.T) {
	fakeOK := func() ([]byte, []byte, error) { return nil, nil, nil }
	fakeErr := func() ([]byte, []byte, error) { return []byte("error"), nil, &fakeexec.FakeExitError{Status: 1} }
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			fakeOK, fakeOK, fakeOK, fakeOK, fakeOK, fakeOK, // Test case 1
			fakeOK, fakeOK, fakeOK, fakeErr, fakeOK, fakeOK,
			fakeErr, fakeOK, fakeOK, fakeErr, fakeOK,
			fakeOK, fakeOK, fakeOK, fakeOK, fakeOK, fakeOK,
			fakeOK, fakeOK, fakeOK, fakeErr, fakeOK, fakeOK,
			fakeErr, fakeOK, fakeOK, fakeErr, fakeOK,
		},
	}
	execer := fakeexec.FakeExec{
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
		{"invalid: CRI rmp failure", "unix:///var/run/crio/crio.sock", []string{"k8s_p1", "k8s_p2", "k8s_p3"}, true},
		{"invalid: CRI stopp failure", "unix:///var/run/crio/crio.sock", []string{"k8s_p1", "k8s_p2", "k8s_p3"}, true},
		{"valid: remove containers using docker", constants.DefaultDockerCRISocket, []string{"k8s_c1", "k8s_c2", "k8s_c3"}, false},
		{"invalid: docker rm failure", constants.DefaultDockerCRISocket, []string{"k8s_c1", "k8s_c2", "k8s_c3"}, true},
		{"invalid: docker stop failure", constants.DefaultDockerCRISocket, []string{"k8s_c1", "k8s_c2", "k8s_c3"}, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(&execer, tc.criSocket)
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
	execer := fakeexec.FakeExec{
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
		{"valid: pull image using docker", constants.DefaultDockerCRISocket, "image1", false},
		{"invalid: docker pull error", constants.DefaultDockerCRISocket, "image2", true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(&execer, tc.criSocket)
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
	execer := fakeexec.FakeExec{
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
		{"invalid: CRI inspecti failure", "unix:///var/run/crio/crio.sock", "image2", true},
		{"valid: test if image exists using docker", constants.DefaultDockerCRISocket, "image1", false},
		{"invalid: docker inspect failure", constants.DefaultDockerCRISocket, "image2", true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runtime, err := NewContainerRuntime(&execer, tc.criSocket)
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
				tmpFile, err := ioutil.TempFile("", tempPrefix)
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

				if !isExistingSocket(theSocket) {
					t.Fatalf("isExistingSocket(%q) gave unexpected result. Should have been true, instead of false", theSocket)
				}
			},
		},
		{
			name: "Regular file is not a domain socket",
			proc: func(t *testing.T) {
				tmpFile, err := ioutil.TempFile("", tempPrefix)
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
			name:            "No existing sockets, use Docker",
			existingSockets: []string{},
			expectedError:   false,
			expectedSocket:  constants.DefaultDockerCRISocket,
		},
		{
			name:            "One valid CRI socket leads to success",
			existingSockets: []string{"/var/run/crio/crio.sock"},
			expectedError:   false,
			expectedSocket:  "/var/run/crio/crio.sock",
		},
		{
			name:            "Correct Docker CRI socket is returned",
			existingSockets: []string{"/var/run/docker.sock"},
			expectedError:   false,
			expectedSocket:  constants.DefaultDockerCRISocket,
		},
		{
			name: "CRI and Docker sockets lead to an error",
			existingSockets: []string{
				"/var/run/docker.sock",
				"/var/run/crio/crio.sock",
			},
			expectedError: true,
		},
		{
			name: "Docker and containerd lead to Docker being used",
			existingSockets: []string{
				"/var/run/docker.sock",
				"/run/containerd/containerd.sock",
			},
			expectedError:  false,
			expectedSocket: constants.DefaultDockerCRISocket,
		},
		{
			name: "A couple of CRI sockets lead to an error",
			existingSockets: []string{
				"/var/run/crio/crio.sock",
				"/run/containerd/containerd.sock",
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
			})
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
