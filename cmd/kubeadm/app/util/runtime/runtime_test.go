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
	"reflect"
	"testing"

	"github.com/pkg/errors"

	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
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
		{"valid: default cri socket", execLookPathOK, kubeadmapiv1beta1.DefaultCRISocket, true, false},
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
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return nil, nil },
			func() ([]byte, error) { return []byte("error"), &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, error) { return nil, nil },
			func() ([]byte, error) { return []byte("error"), &fakeexec.FakeExitError{Status: 1} },
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
		{"valid: docker is running", kubeadmapiv1beta1.DefaultCRISocket, dockerExecer, false},
		{"invalid: docker is not running", kubeadmapiv1beta1.DefaultCRISocket, dockerExecer, true},
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
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("k8s_p1\nk8s_p2"), nil },
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, error) { return []byte("k8s_p1\nk8s_p2"), nil },
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
		{"valid: list containers using docker", kubeadmapiv1beta1.DefaultCRISocket, false},
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
	fakeOK := func() ([]byte, error) { return nil, nil }
	fakeErr := func() ([]byte, error) { return []byte("error"), &fakeexec.FakeExitError{Status: 1} }
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			fakeOK, fakeOK, fakeOK, fakeOK, fakeOK, fakeOK, // Test case 1
			fakeOK, fakeOK, fakeOK, fakeErr, fakeOK, fakeOK,
			fakeErr, fakeOK, fakeOK, fakeErr, fakeOK,
			fakeOK, fakeOK, fakeOK,
			fakeOK, fakeErr, fakeOK,
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
		{"valid: remove containers using docker", kubeadmapiv1beta1.DefaultCRISocket, []string{"k8s_c1", "k8s_c2", "k8s_c3"}, false},
		{"invalid: remove containers using docker", kubeadmapiv1beta1.DefaultCRISocket, []string{"k8s_c1", "k8s_c2", "k8s_c3"}, true},
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
				t.Errorf("unexpected RemoveContnainers success, criSocket: %s, containers: %v", tc.criSocket, tc.containers)
			}
		})
	}
}

func TestPullImage(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return nil, nil },
			func() ([]byte, error) { return []byte("error"), &fakeexec.FakeExitError{Status: 1} },
			func() ([]byte, error) { return nil, nil },
			func() ([]byte, error) { return []byte("error"), &fakeexec.FakeExitError{Status: 1} },
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
		{"valid: pull image using docker", kubeadmapiv1beta1.DefaultCRISocket, "image1", false},
		{"invalide: docer pull error", kubeadmapiv1beta1.DefaultCRISocket, "image2", true},
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
		RunScript: []fakeexec.FakeRunAction{
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
		{"valid: test if image exists using docker", kubeadmapiv1beta1.DefaultCRISocket, "image1", false},
		{"invalid: docker inspect failure", kubeadmapiv1beta1.DefaultCRISocket, "image2", true},
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
