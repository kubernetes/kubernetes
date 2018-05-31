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

package images_test

import (
	"context"
	"errors"
	"io"
	"testing"

	kubeadmapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/utils/exec"
)

type fakeCmd struct {
	err error
}

func (f *fakeCmd) Run() error {
	return f.err
}
func (f *fakeCmd) CombinedOutput() ([]byte, error) { return nil, nil }
func (f *fakeCmd) Output() ([]byte, error)         { return nil, nil }
func (f *fakeCmd) SetDir(dir string)               {}
func (f *fakeCmd) SetStdin(in io.Reader)           {}
func (f *fakeCmd) SetStdout(out io.Writer)         {}
func (f *fakeCmd) SetStderr(out io.Writer)         {}
func (f *fakeCmd) Stop()                           {}

type fakeExecer struct {
	cmd        exec.Cmd
	findCrictl bool
	findDocker bool
}

func (f *fakeExecer) Command(cmd string, args ...string) exec.Cmd { return f.cmd }
func (f *fakeExecer) CommandContext(ctx context.Context, cmd string, args ...string) exec.Cmd {
	return f.cmd
}
func (f *fakeExecer) LookPath(file string) (string, error) {
	if file == "crictl" {
		if f.findCrictl {
			return "/path", nil
		}
		return "", errors.New("no crictl for you")
	}
	if file == "docker" {
		if f.findDocker {
			return "/path", nil
		}
		return "", errors.New("no docker for you")
	}
	return "", errors.New("unknown binary")
}

func TestNewCRInterfacer(t *testing.T) {
	testcases := []struct {
		name        string
		criSocket   string
		findCrictl  bool
		findDocker  bool
		expectError bool
	}{
		{
			name:        "need crictl but can only find docker should return an error",
			criSocket:   "/not/docker",
			findCrictl:  false,
			findDocker:  true,
			expectError: true,
		},
		{
			name:        "need crictl and cannot find either should return an error",
			criSocket:   "/not/docker",
			findCrictl:  false,
			findDocker:  false,
			expectError: true,
		},
		{
			name:        "need crictl and cannot find docker should return no error",
			criSocket:   "/not/docker",
			findCrictl:  true,
			findDocker:  false,
			expectError: false,
		},
		{
			name:        "need crictl and can find both should return no error",
			criSocket:   "/not/docker",
			findCrictl:  true,
			findDocker:  true,
			expectError: false,
		},
		{
			name:        "need docker and cannot find crictl should return no error",
			criSocket:   kubeadmapiv1alpha2.DefaultCRISocket,
			findCrictl:  false,
			findDocker:  true,
			expectError: false,
		},
		{
			name:        "need docker and cannot find docker should return an error",
			criSocket:   kubeadmapiv1alpha2.DefaultCRISocket,
			findCrictl:  false,
			findDocker:  false,
			expectError: true,
		},
		{
			name:        "need docker and can find both should return no error",
			criSocket:   kubeadmapiv1alpha2.DefaultCRISocket,
			findCrictl:  true,
			findDocker:  true,
			expectError: false,
		},
		{
			name:        "need docker and can only find crictl should return an error",
			criSocket:   kubeadmapiv1alpha2.DefaultCRISocket,
			findCrictl:  true,
			findDocker:  false,
			expectError: true,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			fe := &fakeExecer{
				findCrictl: tc.findCrictl,
				findDocker: tc.findDocker,
			}
			_, err := images.NewCRInterfacer(fe, tc.criSocket)
			if tc.expectError && err == nil {
				t.Fatal("expected an error but did not get one")
			}
			if !tc.expectError && err != nil {
				t.Fatalf("did not expedt an error but got an error: %v", err)
			}
		})
	}
}

func TestImagePuller(t *testing.T) {
	testcases := []struct {
		name          string
		criSocket     string
		pullFails     bool
		errorExpected bool
	}{
		{
			name:          "using docker and pull fails",
			criSocket:     kubeadmapiv1alpha2.DefaultCRISocket,
			pullFails:     true,
			errorExpected: true,
		},
		{
			name:          "using docker and pull succeeds",
			criSocket:     kubeadmapiv1alpha2.DefaultCRISocket,
			pullFails:     false,
			errorExpected: false,
		},
		{
			name:          "using crictl pull fails",
			criSocket:     "/not/default",
			pullFails:     true,
			errorExpected: true,
		},
		{
			name:          "using crictl and pull succeeds",
			criSocket:     "/not/default",
			pullFails:     false,
			errorExpected: false,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			var err error
			if tc.pullFails {
				err = errors.New("error")
			}

			fe := &fakeExecer{
				cmd:        &fakeCmd{err},
				findCrictl: true,
				findDocker: true,
			}
			ip, _ := images.NewCRInterfacer(fe, tc.criSocket)

			err = ip.Pull("imageName")
			if tc.errorExpected && err == nil {
				t.Fatal("expected an error and did not get one")
			}
			if !tc.errorExpected && err != nil {
				t.Fatalf("expected no error but got one: %v", err)
			}
		})
	}
}

func TestImageExists(t *testing.T) {
	testcases := []struct {
		name          string
		criSocket     string
		existFails    bool
		errorExpected bool
	}{
		{
			name:          "using docker and exist fails",
			criSocket:     kubeadmapiv1alpha2.DefaultCRISocket,
			existFails:    true,
			errorExpected: true,
		},
		{
			name:          "using docker and exist succeeds",
			criSocket:     kubeadmapiv1alpha2.DefaultCRISocket,
			existFails:    false,
			errorExpected: false,
		},
		{
			name:          "using crictl exist fails",
			criSocket:     "/not/default",
			existFails:    true,
			errorExpected: true,
		},
		{
			name:          "using crictl and exist succeeds",
			criSocket:     "/not/default",
			existFails:    false,
			errorExpected: false,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			var err error
			if tc.existFails {
				err = errors.New("error")
			}

			fe := &fakeExecer{
				cmd:        &fakeCmd{err},
				findCrictl: true,
				findDocker: true,
			}
			ip, _ := images.NewCRInterfacer(fe, tc.criSocket)

			err = ip.Exists("imageName")
			if tc.errorExpected && err == nil {
				t.Fatal("expected an error and did not get one")
			}
			if !tc.errorExpected && err != nil {
				t.Fatalf("expected no error but got one: %v", err)
			}
		})
	}
}
