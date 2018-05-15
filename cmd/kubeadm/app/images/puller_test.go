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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	kubeadmdefaults "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/utils/exec"
)

type fakeCmd struct {
	cmd  string
	args []string
	out  io.Writer
}

func (f *fakeCmd) Run() error {
	fmt.Fprintf(f.out, "%v %v", f.cmd, strings.Join(f.args, " "))
	return nil
}
func (f *fakeCmd) CombinedOutput() ([]byte, error) { return nil, nil }
func (f *fakeCmd) Output() ([]byte, error)         { return nil, nil }
func (f *fakeCmd) SetDir(dir string)               {}
func (f *fakeCmd) SetStdin(in io.Reader)           {}
func (f *fakeCmd) SetStdout(out io.Writer) {
	f.out = out
}
func (f *fakeCmd) SetStderr(out io.Writer) {}
func (f *fakeCmd) Stop()                   {}

type fakeExecer struct {
	cmd              exec.Cmd
	lookPathSucceeds bool
}

func (f *fakeExecer) Command(cmd string, args ...string) exec.Cmd { return f.cmd }
func (f *fakeExecer) CommandContext(ctx context.Context, cmd string, args ...string) exec.Cmd {
	return f.cmd
}
func (f *fakeExecer) LookPath(file string) (string, error) {
	if f.lookPathSucceeds {
		return file, nil
	}
	return "", &os.PathError{Err: errors.New("does not exist")}
}

func TestImagePuller(t *testing.T) {
	testcases := []struct {
		name          string
		criSocket     string
		cmd           exec.Cmd
		findCrictl    bool
		expected      string
		errorExpected bool
	}{
		{
			name:      "New succeeds even if crictl is not in path",
			criSocket: kubeadmdefaults.DefaultCRISocket,
			cmd: &fakeCmd{
				cmd:  "hello",
				args: []string{"world", "and", "friends"},
			},
			findCrictl: false,
			expected:   "hello world and friends",
		},
		{
			name:      "New succeeds with crictl in path",
			criSocket: "/not/default",
			cmd: &fakeCmd{
				cmd:  "crictl",
				args: []string{"-r", "/some/socket", "imagename"},
			},
			findCrictl: true,
			expected:   "crictl -r /some/socket imagename",
		},
		{
			name:      "New fails with crictl not in path but is required",
			criSocket: "/not/docker",
			cmd: &fakeCmd{
				cmd:  "crictl",
				args: []string{"-r", "/not/docker", "an image"},
			},
			findCrictl:    false,
			errorExpected: true,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			var b bytes.Buffer
			tc.cmd.SetStdout(&b)
			fe := &fakeExecer{
				cmd:              tc.cmd,
				lookPathSucceeds: tc.findCrictl,
			}
			ip, err := images.NewImagePuller(fe, tc.criSocket)

			if tc.errorExpected {
				if err == nil {
					t.Fatalf("expected an error but found nil: %v", fe)
				}
				return
			}

			if err != nil {
				t.Fatalf("expected nil but found an error: %v", err)
			}
			if err = ip.Pull("imageName"); err != nil {
				t.Fatalf("expected nil pulling an image but found: %v", err)
			}
			if b.String() != tc.expected {
				t.Fatalf("expected %v but got: %v", tc.expected, b.String())
			}
		})
	}
}
