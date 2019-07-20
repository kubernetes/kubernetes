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

package kubelet

import (
	"context"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/utils/exec"
)

type fakeCmd struct {
	b   []byte
	err error
}

func (f fakeCmd) Run() error                         { return f.err }
func (f fakeCmd) CombinedOutput() ([]byte, error)    { return f.b, f.err }
func (f fakeCmd) Output() ([]byte, error)            { return f.b, f.err }
func (f fakeCmd) SetDir(dir string)                  {}
func (f fakeCmd) SetStdin(in io.Reader)              {}
func (f fakeCmd) SetStdout(out io.Writer)            {}
func (f fakeCmd) SetStderr(out io.Writer)            {}
func (f fakeCmd) SetEnv([]string)                    {}
func (f fakeCmd) Stop()                              {}
func (f fakeCmd) Start() error                       { return nil }
func (f fakeCmd) Wait() error                        { return nil }
func (f fakeCmd) StdoutPipe() (io.ReadCloser, error) { return nil, nil }
func (f fakeCmd) StderrPipe() (io.ReadCloser, error) { return nil, nil }

type fakeExecer struct {
	ioMap map[string]fakeCmd
}

func (f fakeExecer) Command(cmd string, args ...string) exec.Cmd {
	cmds := []string{cmd}
	cmds = append(cmds, args...)
	return f.ioMap[strings.Join(cmds, " ")]
}
func (f fakeExecer) CommandContext(ctx context.Context, cmd string, args ...string) exec.Cmd {
	return f.Command(cmd, args...)
}
func (f fakeExecer) LookPath(file string) (string, error) { return "", errors.New("unknown binary") }

var (
	systemdCgroupExecer = fakeExecer{
		ioMap: map[string]fakeCmd{
			"docker info -f {{.CgroupDriver}}": {
				b: []byte(`systemd`),
			},
		},
	}

	cgroupfsCgroupExecer = fakeExecer{
		ioMap: map[string]fakeCmd{
			"docker info -f {{.CgroupDriver}}": {
				b: []byte(`cgroupfs`),
			},
		},
	}

	errCgroupExecer = fakeExecer{
		ioMap: map[string]fakeCmd{
			"docker info -f {{.CgroupDriver}}": {
				err: errors.New("no such binary: docker"),
			},
		},
	}
)

func serviceIsActiveFunc(_ string) (bool, error) {
	return true, nil
}

func serviceIsNotActiveFunc(_ string) (bool, error) {
	return false, nil
}

func TestBuildKubeletArgMap(t *testing.T) {

	tests := []struct {
		name     string
		opts     kubeletFlagsOpts
		expected map[string]string
	}{
		{
			name: "the simplest case",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/dockershim.sock",
					Name:      "foo",
					Taints: []v1.Taint{ // This should be ignored as registerTaintsUsingFlags is false
						{
							Key:    "foo",
							Value:  "bar",
							Effect: "baz",
						},
					},
				},
				execer:              errCgroupExecer,
				isServiceActiveFunc: serviceIsNotActiveFunc,
				defaultHostname:     "foo",
			},
			expected: map[string]string{
				"network-plugin": "cni",
			},
		},
		{
			name: "nodeRegOpts.Name != default hostname",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/dockershim.sock",
					Name:      "override-name",
				},
				execer:              errCgroupExecer,
				isServiceActiveFunc: serviceIsNotActiveFunc,
				defaultHostname:     "default",
			},
			expected: map[string]string{
				"network-plugin":    "cni",
				"hostname-override": "override-name",
			},
		},
		{
			name: "systemd cgroup driver",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/dockershim.sock",
					Name:      "foo",
				},
				execer:              systemdCgroupExecer,
				isServiceActiveFunc: serviceIsNotActiveFunc,
				defaultHostname:     "foo",
			},
			expected: map[string]string{
				"network-plugin": "cni",
				"cgroup-driver":  "systemd",
			},
		},
		{
			name: "cgroupfs cgroup driver",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/dockershim.sock",
					Name:      "foo",
				},
				execer:              cgroupfsCgroupExecer,
				isServiceActiveFunc: serviceIsNotActiveFunc,
				defaultHostname:     "foo",
			},
			expected: map[string]string{
				"network-plugin": "cni",
				"cgroup-driver":  "cgroupfs",
			},
		},
		{
			name: "external CRI runtime",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/containerd.sock",
					Name:      "foo",
				},
				execer:              cgroupfsCgroupExecer,
				isServiceActiveFunc: serviceIsNotActiveFunc,
				defaultHostname:     "foo",
			},
			expected: map[string]string{
				"container-runtime":          "remote",
				"container-runtime-endpoint": "/var/run/containerd.sock",
			},
		},
		{
			name: "register with taints",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/containerd.sock",
					Name:      "foo",
					Taints: []v1.Taint{
						{
							Key:    "foo",
							Value:  "bar",
							Effect: "baz",
						},
						{
							Key:    "key",
							Value:  "val",
							Effect: "eff",
						},
					},
				},
				registerTaintsUsingFlags: true,
				execer:                   cgroupfsCgroupExecer,
				isServiceActiveFunc:      serviceIsNotActiveFunc,
				defaultHostname:          "foo",
			},
			expected: map[string]string{
				"container-runtime":          "remote",
				"container-runtime-endpoint": "/var/run/containerd.sock",
				"register-with-taints":       "foo=bar:baz,key=val:eff",
			},
		},
		{
			name: "systemd-resolved running",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/containerd.sock",
					Name:      "foo",
				},
				execer:              cgroupfsCgroupExecer,
				isServiceActiveFunc: serviceIsActiveFunc,
				defaultHostname:     "foo",
			},
			expected: map[string]string{
				"container-runtime":          "remote",
				"container-runtime-endpoint": "/var/run/containerd.sock",
				"resolv-conf":                "/run/systemd/resolve/resolv.conf",
			},
		},
		{
			name: "pause image is set",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/dockershim.sock",
					Name:      "foo",
				},
				pauseImage:          "gcr.io/pause:3.1",
				execer:              cgroupfsCgroupExecer,
				isServiceActiveFunc: serviceIsNotActiveFunc,
				defaultHostname:     "foo",
			},
			expected: map[string]string{
				"network-plugin":            "cni",
				"cgroup-driver":             "cgroupfs",
				"pod-infra-container-image": "gcr.io/pause:3.1",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := buildKubeletArgMap(test.opts)
			if !reflect.DeepEqual(actual, test.expected) {
				t.Errorf(
					"failed buildKubeletArgMap:\n\texpected: %v\n\t  actual: %v",
					test.expected,
					actual,
				)
			}
		})
	}
}
