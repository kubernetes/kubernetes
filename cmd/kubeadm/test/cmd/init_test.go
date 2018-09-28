/*
Copyright 2016 The Kubernetes Authors.

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

package kubeadm

import (
	"testing"

	"github.com/renstrom/dedent"
)

func runKubeadmInit(args ...string) (string, string, error) {
	kubeadmPath := getKubeadmPath()
	kubeadmArgs := []string{"init", "--dry-run", "--ignore-preflight-errors=all"}
	kubeadmArgs = append(kubeadmArgs, args...)
	return RunCmd(kubeadmPath, kubeadmArgs...)
}

func TestCmdInitToken(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	initTest := []struct {
		name     string
		args     string
		expected bool
	}{
		{
			name:     "invalid token size",
			args:     "--token=abcd:1234567890abcd",
			expected: false,
		},
		{
			name:     "invalid token non-lowercase",
			args:     "--token=Abcdef:1234567890abcdef",
			expected: false,
		},
		{
			name:     "valid token is accepted",
			args:     "--token=abcdef.0123456789abcdef",
			expected: true,
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, err := runKubeadmInit(rt.args)
			if (err == nil) != rt.expected {
				t.Fatalf(dedent.Dedent(`
					CmdInitToken test case %q failed with an error: %v
					command 'kubeadm init %s'
						expected: %t
						err: %t
					`),
					rt.name,
					err,
					rt.args,
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}

func TestCmdInitKubernetesVersion(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	initTest := []struct {
		name     string
		args     string
		expected bool
	}{
		{
			name:     "invalid semantic version string is detected",
			args:     "--kubernetes-version=v1.1",
			expected: false,
		},
		{
			name:     "valid version is accepted",
			args:     "--kubernetes-version=1.11.0",
			expected: true,
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, err := runKubeadmInit(rt.args)
			if (err == nil) != rt.expected {
				t.Fatalf(dedent.Dedent(`
					CmdInitKubernetesVersion test case %q failed with an error: %v
					command 'kubeadm init %s'
						expected: %t
						err: %t
					`),
					rt.name,
					err,
					rt.args,
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}

func TestCmdInitConfig(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	initTest := []struct {
		name     string
		args     string
		expected bool
	}{
		{
			name:     "fail on non existing path",
			args:     "--config=/does/not/exist/foo/bar",
			expected: false,
		},
		{
			name:     "can't load v1alpha1 config",
			args:     "--config=testdata/init/v1alpha1.yaml",
			expected: false,
		},
		{
			name:     "can't load v1alpha2 config",
			args:     "--config=testdata/init/v1alpha2.yaml",
			expected: false,
		},
		{
			name:     "can load v1alpha3 config",
			args:     "--config=testdata/init/v1alpha3.yaml",
			expected: true,
		},
		// TODO: implement v1beta1 tests after introducing v1beta1
		{
			name:     "don't allow mixed arguments",
			args:     "--kubernetes-version=1.11.0 --config=testdata/init/v1alpha3.yaml",
			expected: false,
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, err := runKubeadmInit(rt.args)
			if (err == nil) != rt.expected {
				t.Fatalf(dedent.Dedent(`
						CmdInitConfig test case %q failed with an error: %v
						command 'kubeadm init %s'
							expected: %t
							err: %t
						`),
					rt.name,
					err,
					rt.args,
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}

func TestCmdInitAPIPort(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	initTest := []struct {
		name     string
		args     string
		expected bool
	}{
		{
			name:     "fail on non-string port",
			args:     "--apiserver-bind-port=foobar",
			expected: false,
		},
		{
			name:     "fail on too large port number",
			args:     "--apiserver-bind-port=100000",
			expected: false,
		},
		{
			name:     "fail on negative port number",
			args:     "--apiserver-bind-port=-6000",
			expected: false,
		},
		{
			name:     "accept a valid port number",
			args:     "--apiserver-bind-port=6000",
			expected: true,
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, err := runKubeadmInit(rt.args)
			if (err == nil) != rt.expected {
				t.Fatalf(dedent.Dedent(`
							CmdInitAPIPort test case %q failed with an error: %v
							command 'kubeadm init %s'
								expected: %t
								err: %t
							`),
					rt.name,
					err,
					rt.args,
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}
