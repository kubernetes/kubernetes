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

import "testing"

// kubeadmReset executes "kubeadm reset" and restarts kubelet.
func kubeadmReset() error {
	_, _, err := RunCmd(*kubeadmPath, "reset")
	return err
}

func TestCmdInitToken(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	var initTest = []struct {
		args     string
		expected bool
	}{
		{"--discovery=token://abcd:1234567890abcd", false},     // invalid token size
		{"--discovery=token://Abcdef:1234567890abcdef", false}, // invalid token non-lowercase
	}

	for _, rt := range initTest {
		_, _, actual := RunCmd(*kubeadmPath, "init", rt.args, "--skip-preflight-checks")
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CmdInitToken running 'kubeadm init %s' with an error: %v\n\texpected: %t\n\t  actual: %t",
				rt.args,
				actual,
				rt.expected,
				(actual == nil),
			)
		}
		kubeadmReset()
	}
}

func TestCmdInitKubernetesVersion(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	var initTest = []struct {
		args     string
		expected bool
	}{
		{"--kubernetes-version=foobar", false},
	}

	for _, rt := range initTest {
		_, _, actual := RunCmd(*kubeadmPath, "init", rt.args, "--skip-preflight-checks")
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CmdInitKubernetesVersion running 'kubeadm init %s' with an error: %v\n\texpected: %t\n\t  actual: %t",
				rt.args,
				actual,
				rt.expected,
				(actual == nil),
			)
		}
		kubeadmReset()
	}
}

func TestCmdInitConfig(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	var initTest = []struct {
		args     string
		expected bool
	}{
		{"--config=foobar", false},
		{"--config=/does/not/exist/foo/bar", false},
	}

	for _, rt := range initTest {
		_, _, actual := RunCmd(*kubeadmPath, "init", rt.args, "--skip-preflight-checks")
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CmdInitConfig running 'kubeadm init %s' with an error: %v\n\texpected: %t\n\t  actual: %t",
				rt.args,
				actual,
				rt.expected,
				(actual == nil),
			)
		}
		kubeadmReset()
	}
}

func TestCmdInitAPIPort(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	var initTest = []struct {
		args     string
		expected bool
	}{
		{"--api-port=foobar", false},
	}

	for _, rt := range initTest {
		_, _, actual := RunCmd(*kubeadmPath, "init", rt.args, "--skip-preflight-checks")
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CmdInitAPIPort running 'kubeadm init %s' with an error: %v\n\texpected: %t\n\t  actual: %t",
				rt.args,
				actual,
				rt.expected,
				(actual == nil),
			)
		}
		kubeadmReset()
	}
}
