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
	"regexp"
	"testing"
)

const (
	TokenExpectedRegex = "^\\S{6}\\.\\S{16}\n$"
)

func TestCmdTokenGenerate(t *testing.T) {
	kubeadmPath := getKubeadmPath()
	stdout, _, _, err := RunCmd(kubeadmPath, "token", "generate")
	if err != nil {
		t.Fatalf("'kubeadm token generate' exited uncleanly: %v", err)
	}

	matched, err := regexp.MatchString(TokenExpectedRegex, stdout)
	if err != nil {
		t.Fatalf("encountered an error while trying to match 'kubeadm token generate' stdout: %v", err)
	}
	if !matched {
		t.Errorf("'kubeadm token generate' stdout did not match expected regex; wanted: [%q], got: [%s]", TokenExpectedRegex, stdout)
	}
}

func TestCmdTokenGenerateTypoError(t *testing.T) {
	/*
		Since we expect users to do things like this:

			$ TOKEN=$(kubeadm token generate)

		we want to make sure that if they have a typo in their command, we exit
		with a non-zero status code after showing the command's usage, so that
		the usage itself isn't captured as a token without the user noticing.
	*/
	kubeadmPath := getKubeadmPath()
	_, _, _, err := RunCmd(kubeadmPath, "token", "genorate") // subtle typo
	if err == nil {
		t.Error("'kubeadm token genorate' (a deliberate typo) exited without an error when we expected non-zero exit status")
	}
}
func TestCmdTokenDelete(t *testing.T) {
	var tests = []struct {
		name     string
		args     string
		expected bool
	}{
		{"no token provided", "", false},
		{"invalid token", "foobar", false},
	}

	kubeadmPath := getKubeadmPath()
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			_, _, _, actual := RunCmd(kubeadmPath, "token", "delete", rt.args)
			if (actual == nil) != rt.expected {
				t.Errorf(
					"failed CmdTokenDelete running 'kubeadm token %s' with an error: %v\n\texpected: %t\n\t  actual: %t",
					rt.args,
					actual,
					rt.expected,
					(actual == nil),
				)
			}
			kubeadmReset()
		})
	}
}
