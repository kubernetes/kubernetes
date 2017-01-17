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
	"flag"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

const (
	TokenExpectedRegex = "^\\S{6}\\:\\S{16}\n$"
)

var kubeadmPath string

func init() {
	flag.StringVar(&kubeadmPath, "kubeadm-path", filepath.Join(os.Getenv("KUBE_ROOT"), "cluster/kubeadm.sh"), "Location of kubeadm")
}

func TestCmdTokenGenerate(t *testing.T) {
	stdout, _, err := RunCmd(kubeadmPath, "ex", "token", "generate")
	if err != nil {
		t.Fatalf("'kubeadm ex token generate' exited uncleanly: %v", err)
	}

	matched, err := regexp.MatchString(TokenExpectedRegex, stdout)
	if err != nil {
		t.Fatalf("encountered an error while trying to match 'kubeadm ex token generate' stdout: %v", err)
	}
	if !matched {
		t.Errorf("'kubeadm ex token generate' stdout did not match expected regex; wanted: [%q], got: [%s]", TokenExpectedRegex, stdout)
	}
}

func TestCmdTokenGenerateTypoError(t *testing.T) {
	/*
		Since we expect users to do things like this:

			$ TOKEN=$(kubeadm ex token generate)

		we want to make sure that if they have a typo in their command, we exit
		with a non-zero status code after showing the command's usage, so that
		the usage itself isn't captured as a token without the user noticing.
	*/

	_, _, err := RunCmd(kubeadmPath, "ex", "token", "genorate") // subtle typo
	if err == nil {
		t.Error("'kubeadm ex token genorate' (a deliberate typo) exited without an error when we expected non-zero exit status")
	}
}

// kubeadmReset executes "kubeadm reset" and restarts kubelet.
func kubeadmReset() error {
	_, _, err := RunCmd(kubeadmPath, "reset")
	return err
}

func TestCmdInitToken(t *testing.T) {
	var initTest = []struct {
		args     string
		expected bool
	}{
		{"--discovery=token://abcd:1234567890abcd", false},     // invalid token size
		{"--discovery=token://Abcdef:1234567890abcdef", false}, // invalid token non-lowercase
		{"--discovery=token://abcdef:1234567890abcdef", true},  // valid token
		{"", true}, // no token provided, so generate
	}

	for _, rt := range initTest {
		_, _, actual := RunCmd(kubeadmPath, "init", rt.args, "--skip-preflight-checks")
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
