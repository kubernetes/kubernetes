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

package cmd

import (
	"bytes"
	"flag"
	"regexp"
	"testing"

	e2e "k8s.io/kubernetes/test/e2e/framework"
)

const (
	TokenExpectedRegex = "^\\S{6}\\.\\S{16}\n$"
)

var kubeadmPath string

func init() {
	flag.StringVar(&kubeadmPath, "kubeadm-path", "", "Location of kubeadm")
}

func TestRunGenerateToken(t *testing.T) {
	var buf bytes.Buffer

	err := RunGenerateToken(&buf)
	if err != nil {
		t.Errorf("RunGenerateToken returned an error: %v", err)
	}

	output := buf.String()

	matched, err := regexp.MatchString(TokenExpectedRegex, output)
	if err != nil {
		t.Fatalf("encountered an error while trying to match RunGenerateToken's output: %v", err)
	}
	if !matched {
		t.Errorf("RunGenerateToken's output did not match expected regex; wanted: [%s], got: [%s]", TokenExpectedRegex, output)
	}
}

func TestCmdTokenGenerate(t *testing.T) {
	if !cmdTestsEnabled() {
		return
	}

	stdout, _, err := e2e.RunCmd(kubeadmPath, "token", "generate")
	if err != nil {
		t.Errorf("'kubeadm token generate' exited uncleanly: %v", err)
	}

	matched, err := regexp.MatchString(TokenExpectedRegex, stdout)
	if err != nil {
		t.Fatalf("encountered an error while trying to match 'kubeadm token generate' stdout: %v", err)
	}
	if !matched {
		t.Errorf("'kubeadm token generate' stdout did not match expected regex; wanted: [%s], got: [%s]", TokenExpectedRegex, stdout)
	}
}

func TestCmdTokenGenerateTypoError(t *testing.T) {
	if !cmdTestsEnabled() {
		return
	}

	/*
		Since we expect users to do things like this:

			$ TOKEN=$(kubeadm token generate)

		we want to make sure that if they have a typo in their command, we exit
		with a non-zero status code after showing the command's usage, so that
		the usage itself isn't captured as a token without the user noticing.
	*/

	_, _, err := e2e.RunCmd(kubeadmPath, "token", "genorate") // subtle typo
	if err == nil {
		t.Error("'kubeadm token genorate' (a deliberate typo) exited without an error when we expected non-zero exit status")
	}
}

// Determine if our "cmd" tests are enabled or not. These tests are run on the
// CLI itself, so they depend on having the kubeadm binary built, and are run
// as part of the "test-cmd" make target instead of the normal "test" target.
func cmdTestsEnabled() bool {
	return kubeadmPath != ""
}
