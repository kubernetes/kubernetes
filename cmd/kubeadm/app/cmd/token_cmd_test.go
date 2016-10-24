// +build cmd

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
	"flag"
	"regexp"
	"testing"

	e2e "k8s.io/kubernetes/test/e2e/framework"
)

var kubeadmPath string

func init() {
	flag.StringVar(&kubeadmPath, "kubeadm-path", "cluster/kubeadm.sh", "Location of kubeadm")
}

func TestTokenGenerate(t *testing.T) {
	stdout, _, err := e2e.RunCmd(kubeadmPath, "token", "generate")
	if err != nil {
		t.Errorf("'kubeadm token generate' exited uncleanly: %v", err)
	}

	expectedOutputRegex := "^\\S{6}\\.\\S{16}\n$"

	matched, err := regexp.MatchString(expectedOutputRegex, stdout)
	if err != nil {
		t.Fatalf("encountered an error while trying to match 'kubeadm token generate' stdout: %v", err)
	}
	if !matched {
		t.Errorf("'kubeadm token generate' stdout did not match expected regex; wanted: [%s], got: [%s]", expectedOutputRegex, stdout)
	}
}

func TestTokenGenerateTypoError(t *testing.T) {
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
