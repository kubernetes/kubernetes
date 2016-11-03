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

package master

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestGenerateTokenIfNeeded(t *testing.T) {
	var tests = []struct {
		s        kubeadmapi.Secrets
		expected bool
	}{
		{kubeadmapi.Secrets{GivenToken: "noperiod"}, false}, // not 2-part '.' format
		{kubeadmapi.Secrets{GivenToken: "abcd.a"}, false},   // len(tokenID) != 6
		{kubeadmapi.Secrets{GivenToken: "abcdef.a"}, true},
		{kubeadmapi.Secrets{GivenToken: ""}, true},
	}

	for _, rt := range tests {
		actual := generateTokenIfNeeded(&rt.s)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed UseGivenTokenIfValid:\n\texpected: %t\n\t  actual: %t\n\t token:%s",
				rt.expected,
				(actual == nil),
				rt.s.GivenToken,
			)
		}
	}
}

func TestCreateTokenAuthFile(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.Remove(tmpdir)

	// set up tmp GlobalEnvParams values for testing
	oldEnv := kubeadmapi.GlobalEnvParams
	kubeadmapi.GlobalEnvParams.HostPKIPath = fmt.Sprintf("%s/etc/kubernetes/pki", tmpdir)
	defer func() { kubeadmapi.GlobalEnvParams = oldEnv }()

	var tests = []struct {
		s        kubeadmapi.Secrets
		expected bool
	}{
		{kubeadmapi.Secrets{GivenToken: "noperiod"}, false}, // not 2-part '.' format
		{kubeadmapi.Secrets{GivenToken: "abcd.a"}, false},   // len(tokenID) != 6
		{kubeadmapi.Secrets{GivenToken: "abcdef.a"}, true},
		{kubeadmapi.Secrets{GivenToken: ""}, true},
	}
	for _, rt := range tests {
		actual := CreateTokenAuthFile(&rt.s)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed WriteKubeconfigIfNotExists with an error:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
