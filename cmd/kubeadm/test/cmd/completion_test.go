/*
Copyright 2017 The Kubernetes Authors.

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

func TestCmdCompletion(t *testing.T) {
	kubeadmPath := getKubeadmPath()

	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	var tests = []struct {
		name     string
		args     string
		expected bool
	}{
		{"shell not expected", "", false},
		{"unsupported shell type", "foo", false},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			_, _, actual := RunCmd(kubeadmPath, "completion", rt.args)
			if (actual == nil) != rt.expected {
				t.Errorf(
					"failed CmdCompletion running 'kubeadm completion %s' with an error: %v\n\texpected: %t\n\t  actual: %t",
					rt.args,
					actual,
					rt.expected,
					(actual == nil),
				)
			}
		})
	}
}
