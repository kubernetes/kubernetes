/*
Copyright 2023 The Kubernetes Authors.

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
	"strings"
	"testing"
)

func TestCmdConfigImagesList(t *testing.T) {
	var tests = []struct {
		name     string
		args     string
		expected bool
	}{
		{
			name:     "valid: latest stable Kubernetes version should not throw the warning that a supported etcd version cannot be found",
			args:     "--kubernetes-version=stable-1",
			expected: false,
		},
	}

	kubeadmPath := getKubeadmPath()
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			_, stderr, _, err := RunCmd(kubeadmPath, "config", "images", "list", "--v=1", rt.args)
			if err != nil {
				t.Fatalf("failed to run 'kubeadm config images list --v=1 %s', err: %v", rt.args, err)
			}
			actual := strings.Contains(stderr, "could not find officially supported version of etcd")
			if actual != rt.expected {
				t.Errorf(
					"failed CmdConfigImagesList running 'kubeadm config images list --v=1 %s' with stderr output:\n%v\n\t  expected: %t\n\t  actual: %t\n\n"+
						"FYI: This usually indicates that the 'SupportedEtcdVersion' map defined in 'cmd/kubeadm/app/constants/constants.go' needs to be updated.\n",
					rt.args,
					stderr,
					rt.expected,
					actual,
				)
			}
		})
	}
}
