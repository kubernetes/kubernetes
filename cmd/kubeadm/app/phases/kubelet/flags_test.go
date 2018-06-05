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
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

func TestBuildKubeletArgMap(t *testing.T) {

	tests := []struct {
		name             string
		hostname         string
		expectedHostname string
	}{
		{
			name:             "manually set to current hostname",
			hostname:         nodeutil.GetHostname(""),
			expectedHostname: "",
		},
		{
			name:             "unset hostname",
			hostname:         "",
			expectedHostname: "",
		},
		{
			name:             "override hostname",
			hostname:         "my-node",
			expectedHostname: "my-node",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			opts := &kubeadmapi.NodeRegistrationOptions{
				Name: test.hostname,
			}

			m := buildKubeletArgMap(opts, false)
			if m["hostname-override"] != test.expectedHostname {
				t.Errorf("expected hostname %q, got %q", test.expectedHostname, m["hostname-override"])
			}
		})
	}
}
