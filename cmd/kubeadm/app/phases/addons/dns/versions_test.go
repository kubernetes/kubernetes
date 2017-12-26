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

package dns

import (
	"testing"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/pkg/util/version"
)

func TestGetKubeDNSVersion(t *testing.T) {
	var tests = []struct {
		k8sVersion string
		dns        string
		expected   string
	}{
		{
			k8sVersion: "v1.9.0",
			dns:        kubeadmconstants.KubeDNS,
			expected:   kubeDNSv190AndAboveVersion,
		},
		{
			k8sVersion: "v1.10.0",
			dns:        kubeadmconstants.KubeDNS,
			expected:   kubeDNSv190AndAboveVersion,
		},
		{
			k8sVersion: "v1.9.0",
			dns:        kubeadmconstants.CoreDNS,
			expected:   coreDNSVersion,
		},
		{
			k8sVersion: "v1.10.0",
			dns:        kubeadmconstants.CoreDNS,
			expected:   coreDNSVersion,
		},
	}
	for _, rt := range tests {
		k8sVersion, err := version.ParseSemantic(rt.k8sVersion)
		if err != nil {
			t.Fatalf("couldn't parse kubernetes version %q: %v", rt.k8sVersion, err)
		}

		actualDNSVersion := GetDNSVersion(k8sVersion, rt.dns)
		if actualDNSVersion != rt.expected {
			t.Errorf(
				"failed GetDNSVersion:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actualDNSVersion,
			)
		}
	}
}
