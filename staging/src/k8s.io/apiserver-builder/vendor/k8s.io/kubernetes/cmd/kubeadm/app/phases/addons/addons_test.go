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

package addons

import (
	"testing"

	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func TestGetClusterCIDR(t *testing.T) {
	emptyClusterCIDR := getClusterCIDR("")
	if emptyClusterCIDR != "" {
		t.Errorf("Invalid format: %s", emptyClusterCIDR)
	}

	clusterCIDR := getClusterCIDR("10.244.0.0/16")
	if clusterCIDR != "- --cluster-cidr=10.244.0.0/16" {
		t.Errorf("Invalid format: %s", clusterCIDR)
	}
}

func TestCompileManifests(t *testing.T) {
	var tests = []struct {
		manifest string
		data     interface{}
		expected bool
	}{
		{
			manifest: KubeProxyConfigMap,
			data: struct{ MasterEndpoint string }{
				MasterEndpoint: "foo",
			},
			expected: true,
		},
		{
			manifest: KubeProxyDaemonSet,
			data: struct{ Image, ClusterCIDR, MasterTaintKey string }{
				Image:          "foo",
				ClusterCIDR:    "foo",
				MasterTaintKey: "foo",
			},
			expected: true,
		},
		{
			manifest: KubeDNSDeployment,
			data: struct{ ImageRepository, Arch, Version, DNSDomain, MasterTaintKey string }{
				ImageRepository: "foo",
				Arch:            "foo",
				Version:         "foo",
				DNSDomain:       "foo",
				MasterTaintKey:  "foo",
			},
			expected: true,
		},
		{
			manifest: KubeDNSService,
			data: struct{ DNSIP string }{
				DNSIP: "foo",
			},
			expected: true,
		},
	}
	for _, rt := range tests {
		_, actual := kubeadmutil.ParseTemplate(rt.manifest, rt.data)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CompileManifests:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
