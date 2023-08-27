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

package phases

import (
	"reflect"
	"testing"
)

func TestGetAddonPhaseFlags(t *testing.T) {
	tests := []struct {
		testName string
		name     string
		expected []string
	}{
		{
			testName: "name is all",
			name:     "all",
			expected: []string{"config", "kubeconfig", "kubernetes-version", "image-repository", "dry-run", "apiserver-advertise-address", "control-plane-endpoint", "apiserver-bind-port", "pod-network-cidr", "feature-gates", "service-dns-domain", "service-cidr"},
		},
		{
			testName: "name is kube-proxy",
			name:     "kube-proxy",
			expected: []string{"config", "kubeconfig", "kubernetes-version", "image-repository", "dry-run", "apiserver-advertise-address", "control-plane-endpoint", "apiserver-bind-port", "pod-network-cidr"},
		},
		{
			testName: "name is coredns",
			name:     "coredns",
			expected: []string{"config", "kubeconfig", "kubernetes-version", "image-repository", "dry-run", "feature-gates", "service-dns-domain", "service-cidr"},
		},
		{
			testName: "name is others",
			name:     "foo",
			expected: []string{"config", "kubeconfig", "kubernetes-version", "image-repository", "dry-run"},
		},
	}

	for _, test := range tests {
		t.Run(test.testName, func(t *testing.T) {
			got := getAddonPhaseFlags(test.name)
			if !reflect.DeepEqual(got, test.expected) {
				t.Errorf("expected %s, got %s", test.expected, got)
			}
		})
	}
}
