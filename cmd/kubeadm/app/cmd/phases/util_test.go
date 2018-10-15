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

package phases

import (
	"testing"

	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/pkg/version"
)

func TestSetKubernetesVersion(t *testing.T) {

	ver := version.Get().String()

	tests := []struct {
		name   string
		input  string
		output string
	}{
		{
			name:   "empty version is processed",
			input:  "",
			output: ver,
		},
		{
			name:   "default version is processed",
			input:  kubeadmapiv1beta1.DefaultKubernetesVersion,
			output: ver,
		},
		{
			name:   "any other version is skipped",
			input:  "v1.12.0",
			output: "v1.12.0",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &kubeadmapiv1beta1.InitConfiguration{ClusterConfiguration: kubeadmapiv1beta1.ClusterConfiguration{KubernetesVersion: test.input}}
			SetKubernetesVersion(cfg)
			if cfg.KubernetesVersion != test.output {
				t.Fatalf("expected %q, got %q", test.output, cfg.KubernetesVersion)
			}
		})
	}
}
