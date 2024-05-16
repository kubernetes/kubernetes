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

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
)

func TestGetAddonPhaseFlags(t *testing.T) {
	tests := []struct {
		name string
		want []string
	}{
		{
			name: "all",
			want: []string{options.CfgPath,
				options.KubeconfigPath,
				options.KubernetesVersion,
				options.ImageRepository,
				options.DryRun,
				options.APIServerAdvertiseAddress,
				options.ControlPlaneEndpoint,
				options.APIServerBindPort,
				options.NetworkingPodSubnet,
				options.FeatureGatesString,
				options.NetworkingDNSDomain,
				options.NetworkingServiceSubnet,
			},
		}, {
			name: "kube-proxy",
			want: []string{options.CfgPath,
				options.KubeconfigPath,
				options.KubernetesVersion,
				options.ImageRepository,
				options.DryRun,
				options.APIServerAdvertiseAddress,
				options.ControlPlaneEndpoint,
				options.APIServerBindPort,
				options.NetworkingPodSubnet,
			},
		}, {
			name: "coredns",
			want: []string{options.CfgPath,
				options.KubeconfigPath,
				options.KubernetesVersion,
				options.ImageRepository,
				options.DryRun,
				options.FeatureGatesString,
				options.NetworkingDNSDomain,
				options.NetworkingServiceSubnet,
			},
		}, {
			name: "invalid_name",
			want: []string{options.CfgPath,
				options.KubeconfigPath,
				options.KubernetesVersion,
				options.ImageRepository,
				options.DryRun,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getAddonPhaseFlags(tt.name)
			if ok := reflect.DeepEqual(got, tt.want); !ok {
				t.Errorf("phase init addons.getAddonPhaseFlags() = %v, want %v", got, tt.want)
			}
		})
	}
}
