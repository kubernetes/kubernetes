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
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestBuildKubeletArgMap(t *testing.T) {

	tests := []struct {
		name     string
		opts     kubeletFlagsOpts
		expected map[string]string
	}{
		{
			name: "the simplest case",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/dockershim.sock",
					Taints: []v1.Taint{ // This should be ignored as registerTaintsUsingFlags is false
						{
							Key:    "foo",
							Value:  "bar",
							Effect: "baz",
						},
					},
				},
			},
			expected: map[string]string{
				"network-plugin": "cni",
			},
		},
		{
			name: "hostname override from NodeRegistrationOptions.Name",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/dockershim.sock",
					Name:      "override-name",
				},
			},
			expected: map[string]string{
				"network-plugin":    "cni",
				"hostname-override": "override-name",
			},
		},
		{
			name: "hostname override from NodeRegistrationOptions.KubeletExtraArgs",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket:        "/var/run/dockershim.sock",
					KubeletExtraArgs: map[string]string{"hostname-override": "override-name"},
				},
			},
			expected: map[string]string{
				"network-plugin":    "cni",
				"hostname-override": "override-name",
			},
		},
		{
			name: "external CRI runtime",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/containerd.sock",
				},
			},
			expected: map[string]string{
				"container-runtime":          "remote",
				"container-runtime-endpoint": "/var/run/containerd.sock",
			},
		},
		{
			name: "register with taints",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/containerd.sock",
					Taints: []v1.Taint{
						{
							Key:    "foo",
							Value:  "bar",
							Effect: "baz",
						},
						{
							Key:    "key",
							Value:  "val",
							Effect: "eff",
						},
					},
				},
				registerTaintsUsingFlags: true,
			},
			expected: map[string]string{
				"container-runtime":          "remote",
				"container-runtime-endpoint": "/var/run/containerd.sock",
				"register-with-taints":       "foo=bar:baz,key=val:eff",
			},
		},
		{
			name: "pause image is set",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "/var/run/dockershim.sock",
				},
				pauseImage: "gcr.io/pause:3.2",
			},
			expected: map[string]string{
				"network-plugin":            "cni",
				"pod-infra-container-image": "gcr.io/pause:3.2",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := buildKubeletArgMap(test.opts)
			if !reflect.DeepEqual(actual, test.expected) {
				t.Errorf(
					"failed buildKubeletArgMap:\n\texpected: %v\n\t  actual: %v",
					test.expected,
					actual,
				)
			}
		})
	}
}
