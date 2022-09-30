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
	"os"
	"reflect"
	"strings"
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
			name: "hostname override",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket:        "unix:///var/run/containerd/containerd.sock",
					KubeletExtraArgs: map[string]string{"hostname-override": "override-name"},
				},
			},
			expected: map[string]string{
				"container-runtime-endpoint": "unix:///var/run/containerd/containerd.sock",
				"hostname-override":          "override-name",
			},
		},
		{
			name: "register with taints",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "unix:///var/run/containerd/containerd.sock",
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
				"container-runtime-endpoint": "unix:///var/run/containerd/containerd.sock",
				"register-with-taints":       "foo=bar:baz,key=val:eff",
			},
		},
		{
			name: "pause image is set",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					CRISocket: "unix:///var/run/containerd/containerd.sock",
				},
				pauseImage: "registry.k8s.io/pause:3.8",
			},
			expected: map[string]string{
				"container-runtime-endpoint": "unix:///var/run/containerd/containerd.sock",
				"pod-infra-container-image":  "registry.k8s.io/pause:3.8",
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

func TestGetNodeNameAndHostname(t *testing.T) {
	hostname, err := os.Hostname()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	testCases := []struct {
		name             string
		opts             kubeletFlagsOpts
		expectedNodeName string
		expectedHostName string
	}{
		{
			name: "overridden hostname",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					KubeletExtraArgs: map[string]string{"hostname-override": "override-name"},
				},
			},
			expectedNodeName: "override-name",
			expectedHostName: strings.ToLower(hostname),
		},
		{
			name: "overridden hostname uppercase",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					KubeletExtraArgs: map[string]string{"hostname-override": "OVERRIDE-NAME"},
				},
			},
			expectedNodeName: "OVERRIDE-NAME",
			expectedHostName: strings.ToLower(hostname),
		},
		{
			name: "hostname contains only spaces",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					KubeletExtraArgs: map[string]string{"hostname-override": " "},
				},
			},
			expectedNodeName: " ",
			expectedHostName: strings.ToLower(hostname),
		},
		{
			name: "empty parameter",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					KubeletExtraArgs: map[string]string{"hostname-override": ""},
				},
			},
			expectedNodeName: "",
			expectedHostName: strings.ToLower(hostname),
		},
		{
			name: "nil parameter",
			opts: kubeletFlagsOpts{
				nodeRegOpts: &kubeadmapi.NodeRegistrationOptions{
					KubeletExtraArgs: nil,
				},
			},
			expectedNodeName: strings.ToLower(hostname),
			expectedHostName: strings.ToLower(hostname),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nodeName, hostname, err := GetNodeNameAndHostname(tc.opts.nodeRegOpts)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if nodeName != tc.expectedNodeName {
				t.Errorf("expected nodeName: %v, got %v", tc.expectedNodeName, nodeName)
			}
			if hostname != tc.expectedHostName {
				t.Errorf("expected hostname: %v, got %v", tc.expectedHostName, hostname)
			}
		})
	}
}
