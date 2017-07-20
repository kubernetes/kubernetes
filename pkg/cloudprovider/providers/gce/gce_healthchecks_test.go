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

package gce

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
)

func TestIsAtLeastMinNodesHealthCheckVersion(t *testing.T) {
	testCases := []struct {
		version string
		expect  bool
	}{
		{"v1.7.3", true},
		{"v1.7.2", true},
		{"v1.7.2-alpha.2.597+276d289b90d322", true},
		{"v1.6.0-beta.3.472+831q821c907t31a", false},
		{"v1.5.2", false},
	}

	for _, tc := range testCases {
		if res := isAtLeastMinNodesHealthCheckVersion(tc.version); res != tc.expect {
			t.Errorf("%v: want %v, got %v", tc.version, tc.expect, res)
		}
	}
}

func TestSupportsNodesHealthCheck(t *testing.T) {
	testCases := []struct {
		desc   string
		nodes  []*v1.Node
		expect bool
	}{
		{
			"All nodes support nodes health check",
			[]*v1.Node{
				{
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeProxyVersion: "v1.7.2",
						},
					},
				},
				{
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeProxyVersion: "v1.7.2-alpha.2.597+276d289b90d322",
						},
					},
				},
			},
			true,
		},
		{
			"All nodes don't support nodes health check",
			[]*v1.Node{
				{
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeProxyVersion: "v1.6.0-beta.3.472+831q821c907t31a",
						},
					},
				},
				{
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeProxyVersion: "v1.5.2",
						},
					},
				},
			},
			false,
		},
		{
			"One node doesn't support nodes health check",
			[]*v1.Node{
				{
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeProxyVersion: "v1.7.3",
						},
					},
				},
				{
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeProxyVersion: "v1.7.2-alpha.2.597+276d289b90d322",
						},
					},
				},
				{
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeProxyVersion: "v1.5.2",
						},
					},
				},
			},
			false,
		},
	}

	for _, tc := range testCases {
		if res := supportsNodesHealthCheck(tc.nodes); res != tc.expect {
			t.Errorf("%v: want %v, got %v", tc.desc, tc.expect, res)
		}
	}
}
