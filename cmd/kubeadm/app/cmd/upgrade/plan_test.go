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

package upgrade

import (
	"bytes"
	"reflect"
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
)

func TestSortedSliceFromStringIntMap(t *testing.T) {
	var tests = []struct {
		strMap        map[string]uint16
		expectedSlice []string
	}{ // The returned slice should be alphabetically sorted based on the string keys in the map
		{
			strMap:        map[string]uint16{"foo": 1, "bar": 2},
			expectedSlice: []string{"bar", "foo"},
		},
		{ // The int value should not affect this func
			strMap:        map[string]uint16{"foo": 2, "bar": 1},
			expectedSlice: []string{"bar", "foo"},
		},
		{
			strMap:        map[string]uint16{"b": 2, "a": 1, "cb": 0, "ca": 1000},
			expectedSlice: []string{"a", "b", "ca", "cb"},
		},
		{ // This should work for version numbers as well; and the lowest version should come first
			strMap:        map[string]uint16{"v1.7.0": 1, "v1.6.1": 1, "v1.6.2": 1, "v1.8.0": 1, "v1.8.0-alpha.1": 1},
			expectedSlice: []string{"v1.6.1", "v1.6.2", "v1.7.0", "v1.8.0", "v1.8.0-alpha.1"},
		},
	}
	for _, rt := range tests {
		actualSlice := sortedSliceFromStringIntMap(rt.strMap)
		if !reflect.DeepEqual(actualSlice, rt.expectedSlice) {
			t.Errorf(
				"failed SortedSliceFromStringIntMap:\n\texpected: %v\n\t  actual: %v",
				rt.expectedSlice,
				actualSlice,
			)
		}
	}
}

// TODO Think about modifying this test to be less verbose checking b/c it can be brittle.
func TestPrintAvailableUpgrades(t *testing.T) {
	featureGates := make(map[string]bool)
	var tests = []struct {
		upgrades      []upgrade.Upgrade
		buf           *bytes.Buffer
		expectedBytes []byte
	}{
		{
			upgrades: []upgrade.Upgrade{},
			expectedBytes: []byte(`Awesome, you're up-to-date! Enjoy!
`),
		},
		{
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.8 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.1",
						KubeletVersions: map[string]uint16{
							"v1.8.1": 1,
						},
						KubeadmVersion: "v1.8.2",
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.8.3",
						KubeadmVersion: "v1.8.3",
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT      AVAILABLE
Kubelet     1 x v1.8.1   v1.8.3

Upgrade to the latest version in the v1.8 series:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.8.1    v1.8.3
Controller Manager   v1.8.1    v1.8.3
Scheduler            v1.8.1    v1.8.3
Kube Proxy           v1.8.1    v1.8.3
Kube DNS             1.14.5    1.14.5
Etcd                 3.0.17    3.0.17

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.8.3

Note: Before you can perform this upgrade, you have to update kubeadm to v1.8.3.

_____________________________________________________________________

`),
		},
		{
			upgrades: []upgrade.Upgrade{
				{
					Description: "stable version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.3",
						KubeletVersions: map[string]uint16{
							"v1.8.3": 1,
						},
						KubeadmVersion: "v1.9.0",
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.0",
						KubeadmVersion: "v1.9.0",
						DNSVersion:     "1.14.7",
						EtcdVersion:    "3.1.10",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT      AVAILABLE
Kubelet     1 x v1.8.3   v1.9.0

Upgrade to the latest stable version:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.8.3    v1.9.0
Controller Manager   v1.8.3    v1.9.0
Scheduler            v1.8.3    v1.9.0
Kube Proxy           v1.8.3    v1.9.0
Kube DNS             1.14.5    1.14.7
Etcd                 3.0.17    3.1.10

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.0

_____________________________________________________________________

`),
		},
		{
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.8 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.3",
						KubeletVersions: map[string]uint16{
							"v1.8.3": 1,
						},
						KubeadmVersion: "v1.8.3",
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.8.5",
						KubeadmVersion: "v1.8.3",
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
				},
				{
					Description: "stable version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.3",
						KubeletVersions: map[string]uint16{
							"v1.8.3": 1,
						},
						KubeadmVersion: "v1.8.3",
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.0",
						KubeadmVersion: "v1.9.0",
						DNSVersion:     "1.14.7",
						EtcdVersion:    "3.1.10",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT      AVAILABLE
Kubelet     1 x v1.8.3   v1.8.5

Upgrade to the latest version in the v1.8 series:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.8.3    v1.8.5
Controller Manager   v1.8.3    v1.8.5
Scheduler            v1.8.3    v1.8.5
Kube Proxy           v1.8.3    v1.8.5
Kube DNS             1.14.5    1.14.5
Etcd                 3.0.17    3.0.17

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.8.5

_____________________________________________________________________

Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT      AVAILABLE
Kubelet     1 x v1.8.3   v1.9.0

Upgrade to the latest stable version:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.8.3    v1.9.0
Controller Manager   v1.8.3    v1.9.0
Scheduler            v1.8.3    v1.9.0
Kube Proxy           v1.8.3    v1.9.0
Kube DNS             1.14.5    1.14.7
Etcd                 3.0.17    3.1.10

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.0

Note: Before you can perform this upgrade, you have to update kubeadm to v1.9.0.

_____________________________________________________________________

`),
		},
		{
			upgrades: []upgrade.Upgrade{
				{
					Description: "experimental version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.5",
						KubeletVersions: map[string]uint16{
							"v1.8.5": 1,
						},
						KubeadmVersion: "v1.8.5",
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.0-beta.1",
						KubeadmVersion: "v1.9.0-beta.1",
						DNSVersion:     "1.14.7",
						EtcdVersion:    "3.1.10",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT      AVAILABLE
Kubelet     1 x v1.8.5   v1.9.0-beta.1

Upgrade to the latest experimental version:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.8.5    v1.9.0-beta.1
Controller Manager   v1.8.5    v1.9.0-beta.1
Scheduler            v1.8.5    v1.9.0-beta.1
Kube Proxy           v1.8.5    v1.9.0-beta.1
Kube DNS             1.14.5    1.14.7
Etcd                 3.0.17    3.1.10

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.0-beta.1

Note: Before you can perform this upgrade, you have to update kubeadm to v1.9.0-beta.1.

_____________________________________________________________________

`),
		},
		{
			upgrades: []upgrade.Upgrade{
				{
					Description: "release candidate version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.5",
						KubeletVersions: map[string]uint16{
							"v1.8.5": 1,
						},
						KubeadmVersion: "v1.8.5",
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.0-rc.1",
						KubeadmVersion: "v1.9.0-rc.1",
						DNSVersion:     "1.14.7",
						EtcdVersion:    "3.1.10",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT      AVAILABLE
Kubelet     1 x v1.8.5   v1.9.0-rc.1

Upgrade to the latest release candidate version:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.8.5    v1.9.0-rc.1
Controller Manager   v1.8.5    v1.9.0-rc.1
Scheduler            v1.8.5    v1.9.0-rc.1
Kube Proxy           v1.8.5    v1.9.0-rc.1
Kube DNS             1.14.5    1.14.7
Etcd                 3.0.17    3.1.10

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.0-rc.1

Note: Before you can perform this upgrade, you have to update kubeadm to v1.9.0-rc.1.

_____________________________________________________________________

`),
		},
	}
	for _, rt := range tests {
		rt.buf = bytes.NewBufferString("")
		printAvailableUpgrades(rt.upgrades, rt.buf, featureGates)
		actualBytes := rt.buf.Bytes()
		if !bytes.Equal(actualBytes, rt.expectedBytes) {
			t.Errorf(
				"failed PrintAvailableUpgrades:\n\texpected: %q\n\t  actual: %q",
				string(rt.expectedBytes),
				string(actualBytes),
			)
		}
	}
}
