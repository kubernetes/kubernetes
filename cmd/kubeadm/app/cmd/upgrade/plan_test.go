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
		name          string
		strMap        map[string]uint16
		expectedSlice []string
	}{
		{
			name:          "the returned slice should be alphabetically sorted based on the string keys in the map",
			strMap:        map[string]uint16{"foo": 1, "bar": 2},
			expectedSlice: []string{"bar", "foo"},
		},
		{
			name:          "the int value should not affect this func",
			strMap:        map[string]uint16{"foo": 2, "bar": 1},
			expectedSlice: []string{"bar", "foo"},
		},
		{
			name:          "slice with 4 keys and different values",
			strMap:        map[string]uint16{"b": 2, "a": 1, "cb": 0, "ca": 1000},
			expectedSlice: []string{"a", "b", "ca", "cb"},
		},
		{
			name:          "this should work for version numbers as well; and the lowest version should come first",
			strMap:        map[string]uint16{"v1.7.0": 1, "v1.6.1": 1, "v1.6.2": 1, "v1.8.0": 1, "v1.8.0-alpha.1": 1},
			expectedSlice: []string{"v1.6.1", "v1.6.2", "v1.7.0", "v1.8.0", "v1.8.0-alpha.1"},
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actualSlice := sortedSliceFromStringIntMap(rt.strMap)
			if !reflect.DeepEqual(actualSlice, rt.expectedSlice) {
				t.Errorf(
					"failed SortedSliceFromStringIntMap:\n\texpected: %v\n\t  actual: %v",
					rt.expectedSlice,
					actualSlice,
				)
			}
		})
	}
}

// TODO Think about modifying this test to be less verbose checking b/c it can be brittle.
func TestPrintAvailableUpgrades(t *testing.T) {
	var tests = []struct {
		name          string
		upgrades      []upgrade.Upgrade
		buf           *bytes.Buffer
		expectedBytes []byte
		externalEtcd  bool
	}{
		{
			name: "Patch version available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.18 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.18.1",
						KubeletVersions: map[string]uint16{
							"v1.18.1": 1,
						},
						KubeadmVersion: "v1.18.1",
						DNSVersion:     "1.6.7",
						EtcdVersion:    "3.4.3-0",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.18.4",
						KubeadmVersion: "v1.18.4",
						DNSVersion:     "1.6.7",
						EtcdVersion:    "3.4.3-0",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       TARGET
kubelet     1 x v1.18.1   v1.18.4

Upgrade to the latest version in the v1.18 series:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.18.1   v1.18.4
kube-controller-manager   v1.18.1   v1.18.4
kube-scheduler            v1.18.1   v1.18.4
kube-proxy                v1.18.1   v1.18.4
CoreDNS                   1.6.7     1.6.7
etcd                      3.4.3-0   3.4.3-0

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.18.4

Note: Before you can perform this upgrade, you have to update kubeadm to v1.18.4.

_____________________________________________________________________

`),
		},
		{
			name: "minor version available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "stable version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.18.4",
						KubeletVersions: map[string]uint16{
							"v1.18.4": 1,
						},
						KubeadmVersion: "v1.18.4",
						DNSVersion:     "1.6.7",
						EtcdVersion:    "3.4.3-0",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.19.0",
						KubeadmVersion: "v1.19.0",
						DNSVersion:     "1.7.0",
						EtcdVersion:    "3.4.7-0",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       TARGET
kubelet     1 x v1.18.4   v1.19.0

Upgrade to the latest stable version:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.18.4   v1.19.0
kube-controller-manager   v1.18.4   v1.19.0
kube-scheduler            v1.18.4   v1.19.0
kube-proxy                v1.18.4   v1.19.0
CoreDNS                   1.6.7     1.7.0
etcd                      3.4.3-0   3.4.7-0

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.19.0

Note: Before you can perform this upgrade, you have to update kubeadm to v1.19.0.

_____________________________________________________________________

`),
		},
		{
			name: "patch and minor version available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.18 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.18.3",
						KubeletVersions: map[string]uint16{
							"v1.18.3": 1,
						},
						KubeadmVersion: "v1.18.3",
						DNSVersion:     "1.6.7",
						EtcdVersion:    "3.4.3-0",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.18.5",
						KubeadmVersion: "v1.18.3",
						DNSVersion:     "1.6.7",
						EtcdVersion:    "3.4.3-0",
					},
				},
				{
					Description: "stable version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.18.3",
						KubeletVersions: map[string]uint16{
							"v1.18.3": 1,
						},
						KubeadmVersion: "v1.18.3",
						DNSVersion:     "1.6.7",
						EtcdVersion:    "3.4.3-0",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.19.0",
						KubeadmVersion: "v1.19.0",
						DNSVersion:     "1.7.0",
						EtcdVersion:    "3.4.7-0",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       TARGET
kubelet     1 x v1.18.3   v1.18.5

Upgrade to the latest version in the v1.18 series:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.18.3   v1.18.5
kube-controller-manager   v1.18.3   v1.18.5
kube-scheduler            v1.18.3   v1.18.5
kube-proxy                v1.18.3   v1.18.5
CoreDNS                   1.6.7     1.6.7
etcd                      3.4.3-0   3.4.3-0

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.18.5

_____________________________________________________________________

Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       TARGET
kubelet     1 x v1.18.3   v1.19.0

Upgrade to the latest stable version:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.18.3   v1.19.0
kube-controller-manager   v1.18.3   v1.19.0
kube-scheduler            v1.18.3   v1.19.0
kube-proxy                v1.18.3   v1.19.0
CoreDNS                   1.6.7     1.7.0
etcd                      3.4.3-0   3.4.7-0

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.19.0

Note: Before you can perform this upgrade, you have to update kubeadm to v1.19.0.

_____________________________________________________________________

`),
		},
		{
			name: "experimental version available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "experimental version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.18.5",
						KubeletVersions: map[string]uint16{
							"v1.18.5": 1,
						},
						KubeadmVersion: "v1.18.5",
						DNSVersion:     "1.6.7",
						EtcdVersion:    "3.4.3-0",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.19.0-beta.1",
						KubeadmVersion: "v1.19.0-beta.1",
						DNSVersion:     "1.7.0",
						EtcdVersion:    "3.4.7-0",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       TARGET
kubelet     1 x v1.18.5   v1.19.0-beta.1

Upgrade to the latest experimental version:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.18.5   v1.19.0-beta.1
kube-controller-manager   v1.18.5   v1.19.0-beta.1
kube-scheduler            v1.18.5   v1.19.0-beta.1
kube-proxy                v1.18.5   v1.19.0-beta.1
CoreDNS                   1.6.7     1.7.0
etcd                      3.4.3-0   3.4.7-0

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.19.0-beta.1 --allow-experimental-upgrades

Note: Before you can perform this upgrade, you have to update kubeadm to v1.19.0-beta.1.

_____________________________________________________________________

`),
		},
		{
			name: "release candidate available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "release candidate version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.18.5",
						KubeletVersions: map[string]uint16{
							"v1.18.5": 1,
						},
						KubeadmVersion: "v1.18.5",
						DNSVersion:     "1.6.7",
						EtcdVersion:    "3.4.3-0",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.19.0-rc.1",
						KubeadmVersion: "v1.19.0-rc.1",
						DNSVersion:     "1.7.0",
						EtcdVersion:    "3.4.7-0",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       TARGET
kubelet     1 x v1.18.5   v1.19.0-rc.1

Upgrade to the latest release candidate version:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.18.5   v1.19.0-rc.1
kube-controller-manager   v1.18.5   v1.19.0-rc.1
kube-scheduler            v1.18.5   v1.19.0-rc.1
kube-proxy                v1.18.5   v1.19.0-rc.1
CoreDNS                   1.6.7     1.7.0
etcd                      3.4.3-0   3.4.7-0

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.19.0-rc.1 --allow-release-candidate-upgrades

Note: Before you can perform this upgrade, you have to update kubeadm to v1.19.0-rc.1.

_____________________________________________________________________

`),
		},
		{
			name: "multiple kubelet versions",
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.19 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.19.2",
						KubeletVersions: map[string]uint16{
							"v1.19.2": 1,
							"v1.19.3": 2,
						},
						KubeadmVersion: "v1.19.2",
						DNSVersion:     "1.7.0",
						EtcdVersion:    "3.4.7-0",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.19.3",
						KubeadmVersion: "v1.19.3",
						DNSVersion:     "1.7.0",
						EtcdVersion:    "3.4.7-0",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       TARGET
kubelet     1 x v1.19.2   v1.19.3
            2 x v1.19.3   v1.19.3

Upgrade to the latest version in the v1.19 series:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.19.2   v1.19.3
kube-controller-manager   v1.19.2   v1.19.3
kube-scheduler            v1.19.2   v1.19.3
kube-proxy                v1.19.2   v1.19.3
CoreDNS                   1.7.0     1.7.0
etcd                      3.4.7-0   3.4.7-0

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.19.3

Note: Before you can perform this upgrade, you have to update kubeadm to v1.19.3.

_____________________________________________________________________

`),
		},

		{
			name: "external etcd upgrade available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.19 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.19.2",
						KubeletVersions: map[string]uint16{
							"v1.19.2": 1,
						},
						KubeadmVersion: "v1.19.2",
						DNSVersion:     "1.7.0",
						EtcdVersion:    "3.4.7-0",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.19.3",
						KubeadmVersion: "v1.19.3",
						DNSVersion:     "1.7.0",
						EtcdVersion:    "3.4.7-0",
					},
				},
			},
			externalEtcd: true,
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       TARGET
kubelet     1 x v1.19.2   v1.19.3

Upgrade to the latest version in the v1.19 series:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.19.2   v1.19.3
kube-controller-manager   v1.19.2   v1.19.3
kube-scheduler            v1.19.2   v1.19.3
kube-proxy                v1.19.2   v1.19.3
CoreDNS                   1.7.0     1.7.0

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.19.3

Note: Before you can perform this upgrade, you have to update kubeadm to v1.19.3.

_____________________________________________________________________

`),
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			rt.buf = bytes.NewBufferString("")
			// Generate and print upgrade plans
			for _, up := range rt.upgrades {
				plan, unstableVersionFlag, err := genUpgradePlan(&up, rt.externalEtcd)
				if err != nil {
					t.Errorf("failed genUpgradePlan, err: %+v", err)
				}
				printUpgradePlan(&up, plan, unstableVersionFlag, rt.externalEtcd, rt.buf)
			}
			actualBytes := rt.buf.Bytes()
			if !bytes.Equal(actualBytes, rt.expectedBytes) {
				t.Errorf(
					"failed PrintAvailableUpgrades:\n\texpected: %q\n\n\tactual  : %q",
					string(rt.expectedBytes),
					string(actualBytes),
				)
			}
		})
	}
}
