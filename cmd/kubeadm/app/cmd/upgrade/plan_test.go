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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
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
			name:     "Up to date",
			upgrades: []upgrade.Upgrade{},
			expectedBytes: []byte(`Awesome, you're up-to-date! Enjoy!
`),
		},
		{
			name:         "Up to date external etcd",
			externalEtcd: true,
			upgrades:     []upgrade.Upgrade{},
			expectedBytes: []byte(`Awesome, you're up-to-date! Enjoy!
`),
		},
		{
			name: "Patch version available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.8 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.1",
						KubeletVersions: map[string]uint16{
							"v1.8.1": 1,
						},
						KubeadmVersion: "v1.8.2",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.8.3",
						KubeadmVersion: "v1.8.3",
						DNSType:        kubeadmapi.KubeDNS,
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
			name: "minor version available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "stable version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.3",
						KubeletVersions: map[string]uint16{
							"v1.8.3": 1,
						},
						KubeadmVersion: "v1.9.0",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.0",
						KubeadmVersion: "v1.9.0",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.13",
						EtcdVersion:    "3.1.12",
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
Kube DNS             1.14.5    1.14.13
Etcd                 3.0.17    3.1.12

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.0

_____________________________________________________________________

`),
		},
		{
			name: "patch and minor version available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.8 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.3",
						KubeletVersions: map[string]uint16{
							"v1.8.3": 1,
						},
						KubeadmVersion: "v1.8.3",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.8.5",
						KubeadmVersion: "v1.8.3",
						DNSType:        kubeadmapi.KubeDNS,
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
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.0",
						KubeadmVersion: "v1.9.0",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.13",
						EtcdVersion:    "3.1.12",
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
Kube DNS             1.14.5    1.14.13
Etcd                 3.0.17    3.1.12

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.0

Note: Before you can perform this upgrade, you have to update kubeadm to v1.9.0.

_____________________________________________________________________

`),
		},
		{
			name: "experimental version available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "experimental version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.5",
						KubeletVersions: map[string]uint16{
							"v1.8.5": 1,
						},
						KubeadmVersion: "v1.8.5",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.0-beta.1",
						KubeadmVersion: "v1.9.0-beta.1",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.13",
						EtcdVersion:    "3.1.12",
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
Kube DNS             1.14.5    1.14.13
Etcd                 3.0.17    3.1.12

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.0-beta.1 --allow-experimental-upgrades

Note: Before you can perform this upgrade, you have to update kubeadm to v1.9.0-beta.1.

_____________________________________________________________________

`),
		},
		{
			name: "release candidate available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "release candidate version",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.8.5",
						KubeletVersions: map[string]uint16{
							"v1.8.5": 1,
						},
						KubeadmVersion: "v1.8.5",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.0-rc.1",
						KubeadmVersion: "v1.9.0-rc.1",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.13",
						EtcdVersion:    "3.1.12",
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
Kube DNS             1.14.5    1.14.13
Etcd                 3.0.17    3.1.12

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.0-rc.1 --allow-release-candidate-upgrades

Note: Before you can perform this upgrade, you have to update kubeadm to v1.9.0-rc.1.

_____________________________________________________________________

`),
		},
		{
			name: "multiple kubelet versions",
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.9 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.9.2",
						KubeletVersions: map[string]uint16{
							"v1.9.2": 1,
							"v1.9.3": 2,
						},
						KubeadmVersion: "v1.9.2",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.3",
						KubeadmVersion: "v1.9.3",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.8",
						EtcdVersion:    "3.1.12",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT      AVAILABLE
Kubelet     1 x v1.9.2   v1.9.3
            2 x v1.9.3   v1.9.3

Upgrade to the latest version in the v1.9 series:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.9.2    v1.9.3
Controller Manager   v1.9.2    v1.9.3
Scheduler            v1.9.2    v1.9.3
Kube Proxy           v1.9.2    v1.9.3
Kube DNS             1.14.5    1.14.8
Etcd                 3.0.17    3.1.12

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.3

Note: Before you can perform this upgrade, you have to update kubeadm to v1.9.3.

_____________________________________________________________________

`),
		},

		{
			name: "external etcd upgrade available",
			upgrades: []upgrade.Upgrade{
				{
					Description: "version in the v1.9 series",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.9.2",
						KubeletVersions: map[string]uint16{
							"v1.9.2": 1,
						},
						KubeadmVersion: "v1.9.2",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.5",
						EtcdVersion:    "3.0.17",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.9.3",
						KubeadmVersion: "v1.9.3",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.8",
						EtcdVersion:    "3.1.12",
					},
				},
			},
			externalEtcd: true,
			expectedBytes: []byte(`External components that should be upgraded manually before you upgrade the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT   AVAILABLE
Etcd        3.0.17    3.1.12

Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT      AVAILABLE
Kubelet     1 x v1.9.2   v1.9.3

Upgrade to the latest version in the v1.9 series:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.9.2    v1.9.3
Controller Manager   v1.9.2    v1.9.3
Scheduler            v1.9.2    v1.9.3
Kube Proxy           v1.9.2    v1.9.3
Kube DNS             1.14.5    1.14.8

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.9.3

Note: Before you can perform this upgrade, you have to update kubeadm to v1.9.3.

_____________________________________________________________________

`),
		},
		{
			name: "kubedns to coredns",
			upgrades: []upgrade.Upgrade{
				{
					Description: "kubedns to coredns",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.10.2",
						KubeletVersions: map[string]uint16{
							"v1.10.2": 1,
						},
						KubeadmVersion: "v1.11.0",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.7",
						EtcdVersion:    "3.1.11",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.11.0",
						KubeadmVersion: "v1.11.0",
						DNSType:        kubeadmapi.CoreDNS,
						DNSVersion:     "1.0.6",
						EtcdVersion:    "3.2.18",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       AVAILABLE
Kubelet     1 x v1.10.2   v1.11.0

Upgrade to the latest kubedns to coredns:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.10.2   v1.11.0
Controller Manager   v1.10.2   v1.11.0
Scheduler            v1.10.2   v1.11.0
Kube Proxy           v1.10.2   v1.11.0
CoreDNS                        1.0.6
Kube DNS             1.14.7    
Etcd                 3.1.11    3.2.18

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.11.0

_____________________________________________________________________

`),
		},
		{
			name: "coredns",
			upgrades: []upgrade.Upgrade{
				{
					Description: "coredns",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.10.2",
						KubeletVersions: map[string]uint16{
							"v1.10.2": 1,
						},
						KubeadmVersion: "v1.11.0",
						DNSType:        kubeadmapi.CoreDNS,
						DNSVersion:     "1.0.5",
						EtcdVersion:    "3.1.11",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.11.0",
						KubeadmVersion: "v1.11.0",
						DNSType:        kubeadmapi.CoreDNS,
						DNSVersion:     "1.0.6",
						EtcdVersion:    "3.2.18",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       AVAILABLE
Kubelet     1 x v1.10.2   v1.11.0

Upgrade to the latest coredns:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.10.2   v1.11.0
Controller Manager   v1.10.2   v1.11.0
Scheduler            v1.10.2   v1.11.0
Kube Proxy           v1.10.2   v1.11.0
CoreDNS              1.0.5     1.0.6
Etcd                 3.1.11    3.2.18

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.11.0

_____________________________________________________________________

`),
		},
		{
			name: "coredns to kubedns",
			upgrades: []upgrade.Upgrade{
				{
					Description: "coredns to kubedns",
					Before: upgrade.ClusterState{
						KubeVersion: "v1.10.2",
						KubeletVersions: map[string]uint16{
							"v1.10.2": 1,
						},
						KubeadmVersion: "v1.11.0",
						DNSType:        kubeadmapi.CoreDNS,
						DNSVersion:     "1.0.6",
						EtcdVersion:    "3.1.11",
					},
					After: upgrade.ClusterState{
						KubeVersion:    "v1.11.0",
						KubeadmVersion: "v1.11.0",
						DNSType:        kubeadmapi.KubeDNS,
						DNSVersion:     "1.14.9",
						EtcdVersion:    "3.2.18",
					},
				},
			},
			expectedBytes: []byte(`Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       AVAILABLE
Kubelet     1 x v1.10.2   v1.11.0

Upgrade to the latest coredns to kubedns:

COMPONENT            CURRENT   AVAILABLE
API Server           v1.10.2   v1.11.0
Controller Manager   v1.10.2   v1.11.0
Scheduler            v1.10.2   v1.11.0
Kube Proxy           v1.10.2   v1.11.0
CoreDNS              1.0.6     
Kube DNS                       1.14.9
Etcd                 3.1.11    3.2.18

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.11.0

_____________________________________________________________________

`),
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			rt.buf = bytes.NewBufferString("")
			printAvailableUpgrades(rt.upgrades, rt.buf, rt.externalEtcd)
			actualBytes := rt.buf.Bytes()
			if !bytes.Equal(actualBytes, rt.expectedBytes) {
				t.Errorf(
					"failed PrintAvailableUpgrades:\n\texpected: %q\n\t  actual: %q",
					string(rt.expectedBytes),
					string(actualBytes),
				)
			}
		})
	}
}
