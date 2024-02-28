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

	"k8s.io/apimachinery/pkg/util/diff"
	outputapischeme "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/scheme"
	outputapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
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
func TestPrintUpgradePlan(t *testing.T) {
	versionStates := []outputapiv1alpha3.ComponentConfigVersionState{
		{
			Group:                 "kubeproxy.config.k8s.io",
			CurrentVersion:        "v1alpha1",
			PreferredVersion:      "v1alpha1",
			ManualUpgradeRequired: false,
		},
		{
			Group:                 "kubelet.config.k8s.io",
			CurrentVersion:        "v1beta1",
			PreferredVersion:      "v1beta1",
			ManualUpgradeRequired: false,
		},
	}

	var tests = []struct {
		name          string
		upgrades      []upgrade.Upgrade
		versionStates []outputapiv1alpha3.ComponentConfigVersionState
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
			versionStates: versionStates,
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


The table below shows the current state of component configs as understood by this version of kubeadm.
Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
upgrade to is denoted in the "PREFERRED VERSION" column.

API GROUP                 CURRENT VERSION   PREFERRED VERSION   MANUAL UPGRADE REQUIRED
kubeproxy.config.k8s.io   v1alpha1          v1alpha1            no
kubelet.config.k8s.io     v1beta1           v1beta1             no
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
			versionStates: versionStates,
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


The table below shows the current state of component configs as understood by this version of kubeadm.
Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
upgrade to is denoted in the "PREFERRED VERSION" column.

API GROUP                 CURRENT VERSION   PREFERRED VERSION   MANUAL UPGRADE REQUIRED
kubeproxy.config.k8s.io   v1alpha1          v1alpha1            no
kubelet.config.k8s.io     v1beta1           v1beta1             no
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
			versionStates: versionStates,
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


The table below shows the current state of component configs as understood by this version of kubeadm.
Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
upgrade to is denoted in the "PREFERRED VERSION" column.

API GROUP                 CURRENT VERSION   PREFERRED VERSION   MANUAL UPGRADE REQUIRED
kubeproxy.config.k8s.io   v1alpha1          v1alpha1            no
kubelet.config.k8s.io     v1beta1           v1beta1             no
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
			versionStates: versionStates,
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


The table below shows the current state of component configs as understood by this version of kubeadm.
Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
upgrade to is denoted in the "PREFERRED VERSION" column.

API GROUP                 CURRENT VERSION   PREFERRED VERSION   MANUAL UPGRADE REQUIRED
kubeproxy.config.k8s.io   v1alpha1          v1alpha1            no
kubelet.config.k8s.io     v1beta1           v1beta1             no
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
			versionStates: versionStates,
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


The table below shows the current state of component configs as understood by this version of kubeadm.
Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
upgrade to is denoted in the "PREFERRED VERSION" column.

API GROUP                 CURRENT VERSION   PREFERRED VERSION   MANUAL UPGRADE REQUIRED
kubeproxy.config.k8s.io   v1alpha1          v1alpha1            no
kubelet.config.k8s.io     v1beta1           v1beta1             no
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
			versionStates: versionStates,
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


The table below shows the current state of component configs as understood by this version of kubeadm.
Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
upgrade to is denoted in the "PREFERRED VERSION" column.

API GROUP                 CURRENT VERSION   PREFERRED VERSION   MANUAL UPGRADE REQUIRED
kubeproxy.config.k8s.io   v1alpha1          v1alpha1            no
kubelet.config.k8s.io     v1beta1           v1beta1             no
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
			versionStates: versionStates,
			externalEtcd:  true,
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


The table below shows the current state of component configs as understood by this version of kubeadm.
Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
upgrade to is denoted in the "PREFERRED VERSION" column.

API GROUP                 CURRENT VERSION   PREFERRED VERSION   MANUAL UPGRADE REQUIRED
kubeproxy.config.k8s.io   v1alpha1          v1alpha1            no
kubelet.config.k8s.io     v1beta1           v1beta1             no
_____________________________________________________________________

`),
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			rt.buf = bytes.NewBufferString("")
			outputFlags := output.NewOutputFlags(&upgradePlanTextPrintFlags{}).WithTypeSetter(outputapischeme.Scheme).WithDefaultOutput(output.TextOutput)
			printer, err := outputFlags.ToPrinter()
			if err != nil {
				t.Errorf("failed ToPrinter, err: %+v", err)
			}

			plan := genUpgradePlan(rt.upgrades, rt.versionStates, rt.externalEtcd)
			if err := printer.PrintObj(plan, rt.buf); err != nil {
				t.Errorf("unexpected error when print object: %v", err)
			}

			actualBytes := rt.buf.Bytes()
			if !bytes.Equal(actualBytes, rt.expectedBytes) {
				t.Errorf(
					"failed PrintUpgradePlan:\n\texpected: %q\n\n\tactual  : %q",
					string(rt.expectedBytes),
					string(actualBytes),
				)
			}
		})
	}
}

func TestPrintUpgradePlanStructured(t *testing.T) {
	upgrades := []upgrade.Upgrade{
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
	}

	versionStates := []outputapiv1alpha3.ComponentConfigVersionState{
		{
			Group:                 "kubeproxy.config.k8s.io",
			CurrentVersion:        "v1alpha1",
			PreferredVersion:      "v1alpha1",
			ManualUpgradeRequired: false,
		},
		{
			Group:                 "kubelet.config.k8s.io",
			CurrentVersion:        "v1beta1",
			PreferredVersion:      "v1beta1",
			ManualUpgradeRequired: false,
		},
	}

	var tests = []struct {
		name         string
		outputFormat string
		buf          *bytes.Buffer
		expected     string
		externalEtcd bool
	}{
		{
			name:         "JSON output",
			outputFormat: "json",
			expected: `{
    "kind": "UpgradePlan",
    "apiVersion": "output.kubeadm.k8s.io/v1alpha3",
    "availableUpgrades": [
        {
            "description": "version in the v1.8 series",
            "components": [
                {
                    "name": "kubelet",
                    "currentVersion": "1 x v1.8.1",
                    "newVersion": "v1.8.3"
                },
                {
                    "name": "kube-apiserver",
                    "currentVersion": "v1.8.1",
                    "newVersion": "v1.8.3"
                },
                {
                    "name": "kube-controller-manager",
                    "currentVersion": "v1.8.1",
                    "newVersion": "v1.8.3"
                },
                {
                    "name": "kube-scheduler",
                    "currentVersion": "v1.8.1",
                    "newVersion": "v1.8.3"
                },
                {
                    "name": "kube-proxy",
                    "currentVersion": "v1.8.1",
                    "newVersion": "v1.8.3"
                },
                {
                    "name": "CoreDNS",
                    "currentVersion": "1.14.5",
                    "newVersion": "1.14.5"
                },
                {
                    "name": "kubeadm",
                    "currentVersion": "v1.8.2",
                    "newVersion": "v1.8.3"
                },
                {
                    "name": "etcd",
                    "currentVersion": "3.0.17",
                    "newVersion": "3.0.17"
                }
            ]
        }
    ],
    "configVersions": [
        {
            "group": "kubeproxy.config.k8s.io",
            "currentVersion": "v1alpha1",
            "preferredVersion": "v1alpha1",
            "manualUpgradeRequired": false
        },
        {
            "group": "kubelet.config.k8s.io",
            "currentVersion": "v1beta1",
            "preferredVersion": "v1beta1",
            "manualUpgradeRequired": false
        }
    ]
}
`,
		},
		{
			name:         "YAML output",
			outputFormat: "yaml",
			expected: `apiVersion: output.kubeadm.k8s.io/v1alpha3
availableUpgrades:
- components:
  - currentVersion: 1 x v1.8.1
    name: kubelet
    newVersion: v1.8.3
  - currentVersion: v1.8.1
    name: kube-apiserver
    newVersion: v1.8.3
  - currentVersion: v1.8.1
    name: kube-controller-manager
    newVersion: v1.8.3
  - currentVersion: v1.8.1
    name: kube-scheduler
    newVersion: v1.8.3
  - currentVersion: v1.8.1
    name: kube-proxy
    newVersion: v1.8.3
  - currentVersion: 1.14.5
    name: CoreDNS
    newVersion: 1.14.5
  - currentVersion: v1.8.2
    name: kubeadm
    newVersion: v1.8.3
  - currentVersion: 3.0.17
    name: etcd
    newVersion: 3.0.17
  description: version in the v1.8 series
configVersions:
- currentVersion: v1alpha1
  group: kubeproxy.config.k8s.io
  manualUpgradeRequired: false
  preferredVersion: v1alpha1
- currentVersion: v1beta1
  group: kubelet.config.k8s.io
  manualUpgradeRequired: false
  preferredVersion: v1beta1
kind: UpgradePlan
`,
		},
		{
			name:         "Text output",
			outputFormat: "text",
			expected: `Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT      TARGET
kubelet     1 x v1.8.1   v1.8.3

Upgrade to the latest version in the v1.8 series:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.8.1    v1.8.3
kube-controller-manager   v1.8.1    v1.8.3
kube-scheduler            v1.8.1    v1.8.3
kube-proxy                v1.8.1    v1.8.3
CoreDNS                   1.14.5    1.14.5
etcd                      3.0.17    3.0.17

You can now apply the upgrade by executing the following command:

	kubeadm upgrade apply v1.8.3

Note: Before you can perform this upgrade, you have to update kubeadm to v1.8.3.

_____________________________________________________________________


The table below shows the current state of component configs as understood by this version of kubeadm.
Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
upgrade to is denoted in the "PREFERRED VERSION" column.

API GROUP                 CURRENT VERSION   PREFERRED VERSION   MANUAL UPGRADE REQUIRED
kubeproxy.config.k8s.io   v1alpha1          v1alpha1            no
kubelet.config.k8s.io     v1beta1           v1beta1             no
_____________________________________________________________________

`,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			rt.buf = bytes.NewBufferString("")
			outputFlags := output.NewOutputFlags(&upgradePlanTextPrintFlags{}).WithTypeSetter(outputapischeme.Scheme).WithDefaultOutput(rt.outputFormat)
			printer, err := outputFlags.ToPrinter()
			if err != nil {
				t.Errorf("failed ToPrinter, err: %+v", err)
			}

			plan := genUpgradePlan(upgrades, versionStates, false)
			if err := printer.PrintObj(plan, rt.buf); err != nil {
				t.Errorf("unexpected error when print object: %v", err)
			}

			actual := rt.buf.String()
			if actual != rt.expected {

				t.Errorf("failed PrintUpgradePlan:\n\nexpected:\n%s\n\nactual:\n%s\n\ndiff:\n%s", rt.expected, actual, diff.StringDiff(actual, rt.expected))
			}
		})
	}
}
