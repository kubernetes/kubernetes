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
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	versionutil "k8s.io/apimachinery/pkg/util/version"
	clientsetfake "k8s.io/client-go/kubernetes/fake"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

type fakeVersionGetter struct {
	clusterVersion         string
	kubeadmVersion         string
	stableVersion          string
	latestVersion          string
	latestDevBranchVersion string
	stablePatchVersion     string
	kubeletVersion         string
	componentVersion       string
	etcdVersion            string
	isExternalEtcd         bool
}

var _ VersionGetter = &fakeVersionGetter{}

// ClusterVersion gets a fake API server version
func (f *fakeVersionGetter) ClusterVersion() (string, *versionutil.Version, error) {
	return f.clusterVersion, versionutil.MustParseSemantic(f.clusterVersion), nil
}

// KubeadmVersion gets a fake kubeadm version
func (f *fakeVersionGetter) KubeadmVersion() (string, *versionutil.Version, error) {
	if f.kubeadmVersion == "" {
		return "", nil, errors.New("get kubeadm version error")
	}
	return f.kubeadmVersion, versionutil.MustParseSemantic(f.kubeadmVersion), nil
}

// VersionFromCILabel gets fake latest versions from CI
func (f *fakeVersionGetter) VersionFromCILabel(ciVersionLabel, _ string) (string, *versionutil.Version, error) {
	if f.stableVersion == "" {
		return "", nil, errors.New("fake error")
	}
	if ciVersionLabel == "stable" {
		return f.stableVersion, versionutil.MustParseSemantic(f.stableVersion), nil
	}
	if ciVersionLabel == "latest" {
		return f.latestVersion, versionutil.MustParseSemantic(f.latestVersion), nil
	}
	if f.latestDevBranchVersion != "" && strings.HasPrefix(ciVersionLabel, "latest-") {
		return f.latestDevBranchVersion, versionutil.MustParseSemantic(f.latestDevBranchVersion), nil
	}
	return f.stablePatchVersion, versionutil.MustParseSemantic(f.stablePatchVersion), nil
}

// KubeletVersions should return a map with a version and a list of node names that describes how many kubelets there are for that version
func (f *fakeVersionGetter) KubeletVersions() (map[string][]string, error) {
	if f.kubeletVersion == "" {
		return nil, errors.New("get kubelet version failed")
	}
	return map[string][]string{
		f.kubeletVersion: {"node1"},
	}, nil
}

// ComponentVersions should return a map with a version and a list of node names that describes how many a given control-plane components there are for that version
func (f *fakeVersionGetter) ComponentVersions(name string) (map[string][]string, error) {
	if name == constants.Etcd {
		if f.isExternalEtcd {
			return map[string][]string{}, nil
		}
		return map[string][]string{
			f.etcdVersion: {"node1"},
		}, nil
	}

	if f.componentVersion == "" {
		return nil, errors.New("get component version failed")
	}

	if f.componentVersion == "multiVersion" {
		return map[string][]string{
			"node1": {"node1"},
			"node2": {"node2"},
		}, nil
	}

	return map[string][]string{
		f.componentVersion: {"node1"},
	}, nil
}

const fakeCurrentEtcdVersion = "3.1.12"

func getEtcdVersion(v *versionutil.Version) string {
	etcdVer, _, _ := constants.EtcdSupportedVersion(constants.SupportedEtcdVersion, v.String())
	return etcdVer.String()
}

const fakeCurrentCoreDNSVersion = "1.0.6"

func TestGetAvailableUpgrades(t *testing.T) {

	// constants for test cases
	// variables are in the form v{MAJOR}{MINOR}{PATCH}, where MINOR is a variable so test are automatically uptodate to the latest MinimumControlPlaneVersion/

	// v1.X series, e.g. v1.14
	v1X0 := versionutil.MustParseSemantic("v1.14.0")
	v1X5 := v1X0.WithPatch(5)

	// v1.Y series, where Y = X+1, e.g. v1.15
	v1Y0 := versionutil.MustParseSemantic("v1.15.0")
	v1Y0alpha0 := v1Y0.WithPreRelease("alpha.0")
	v1Y0alpha1 := v1Y0.WithPreRelease("alpha.1")
	v1Y1 := v1Y0.WithPatch(1)
	v1Y2 := v1Y0.WithPatch(2)
	v1Y3 := v1Y0.WithPatch(3)
	v1Y5 := v1Y0.WithPatch(5)

	// v1.Z series, where Z = Y+1, e.g. v1.16
	v1Z0 := versionutil.MustParseSemantic("v1.16.0")
	v1Z0alpha1 := v1Z0.WithPreRelease("alpha.1")
	v1Z0alpha2 := v1Z0.WithPreRelease("alpha.2")
	v1Z0beta1 := v1Z0.WithPreRelease("beta.1")
	v1Z0rc1 := v1Z0.WithPreRelease("rc.1")
	v1Z1 := v1Z0.WithPatch(1)

	tests := []struct {
		name                        string
		vg                          VersionGetter
		expectedUpgrades            []Upgrade
		allowExperimental, allowRCs bool
		errExpected                 bool
		beforeDNSVersion            string
		deployDNSFailed             bool
	}{
		{
			name: "no action needed, already up-to-date",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y0.String(),
				componentVersion: v1Y0.String(),
				kubeletVersion:   v1Y0.String(),
				kubeadmVersion:   v1Y0.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y0.String(),
				stableVersion:      v1Y0.String(),
			},
			beforeDNSVersion:  fakeCurrentCoreDNSVersion,
			expectedUpgrades:  nil,
			allowExperimental: false,
			errExpected:       false,
		},
		{
			name: "get component version failed",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y0.String(),
				componentVersion: "",
				kubeletVersion:   v1Y0.String(),
				kubeadmVersion:   v1Y0.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y0.String(),
				stableVersion:      v1Y0.String(),
			},
			beforeDNSVersion:  fakeCurrentCoreDNSVersion,
			expectedUpgrades:  nil,
			allowExperimental: false,
			errExpected:       true,
		},
		{
			name: "there is version information about multiple components",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y0.String(),
				componentVersion: "multiVersion",
				kubeletVersion:   v1Y0.String(),
				kubeadmVersion:   v1Y0.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y0.String(),
				stableVersion:      v1Y0.String(),
			},
			beforeDNSVersion:  fakeCurrentCoreDNSVersion,
			expectedUpgrades:  nil,
			allowExperimental: false,
			errExpected:       true,
		},
		{
			name: "get kubeadm version failed",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y0.String(),
				componentVersion: v1Y0.String(),
				kubeletVersion:   v1Y0.String(),
				kubeadmVersion:   "",
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y0.String(),
				stableVersion:      v1Y0.String(),
			},
			beforeDNSVersion:  fakeCurrentCoreDNSVersion,
			expectedUpgrades:  nil,
			allowExperimental: false,
			errExpected:       true,
		},
		{
			name: "get kubelet version failed",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y0.String(),
				componentVersion: v1Y0.String(),
				kubeletVersion:   "",
				kubeadmVersion:   v1Y0.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y0.String(),
				stableVersion:      v1Y0.String(),
			},
			beforeDNSVersion:  fakeCurrentCoreDNSVersion,
			expectedUpgrades:  nil,
			allowExperimental: false,
			errExpected:       true,
		},
		{
			name: "deploy DNS failed",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y0.String(),
				componentVersion: v1Y0.String(),
				kubeletVersion:   v1Y0.String(),
				kubeadmVersion:   v1Y0.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y0.String(),
				stableVersion:      v1Y0.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,

			expectedUpgrades:  nil,
			allowExperimental: false,
			errExpected:       false,
		},
		{
			name: "get stable version from CI label failed",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y0.String(),
				componentVersion: v1Y0.String(),
				kubeletVersion:   v1Y0.String(),
				kubeadmVersion:   v1Y0.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y0.String(),
				stableVersion:      "",
			},
			beforeDNSVersion:  fakeCurrentCoreDNSVersion,
			deployDNSFailed:   true,
			expectedUpgrades:  nil,
			allowExperimental: false,
			errExpected:       true,
		},
		{
			name: "simple patch version upgrade",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y1.String(),
				componentVersion: v1Y1.String(),
				kubeletVersion:   v1Y1.String(), // the kubelet are on the same version as the control plane
				kubeadmVersion:   v1Y2.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y3.String(),
				stableVersion:      v1Y3.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: fmt.Sprintf("version in the v%d.%d series", v1Y0.Major(), v1Y0.Minor()),
					Before: ClusterState{
						KubeVersion: v1Y1.String(),
						KubeAPIServerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeadmVersion: v1Y2.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Y3.String(),
						KubeadmVersion: v1Y3.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Y3),
					},
				},
			},
			allowExperimental: false,
			errExpected:       false,
		},
		{
			name: "simple patch version upgrade with external etcd",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y1.String(),
				componentVersion: v1Y1.String(),
				kubeletVersion:   v1Y1.String(), // the kubelet are on the same version as the control plane
				kubeadmVersion:   v1Y2.String(),
				isExternalEtcd:   true,

				stablePatchVersion: v1Y3.String(),
				stableVersion:      v1Y3.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: fmt.Sprintf("version in the v%d.%d series", v1Y0.Major(), v1Y0.Minor()),
					Before: ClusterState{
						KubeVersion: v1Y1.String(),
						KubeAPIServerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						EtcdVersions:   map[string][]string{},
						KubeadmVersion: v1Y2.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
					},
					After: ClusterState{
						KubeVersion:    v1Y3.String(),
						KubeadmVersion: v1Y3.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    "",
					},
				},
			},
			allowExperimental: false,
			errExpected:       false,
		},
		{
			name: "no version provided to offline version getter does not change behavior",
			vg: NewOfflineVersionGetter(&fakeVersionGetter{
				clusterVersion:   v1Y1.String(),
				componentVersion: v1Y1.String(),
				kubeletVersion:   v1Y1.String(), // the kubelet are on the same version as the control plane
				kubeadmVersion:   v1Y2.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y3.String(),
				stableVersion:      v1Y3.String(),
			}, ""),
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: fmt.Sprintf("version in the v%d.%d series", v1Y0.Major(), v1Y0.Minor()),
					Before: ClusterState{
						KubeVersion: v1Y1.String(),
						KubeAPIServerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeadmVersion: v1Y2.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Y3.String(),
						KubeadmVersion: v1Y3.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Y3),
					},
				},
			},
			allowExperimental: false,
			errExpected:       false,
		},
		{
			name: "minor version upgrade only",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y1.String(),
				componentVersion: v1Y1.String(),
				kubeletVersion:   v1Y1.String(), // the kubelet are on the same version as the control plane
				kubeadmVersion:   v1Z0.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y1.String(),
				stableVersion:      v1Z0.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: "stable version",
					Before: ClusterState{
						KubeVersion: v1Y1.String(),
						KubeAPIServerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeadmVersion: v1Z0.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Z0.String(),
						KubeadmVersion: v1Z0.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Z0),
					},
				},
			},
			allowExperimental: false,
			errExpected:       false,
		},
		{
			name: "both minor version upgrade and patch version upgrade available",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y3.String(),
				componentVersion: v1Y3.String(),
				kubeletVersion:   v1Y3.String(), // the kubelet are on the same version as the control plane
				kubeadmVersion:   v1Y5.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y5.String(),
				stableVersion:      v1Z1.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: fmt.Sprintf("version in the v%d.%d series", v1Y0.Major(), v1Y0.Minor()),
					Before: ClusterState{
						KubeVersion: v1Y3.String(),
						KubeAPIServerVersions: map[string][]string{
							v1Y3.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1Y3.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1Y3.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1Y3.String(): {"node1"},
						},
						KubeadmVersion: v1Y5.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Y5.String(),
						KubeadmVersion: v1Y5.String(), // Note: The kubeadm version mustn't be "downgraded" here
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Y5),
					},
				},
				{
					Description: "stable version",
					Before: ClusterState{
						KubeVersion: v1Y3.String(),
						KubeAPIServerVersions: map[string][]string{
							v1Y3.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1Y3.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1Y3.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1Y3.String(): {"node1"},
						},
						KubeadmVersion: v1Y5.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Z1.String(),
						KubeadmVersion: v1Z1.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Z1),
					},
				},
			},
			allowExperimental: false,
			errExpected:       false,
		},
		{
			name: "allow experimental upgrades, but no upgrade available",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Z0alpha2.String(),
				componentVersion: v1Z0alpha2.String(),
				kubeletVersion:   v1Y5.String(),
				kubeadmVersion:   v1Y5.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y5.String(),
				stableVersion:      v1Y5.String(),
				latestVersion:      v1Z0alpha2.String(),
			},
			beforeDNSVersion:  fakeCurrentCoreDNSVersion,
			expectedUpgrades:  nil,
			allowExperimental: true,
			errExpected:       false,
		},
		{
			name: "upgrade to an unstable version should be supported",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Y5.String(),
				componentVersion: v1Y5.String(),
				kubeletVersion:   v1Y5.String(),
				kubeadmVersion:   v1Y5.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y5.String(),
				stableVersion:      v1Y5.String(),
				latestVersion:      v1Z0alpha2.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: "experimental version",
					Before: ClusterState{
						KubeVersion: v1Y5.String(),
						KubeAPIServerVersions: map[string][]string{
							v1Y5.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1Y5.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1Y5.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1Y5.String(): {"node1"},
						},
						KubeadmVersion: v1Y5.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Z0alpha2.String(),
						KubeadmVersion: v1Z0alpha2.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Z0alpha2),
					},
				},
			},
			allowExperimental: true,
			errExpected:       false,
		},
		{
			name: "upgrade from an unstable version to an unstable version should be supported",
			vg: &fakeVersionGetter{
				clusterVersion:   v1Z0alpha1.String(),
				componentVersion: v1Z0alpha1.String(),
				kubeletVersion:   v1Y5.String(),
				kubeadmVersion:   v1Y5.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion: v1Y5.String(),
				stableVersion:      v1Y5.String(),
				latestVersion:      v1Z0alpha2.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: "experimental version",
					Before: ClusterState{
						KubeVersion: v1Z0alpha1.String(),
						KubeAPIServerVersions: map[string][]string{
							v1Z0alpha1.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1Z0alpha1.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1Z0alpha1.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1Y5.String(): {"node1"},
						},
						KubeadmVersion: v1Y5.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Z0alpha2.String(),
						KubeadmVersion: v1Z0alpha2.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Z0alpha2),
					},
				},
			},
			allowExperimental: true,
			errExpected:       false,
		},
		{
			name: "v1.X.0-alpha.0 should be ignored",
			vg: &fakeVersionGetter{
				clusterVersion:   v1X5.String(),
				componentVersion: v1X5.String(),
				kubeletVersion:   v1X5.String(),
				kubeadmVersion:   v1X5.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion:     v1X5.String(),
				stableVersion:          v1X5.String(),
				latestDevBranchVersion: v1Z0beta1.String(),
				latestVersion:          v1Y0alpha0.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: "experimental version",
					Before: ClusterState{
						KubeVersion: v1X5.String(),
						KubeAPIServerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeadmVersion: v1X5.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Z0beta1.String(),
						KubeadmVersion: v1Z0beta1.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Z0beta1),
					},
				},
			},
			allowExperimental: true,
			errExpected:       false,
		},
		{
			name: "upgrade to an RC version should be supported",
			vg: &fakeVersionGetter{
				clusterVersion:   v1X5.String(),
				componentVersion: v1X5.String(),
				kubeletVersion:   v1X5.String(),
				kubeadmVersion:   v1X5.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion:     v1X5.String(),
				stableVersion:          v1X5.String(),
				latestDevBranchVersion: v1Z0rc1.String(),
				latestVersion:          v1Y0alpha1.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: "release candidate version",
					Before: ClusterState{
						KubeVersion: v1X5.String(),
						KubeAPIServerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeadmVersion: v1X5.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Z0rc1.String(),
						KubeadmVersion: v1Z0rc1.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Z0rc1),
					},
				},
			},
			allowRCs:    true,
			errExpected: false,
		},
		{
			name: "it is possible (but very uncommon) that the latest version from the previous branch is an rc and the current latest version is alpha.0. In that case, show the RC",
			vg: &fakeVersionGetter{
				clusterVersion:   v1X5.String(),
				componentVersion: v1X5.String(),
				kubeletVersion:   v1X5.String(),
				kubeadmVersion:   v1X5.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion:     v1X5.String(),
				stableVersion:          v1X5.String(),
				latestDevBranchVersion: v1Z0rc1.String(),
				latestVersion:          v1Y0alpha0.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: "experimental version", // Note that this is considered an experimental version in this uncommon scenario
					Before: ClusterState{
						KubeVersion: v1X5.String(),
						KubeAPIServerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeadmVersion: v1X5.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Z0rc1.String(),
						KubeadmVersion: v1Z0rc1.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Z0rc1),
					},
				},
			},
			allowExperimental: true,
			errExpected:       false,
		},
		{
			name: "upgrade to an RC version should be supported. There may also be an even newer unstable version.",
			vg: &fakeVersionGetter{
				clusterVersion:   v1X5.String(),
				componentVersion: v1X5.String(),
				kubeletVersion:   v1X5.String(),
				kubeadmVersion:   v1X5.String(),
				etcdVersion:      fakeCurrentEtcdVersion,

				stablePatchVersion:     v1X5.String(),
				stableVersion:          v1X5.String(),
				latestDevBranchVersion: v1Z0rc1.String(),
				latestVersion:          v1Y0alpha1.String(),
			},
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: "release candidate version",
					Before: ClusterState{
						KubeVersion: v1X5.String(),
						KubeAPIServerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeadmVersion: v1X5.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Z0rc1.String(),
						KubeadmVersion: v1Z0rc1.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Z0rc1),
					},
				},
				{
					Description: "experimental version",
					Before: ClusterState{
						KubeVersion: v1X5.String(),
						KubeAPIServerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1X5.String(): {"node1"},
						},
						KubeadmVersion: v1X5.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Y0alpha1.String(),
						KubeadmVersion: v1Y0alpha1.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Y0alpha1),
					},
				},
			},
			allowRCs:          true,
			allowExperimental: true,
			errExpected:       false,
		},
		{
			name: "offline version getter",
			vg: NewOfflineVersionGetter(&fakeVersionGetter{
				clusterVersion:   v1Y1.String(),
				componentVersion: v1Y1.String(),
				kubeletVersion:   v1Y0.String(),
				kubeadmVersion:   v1Y1.String(),
				etcdVersion:      fakeCurrentEtcdVersion,
			}, v1Z1.String()),
			beforeDNSVersion: fakeCurrentCoreDNSVersion,
			expectedUpgrades: []Upgrade{
				{
					Description: fmt.Sprintf("version in the v%d.%d series", v1Y0.Major(), v1Y0.Minor()),
					Before: ClusterState{
						KubeVersion: v1Y1.String(),
						KubeAPIServerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeControllerManagerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeSchedulerVersions: map[string][]string{
							v1Y1.String(): {"node1"},
						},
						KubeletVersions: map[string][]string{
							v1Y0.String(): {"node1"},
						},
						KubeadmVersion: v1Y1.String(),
						DNSVersion:     fakeCurrentCoreDNSVersion,
						EtcdVersions:   map[string][]string{fakeCurrentEtcdVersion: {"node1"}},
					},
					After: ClusterState{
						KubeVersion:    v1Z1.String(),
						KubeadmVersion: v1Z1.String(),
						DNSVersion:     constants.CoreDNSVersion,
						EtcdVersion:    getEtcdVersion(v1Z1),
					},
				},
			},
		},
	}

	// Instantiating a fake etcd cluster for being able to get etcd version for a corresponding
	// Kubernetes release.
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {

			dnsName := constants.CoreDNSDeploymentName

			client := newMockClientForTest(t, dnsName, rt.beforeDNSVersion, rt.deployDNSFailed)

			actualUpgrades, actualErr := GetAvailableUpgrades(rt.vg, rt.allowExperimental, rt.allowRCs, client, &output.TextPrinter{})
			if diff := cmp.Diff(rt.expectedUpgrades, actualUpgrades); len(diff) > 0 {
				t.Errorf("failed TestGetAvailableUpgrades\n\texpected upgrades:\n%v\n\tgot:\n%v\n\tdiff:\n%v", rt.expectedUpgrades, actualUpgrades, diff)
			}
			if rt.errExpected && actualErr == nil {
				t.Error("unexpected success")
			} else if !rt.errExpected && actualErr != nil {
				t.Errorf("unexpected failure: %v", actualErr)
			}
			if diff := cmp.Diff(rt.expectedUpgrades, actualUpgrades); len(diff) > 0 {
				t.Logf("diff: %s", cmp.Diff(rt.expectedUpgrades, actualUpgrades))
				t.Errorf("failed TestGetAvailableUpgrades\n\texpected upgrades:\n%v\n\tgot:\n%v\n\tdiff:\n%v", rt.expectedUpgrades, actualUpgrades, diff)
			}
		})
	}
}

func TestKubeletUpgrade(t *testing.T) {
	tests := []struct {
		name     string
		before   map[string][]string
		after    string
		expected bool
	}{
		{
			name: "upgrade from v1.10.1 to v1.10.3 is available",
			before: map[string][]string{
				"v1.10.1": {"node1"},
			},
			after:    "v1.10.3",
			expected: true,
		},
		{
			name: "upgrade from v1.10.1 and v1.10.3/2 to v1.10.3 is available",
			before: map[string][]string{
				"v1.10.1": {"node1"},
				"v1.10.3": {"node2", "node3"},
			},
			after:    "v1.10.3",
			expected: true,
		},
		{
			name: "upgrade from v1.10.3 to v1.10.3 is not available",
			before: map[string][]string{
				"v1.10.3": {"node1"},
			},
			after:    "v1.10.3",
			expected: false,
		},
		{
			name: "upgrade from v1.10.3/2 to v1.10.3 is not available",
			before: map[string][]string{
				"v1.10.3": {"node1", "node2"},
			},
			after:    "v1.10.3",
			expected: false,
		},
		{
			name:     "upgrade is not available if we don't know anything about the earlier state",
			before:   map[string][]string{},
			after:    "v1.10.3",
			expected: false,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			upgrade := Upgrade{
				Before: ClusterState{
					KubeletVersions: rt.before,
				},
				After: ClusterState{
					KubeVersion: rt.after,
				},
			}
			actual := upgrade.CanUpgradeKubelets()
			if actual != rt.expected {
				t.Errorf("failed TestKubeletUpgrade\n\texpected: %t\n\tgot: %t\n\ttest object: %v", rt.expected, actual, upgrade)
			}
		})
	}
}

func TestGetBranchFromVersion(t *testing.T) {
	testCases := []struct {
		version         string
		expectedVersion string
	}{
		{
			version:         "v1.9.5",
			expectedVersion: "1.9",
		},
		{
			version:         "v1.9.0-alpha.2",
			expectedVersion: "1.9",
		},
		{
			version:         "v1.9.0-beta.0",
			expectedVersion: "1.9",
		},
		{
			version:         "v1.9.0-rc.1",
			expectedVersion: "1.9",
		},
		{
			version:         "v1.11.0-alpha.0",
			expectedVersion: "1.11",
		},

		{
			version:         "v1.11.0-beta.1",
			expectedVersion: "1.11",
		},
		{
			version:         "v1.11.0-rc.0",
			expectedVersion: "1.11",
		},
		{
			version:         "1.12.5",
			expectedVersion: "1.12",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.version, func(t *testing.T) {
			v := getBranchFromVersion(tc.version)
			if v != tc.expectedVersion {
				t.Errorf("expected version %s, got %s", tc.expectedVersion, v)
			}
		})
	}
}

func TestGetSuggestedEtcdVersion(t *testing.T) {
	constants.SupportedEtcdVersion = map[uint8]string{
		16: "3.3.17-0",
		17: "3.4.3-0",
		18: "3.4.3-0",
		19: "3.4.13-0",
		20: "3.4.13-0",
		21: "3.4.13-0",
		22: "3.5.5-0",
	}

	tests := []struct {
		name              string
		externalEtcd      bool
		kubernetesVersion string
		expectedVersion   string
	}{
		{
			name:              "external etcd: no version",
			externalEtcd:      true,
			kubernetesVersion: "1.1.0",
			expectedVersion:   "",
		},
		{
			name:              "local etcd: illegal kubernetes version",
			externalEtcd:      false,
			kubernetesVersion: "1.x.5",
			expectedVersion:   "N/A",
		},
		{
			name:              "local etcd: no supported version for 1.10.5, return the nearest version",
			externalEtcd:      false,
			kubernetesVersion: "1.10.5",
			expectedVersion:   "3.3.17-0",
		},
		{
			name:              "local etcd: has supported version for 1.17.0",
			externalEtcd:      false,
			kubernetesVersion: "1.17.0",
			expectedVersion:   "3.4.3-0",
		},
		{
			name:              "local etcd: no supported version for v1.99.0, return the nearest version",
			externalEtcd:      false,
			kubernetesVersion: "v1.99.0",
			expectedVersion:   "3.5.5-0",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getSuggestedEtcdVersion(tt.externalEtcd, tt.kubernetesVersion); got != tt.expectedVersion {
				t.Errorf("getSuggestedEtcdVersion() want %v, got %v", tt.expectedVersion, got)
			}
		})
	}
}

func newMockClientForTest(t *testing.T, dnsName string, dnsVersion string, multiDNS bool) *clientsetfake.Clientset {
	client := clientsetfake.NewSimpleClientset()
	createDeployment := func(name string) {
		_, err := client.AppsV1().Deployments(metav1.NamespaceSystem).Create(context.TODO(), &apps.Deployment{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      dnsName + name,
				Namespace: "kube-system",
				Labels: map[string]string{
					"k8s-app": "kube-dns",
				},
			},
			Spec: apps.DeploymentSpec{
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Image: "test:" + dnsVersion + name,
							},
						},
					},
				},
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("error creating deployment: %v", err)
		}
	}

	createDeployment("")
	if multiDNS {
		createDeployment("dns2")
	}
	return client
}
