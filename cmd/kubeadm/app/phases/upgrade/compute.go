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
	"fmt"
	"strings"

	versionutil "k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

// Upgrade defines an upgrade possibility to upgrade from a current version to a new one
type Upgrade struct {
	Description string
	Before      ClusterState
	After       ClusterState
}

// CanUpgradeKubelets returns whether an upgrade of any kubelet in the cluster is possible
func (u *Upgrade) CanUpgradeKubelets() bool {
	// If there are multiple different versions now, an upgrade is possible (even if only for a subset of the nodes)
	if len(u.Before.KubeletVersions) > 1 {
		return true
	}
	// Don't report something available for upgrade if we don't know the current state
	if len(u.Before.KubeletVersions) == 0 {
		return false
	}

	// if the same version number existed both before and after, we don't have to upgrade it
	_, sameVersionFound := u.Before.KubeletVersions[u.After.KubeVersion]
	return !sameVersionFound
}

// ClusterState describes the state of certain versions for a cluster during an upgrade
type ClusterState struct {
	// KubeVersion describes the version of latest Kubernetes API Server in the cluster.
	KubeVersion string
	// DNSVersion describes the version of the DNS add-on.
	DNSVersion string
	// KubeadmVersion describes the version of the kubeadm CLI
	KubeadmVersion string
	// EtcdVersion represents the version of etcd used in the cluster
	EtcdVersion string

	// The following maps describe the versions of the different components in the cluster.
	// The key is the version string and the value is a list of nodes that have that version.
	KubeAPIServerVersions         map[string][]string
	KubeControllerManagerVersions map[string][]string
	KubeSchedulerVersions         map[string][]string
	EtcdVersions                  map[string][]string
	KubeletVersions               map[string][]string
}

// GetAvailableUpgrades fetches all versions from the specified VersionGetter and computes which
// kinds of upgrades can be performed
func GetAvailableUpgrades(versionGetterImpl VersionGetter, experimentalUpgradesAllowed, rcUpgradesAllowed bool, client clientset.Interface, printer output.Printer) ([]Upgrade, error) {
	printer.Printf("[upgrade] Fetching available versions to upgrade to\n")

	// Collect the upgrades kubeadm can do in this list
	var upgrades []Upgrade

	// Get the kube-apiserver versions in the cluster
	kubeAPIServerVersions, err := versionGetterImpl.ComponentVersions(kubeadmconstants.KubeAPIServer)
	if err != nil {
		return upgrades, err
	}
	if len(kubeAPIServerVersions) > 1 {
		verMsg := []string{}
		for version, nodes := range kubeAPIServerVersions {
			verMsg = append(verMsg, fmt.Sprintf("%s on nodes %v", version, nodes))
		}
		klog.Warningf("Different API server versions in the cluster were discovered: %v. Please upgrade your control plane"+
			" nodes to the same version of Kubernetes", strings.Join(verMsg, ", "))
	}

	// Get the latest cluster version
	clusterVersion, err := getLatestClusterVersion(kubeAPIServerVersions)
	if err != nil {
		return upgrades, err
	}
	clusterVersionStr := clusterVersion.String()

	printer.Printf("[upgrade/versions] Cluster version: %s\n", clusterVersionStr)

	// Get current kubeadm CLI version
	kubeadmVersionStr, kubeadmVersion, err := versionGetterImpl.KubeadmVersion()
	if err != nil {
		return upgrades, err
	}
	printer.Printf("[upgrade/versions] kubeadm version: %s\n", kubeadmVersionStr)

	// Get and output the current latest stable version
	stableVersionStr, stableVersion, err := versionGetterImpl.VersionFromCILabel("stable", "stable version")
	if err != nil {
		klog.Warningf("[upgrade/versions] WARNING: %v\n", err)
		klog.Warningf("[upgrade/versions] WARNING: Falling back to current kubeadm version as latest stable version")
		stableVersionStr, stableVersion = kubeadmVersionStr, kubeadmVersion
	} else {
		printer.Printf("[upgrade/versions] Target version: %s\n", stableVersionStr)
	}

	// Get the kubelet versions in the cluster
	kubeletVersions, err := versionGetterImpl.KubeletVersions()
	if err != nil {
		return upgrades, err
	}

	// Get the kube-controller-manager versions in the cluster
	kubeControllerManagerVersions, err := versionGetterImpl.ComponentVersions(kubeadmconstants.KubeControllerManager)
	if err != nil {
		return upgrades, err
	}

	// Get the kube-scheduler versions in the cluster
	kubeSchedulerVersions, err := versionGetterImpl.ComponentVersions(kubeadmconstants.KubeScheduler)
	if err != nil {
		return upgrades, err
	}

	// Get the etcd versions in the cluster
	etcdVersions, err := versionGetterImpl.ComponentVersions(kubeadmconstants.Etcd)
	if err != nil {
		return upgrades, err
	}
	isExternalEtcd := len(etcdVersions) == 0

	dnsVersion, err := dns.DeployedDNSAddon(client)
	if err != nil {
		return nil, err
	}

	// Construct a descriptor for the current state of the world
	beforeState := ClusterState{
		KubeVersion:                   clusterVersionStr,
		DNSVersion:                    dnsVersion,
		KubeadmVersion:                kubeadmVersionStr,
		KubeAPIServerVersions:         kubeAPIServerVersions,
		KubeControllerManagerVersions: kubeControllerManagerVersions,
		KubeSchedulerVersions:         kubeSchedulerVersions,
		KubeletVersions:               kubeletVersions,
		EtcdVersions:                  etcdVersions,
	}

	// Do a "dumb guess" that a new minor upgrade is available just because the latest stable version is higher than the cluster version
	// This guess will be corrected once we know if there is a patch version available
	canDoMinorUpgrade := clusterVersion.LessThan(stableVersion)

	// A patch version doesn't exist if the cluster version is higher than or equal to the current stable version
	// in the case that a user is trying to upgrade from, let's say, v1.8.0-beta.2 to v1.8.0-rc.1 (given we support such upgrades experimentally)
	// a stable-1.8 branch doesn't exist yet. Hence this check.
	if patchVersionBranchExists(clusterVersion, stableVersion) {
		currentBranch := getBranchFromVersion(clusterVersionStr)
		versionLabel := fmt.Sprintf("stable-%s", currentBranch)
		description := fmt.Sprintf("version in the v%s series", currentBranch)

		// Get and output the latest patch version for the cluster branch
		patchVersionStr, patchVersion, err := versionGetterImpl.VersionFromCILabel(versionLabel, description)
		if err != nil {
			klog.Warningf("[upgrade/versions] WARNING: %v\n", err)
		} else {
			printer.Printf("[upgrade/versions] Latest %s: %s\n", description, patchVersionStr)

			// Check if a minor version upgrade is possible when a patch release exists
			// It's only possible if the latest patch version is higher than the current patch version
			// If that's the case, they must be on different branches => a newer minor version can be upgraded to
			canDoMinorUpgrade = minorUpgradePossibleWithPatchRelease(stableVersion, patchVersion)

			// If the cluster version is lower than the newest patch version, we should inform about the possible upgrade
			if patchUpgradePossible(clusterVersion, patchVersion) {

				// The kubeadm version has to be upgraded to the latest patch version
				newKubeadmVer := patchVersionStr
				if kubeadmVersion.AtLeast(patchVersion) {
					// In this case, the kubeadm CLI version is new enough. Don't display an update suggestion for kubeadm by making .NewKubeadmVersion equal .CurrentKubeadmVersion
					newKubeadmVer = kubeadmVersionStr
				}

				upgrades = append(upgrades, Upgrade{
					Description: description,
					Before:      beforeState,
					After: ClusterState{
						KubeVersion:    patchVersionStr,
						DNSVersion:     kubeadmconstants.CoreDNSVersion,
						KubeadmVersion: newKubeadmVer,
						EtcdVersion:    getSuggestedEtcdVersion(isExternalEtcd, patchVersionStr),
					},
				})
			}
		}
	}

	if canDoMinorUpgrade {
		upgrades = append(upgrades, Upgrade{
			Description: "stable version",
			Before:      beforeState,
			After: ClusterState{
				KubeVersion:    stableVersionStr,
				DNSVersion:     kubeadmconstants.CoreDNSVersion,
				KubeadmVersion: stableVersionStr,
				EtcdVersion:    getSuggestedEtcdVersion(isExternalEtcd, stableVersionStr),
			},
		})
	}

	if experimentalUpgradesAllowed || rcUpgradesAllowed {
		// dl.k8s.io/release/latest.txt is ALWAYS an alpha.X version
		// dl.k8s.io/release/latest-1.X.txt is first v1.X.0-alpha.0 -> v1.X.0-alpha.Y, then v1.X.0-beta.0 to v1.X.0-beta.Z, then v1.X.0-rc.1 to v1.X.0-rc.W.
		// After the v1.X.0 release, latest-1.X.txt is always a beta.0 version. Let's say the latest stable version on the v1.7 branch is v1.7.3, then the
		// latest-1.7 version is v1.7.4-beta.0

		// Worth noticing is that when the release-1.X branch is cut; there are two versions tagged: v1.X.0-beta.0 AND v1.(X+1).alpha.0
		// The v1.(X+1).alpha.0 is pretty much useless and should just be ignored, as more betas may be released that have more features than the initial v1.(X+1).alpha.0

		// So what we do below is getting the latest overall version, always an v1.X.0-alpha.Y version. Then we get latest-1.(X-1) version. This version may be anything
		// between v1.(X-1).0-beta.0 and v1.(X-1).Z-beta.0. At some point in time, latest-1.(X-1) will point to v1.(X-1).0-rc.1. Then we should show it.

		// The flow looks like this (with time on the X axis):
		// v1.8.0-alpha.1 -> v1.8.0-alpha.2 -> v1.8.0-alpha.3 | release-1.8 branch | v1.8.0-beta.0 -> v1.8.0-beta.1 -> v1.8.0-beta.2 -> v1.8.0-rc.1 -> v1.8.0 -> v1.8.1
		//                                                                           v1.9.0-alpha.0                                             -> v1.9.0-alpha.1 -> v1.9.0-alpha.2

		// Get and output the current latest unstable version
		latestVersionStr, latestVersion, err := versionGetterImpl.VersionFromCILabel("latest", "experimental version")
		if err != nil {
			return upgrades, err
		}
		_, _ = printer.Printf("[upgrade/versions] Latest %s: %s\n", "experimental version", latestVersionStr)

		minorUnstable := latestVersion.Components()[1]
		// Get and output the current latest unstable version
		previousBranch := fmt.Sprintf("latest-1.%d", minorUnstable-1)
		previousBranchLatestVersionStr, previousBranchLatestVersion, err := versionGetterImpl.VersionFromCILabel(previousBranch, "previous version")
		if err != nil {
			return upgrades, err
		}
		_, _ = printer.Printf("[upgrade/versions] Latest %s: %s\n", "previous version", previousBranchLatestVersionStr)

		// If that previous latest version is an RC, RCs are allowed and the cluster version is lower than the RC version, show the upgrade
		if rcUpgradesAllowed && rcUpgradePossible(clusterVersion, previousBranchLatestVersion) {
			upgrades = append(upgrades, Upgrade{
				Description: "release candidate version",
				Before:      beforeState,
				After: ClusterState{
					KubeVersion:    previousBranchLatestVersionStr,
					DNSVersion:     kubeadmconstants.CoreDNSVersion,
					KubeadmVersion: previousBranchLatestVersionStr,
					EtcdVersion:    getSuggestedEtcdVersion(isExternalEtcd, previousBranchLatestVersionStr),
				},
			})
		}

		// Show the possibility if experimental upgrades are allowed
		if experimentalUpgradesAllowed && clusterVersion.LessThan(latestVersion) {

			// Default to assume that the experimental version to show is the unstable one
			unstableKubeVersion := latestVersionStr

			// Ẃe should not display alpha.0. The previous branch's beta/rc versions are more relevant due how the kube branching process works.
			if latestVersion.PreRelease() == "alpha.0" {
				unstableKubeVersion = previousBranchLatestVersionStr
			}

			upgrades = append(upgrades, Upgrade{
				Description: "experimental version",
				Before:      beforeState,
				After: ClusterState{
					KubeVersion:    unstableKubeVersion,
					DNSVersion:     kubeadmconstants.CoreDNSVersion,
					KubeadmVersion: unstableKubeVersion,
					EtcdVersion:    getSuggestedEtcdVersion(isExternalEtcd, unstableKubeVersion),
				},
			})
		}
	}

	// Add a newline in the end of this output to leave some space to the next output section
	printer.Println()

	return upgrades, nil
}

func getBranchFromVersion(version string) string {
	v := versionutil.MustParseGeneric(version)
	return fmt.Sprintf("%d.%d", v.Major(), v.Minor())
}

func patchVersionBranchExists(clusterVersion, stableVersion *versionutil.Version) bool {
	return stableVersion.AtLeast(clusterVersion)
}

func patchUpgradePossible(clusterVersion, patchVersion *versionutil.Version) bool {
	return clusterVersion.LessThan(patchVersion)
}

func rcUpgradePossible(clusterVersion, previousBranchLatestVersion *versionutil.Version) bool {
	return strings.HasPrefix(previousBranchLatestVersion.PreRelease(), "rc") && clusterVersion.LessThan(previousBranchLatestVersion)
}

func minorUpgradePossibleWithPatchRelease(stableVersion, patchVersion *versionutil.Version) bool {
	return patchVersion.LessThan(stableVersion)
}

func getSuggestedEtcdVersion(isExternalEtcd bool, kubernetesVersion string) string {
	if isExternalEtcd {
		return ""
	}
	etcdVersion, warning, err := kubeadmconstants.EtcdSupportedVersion(kubeadmconstants.SupportedEtcdVersion, kubernetesVersion)
	if err != nil {
		klog.Warningf("[upgrade/versions] could not retrieve an etcd version for the target Kubernetes version: %v", err)
		return "N/A"
	}
	if warning != nil {
		klog.V(1).Infof("[upgrade/versions] WARNING: %v", warning)
	}
	return etcdVersion.String()
}

func getLatestClusterVersion(kubeAPIServerVersions map[string][]string) (*versionutil.Version, error) {
	var latestVersion *versionutil.Version
	for versionStr, nodes := range kubeAPIServerVersions {
		ver, err := versionutil.ParseSemantic(versionStr)
		if err != nil {
			return nil, fmt.Errorf("couldn't parse kube-apiserver version %s from nodes %v", versionStr, nodes)
		}
		if latestVersion == nil || ver.AtLeast(latestVersion) {
			latestVersion = ver
		}
	}

	return latestVersion, nil
}
