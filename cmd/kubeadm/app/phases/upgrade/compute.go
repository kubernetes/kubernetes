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

	clientset "k8s.io/client-go/kubernetes"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	versionutil "k8s.io/kubernetes/pkg/util/version"
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

// CanUpgradeEtcd returns whether an upgrade of etcd is possible
func (u *Upgrade) CanUpgradeEtcd() bool {
	return u.Before.EtcdVersion != u.After.EtcdVersion
}

// ActiveDNSAddon returns the version of CoreDNS or kube-dns
func ActiveDNSAddon(featureGates map[string]bool) string {
	if features.Enabled(featureGates, features.CoreDNS) {
		return kubeadmconstants.CoreDNS
	}
	return kubeadmconstants.KubeDNS
}

// ClusterState describes the state of certain versions for a cluster
type ClusterState struct {
	// KubeVersion describes the version of the Kubernetes API Server, Controller Manager, Scheduler and Proxy.
	KubeVersion string
	// DNSType
	DNSType string
	// DNSVersion describes the version of the kube-dns images used and manifest version
	DNSVersion string
	// KubeadmVersion describes the version of the kubeadm CLI
	KubeadmVersion string
	// KubeletVersions is a map with a version number linked to the amount of kubelets running that version in the cluster
	KubeletVersions map[string]uint16
	// EtcdVersion represents the version of etcd used in the cluster
	EtcdVersion string
}

// GetAvailableUpgrades fetches all versions from the specified VersionGetter and computes which
// kinds of upgrades can be performed
func GetAvailableUpgrades(versionGetterImpl VersionGetter, experimentalUpgradesAllowed, rcUpgradesAllowed bool, etcdClient etcdutil.ClusterInterrogator, featureGates map[string]bool, client clientset.Interface) ([]Upgrade, error) {
	fmt.Println("[upgrade] Fetching available versions to upgrade to")

	// Collect the upgrades kubeadm can do in this list
	upgrades := []Upgrade{}

	// Get the cluster version
	clusterVersionStr, clusterVersion, err := versionGetterImpl.ClusterVersion()
	if err != nil {
		return upgrades, err
	}

	// Get current kubeadm CLI version
	kubeadmVersionStr, kubeadmVersion, err := versionGetterImpl.KubeadmVersion()
	if err != nil {
		return upgrades, err
	}

	// Get and output the current latest stable version
	stableVersionStr, stableVersion, err := versionGetterImpl.VersionFromCILabel("stable", "stable version")
	if err != nil {
		fmt.Printf("[upgrade/versions] WARNING: %v\n", err)
		fmt.Println("[upgrade/versions] WARNING: Falling back to current kubeadm version as latest stable version")
		stableVersionStr, stableVersion = kubeadmVersionStr, kubeadmVersion
	}

	// Get the kubelet versions in the cluster
	kubeletVersions, err := versionGetterImpl.KubeletVersions()
	if err != nil {
		return upgrades, err
	}

	// Get current etcd version
	etcdVersion, err := etcdClient.GetVersion()
	if err != nil {
		return upgrades, err
	}

	dnsType, dnsVersion, err := dns.DeployedDNSAddon(client)
	if err != nil {
		return nil, err
	}

	// Construct a descriptor for the current state of the world
	beforeState := ClusterState{
		KubeVersion:     clusterVersionStr,
		DNSType:         dnsType,
		DNSVersion:      dnsVersion,
		KubeadmVersion:  kubeadmVersionStr,
		KubeletVersions: kubeletVersions,
		EtcdVersion:     etcdVersion,
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
			fmt.Printf("[upgrade/versions] WARNING: %v\n", err)
		} else {
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
						DNSType:        ActiveDNSAddon(featureGates),
						DNSVersion:     kubeadmconstants.GetDNSVersion(ActiveDNSAddon(featureGates)),
						KubeadmVersion: newKubeadmVer,
						EtcdVersion:    getSuggestedEtcdVersion(patchVersionStr),
						// KubeletVersions is unset here as it is not used anywhere in .After
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
				DNSType:        ActiveDNSAddon(featureGates),
				DNSVersion:     kubeadmconstants.GetDNSVersion(ActiveDNSAddon(featureGates)),
				KubeadmVersion: stableVersionStr,
				EtcdVersion:    getSuggestedEtcdVersion(stableVersionStr),
				// KubeletVersions is unset here as it is not used anywhere in .After
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

		minorUnstable := latestVersion.Components()[1]
		// Get and output the current latest unstable version
		previousBranch := fmt.Sprintf("latest-1.%d", minorUnstable-1)
		previousBranchLatestVersionStr, previousBranchLatestVersion, err := versionGetterImpl.VersionFromCILabel(previousBranch, "")
		if err != nil {
			return upgrades, err
		}

		// If that previous latest version is an RC, RCs are allowed and the cluster version is lower than the RC version, show the upgrade
		if rcUpgradesAllowed && rcUpgradePossible(clusterVersion, previousBranchLatestVersion) {
			upgrades = append(upgrades, Upgrade{
				Description: "release candidate version",
				Before:      beforeState,
				After: ClusterState{
					KubeVersion:    previousBranchLatestVersionStr,
					DNSType:        ActiveDNSAddon(featureGates),
					DNSVersion:     kubeadmconstants.GetDNSVersion(ActiveDNSAddon(featureGates)),
					KubeadmVersion: previousBranchLatestVersionStr,
					EtcdVersion:    getSuggestedEtcdVersion(previousBranchLatestVersionStr),
					// KubeletVersions is unset here as it is not used anywhere in .After
				},
			})
		}

		// Show the possibility if experimental upgrades are allowed
		if experimentalUpgradesAllowed && clusterVersion.LessThan(latestVersion) {

			// Default to assume that the experimental version to show is the unstable one
			unstableKubeVersion := latestVersionStr
			unstableKubeDNSVersion := kubeadmconstants.GetDNSVersion(ActiveDNSAddon(featureGates))

			// Ẃe should not display alpha.0. The previous branch's beta/rc versions are more relevant due how the kube branching process works.
			if latestVersion.PreRelease() == "alpha.0" {
				unstableKubeVersion = previousBranchLatestVersionStr
				unstableKubeDNSVersion = kubeadmconstants.GetDNSVersion(ActiveDNSAddon(featureGates))
			}

			upgrades = append(upgrades, Upgrade{
				Description: "experimental version",
				Before:      beforeState,
				After: ClusterState{
					KubeVersion:    unstableKubeVersion,
					DNSType:        ActiveDNSAddon(featureGates),
					DNSVersion:     unstableKubeDNSVersion,
					KubeadmVersion: unstableKubeVersion,
					EtcdVersion:    getSuggestedEtcdVersion(unstableKubeVersion),
					// KubeletVersions is unset here as it is not used anywhere in .After
				},
			})
		}
	}

	// Add a newline in the end of this output to leave some space to the next output section
	fmt.Println("")

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

func getSuggestedEtcdVersion(kubernetesVersion string) string {
	etcdVersion, err := kubeadmconstants.EtcdSupportedVersion(kubernetesVersion)
	if err != nil {
		fmt.Printf("[upgrade/versions] WARNING: No recommended etcd for requested kubernetes version (%s)\n", kubernetesVersion)
		return "N/A"
	}
	return etcdVersion.String()
}
