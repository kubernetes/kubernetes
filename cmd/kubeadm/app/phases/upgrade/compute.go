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

// ClusterState describes the state of certain versions for a cluster
type ClusterState struct {
	// KubeVersion describes the version of the Kubernetes API Server, Controller Manager, Scheduler and Proxy.
	KubeVersion string
	// DNSVersion describes the version of the kube-dns images used and manifest version
	DNSVersion string
	// KubeadmVersion describes the version of the kubeadm CLI
	KubeadmVersion string
	// KubeletVersions is a map with a version number linked to the amount of kubelets running that version in the cluster
	KubeletVersions map[string]uint16
}

// GetAvailableUpgrades fetches all versions from the specified VersionGetter and computes which
// kinds of upgrades can be performed
func GetAvailableUpgrades(_ VersionGetter, _, _ bool) ([]Upgrade, error) {
	fmt.Println("[upgrade] Fetching available versions to upgrade to:")
	return []Upgrade{}, nil
}
