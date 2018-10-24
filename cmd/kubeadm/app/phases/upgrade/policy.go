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

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const (
	// MaximumAllowedMinorVersionUpgradeSkew describes how many minor versions kubeadm can upgrade the control plane version in one go
	MaximumAllowedMinorVersionUpgradeSkew = 1

	// MaximumAllowedMinorVersionDowngradeSkew describes how many minor versions kubeadm can upgrade the control plane version in one go
	MaximumAllowedMinorVersionDowngradeSkew = 1

	// MaximumAllowedMinorVersionKubeletSkew describes how many minor versions the control plane version and the kubelet can skew in a kubeadm cluster
	MaximumAllowedMinorVersionKubeletSkew = 1
)

// VersionSkewPolicyErrors describes version skew errors that might be seen during the validation process in EnforceVersionPolicies
type VersionSkewPolicyErrors struct {
	Mandatory []error
	Skippable []error
}

// EnforceVersionPolicies enforces that the proposed new version is compatible with all the different version skew policies
func EnforceVersionPolicies(versionGetter VersionGetter, newK8sVersionStr string, newK8sVersion *version.Version, allowExperimentalUpgrades, allowRCUpgrades bool) *VersionSkewPolicyErrors {

	skewErrors := &VersionSkewPolicyErrors{
		Mandatory: []error{},
		Skippable: []error{},
	}

	clusterVersionStr, clusterVersion, err := versionGetter.ClusterVersion()
	if err != nil {
		// This case can't be forced: kubeadm has to be able to lookup cluster version for upgrades to work
		skewErrors.Mandatory = append(skewErrors.Mandatory, fmt.Errorf("Unable to fetch cluster version: %v", err))
		return skewErrors
	}

	kubeadmVersionStr, kubeadmVersion, err := versionGetter.KubeadmVersion()
	if err != nil {
		// This case can't be forced: kubeadm has to be able to lookup its version for upgrades to work
		skewErrors.Mandatory = append(skewErrors.Mandatory, fmt.Errorf("Unable to fetch kubeadm version: %v", err))
		return skewErrors
	}

	kubeletVersions, err := versionGetter.KubeletVersions()
	if err != nil {
		// This is a non-critical error; continue although kubeadm couldn't look this up
		skewErrors.Skippable = append(skewErrors.Skippable, fmt.Errorf("Unable to fetch kubelet version: %v", err))
	}

	// Make sure the new version is a supported version (higher than the minimum one supported)
	if constants.MinimumControlPlaneVersion.AtLeast(newK8sVersion) {
		// This must not happen, kubeadm always supports a minimum version; and we can't go below that
		skewErrors.Mandatory = append(skewErrors.Mandatory, fmt.Errorf("Specified version to upgrade to %q is equal to or lower than the minimum supported version %q. Please specify a higher version to upgrade to", newK8sVersionStr, clusterVersionStr))
	}

	// kubeadm doesn't support upgrades between two minor versions; e.g. a v1.7 -> v1.9 upgrade is not supported right away
	if newK8sVersion.Minor() > clusterVersion.Minor()+MaximumAllowedMinorVersionUpgradeSkew {
		tooLargeUpgradeSkewErr := fmt.Errorf("Specified version to upgrade to %q is too high; kubeadm can upgrade only %d minor version at a time", newK8sVersionStr, MaximumAllowedMinorVersionUpgradeSkew)
		// If the version that we're about to upgrade to is a released version, we should fully enforce this policy
		// If the version is a CI/dev/experimental version, it's okay to jump two minor version steps, but then require the -f flag
		if len(newK8sVersion.PreRelease()) == 0 {
			skewErrors.Mandatory = append(skewErrors.Mandatory, tooLargeUpgradeSkewErr)
		} else {
			skewErrors.Skippable = append(skewErrors.Skippable, tooLargeUpgradeSkewErr)
		}
	}

	// kubeadm doesn't support downgrades between two minor versions; e.g. a v1.9 -> v1.7 downgrade is not supported right away
	if newK8sVersion.Minor() < clusterVersion.Minor()-MaximumAllowedMinorVersionDowngradeSkew {
		tooLargeDowngradeSkewErr := fmt.Errorf("Specified version to downgrade to %q is too low; kubeadm can downgrade only %d minor version at a time", newK8sVersionStr, MaximumAllowedMinorVersionDowngradeSkew)
		// If the version that we're about to downgrade to is a released version, we should fully enforce this policy
		// If the version is a CI/dev/experimental version, it's okay to jump two minor version steps, but then require the -f flag
		if len(newK8sVersion.PreRelease()) == 0 {
			skewErrors.Mandatory = append(skewErrors.Mandatory, tooLargeDowngradeSkewErr)
		} else {
			skewErrors.Skippable = append(skewErrors.Skippable, tooLargeDowngradeSkewErr)
		}
	}

	// If the kubeadm version is lower than what we want to upgrade to; error
	if kubeadmVersion.LessThan(newK8sVersion) {
		if newK8sVersion.Minor() > kubeadmVersion.Minor() {
			tooLargeKubeadmSkew := fmt.Errorf("Specified version to upgrade to %q is at least one minor release higher than the kubeadm minor release (%d > %d). Such an upgrade is not supported", newK8sVersionStr, newK8sVersion.Minor(), kubeadmVersion.Minor())
			// This is unsupported; kubeadm has no idea how it should handle a newer minor release than itself
			// If the version is a CI/dev/experimental version though, lower the severity of this check, but then require the -f flag
			if len(newK8sVersion.PreRelease()) == 0 {
				skewErrors.Mandatory = append(skewErrors.Mandatory, tooLargeKubeadmSkew)
			} else {
				skewErrors.Skippable = append(skewErrors.Skippable, tooLargeKubeadmSkew)
			}
		} else {
			// Upgrading to a higher patch version than kubeadm is ok if the user specifies --force. Not recommended, but possible.
			skewErrors.Skippable = append(skewErrors.Skippable, fmt.Errorf("Specified version to upgrade to %q is higher than the kubeadm version %q. Upgrade kubeadm first using the tool you used to install kubeadm", newK8sVersionStr, kubeadmVersionStr))
		}
	}

	if kubeadmVersion.Major() > newK8sVersion.Major() ||
		kubeadmVersion.Minor() > newK8sVersion.Minor() {
		skewErrors.Skippable = append(skewErrors.Skippable, fmt.Errorf("Kubeadm version %s can only be used to upgrade to Kubernetes version %d.%d", kubeadmVersionStr, kubeadmVersion.Major(), kubeadmVersion.Minor()))
	}

	// Detect if the version is unstable and the user didn't allow that
	if err = detectUnstableVersionError(newK8sVersion, newK8sVersionStr, allowExperimentalUpgrades, allowRCUpgrades); err != nil {
		skewErrors.Skippable = append(skewErrors.Skippable, err)
	}

	// Detect if there are too old kubelets in the cluster
	// Check for nil here since this is the only case where kubeletVersions can be nil; if KubeletVersions() returned an error
	// However, it's okay to skip that check
	if kubeletVersions != nil {
		if err = detectTooOldKubelets(newK8sVersion, kubeletVersions); err != nil {
			skewErrors.Skippable = append(skewErrors.Skippable, err)
		}
	}

	// If we did not see any errors, return nil
	if len(skewErrors.Skippable) == 0 && len(skewErrors.Mandatory) == 0 {
		return nil
	}

	// Uh oh, we encountered one or more errors, return them
	return skewErrors
}

// detectUnstableVersionError is a helper function for detecting if the unstable version (if specified) is allowed to be used
func detectUnstableVersionError(newK8sVersion *version.Version, newK8sVersionStr string, allowExperimentalUpgrades, allowRCUpgrades bool) error {
	// Short-circuit quickly if this is not an unstable version
	if len(newK8sVersion.PreRelease()) == 0 {
		return nil
	}
	// If the user has specified that unstable versions are fine, then no error should be returned
	if allowExperimentalUpgrades {
		return nil
	}
	// If this is a release candidate and we allow such ones, everything's fine
	if strings.HasPrefix(newK8sVersion.PreRelease(), "rc") && allowRCUpgrades {
		return nil
	}

	return fmt.Errorf("Specified version to upgrade to %q is an unstable version and such upgrades weren't allowed via setting the --allow-*-upgrades flags", newK8sVersionStr)
}

// detectTooOldKubelets errors out if the kubelet versions are so old that an unsupported skew would happen if the cluster was upgraded
func detectTooOldKubelets(newK8sVersion *version.Version, kubeletVersions map[string]uint16) error {
	tooOldKubeletVersions := []string{}
	for versionStr := range kubeletVersions {

		kubeletVersion, err := version.ParseSemantic(versionStr)
		if err != nil {
			return fmt.Errorf("couldn't parse kubelet version %s", versionStr)
		}

		if newK8sVersion.Minor() > kubeletVersion.Minor()+MaximumAllowedMinorVersionKubeletSkew {
			tooOldKubeletVersions = append(tooOldKubeletVersions, versionStr)
		}
	}
	if len(tooOldKubeletVersions) == 0 {
		return nil
	}

	return fmt.Errorf("There are kubelets in this cluster that are too old that have these versions %v", tooOldKubeletVersions)
}
