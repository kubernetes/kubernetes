/*
Copyright 2020 The Kubernetes Authors.

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

package phases

import (
	"fmt"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

// RunPreflightChecksPhase runs all the preflight checks for upgrades
func RunPreflightChecksPhase(data ApplyData) error {

	client := data.Client()

	err := upgrade.EnforceRequirements(client, data.Cfg(), data.IgnorePreflightErrors(), data.FeatureGates(), data.DryRun())
	if err != nil {
		return err
	}

	// Validate requested and validate actual version
	klog.V(1).Infoln("[upgrade/apply] validating requested and actual version")
	if err := configutil.NormalizeKubernetesVersion(&data.Cfg().ClusterConfiguration); err != nil {
		return err
	}

	newK8sVersion, err := version.ParseSemantic(data.Cfg().KubernetesVersion)
	if err != nil {
		return errors.Errorf("unable to parse normalized version %q as a semantic version", data.Cfg().KubernetesVersion)
	}

	if err := features.ValidateVersion(features.InitFeatureGates, data.Cfg().FeatureGates, data.Cfg().KubernetesVersion); err != nil {
		return err
	}

	// Enforce the version skew policies
	klog.V(1).Infoln("[upgrade/version] enforcing version skew policies")
	if err := EnforceUpgradeVersionPolicies(data.Cfg().KubernetesVersion, newK8sVersion, data.AllowExperimentalUpgrades(), data.AllowRCUpgrades(), data.Force(), data.VersionGetter()); err != nil {
		return errors.Wrap(err, "[upgrade/version] FATAL")
	}

	// Use a prepuller implementation based on creating DaemonSets
	// and block until all DaemonSets are ready; then we know for sure that all control plane images are cached locally
	klog.V(1).Infoln("[upgrade/apply] creating prepuller")
	prepuller := upgrade.NewDaemonSetPrepuller(client, data.Waiter(), &data.Cfg().ClusterConfiguration)
	componentsToPrepull := constants.ControlPlaneComponents
	if data.Cfg().Etcd.External == nil && data.UpgradeETCD() {
		componentsToPrepull = append(componentsToPrepull, constants.Etcd)
	}
	if err := upgrade.PrepullImagesInParallel(prepuller, data.ImagePullTimeout(), componentsToPrepull); err != nil {
		return errors.Wrap(err, "[upgrade/prepull] Failed prepulled the images for the control plane components error")
	}

	return nil
}

// EnforceUpgradeVersionPolicies makes sure that the version the user specified is valid to upgrade to
// There are both fatal and skippable (with --force) errors
func EnforceUpgradeVersionPolicies(newK8sVersionStr string, newK8sVersion *version.Version, allowExperimentalUpgrades bool, allowRCUpgrades bool, force bool, versionGetter upgrade.VersionGetter) error {
	fmt.Printf("[upgrade/version] You have chosen to change the cluster version to %q\n", newK8sVersionStr)

	versionSkewErrs := upgrade.EnforceVersionPolicies(versionGetter, newK8sVersionStr, newK8sVersion, allowExperimentalUpgrades, allowRCUpgrades)
	if versionSkewErrs != nil {

		if len(versionSkewErrs.Mandatory) > 0 {
			return errors.Errorf("the --version argument is invalid due to these fatal errors:\n\n%v\nPlease fix the misalignments highlighted above and try upgrading again",
				kubeadmutil.FormatErrMsg(versionSkewErrs.Mandatory))
		}

		if len(versionSkewErrs.Skippable) > 0 {
			// Return the error if the user hasn't specified the --force flag
			if !force {
				return errors.Errorf("the --version argument is invalid due to these errors:\n\n%v\nCan be bypassed if you pass the --force flag",
					kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
			}
			// Soft errors found, but --force was specified
			fmt.Printf("[upgrade/version] Found %d potential version compatibility errors but skipping since the --force flag is set: \n\n%v", len(versionSkewErrs.Skippable), kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
		}
	}
	return nil
}
